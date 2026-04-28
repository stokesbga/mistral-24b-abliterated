"""Stage 1 of abliteration: compute the per-layer 'refusal direction'.

We feed N harmful and N harmless instructions through the model (with the
chat template applied), capture the residual-stream activation at the LAST
token of every layer, and store the mean-difference vector

    r_l = mean_h(harmful) - mean_h(harmless)

for every layer l.  We also pick a 'best' layer using the heuristic of
maximum |r_l|/|h_l| relative magnitude — empirically a good proxy for the
layer where the direction has the strongest causal effect on refusal.

Output: ``refusal_direction.pt`` — a dict with ``directions`` (list of
length n_layers), ``best_layer`` (int), and metadata.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)

from prompts import (
    HARMFUL_INSTRUCTIONS,
    HARMLESS_INSTRUCTIONS,
    encode_user_prompt,
    load_mistral_tokenizer,
)

DEFAULT_MODEL_DIR = "/home/alex/ablated/models/mistral-small-3.2-24b"
DEFAULT_OUT = "/home/alex/ablated/refusal_direction.pt"


# --------------------------------------------------------------------------- #
# Activation capture
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_last_token_residuals(
    text_model: torch.nn.Module,
    mistral_tok,
    instructions: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Return a tensor of shape [n_layers + 1, n_prompts, hidden_size] holding
    the residual-stream activation at the LAST input token, per layer, per
    prompt.  Layer 0 is the embedding output; layer i (i>=1) is the output of
    decoder block i-1 (i.e. the input to block i).  This convention matches
    HF's ``output_hidden_states``.

    We call the *text* sub-model directly (bypassing any multimodal wrapper)
    so we don't have to plumb pixel_values through.
    """
    all_acts: list[torch.Tensor] = []  # one [n_layers+1, hidden] per prompt
    for instr in tqdm(instructions, desc="forward passes"):
        # The chat-completion encoder ends the sequence with [/INST] — exactly
        # the position where refusal would be emitted by the assistant.
        input_ids = encode_user_prompt(mistral_tok, instr).to(device)

        out = text_model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # hidden_states is a tuple of length n_layers + 1, each [1, T, hidden]
        # We take the LAST token of each.  Move to CPU/float32 immediately to
        # avoid keeping bf16 GPU memory pinned across iterations.
        per_layer = torch.stack(
            [hs[0, -1, :].detach().to("cpu", dtype=torch.float32) for hs in out.hidden_states],
            dim=0,
        )
        all_acts.append(per_layer)
        del out
    # → [n_layers+1, n_prompts, hidden]
    return torch.stack(all_acts, dim=1)


def get_text_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the text decoder backbone regardless of multimodal wrapping."""
    # Mistral3ForConditionalGeneration:
    #   .model (Mistral3Model) -> .language_model (MistralModel)
    inner = getattr(model, "model", model)
    if hasattr(inner, "language_model"):
        return inner.language_model
    # Plain MistralForCausalLM: .model is itself the MistralModel.
    return inner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    print(f"Loading mistral-common tokenizer from {args.model_dir}/tekken.json")
    mistral_tok = load_mistral_tokenizer(args.model_dir)

    cfg = AutoConfig.from_pretrained(args.model_dir)
    is_multimodal = getattr(cfg, "model_type", "") == "mistral3"
    text_cfg = getattr(cfg, "text_config", cfg)

    print(f"Loading model from {args.model_dir} (bf16, device_map=auto, "
          f"multimodal={is_multimodal})")
    Loader = AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
    model = Loader.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()

    # Resolve the device of the embedding so we put input tokens there.
    embed_device = next(model.get_input_embeddings().parameters()).device
    print(f"Embedding device: {embed_device}")

    n_layers = text_cfg.num_hidden_layers
    hidden_size = text_cfg.hidden_size
    print(f"Model has {n_layers} layers, hidden_size={hidden_size}")

    text_model = get_text_model(model)

    print("\nCollecting harmful activations...")
    harmful_acts = collect_last_token_residuals(
        text_model, mistral_tok, HARMFUL_INSTRUCTIONS, embed_device
    )
    print(f"  → shape {tuple(harmful_acts.shape)}")

    print("\nCollecting harmless activations...")
    harmless_acts = collect_last_token_residuals(
        text_model, mistral_tok, HARMLESS_INSTRUCTIONS, embed_device
    )
    print(f"  → shape {tuple(harmless_acts.shape)}")

    # Per-layer mean diff = refusal direction (unnormalised)
    harmful_mean = harmful_acts.mean(dim=1)   # [n_layers+1, hidden]
    harmless_mean = harmless_acts.mean(dim=1) # [n_layers+1, hidden]
    diff = harmful_mean - harmless_mean       # [n_layers+1, hidden]

    # Score each layer: ratio of diff norm to harmful-activation norm (a rough
    # proxy for "how much of the residual stream this direction explains").
    diff_norms = diff.norm(dim=-1)
    harmful_norms = harmful_mean.norm(dim=-1)
    rel_strength = diff_norms / (harmful_norms + 1e-8)

    # We exclude the very first (embedding) and last (post-final-norm) layers
    # because they tend to be poor injection sites in practice.
    candidate_start = 1
    candidate_end = n_layers  # inclusive of layer index n_layers-1 + the +1
    best_layer = int(
        candidate_start + rel_strength[candidate_start:candidate_end].argmax().item()
    )
    print(f"\nPer-layer |refusal_dir| (relative): "
          f"{[f'{v:.3f}' for v in rel_strength.tolist()]}")
    print(f"Selected best layer: {best_layer} "
          f"(|diff|={diff_norms[best_layer]:.3f}, rel={rel_strength[best_layer]:.3f})")

    # Normalise the chosen direction
    chosen = diff[best_layer]
    chosen = chosen / chosen.norm()

    payload = {
        "all_directions": diff,            # [n_layers+1, hidden] unnormalised
        "best_direction": chosen,          # [hidden] unit vector
        "best_layer": best_layer,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "model_dir": args.model_dir,
        "n_harmful": len(HARMFUL_INSTRUCTIONS),
        "n_harmless": len(HARMLESS_INSTRUCTIONS),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"\nSaved refusal direction to {args.out}")


if __name__ == "__main__":
    main()
