"""Stage 2: orthogonalise model weights against the refusal direction.

For every weight matrix that *writes into* the residual stream, replace it
with its projection onto the hyperplane orthogonal to the refusal direction
``r`` (a unit vector in hidden_size dims).  Equivalently::

    W_new = (I - r r^T) W      for matrices whose output dim == hidden_size
    W_new = W (I - r r^T)      for the embedding matrix [vocab, hidden]

Targets in a Mistral-style decoder:
  - ``model.embed_tokens.weight``                  (vocab × hidden)
  - ``model.layers.{i}.self_attn.o_proj.weight``   (hidden × n_heads*head_dim)
  - ``model.layers.{i}.mlp.down_proj.weight``      (hidden × intermediate)

Bias vectors of those matrices, when present, get the same treatment::

    b_new = (I - r r^T) b

Everything happens on CPU in bf16 — no GPU needed for this stage, and the
24B model fits comfortably in 251 GB RAM.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)

DEFAULT_MODEL_DIR = "/home/alex/ablated/models/mistral-small-3.2-24b"
DEFAULT_DIRECTION = "/home/alex/ablated/refusal_direction.pt"
DEFAULT_OUT_DIR = "/home/alex/ablated/models/mistral-small-3.2-24b-abliterated"


def project_out_(weight: torch.Tensor, r: torch.Tensor, *, axis: str) -> None:
    """In-place: subtract the rank-1 component of ``weight`` along ``r``.

    axis='out'   → weight has shape [hidden, K]; project rows onto r and
                  subtract: W -= r (r^T W).
    axis='embed' → weight has shape [V, hidden]; project columns onto r and
                  subtract: W -= (W r) r^T.
    """
    orig_dtype = weight.dtype
    w32 = weight.to(torch.float32)
    r32 = r.to(torch.float32)
    if axis == "out":
        # r: [hidden].  rTW: [K].  r ⊗ rTW: [hidden, K]
        rTW = r32 @ w32                # [K]
        w32.sub_(torch.outer(r32, rTW))
    elif axis == "embed":
        Wr = w32 @ r32                 # [V]
        w32.sub_(torch.outer(Wr, r32))
    else:
        raise ValueError(axis)
    weight.copy_(w32.to(orig_dtype))


def project_out_bias_(bias: torch.Tensor, r: torch.Tensor) -> None:
    orig_dtype = bias.dtype
    b32 = bias.to(torch.float32)
    r32 = r.to(torch.float32)
    coeff = (r32 @ b32).item()
    b32.sub_(r32 * coeff)
    bias.copy_(b32.to(orig_dtype))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--direction", default=DEFAULT_DIRECTION)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    print(f"Loading direction from {args.direction}")
    payload = torch.load(args.direction, map_location="cpu", weights_only=False)
    r: torch.Tensor = payload["best_direction"].to(torch.float32)
    r = r / r.norm()
    print(f"  best_layer={payload['best_layer']}, hidden={r.numel()}")

    cfg = AutoConfig.from_pretrained(args.model_dir)
    is_multimodal = getattr(cfg, "model_type", "") == "mistral3"
    Loader = AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM

    print(f"Loading model from {args.model_dir} on CPU (bf16, "
          f"multimodal={is_multimodal})")
    model = Loader.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    # Find the text-decoder backbone.
    #   Mistral3ForConditionalGeneration: model.model.language_model
    #   plain MistralForCausalLM:         model.model
    inner = getattr(model, "model", model)
    if hasattr(inner, "language_model"):
        base = inner.language_model
    else:
        base = inner

    n_layers = len(base.layers)
    print(f"Orthogonalising {n_layers} text-decoder layers + embedding...")

    embed = base.embed_tokens
    project_out_(embed.weight.data, r, axis="embed")

    for i, layer in enumerate(tqdm(base.layers, desc="layers")):
        # Attention output projection
        o_proj = layer.self_attn.o_proj
        project_out_(o_proj.weight.data, r, axis="out")
        if getattr(o_proj, "bias", None) is not None:
            project_out_bias_(o_proj.bias.data, r)

        # MLP down projection
        down = layer.mlp.down_proj
        project_out_(down.weight.data, r, axis="out")
        if getattr(down, "bias", None) is not None:
            project_out_bias_(down.bias.data, r)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving abliterated model to {out_dir}")
    model.save_pretrained(
        out_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    # Copy auxiliary files: tekken.json (the canonical mistral-common
    # tokenizer), params.json, generation_config.json, system prompt, etc.
    # The HF AutoTokenizer auto-converted from tekken.json is broken for
    # this model (out-of-bounds IDs), so we deliberately do NOT save
    # tokenizer.json — downstream code should use mistral-common against
    # tekken.json, exactly like with the original repo.
    src = Path(args.model_dir)
    for name in [
        "tekken.json",
        "params.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "SYSTEM_PROMPT.txt",
        "README.md",
    ]:
        f = src / name
        if f.exists() and not (out_dir / name).exists():
            shutil.copy2(f, out_dir / name)

    print("Done.")


if __name__ == "__main__":
    main()
