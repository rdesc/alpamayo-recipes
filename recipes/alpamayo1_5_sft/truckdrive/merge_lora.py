# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Merge a LoRA Stage-1 checkpoint into a plain full-weight checkpoint.

A LoRA training run's Trainer checkpoint stores adapted projections as
``...q_proj.base_layer.weight`` plus separate ``lora_A``/``lora_B`` tensors. Any
consumer that loads weights by NAME -- notably ``load_alpamayo1_vlm``, used by the
Stage-2 expert via ``model.stage1_vlm_checkpoint_path`` -- will not match those
keys. Before the load guard existed this failed silently, leaving every adapted
projection randomly initialized.

This script folds ``delta_W = B @ A * (alpha / r)`` into each base weight and
re-saves with the ORIGINAL key names, producing a checkpoint dir that is
indistinguishable from a full fine-tune's output.

Usage:
    a1_5_sft/bin/python -m alpamayo1_5_sft.truckdrive.merge_lora \
        --checkpoint /path/to/output_truckdrive_lora/checkpoint-2060 \
        --output /path/to/merged_checkpoint

The output dir can then be passed to Stage 2 as
``model.stage1_vlm_checkpoint_path``, or to ``evaluate_hf`` as
``evaluate.eval_ckpt`` (though eval handles unmerged adapters directly).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file

from alpamayo1_5_sft.models.lora import (
    ADAPTER_DIRNAME,
    find_adapter_dir,
    load_lora_settings,
)

# Files copied verbatim from the source checkpoint so the merged dir is a drop-in
# replacement (config.json is required by from_alpamayo_checkpoint).
_PASSTHROUGH_FILES = ("config.json", "generation_config.json", "run_config.yaml")


def _scaling(r: int, alpha: int) -> float:
    return alpha / r


def merge_checkpoint(checkpoint: Path, output: Path) -> None:
    adapter_dir = find_adapter_dir(str(checkpoint))
    if adapter_dir is None:
        raise SystemExit(
            f"{checkpoint} has no {ADAPTER_DIRNAME}/ subdir -- it is not a LoRA checkpoint."
        )
    settings = load_lora_settings(adapter_dir)
    scaling = _scaling(settings.r, settings.alpha)
    print(f"LoRA settings: r={settings.r} alpha={settings.alpha} -> scaling={scaling:g}")

    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"Missing {index_path}; expected a Trainer checkpoint dir.")
    with index_path.open("r", encoding="utf-8") as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]

    # Load every shard once. The 10B checkpoint is ~20GB in bf16; this runs on CPU.
    shards: dict[str, dict[str, torch.Tensor]] = {}
    for shard_name in sorted(set(weight_map.values())):
        print(f"Loading shard {shard_name} ...")
        shards[shard_name] = load_safetensors_file(str(checkpoint / shard_name), device="cpu")

    full_sd: dict[str, torch.Tensor] = {}
    for shard in shards.values():
        full_sd.update(shard)

    # Group adapter tensors by the module they belong to.
    # e.g. vlm...q_proj.lora_A.default.weight -> stem "vlm...q_proj"
    stems: set[str] = set()
    for key in full_sd:
        if ".lora_A." in key:
            stems.add(key.split(".lora_A.")[0])

    if not stems:
        raise SystemExit(f"No lora_A tensors found in {checkpoint}; nothing to merge.")

    merged_sd: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    n_merged = 0

    for stem in sorted(stems):
        base_key = f"{stem}.base_layer.weight"
        a_key = f"{stem}.lora_A.default.weight"
        b_key = f"{stem}.lora_B.default.weight"
        missing = [k for k in (base_key, a_key, b_key) if k not in full_sd]
        if missing:
            raise SystemExit(f"Adapted module {stem} is missing tensors: {missing}")

        base = full_sd[base_key]
        lora_a = full_sd[a_key]
        lora_b = full_sd[b_key]

        # delta_W = B @ A * scaling, computed in fp32 for numerical headroom then
        # cast back to the base weight's dtype (bf16).
        delta = (lora_b.float() @ lora_a.float()) * scaling
        if delta.shape != base.shape:
            raise SystemExit(
                f"Shape mismatch merging {stem}: delta {tuple(delta.shape)} vs "
                f"base {tuple(base.shape)}"
            )
        merged_sd[f"{stem}.weight"] = (base.float() + delta).to(base.dtype).contiguous()

        consumed.update({base_key, a_key, b_key})
        n_merged += 1

        # Adapted Linears may also carry a bias, which LoRA leaves untouched but
        # which lives under the base_layer prefix and must be renamed too.
        bias_key = f"{stem}.base_layer.bias"
        if bias_key in full_sd:
            merged_sd[f"{stem}.bias"] = full_sd[bias_key].contiguous()
            consumed.add(bias_key)

    # Everything the adapter did not touch passes through unchanged. Note these
    # include fully-trained non-adapted params (embeddings, norms, vision tower),
    # which LoRA training still updates -- they must be preserved, not reset to base.
    for key, tensor in full_sd.items():
        if key in consumed:
            continue
        if ".lora_" in key or ".base_layer." in key:
            raise SystemExit(f"Unhandled adapter-format key survived the merge: {key}")
        merged_sd[key] = tensor.contiguous()

    output.mkdir(parents=True, exist_ok=True)

    # Re-shard along the original shard boundaries so file sizes stay sane.
    out_index: dict[str, str] = {}
    per_shard: dict[str, dict[str, torch.Tensor]] = {}
    for key in merged_sd:
        # Map merged key back to its source shard via the base_layer name when needed.
        src_key = key
        if src_key not in weight_map:
            stem, _, suffix = key.rpartition(".")
            src_key = f"{stem}.base_layer.{suffix}"
        shard_name = weight_map.get(src_key)
        if shard_name is None:
            raise SystemExit(f"Cannot map merged key back to a shard: {key}")
        per_shard.setdefault(shard_name, {})[key] = merged_sd[key]
        out_index[key] = shard_name

    total_bytes = 0
    for shard_name, tensors in sorted(per_shard.items()):
        print(f"Writing {shard_name} ({len(tensors)} tensors) ...")
        save_file(tensors, str(output / shard_name), metadata={"format": "pt"})
        total_bytes += sum(t.numel() * t.element_size() for t in tensors.values())

    with (output / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": out_index}, f, indent=2)

    for name in _PASSTHROUGH_FILES:
        src = checkpoint / name
        if src.exists():
            shutil.copy2(src, output / name)

    print(
        f"\nMerged {n_merged} LoRA layers -> {output}\n"
        f"  {len(merged_sd)} tensors, {total_bytes / 1e9:.1f} GB\n"
        f"  no `.lora_`/`.base_layer.` keys remain; safe for "
        f"model.stage1_vlm_checkpoint_path"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, type=Path, help="LoRA Trainer checkpoint dir"
    )
    parser.add_argument("--output", required=True, type=Path, help="Destination dir")
    args = parser.parse_args()
    merge_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
