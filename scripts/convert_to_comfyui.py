#!/usr/bin/env python3
"""
Convert MiniMax H3 HuggingFace checkpoint to ComfyUI-compatible format.

ComfyUI uses a specific checkpoint format for model loading. This script
converts the MiniMaxAI/MiniMax-H3 weights from the HuggingFace format
to the format expected by the ComfyUI custom nodes.

Usage:
    python convert_to_comfyui.py --input MiniMax-H3 --output ./comfyui_models/h3/
"""

import argparse
import json
import shutil
import sys
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert MiniMax H3 to ComfyUI format")
    parser.add_argument("--input", required=True, help="HuggingFace model directory")
    parser.add_argument("--output", required=True, help="Output directory for ComfyUI")
    parser.add_argument("--variant", choices=["fl2va", "ref2va", "both"],
                        default="fl2va")
    parser.add_argument("--precision", choices=["bf16", "fp16"],
                        default="bf16")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    print(f"=== Converting MiniMax H3 to ComfyUI format ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Variant: {args.variant}")
    print(f"Precision: {args.precision}")
    print()

    # Copy model_index.json
    index_src = input_dir / "model_index.json"
    if index_src.exists():
        shutil.copy2(index_src, output_dir / "model_index.json")
        print("Copied model_index.json")

    variants = ["FL2VA", "Ref2VA"] if args.variant == "both" else [
        "FL2VA" if args.variant == "fl2va" else "Ref2VA"
    ]

    for variant in variants:
        variant_src = input_dir / variant
        if not variant_src.exists():
            print(f"WARNING: {variant} not found in input directory, skipping")
            continue

        variant_dst = output_dir / variant
        variant_dst.mkdir(parents=True, exist_ok=True)

        print(f"\nConverting {variant}...")

        components = ["transformer", "text_encoder", "visual_vae", "audio_vae",
                      "processor", "tokenizer"]

        for comp in components:
            comp_src = variant_src / comp
            if not comp_src.exists():
                continue

            comp_dst = variant_dst / comp
            comp_dst.mkdir(parents=True, exist_ok=True)

            print(f"  {comp}...")

            for safetensor in comp_src.glob("*.safetensors"):
                print(f"    Converting {safetensor.name}...")

                from safetensors.torch import load_file, save_file

                state_dict = load_file(str(safetensor))

                converted = {}
                for key, tensor in state_dict.items():
                    converted[key] = tensor.to(dtype)

                save_file(converted, str(comp_dst / safetensor.name))

            for json_file in comp_src.glob("*.json"):
                shutil.copy2(json_file, comp_dst / json_file.name)

            for txt_file in comp_src.glob("*.txt"):
                shutil.copy2(txt_file, comp_dst / txt_file.name)

            for model_file in comp_src.glob("*.model"):
                shutil.copy2(model_file, comp_dst / model_file.name)

    # Create ComfyUI-specific config
    config = {
        "model_type": "MiniMax-H3",
        "version": "1.0.0",
        "architecture": "H3-Omni-Transformer",
        "parameters": "33B",
        "precision": args.precision,
        "variants_converted": variants,
        "comfyui_version_required": "0.31.0",
    }

    with open(output_dir / "comfyui_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nConversion complete!")
    print(f"ComfyUI model directory: {output_dir}")
    print(f"\nTo use in ComfyUI:")
    print(f"  Set model_path to: {output_dir}")
    print(f"  Required ComfyUI version: 0.31.0+")

    return 0


if __name__ == "__main__":
    sys.exit(main())
