#!/usr/bin/env python3
"""
Download MiniMax H3 model and Turbo LoRA from HuggingFace.

Downloads the MiniMaxAI/MiniMax-H3 model weights (FL2VA and/or Ref2VA
variants) and optionally the MiniMax H3 Turbo LoRA to a local directory
for use with ComfyUI, SGLang, vLLM, or diffusers.

Usage:
    python download_model.py --variant fl2va --output ./MiniMax-H3 --turbo-lora
    python download_model.py --variant both --output ./MiniMax-H3 --turbo-lora
    python download_model.py --variant fl2va --output ./MiniMax-H3 --no-turbo-lora
"""

import argparse
import subprocess
import sys
from pathlib import Path


MODEL_ID = "MiniMaxAI/MiniMax-H3"
TURBO_LORA_ID = "MiniMaxAI/MiniMax-H3-Turbo-Lora"


def main():
    parser = argparse.ArgumentParser(description="Download MiniMax H3 and Turbo LoRA from HuggingFace")
    parser.add_argument("--variant", choices=["fl2va", "ref2va", "both"],
                        default="fl2va", help="Model variant to download")
    parser.add_argument("--output", default="./MiniMax-H3",
                        help="Output directory for model weights")
    parser.add_argument("--turbo-lora-output", default="./MiniMax-H3-Turbo-Lora",
                        help="Output directory for Turbo LoRA weights")
    parser.add_argument("--turbo-lora", dest="turbo_lora", action="store_true",
                        default=True, help="Download Turbo LoRA (default: enabled)")
    parser.add_argument("--no-turbo-lora", dest="turbo_lora", action="store_false",
                        help="Skip Turbo LoRA download")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use HuggingFace cache (default: local dir)")
    args = parser.parse_args()

    print(f"=== MiniMax H3 Model Downloader ===")
    print(f"Model: {MODEL_ID}")
    print(f"Variant: {args.variant}")
    print(f"Output: {args.output}")
    print(f"Turbo LoRA: {'Yes' if args.turbo_lora else 'No'}")
    if args.turbo_lora:
        print(f"Turbo LoRA ID: {TURBO_LORA_ID}")
        print(f"Turbo LoRA Output: {args.turbo_lora_output}")
    print()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "huggingface_hub[cli]"])
        from huggingface_hub import snapshot_download

    include_patterns = ["model_index.json"]

    if args.variant == "fl2va":
        include_patterns.append("FL2VA/*")
    elif args.variant == "ref2va":
        include_patterns.append("Ref2VA/*")
    elif args.variant == "both":
        include_patterns.append("FL2VA/*")
        include_patterns.append("Ref2VA/*")

    print(f"Downloading model with patterns: {include_patterns}")
    print(f"Estimated size: ~80GB per variant (BF16)")
    print()

    local_dir = None if args.use_cache else str(output_dir)

    snapshot_download(
        repo_id=MODEL_ID,
        include=include_patterns,
        local_dir=local_dir,
        max_workers=4,
    )

    print(f"\nModel download complete!")
    if local_dir:
        print(f"Model saved to: {local_dir}")
    else:
        print(f"Model saved to HuggingFace cache. Use {MODEL_ID} as model path.")

    if args.turbo_lora:
        print(f"\n--- Downloading Turbo LoRA ---")
        turbo_dir = Path(args.turbo_lora_output)
        turbo_dir.mkdir(parents=True, exist_ok=True)

        turbo_local = None if args.use_cache else str(turbo_dir)

        snapshot_download(
            repo_id=TURBO_LORA_ID,
            local_dir=turbo_local,
            max_workers=4,
        )

        print(f"\nTurbo LoRA download complete!")
        if turbo_local:
            print(f"Turbo LoRA saved to: {turbo_local}")
        else:
            print(f"Turbo LoRA saved to HuggingFace cache. Use {TURBO_LORA_ID} as lora path.")

    print(f"\nTo use with ComfyUI:")
    print(f"  Set model_path in H3ModelLoader to: {output_dir}")
    if args.turbo_lora:
        print(f"  Add H3TurboLoraLoader node, set lora_path to: {turbo_dir}")
        print(f"  Set num_inference_steps to 10-15 for full turbo mode")
    print(f"\nTo use with SGLang:")
    print(f"  sglang serve --model-path {output_dir} --num-gpus 4 ...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
