#!/usr/bin/env python3
"""
Deploy MiniMax H3 with SGLang inference server.

Starts an SGLang server hosting the MiniMax H3 model for fast
inference. ComfyUI or standalone scripts can connect to this
server instead of loading the model directly.

Usage:
    python sglang_serve.py --variant fl2va --port 30010
    python sglang_serve.py --variant ref2va --port 30011
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Deploy MiniMax H3 with SGLang")
    parser.add_argument("--model-path", default="MiniMaxAI/MiniMax-H3",
                        help="Path to MiniMax H3 model")
    parser.add_argument("--variant", choices=["fl2va", "ref2va"],
                        default="fl2va", help="Model variant")
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="Number of GPUs to use")
    parser.add_argument("--ulysses-degree", type=int, default=4,
                        help="Ulysses parallelism degree")
    parser.add_argument("--port", type=int, default=30010,
                        help="Server port")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--performance-mode", choices=["speed", "quality"],
                        default="speed")
    args = parser.parse_args()

    cmd = [
        "sglang", "serve",
        "--model-path", args.model_path,
        "--num-gpus", str(args.num_gpus),
        "--ulysses-degree", str(args.ulysses_degree),
        "--performance-mode", args.performance_mode,
        "--host", args.host,
        "--port", str(args.port),
        "--model-variant", args.variant,
    ]

    print("=== MiniMax H3 SGLang Deployment ===")
    print(f"Model: {args.model_path}")
    print(f"Variant: {args.variant}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Endpoint: http://{args.host}:{args.port}")
    print()
    print("Command:")
    print(" ".join(cmd))
    print()
    print("Starting server...")
    print("Press Ctrl+C to stop.")
    print()

    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    except KeyboardInterrupt:
        print("\nStopping server...")
        return 0
    except FileNotFoundError:
        print("ERROR: sglang not found. Install with: pip install sglang[all]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
