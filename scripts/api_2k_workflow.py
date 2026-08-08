#!/usr/bin/env python3
"""
Full 2K Workflow: Local H3-Base + H3-Context-IR API + H3-Regenerate-2K API.

Combines a locally deployed SGLang H3-Base service with the official
H3-Context-IR and H3-Regenerate-2K APIs to reproduce the quality of
2K videos generated directly by the MiniMax platform.

Usage:
    python api_2k_workflow.py --prompt "..." --duration 10 --ratio 16:9
    python api_2k_workflow.py --prompt-file examples/prompt_cinematic.txt

Requirements:
    - SGLang deployment of H3-Base (see sglang_serve.py)
    - MiniMax API token (set MINIMAX_API_KEY env var)
"""

import argparse
import base64
import io
import json
import os
import sys
import requests
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="MiniMax H3 Full 2K Workflow")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--sglang-url", default="http://localhost:30010",
                        help="SGLang H3-Base endpoint URL")
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--ratio", default="16:9")
    parser.add_argument("--api-base", default="https://api.minimax.io",
                        help="MiniMax API base URL")
    parser.add_argument("--output", default="./output")
    parser.add_argument("--skip-2k", action="store_true",
                        help="Skip 2K regeneration (768p only)")
    args = parser.parse_args()

    token = os.environ.get("MINIMAX_API_KEY", "")
    if not token and not args.skip_2k:
        print("ERROR: Set MINIMAX_API_KEY environment variable")
        return 1

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = f.read().strip()

    if not prompt:
        print("ERROR: Provide --prompt or --prompt-file")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== MiniMax H3 Full 2K Workflow ===")
    print(f"SGLang: {args.sglang_url}")
    print(f"API: {args.api_base}")
    print()

    # Step 1: H3-Context-IR
    print("[1/3] H3-Context-IR: Enhancing prompt...")
    enhanced_prompt = call_context_ir(args.api_base, token, prompt, args.duration, args.ratio)
    print(f"  Enhanced prompt length: {len(enhanced_prompt)} chars")

    # Step 2: H3-Base (local SGLang)
    print("[2/3] H3-Base: Generating 768p video locally...")
    video_path_768p, audio_path = generate_with_sglang(
        args.sglang_url, enhanced_prompt, args.duration, args.ratio, output_dir
    )
    print(f"  768p video: {video_path_768p}")

    if args.skip_2k:
        print("\nSkipping 2K regeneration. Output at 768p.")
        return 0

    # Step 3: H3-Regenerate-2K
    print("[3/3] H3-Regenerate-2K: Upscaling to 2K...")
    video_path_2k = call_regenerate_2k(
        args.api_base, token, video_path_768p, enhanced_prompt,
        args.duration, args.ratio, output_dir
    )
    if video_path_2k:
        print(f"  2K video: {video_path_2k}")
    else:
        print("  2K regeneration failed. Keeping 768p output.")

    print(f"\nWorkflow complete! Output in: {output_dir}")
    return 0


def call_context_ir(api_base, token, prompt, duration, ratio):
    url = f"{api_base}/video-generation-v2-h3-context-ir"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": "MiniMax-H3",
        "prompt": prompt,
        "duration": duration,
        "ratio": ratio,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    result = resp.json()

    task = result.get("task", {})
    enhanced = task.get("content", {}).get("prompt", prompt)

    # Save enhanced prompt
    with open("./output/enhanced_prompt.txt", 'w') as f:
        f.write(enhanced)

    return enhanced


def generate_with_sglang(sglang_url, prompt, duration, ratio, output_dir):
    import subprocess
    import tempfile

    temp_script = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    temp_script.write(f'''
import torch
from diffusers import MiniMaxH3ModularPipeline

pipe = MiniMaxH3ModularPipeline.from_pretrained(
    "MiniMaxAI/MiniMax-H3", variant="fl2va", dtype=torch.bfloat16,
)
pipe.enable_sequential_cpu_offload()

num_frames = int({duration} * 24)
output = pipe(prompt="""{prompt}""", num_frames=num_frames,
              output_type="pil", return_dict=True)

if output.frames:
    output.frames[0].save("{output_dir}/t2va_768p.mp4",
        save_all=True, duration=1000//24, loop=0, codec="libx264")
if output.audios:
    import scipy.io.wavfile
    audio = output.audios[0].cpu().numpy()
    if audio.ndim == 1: audio = audio.reshape(1, -1)
    scipy.io.wavfile.write("{output_dir}/t2va_audio.wav", 32000, audio.T)
''')
    temp_script.close()

    subprocess.run([sys.executable, temp_script.name], check=True)
    os.unlink(temp_script.name)

    return f"{output_dir}/t2va_768p.mp4", f"{output_dir}/t2va_audio.wav"


def call_regenerate_2k(api_base, token, video_path, prompt, duration, ratio, output_dir):
    url = f"{api_base}/video-generation-v2-regeneration"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    with open(video_path, 'rb') as f:
        video_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": "MiniMax-H3",
        "prompt": prompt,
        "duration": duration,
        "ratio": ratio,
        "base_video": f"data:video/mp4;base64,{video_b64}",
        "target_resolution": "2K",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=600)
    resp.raise_for_status()
    result = resp.json()

    task = result.get("task", {})
    if task.get("status") == "succeeded":
        video_url = task.get("content", {}).get("video_url", "")
        if video_url:
            video_data = requests.get(video_url).content
            output_path = f"{output_dir}/t2va_2k.mp4"
            with open(output_path, 'wb') as f:
                f.write(video_data)
            return output_path

    return None


if __name__ == "__main__":
    sys.exit(main())
