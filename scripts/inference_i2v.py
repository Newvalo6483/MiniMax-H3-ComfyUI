#!/usr/bin/env python3
"""
MiniMax H3 Image-to-Video inference using diffusers.

Generates video with audio from a first-frame image.

Usage:
    python inference_i2v.py --prompt "..." --first-frame image.jpg
    python inference_i2v.py --prompt "..." --first-frame img1.jpg --last-frame img2.jpg
"""

import argparse
import sys
import torch
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="MiniMax H3 Image-to-Video")
    parser.add_argument("--model-path", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--first-frame", default=None, help="First frame image")
    parser.add_argument("--last-frame", default=None, help="Last frame image")
    parser.add_argument("--output", default="./output")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.first_frame and not args.last_frame:
        print("ERROR: Provide --first-frame and/or --last-frame")
        return 1

    print("=== MiniMax H3 Image-to-Video ===")

    first_frame = None
    last_frame = None

    if args.first_frame:
        first_frame = Image.open(args.first_frame).convert("RGB")
        print(f"First frame: {args.first_frame} ({first_frame.size})")
    if args.last_frame:
        last_frame = Image.open(args.last_frame).convert("RGB")
        print(f"Last frame: {args.last_frame} ({last_frame.size})")

    print(f"Duration: {args.duration}s")

    from diffusers import MiniMaxH3ModularPipeline
    pipe = MiniMaxH3ModularPipeline.from_pretrained(
        args.model_path,
        variant="fl2va",
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    pipe.enable_sequential_cpu_offload()

    num_frames = int(args.duration * 24)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    kwargs = {
        "prompt": args.prompt,
        "num_frames": num_frames,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_steps,
        "generator": generator,
        "output_type": "pil",
        "return_dict": True,
    }
    if first_frame:
        kwargs["first_frame"] = first_frame
    if last_frame:
        kwargs["last_frame"] = last_frame

    print("Generating...")
    output = pipe(**kwargs)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(output, "frames") and output.frames:
        video_path = output_dir / "i2va_video.mp4"
        output.frames[0].save(str(video_path), save_all=True,
                              duration=1000//24, loop=0, codec="libx264")
        print(f"Video: {video_path}")

    if hasattr(output, "audios") and output.audios:
        import scipy.io.wavfile
        audio_path = output_dir / "i2va_audio.wav"
        audio = output.audios[0].cpu().numpy()
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        scipy.io.wavfile.write(str(audio_path), 32000, audio.T)
        print(f"Audio: {audio_path}")

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
