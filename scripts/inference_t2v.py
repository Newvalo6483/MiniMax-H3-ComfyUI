#!/usr/bin/env python3
"""
Standalone MiniMax H3 Text-to-Video inference using diffusers.

Generates video with native stereo audio from a text prompt without
requiring ComfyUI. Uses the MiniMaxH3ModularPipeline from diffusers.

Usage:
    python inference_t2v.py --prompt "A cat playing piano" --output output/
    python inference_t2v.py --prompt-file examples/prompt_cinematic.txt
"""

import argparse
import sys
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="MiniMax H3 Text-to-Video")
    parser.add_argument("--model-path", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--prompt", default=None, help="Text prompt")
    parser.add_argument("--prompt-file", default=None, help="Read prompt from file")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = f.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        print("ERROR: Provide --prompt or --prompt-file")
        return 1

    print(f"=== MiniMax H3 Text-to-Video ===")
    print(f"Model: {args.model_path}")
    print(f"Duration: {args.duration}s")
    print(f"Aspect: {args.aspect_ratio}")
    print(f"Steps: {args.num_steps}")
    print(f"Seed: {args.seed}")
    print()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.precision]

    print("Loading model...")
    from diffusers import MiniMaxH3ModularPipeline
    pipe = MiniMaxH3ModularPipeline.from_pretrained(
        args.model_path,
        variant="fl2va",
        dtype=dtype,
        device_map=args.device,
    )
    pipe.enable_sequential_cpu_offload()

    print("Generating video...")
    ratio_map = {
        "21:9": (21, 9), "16:9": (16, 9), "4:3": (4, 3),
        "1:1": (1, 1), "3:4": (3, 4), "9:16": (9, 16),
    }
    ratio = ratio_map.get(args.aspect_ratio, (16, 9))
    num_frames = int(args.duration * 24)

    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)

    output = pipe(
        prompt=prompt,
        num_frames=num_frames,
        aspect_ratio=ratio,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        generator=generator,
        output_type="pil",
        return_dict=True,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(output, "frames") and output.frames:
        video_path = output_dir / "t2va_video.mp4"
        print(f"Saving video to: {video_path}")
        output.frames[0].save(
            str(video_path),
            save_all=True,
            duration=1000//24,
            loop=0,
            codec="libx264",
        )

    if hasattr(output, "audios") and output.audios:
        import scipy.io.wavfile
        audio_path = output_dir / "t2va_audio.wav"
        print(f"Saving audio to: {audio_path}")
        audio = output.audios[0]
        audio = audio.cpu().numpy()
        sample_rate = 32000
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        scipy.io.wavfile.write(str(audio_path), sample_rate, audio.T)

    print(f"\nDone! Output saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
