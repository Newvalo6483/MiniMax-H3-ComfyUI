#!/usr/bin/env python3
"""
MiniMax H3 Reference-to-Video inference using diffusers.

Generates video with audio from multi-modal references (images,
videos, audio) using the Ref2VA variant.

Usage:
    python inference_r2v.py --prompt "..." --images ref1.jpg ref2.jpg --videos ref.mp4
"""

import argparse
import sys
import torch
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="MiniMax H3 Reference-to-Video")
    parser.add_argument("--model-path", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--images", nargs="*", default=[], help="Reference images (max 9)")
    parser.add_argument("--videos", nargs="*", default=[], help="Reference videos (max 3)")
    parser.add_argument("--audio", nargs="*", default=[], help="Reference audio (max 3)")
    parser.add_argument("--output", default="./output")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total = len(args.images) + len(args.videos) + len(args.audio)
    if total > 12:
        print(f"ERROR: Too many references ({total}). Max 12 total.")
        return 1
    if len(args.images) > 9:
        print(f"ERROR: Too many images ({len(args.images)}). Max 9.")
        return 1
    if len(args.videos) > 3:
        print(f"ERROR: Too many videos ({len(args.videos)}). Max 3.")
        return 1
    if len(args.audio) > 3:
        print(f"ERROR: Too many audio clips ({len(args.audio)}). Max 3.")
        return 1

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = f.read().strip()

    print("=== MiniMax H3 Reference-to-Video ===")
    print(f"Images: {len(args.images)}, Videos: {len(args.videos)}, Audio: {len(args.audio)}")

    ref_images = [Image.open(img).convert("RGB") for img in args.images]

    import torchvision.io as tio
    ref_videos = []
    for vid_path in args.videos:
        frames, _, _ = tio.read_video(vid_path, pts_unit="sec")
        ref_videos.append(frames)

    import torchaudio
    ref_audio = []
    for aud_path in args.audio:
        waveform, sr = torchaudio.load(aud_path)
        ref_audio.append(waveform)

    from diffusers import MiniMaxH3ModularPipeline
    pipe = MiniMaxH3ModularPipeline.from_pretrained(
        args.model_path,
        variant="ref2va",
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    pipe.enable_sequential_cpu_offload()

    num_frames = int(args.duration * 24)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    kwargs = {
        "prompt": prompt,
        "num_frames": num_frames,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_steps,
        "generator": generator,
        "output_type": "pil",
        "return_dict": True,
    }
    if ref_images:
        kwargs["reference_images"] = ref_images
    if ref_videos:
        kwargs["reference_videos"] = ref_videos
    if ref_audio:
        kwargs["reference_audio"] = ref_audio

    print("Generating...")
    output = pipe(**kwargs)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(output, "frames") and output.frames:
        video_path = output_dir / "r2va_video.mp4"
        output.frames[0].save(str(video_path), save_all=True,
                              duration=1000//24, loop=0, codec="libx264")
        print(f"Video: {video_path}")

    if hasattr(output, "audios") and output.audios:
        import scipy.io.wavfile
        audio_path = output_dir / "r2va_audio.wav"
        audio = output.audios[0].cpu().numpy()
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        scipy.io.wavfile.write(str(audio_path), 32000, audio.T)
        print(f"Audio: {audio_path}")

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
