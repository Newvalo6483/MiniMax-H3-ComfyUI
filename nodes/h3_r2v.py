"""MiniMax H3 Reference-to-Video Node.

Generates video with audio from multi-modal references (images, videos,
audio) using the MiniMax H3 Ref2VA (Reference-to-Video-Audio) variant.
Supports up to 9 images, 3 videos, and 3 audio clips as references.
"""

import torch
from typing import List, Optional


class H3ReferenceToVideo:
    """ComfyUI node for reference-to-video generation with MiniMax H3."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("H3_MODEL",),
                "vae": ("H3_VAE",),
                "processor": ("H3_PROCESSOR",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text prompt describing the desired video output. "
                               "Reference subject definitions, retention analysis, "
                               "and detailed descriptions for best results."
                }),
            },
            "optional": {
                "ref_images": ("IMAGE", {
                    "tooltip": "Reference images (up to 9). Connect multiple images."
                }),
                "ref_videos": ("VIDEO", {
                    "tooltip": "Reference videos (up to 3, each 2-15s)"
                }),
                "ref_audio": ("AUDIO", {
                    "tooltip": "Reference audio (up to 3 clips, must accompany image/video)"
                }),
                "duration": ("FLOAT", {
                    "default": 5.0,
                    "min": 4.0,
                    "max": 15.0,
                    "step": 0.5,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                    "step": 5,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**32 - 1,
                }),
            },
        }

    RETURN_TYPES = ("H3_LATENT", "AUDIO_LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent")
    FUNCTION = "generate"
    CATEGORY = "MiniMax H3"

    def generate(self, model, vae, processor, prompt,
                 ref_images=None, ref_videos=None, ref_audio=None,
                 duration=5.0, guidance_scale=5.0,
                 num_inference_steps=50, seed=0):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)

        num_frames = int(duration * 24)

        references = {}

        if ref_images is not None:
            images = self._normalize_inputs(ref_images, max_count=9)
            references["images"] = images

        if ref_videos is not None:
            videos = self._normalize_inputs(ref_videos, max_count=3)
            references["videos"] = videos

        if ref_audio is not None:
            audio = self._normalize_inputs(ref_audio, max_count=3)
            references["audio"] = audio

        total_files = (
            len(references.get("images", [])) +
            len(references.get("videos", [])) +
            len(references.get("audio", []))
        )

        if total_files > 12:
            raise ValueError(
                f"Too many reference files: {total_files}. Maximum is 12 total "
                f"(9 images + 3 videos + 3 audio)."
            )

        kwargs = {
            "prompt": prompt,
            "num_frames": num_frames,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "output_type": "latent",
            "return_dict": True,
        }
        kwargs.update(references)

        result = model.generate(**kwargs)

        return (result.video_latents, result.audio_latents)

    def _normalize_inputs(self, inputs, max_count):
        if isinstance(inputs, list):
            return inputs[:max_count]
        return [inputs]
