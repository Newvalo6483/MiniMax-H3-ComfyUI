"""MiniMax H3 Text-to-Video Node.

Generates video with native stereo audio from a text prompt using
the MiniMax H3 model. Supports configurable duration, aspect ratio,
and sampling parameters.
"""

import torch


class H3TextToVideo:
    """ComfyUI node for text-to-video generation with MiniMax H3."""

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
                    "tooltip": "Text prompt describing the desired video. "
                               "Include shot descriptions, soundscape, and "
                               "non-diegetic music for best results."
                }),
                "duration": ("FLOAT", {
                    "default": 8.0,
                    "min": 4.0,
                    "max": 15.0,
                    "step": 0.5,
                    "tooltip": "Video duration in seconds (4-15s)"
                }),
                "aspect_ratio": ([
                    "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"
                ], {
                    "default": "16:9"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Higher values follow prompt more closely"
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
            "optional": {
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("H3_LATENT", "AUDIO_LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent")
    FUNCTION = "generate"
    CATEGORY = "MiniMax H3"

    def generate(self, model, vae, processor, prompt, duration, aspect_ratio,
                 guidance_scale, num_inference_steps, seed,
                 negative_prompt=""):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)

        ratio_map = {
            "21:9": (21, 9),
            "16:9": (16, 9),
            "4:3": (4, 3),
            "1:1": (1, 1),
            "3:4": (3, 4),
            "9:16": (9, 16),
        }
        ratio = ratio_map.get(aspect_ratio, (16, 9))

        num_frames = int(duration * 24)

        result = model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_frames=num_frames,
            aspect_ratio=ratio,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="latent",
            return_dict=True,
        )

        video_latent = result.video_latents
        audio_latent = result.audio_latents

        return (video_latent, audio_latent)
