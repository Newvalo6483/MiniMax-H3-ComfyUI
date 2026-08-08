"""MiniMax H3 Image-to-Video Node.

Generates video with audio from a first-frame and/or last-frame image
using the MiniMax H3 FL2VA (First-Last-frame-to-Video-Audio) model variant.
"""

import torch


class H3ImageToVideo:
    """ComfyUI node for image-to-video generation with MiniMax H3."""

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
                    "tooltip": "Text prompt describing how the image should animate"
                }),
            },
            "optional": {
                "first_frame": ("IMAGE", {
                    "tooltip": "First frame image to animate from"
                }),
                "last_frame": ("IMAGE", {
                    "tooltip": "Last frame image to animate towards"
                }),
                "duration": ("FLOAT", {
                    "default": 8.0,
                    "min": 4.0,
                    "max": 15.0,
                    "step": 0.5,
                }),
                "aspect_ratio": ([
                    "auto", "16:9", "4:3", "1:1", "9:16"
                ], {
                    "default": "auto",
                    "tooltip": "Auto uses the first frame's aspect ratio"
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
                 first_frame=None, last_frame=None,
                 duration=8.0, aspect_ratio="auto",
                 guidance_scale=5.0, num_inference_steps=50, seed=0):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)

        num_frames = int(duration * 24)

        kwargs = {
            "prompt": prompt,
            "num_frames": num_frames,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "output_type": "latent",
            "return_dict": True,
        }

        if first_frame is not None and last_frame is not None:
            kwargs["first_frame"] = self._preprocess_image(first_frame)
            kwargs["last_frame"] = self._preprocess_image(last_frame)
        elif first_frame is not None:
            kwargs["first_frame"] = self._preprocess_image(first_frame)
        elif last_frame is not None:
            kwargs["last_frame"] = self._preprocess_image(last_frame)
        else:
            kwargs["prompt"] = prompt

        if aspect_ratio != "auto":
            ratio_map = {
                "16:9": (16, 9), "4:3": (4, 3),
                "1:1": (1, 1), "9:16": (9, 16),
            }
            kwargs["aspect_ratio"] = ratio_map.get(aspect_ratio, (16, 9))

        result = model.generate(**kwargs)

        return (result.video_latents, result.audio_latents)

    def _preprocess_image(self, image_tensor):
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
        if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:
            image_tensor = image_tensor.permute(1, 2, 0)
        return image_tensor
