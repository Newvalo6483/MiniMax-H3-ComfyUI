"""MiniMax H3 Model Loader Node.

Loads the MiniMax H3 model (FL2VA or Ref2VA variant) for use in
text-to-video, image-to-video, and reference-to-video generation.
Supports BF16 precision and configurable device placement.
"""

import torch
import folder_paths
from pathlib import Path


class H3ModelLoader:
    """ComfyUI node for loading the MiniMax H3 omni-modal model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "MiniMaxAI/MiniMax-H3",
                    "tooltip": "Path to MiniMax H3 model directory or HuggingFace repo ID"
                }),
                "variant": (["fl2va", "ref2va"], {
                    "default": "fl2va",
                    "tooltip": "FL2VA: text/first-last-frame. Ref2VA: reference inputs."
                }),
                "precision": (["bf16", "fp16", "fp32"], {
                    "default": "bf16",
                    "tooltip": "BF16 recommended for best quality/speed balance"
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda"
                }),
                "offload": (["none", "model", "full"], {
                    "default": "full",
                    "tooltip": "Full offload moves inactive components to CPU"
                }),
            },
            "optional": {
                "use_int8_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use int8_convrot VAE for reduced VRAM (by kijai)"
                }),
            }
        }

    RETURN_TYPES = ("H3_MODEL", "H3_VAE", "H3_PROCESSOR")
    RETURN_NAMES = ("model", "vae", "processor")
    FUNCTION = "load_model"
    CATEGORY = "MiniMax H3"

    def load_model(self, model_path, variant, precision, device, offload,
                   use_int8_vae=False):
        from diffusers import MiniMaxH3ModularPipeline

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(precision, torch.bfloat16)

        pipe = MiniMaxH3ModularPipeline.from_pretrained(
            model_path,
            variant=variant,
            dtype=dtype,
            device_map=device if device == "cuda" else None,
        )

        if offload == "model":
            pipe.enable_model_cpu_offload()
        elif offload == "full":
            pipe.enable_sequential_cpu_offload()

        if use_int8_vae:
            from comfy.minimax_h3 import load_int8_convrot_vae
            load_int8_convrot_vae(pipe)

        model = pipe.transformer
        vae = {
            "visual": pipe.visual_vae,
            "audio": pipe.audio_vae,
        }
        processor = pipe.processor

        return (model, vae, processor)
