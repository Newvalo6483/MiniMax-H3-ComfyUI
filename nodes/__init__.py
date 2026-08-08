"""MiniMax H3 ComfyUI Custom Nodes package.

Provides ComfyUI node definitions for running the MiniMax H3 33B
omni-modal generative model (with optional Turbo LoRA) for
text-to-video, image-to-video, and reference-to-video generation
with native stereo audio.
"""

from .nodes.h3_loader import H3ModelLoader
from .nodes.h3_t2v import H3TextToVideo
from .nodes.h3_i2v import H3ImageToVideo
from .nodes.h3_r2v import H3ReferenceToVideo
from .nodes.h3_vae import H3VAEDecode
from .nodes.h3_context import H3ContextIR
from .nodes.h3_2k import H3Regenerate2K
from .nodes.h3_turbo_lora import H3TurboLoraLoader

NODE_CLASS_MAPPINGS = {
    "H3ModelLoader": H3ModelLoader,
    "H3TextToVideo": H3TextToVideo,
    "H3ImageToVideo": H3ImageToVideo,
    "H3ReferenceToVideo": H3ReferenceToVideo,
    "H3VAEDecode": H3VAEDecode,
    "H3ContextIR": H3ContextIR,
    "H3Regenerate2K": H3Regenerate2K,
    "H3TurboLoraLoader": H3TurboLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ModelLoader": "MiniMax H3 Model Loader",
    "H3TextToVideo": "MiniMax H3 Text-to-Video",
    "H3ImageToVideo": "MiniMax H3 Image-to-Video",
    "H3ReferenceToVideo": "MiniMax H3 Reference-to-Video",
    "H3VAEDecode": "MiniMax H3 VAE Decode",
    "H3ContextIR": "MiniMax H3 Context-IR (API)",
    "H3Regenerate2K": "MiniMax H3 Regenerate 2K (API)",
    "H3TurboLoraLoader": "MiniMax H3 Turbo LoRA Loader",
}

__version__ = "1.0.0"
__author__ = "minimaxh3comfyui"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
