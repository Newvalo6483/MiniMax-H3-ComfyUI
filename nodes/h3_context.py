"""MiniMax H3 Context-IR API Node.

Calls the official H3-Context-IR API to enhance prompts before generation.
H3-Context-IR interprets relationships among text, images, audio, and
reference videos, converting them into a structured Context Intermediate
Representation for optimal H3-Base generation.
"""

import os
import json
import requests
from typing import Optional


class H3ContextIR:
    """ComfyUI node for calling the H3-Context-IR API."""

    GLOBAL_API_BASE = "https://api.minimax.io"
    CN_API_BASE = "https://api.minimaxi.com"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Raw user prompt to be enhanced by H3-Context-IR"
                }),
                "duration": ("INT", {
                    "default": 10,
                    "min": 4,
                    "max": 15,
                    "tooltip": "Target video duration in seconds"
                }),
                "aspect_ratio": ([
                    "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"
                ], {
                    "default": "16:9"
                }),
            },
            "optional": {
                "api_token": ("STRING", {
                    "default": "",
                    "tooltip": "MiniMax API token. If empty, uses MINIMAX_API_KEY env var."
                }),
                "region": (["global", "cn"], {
                    "default": "global",
                    "tooltip": "Global: api.minimax.io | CN: api.minimaxi.com"
                }),
                "reference_image": ("IMAGE", {
                    "tooltip": "Optional reference image for context analysis"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "raw_response")
    FUNCTION = "call_api"
    CATEGORY = "MiniMax H3/API"

    def call_api(self, prompt, duration, aspect_ratio,
                 api_token="", region="global", reference_image=None):
        token = api_token or os.environ.get("MINIMAX_API_KEY", "")
        if not token:
            return (prompt, '{"error": "No API token provided. Set MINIMAX_API_KEY env var."}')

        api_base = self.CN_API_BASE if region == "cn" else self.GLOBAL_API_BASE
        url = f"{api_base}/video-generation-v2-h3-context-ir"

        payload = {
            "model": "MiniMax-H3",
            "prompt": prompt,
            "duration": duration,
            "ratio": aspect_ratio,
        }

        if reference_image is not None:
            import base64
            import io
            from PIL import Image

            if isinstance(reference_image, torch.Tensor):
                img = Image.fromarray(
                    (reference_image.cpu().numpy() * 255).astype(np.uint8)
                )
            else:
                img = reference_image

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            payload["image"] = f"data:image/png;base64,{img_b64}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()

            task = result.get("task", {})
            enhanced = task.get("content", {}).get("prompt", prompt)

            return (enhanced, json.dumps(result, indent=2))

        except requests.exceptions.RequestException as e:
            error_msg = json.dumps({"error": str(e)}, indent=2)
            return (prompt, error_msg)

    @classmethod
    def IS_CHANGED(cls, prompt, duration, aspect_ratio, **kwargs):
        return float("NaN")
