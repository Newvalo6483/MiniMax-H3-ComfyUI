"""MiniMax H3 Regenerate 2K API Node.

Calls the official H3-Regenerate-2K API to upscale 768p video output
to 2K resolution. Uses in-context regeneration to leverage H3's
generative capabilities for superior detail recovery.
"""

import os
import json
import requests
import torch


class H3Regenerate2K:
    """ComfyUI node for calling the H3-Regenerate-2K API."""

    GLOBAL_API_BASE = "https://api.minimax.io"
    CN_API_BASE = "https://api.minimaxi.com"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {
                    "tooltip": "768p video frames from H3VAEDecode to upscale to 2K"
                }),
                "original_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Original text prompt used for generation"
                }),
                "duration": ("INT", {
                    "default": 10,
                    "min": 4,
                    "max": 15,
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
                    "default": "global"
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Audio from H3VAEDecode (optional, for audio-preserving 2K)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("upscaled_frames", "upscaled_audio", "status")
    FUNCTION = "upscale"
    CATEGORY = "MiniMax H3/API"

    def upscale(self, video_frames, original_prompt, duration, aspect_ratio,
                api_token="", region="global", audio=None):
        token = api_token or os.environ.get("MINIMAX_API_KEY", "")
        if not token:
            return (video_frames, audio, '{"error": "No API token provided."}')

        api_base = self.CN_API_BASE if region == "cn" else self.GLOBAL_API_BASE
        url = f"{api_base}/video-generation-v2-regeneration"

        import base64
        import io
        from PIL import Image
        import numpy as np

        if isinstance(video_frames, torch.Tensor):
            frames_np = video_frames.cpu().numpy()
        else:
            frames_np = np.array(video_frames)

        if frames_np.dim() == 4:
            frames_list = [frames_np[i] for i in range(frames_np.shape[0])]
        else:
            frames_list = [frames_np]

        first_frame = Image.fromarray((frames_list[0] * 255).astype(np.uint8))
        buffer = io.BytesIO()
        first_frame.save(buffer, format="PNG")
        first_b64 = base64.b64encode(buffer.getvalue()).decode()

        last_frame = Image.fromarray((frames_list[-1] * 255).astype(np.uint8))
        buffer = io.BytesIO()
        last_frame.save(buffer, format="PNG")
        last_b64 = base64.b64encode(buffer.getvalue()).decode()

        payload = {
            "model": "MiniMax-H3",
            "prompt": original_prompt,
            "duration": duration,
            "ratio": aspect_ratio,
            "base_video_first_frame": f"data:image/png;base64,{first_b64}",
            "base_video_last_frame": f"data:image/png;base64,{last_b64}",
            "target_resolution": "2K",
        }

        if audio is not None:
            waveform = audio.get("waveform")
            if waveform is not None:
                import torchaudio
                buffer = io.BytesIO()
                torchaudio.save(buffer, waveform.unsqueeze(0) if waveform.dim() == 2 else waveform,
                                audio.get("sample_rate", 32000), format="WAV")
                audio_b64 = base64.b64encode(buffer.getvalue()).decode()
                payload["base_audio"] = f"data:audio/wav;base64,{audio_b64}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=300)
            response.raise_for_status()
            result = response.json()

            task = result.get("task", {})
            status = task.get("status", "unknown")

            if status == "succeeded":
                content = task.get("content", {})
                video_url = content.get("video_url", "")
                if video_url:
                    import torchvision.io as tio
                    video_data = requests.get(video_url).content
                    temp_path = "/tmp/h3_2k_output.mp4"
                    with open(temp_path, "wb") as f:
                        f.write(video_data)
                    frames, _, _ = tio.read_video(temp_path, pts_unit="sec")
                    frames = frames.permute(0, 2, 3, 1).float() / 255.0
                    return (frames, audio, json.dumps(result, indent=2))

            return (video_frames, audio, json.dumps(result, indent=2))

        except requests.exceptions.RequestException as e:
            error_msg = json.dumps({"error": str(e)}, indent=2)
            return (video_frames, audio, error_msg)
