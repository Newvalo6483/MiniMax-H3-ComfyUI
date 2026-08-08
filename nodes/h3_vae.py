"""MiniMax H3 VAE Decode Node.

Decodes MiniMax H3 visual and audio latents into video frames and
stereo audio. Uses H3-VisualVAE (f16t4d24) for video and H3-AudioVAE
(32 kHz stereo, 40 Hz latent rate) for audio.
"""

import torch
import numpy as np


class H3VAEDecode:
    """ComfyUI node for decoding H3 latents to video and audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("H3_VAE",),
                "video_latent": ("H3_LATENT",),
            },
            "optional": {
                "audio_latent": ("AUDIO_LATENT", {
                    "tooltip": "Audio latent for stereo audio output"
                }),
                "decode_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Decode audio latent if present"
                }),
                "output_fps": ("INT", {
                    "default": 24,
                    "tooltip": "Output video frame rate (model generates at 24 FPS)"
                }),
                "audio_sample_rate": ("INT", {
                    "default": 32000,
                    "tooltip": "Audio sample rate (32 kHz native)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "METADATA")
    RETURN_NAMES = ("frames", "audio", "metadata")
    FUNCTION = "decode"
    CATEGORY = "MiniMax H3"

    def decode(self, vae, video_latent, audio_latent=None,
               decode_audio=True, output_fps=24, audio_sample_rate=32000):
        visual_vae = vae["visual"]
        audio_vae = vae["audio"] if "audio" in vae else None

        frames = self._decode_video(visual_vae, video_latent)

        audio_output = None
        if decode_audio and audio_latent is not None and audio_vae is not None:
            audio_output = self._decode_audio(audio_vae, audio_latent, audio_sample_rate)

        num_frames = frames.shape[0] if frames.dim() == 4 else frames.shape[1]
        duration_seconds = num_frames / output_fps

        metadata = {
            "num_frames": num_frames,
            "fps": output_fps,
            "duration_seconds": duration_seconds,
            "resolution": f"{frames.shape[-2]}x{frames.shape[-1]}" if frames.dim() == 4 else "unknown",
            "has_audio": audio_output is not None,
            "audio_sample_rate": audio_sample_rate if audio_output is not None else 0,
            "audio_channels": 2 if audio_output is not None else 0,
        }

        return (frames, audio_output, metadata)

    def _decode_video(self, visual_vae, video_latent):
        if hasattr(visual_vae, 'decode'):
            with torch.no_grad():
                video = visual_vae.decode(video_latent)
            if video.dim() == 5:
                video = video.squeeze(0)
            if video.dim() == 4 and video.shape[1] <= 4:
                video = video.permute(0, 2, 3, 1)
            return video.float()
        return video_latent

    def _decode_audio(self, audio_vae, audio_latent, sample_rate):
        if hasattr(audio_vae, 'decode'):
            with torch.no_grad():
                audio = audio_vae.decode(audio_latent)
            if audio.dim() == 3:
                audio = audio.squeeze(0)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0).repeat(2, 1)
            return {
                "waveform": audio.cpu().float(),
                "sample_rate": sample_rate,
            }
        return None
