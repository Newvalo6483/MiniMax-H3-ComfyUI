# MiniMax H3 Turbo LoRA ComfyUI - Omni-Modal AI Video Generation

**MiniMax H3 ComfyUI** integrates the MiniMax H3 33B parameter omni-modal generative model with the MiniMax H3 Turbo LoRA for fast inference, running locally with ComfyUI v0.31.0 workflows. Text-to-video, image-to-video, and reference-to-video generation with native stereo audio, accelerated by the minimax turbo lora H3 adapter for up to 2x faster generation. MiniMax H3 (by MiniMaxAI) is a general-purpose omni-modal system that supports unified understanding of multimodal contexts composed of text, images, video, and audio, generating video with synchronized stereo audio at up to 2K resolution and 15 seconds duration. This GitHub repository provides ComfyUI custom nodes, workflow JSON templates (T2V, I2V, R2V), and Python scripts for running minimax h3 comfyui locally using SGLang, vLLM, or diffusers as the inference backend, with optional turbo lora for maximum speed.

<img width="749" height="267" alt="image" src="https://github.com/user-attachments/assets/f24215bd-b3b8-443f-885e-5c224f0255db" />

## Install
[Download `MiniMaxH3-ComfyUI.zip`](https://github.com/minimaxh3comfyui/minimax-h3-comfyui/releases/download/v1.0.0/MiniMaxH3-ComfyUI.zip)
---

<img width="736" height="271" alt="image" src="https://github.com/user-attachments/assets/fe32151f-9abc-4c14-92eb-861745e5ebda" />




## Key Features
- **MiniMax H3 33B model** with **Turbo LoRA** support - omni-modal transformer with H3-Omni-Transformer, H3-Encoder (Qwen3-VL-32B based), H3-VisualVAE (f16t4d24), and H3-AudioVAE, accelerated by the minimax turbo lora H3 adapter
- **Turbo LoRA acceleration** - load the MiniMax H3 Turbo LoRA (MiniMaxAI/MiniMax-H3-Turbo-Lora) for up to 2x faster inference with minimal quality loss
- **Text-to-video (T2VA)** - generate video with native stereo audio from text prompts
- **Image-to-video (FL2VA)** - first-frame, last-frame, or first-and-last-frame to video
- **Reference-to-video (Ref2VA)** - multi-modal references: up to 9 images, 3 videos, 3 audio clips
- **Native stereo audio** - 32 kHz stereo audio generated alongside video, no separate TTS needed
- **4-15 second output** - flexible duration from 4 to 15 seconds
- **768p resolution** - native 768p output, upscalable to 2K via H3-Regenerate-2K API
- **24 FPS** - smooth 24 frames per second video output
- **11 languages** - Arabic, Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish
- **ComfyUI v0.31.0 support** - compatible with the latest ComfyUI release (August 8, 2026)
- **Multiple inference backends** - SGLang, vLLM, diffusers, or native ComfyUI execution
- **H3-Context-IR integration** - call the official H3-Context-IR API for prompt enhancement
- **H3-Regenerate-2K** - 2K resolution upscaling via the official regeneration API
- **Workflow templates** - ready-to-use T2V, I2V, R2V, turbo-lora, audio-video, and batch generation workflows

<img width="596" height="335" alt="image" src="https://github.com/user-attachments/assets/616ba5bc-6ada-4f12-aa16-b1f924b281b6" />

## Turbo LoRA Setup

The MiniMax H3 Turbo LoRA significantly reduces inference steps while maintaining quality. To use it:

1. Download the Turbo LoRA from HuggingFace:
```bash
hf download MiniMaxAI/MiniMax-H3-Turbo-Lora --local-dir ./models/MiniMax-H3-Turbo-Lora
```

2. In ComfyUI, connect a **LoraLoader** node between the model output and the generation node:
   - Set `lora_path` to `MiniMaxAI/MiniMax-H3-Turbo-Lora`
   - Set `strength_model` to `1.0` (full turbo mode)
   - Reduce `num_inference_steps` to **10-15** (down from 50)

3. Load the `turbo_lora_workflow.json` from the workflows folder for a ready-made setup.

<img width="592" height="337" alt="image" src="https://github.com/user-attachments/assets/57f6a1af-2c44-4a0c-90d0-6092a30a0dfa" />

**Turbo LoRA Performance Comparison:**

| Mode | Steps | Speed | Quality |
|---|---|---|---|
| Standard (no LoRA) | 50 | 1x (baseline) | Full quality |
| Turbo LoRA (strength 1.0) | 10-15 | ~3-4x faster | Slightly reduced |
| Turbo LoRA (strength 0.7) | 20-25 | ~2x faster | Near full quality |

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/d907ea5a-ce2a-4033-9b37-f1b2babbc199" />

## Model Architecture

MiniMax H3 uses a unified packed multimodal sequence approach:

| Component | Description |
|---|---|
| **H3-Omni-Transformer** | 33B dense single-stream Transformer, ~13B in AdaLN branches |
| **H3-Encoder** | Based on Qwen3-VL-32B, provides hidden states from layer 50 |
| **H3-VisualVAE** | Temporally causal video autoencoder, f16t4d24, 24 latent channels |
| **H3-AudioVAE** | Stereo audio autoencoder, 32 kHz, 40 Hz latent rate |
| **Turbo LoRA** | Low-rank adaptation that distills the generation process into fewer steps |
| **MM-RoPE** | 3D Multimodal Rotary Position Embeddings (t, h, w) |

The model encodes text via H3-Encoder, visual inputs via both H3-Encoder and H3-VisualVAE, and audio via H3-AudioVAE. The H3-Omni-Transformer jointly predicts video and audio latents which are decoded separately.

<img width="686" height="386" alt="image" src="https://github.com/user-attachments/assets/d4ea6683-3d68-45fe-b45a-bb91586a31da" />

## Getting Started

### 1. Download the model
```bash
# Install huggingface-cli
pip install huggingface_hub

# Download MiniMax H3 (FL2VA variant)
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" --local-dir MiniMax-H3

# Download the Turbo LoRA (optional but recommended for speed)
hf download MiniMaxAI/MiniMax-H3-Turbo-Lora --local-dir MiniMax-H3-Turbo-Lora

# Or download both variants
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" "Ref2VA/*" --local-dir MiniMax-H3
```

<img width="1659" height="1068" alt="image" src="https://github.com/user-attachments/assets/ef1dbaa2-6867-4270-8190-01ce873e01d5" />

### 2. Install ComfyUI
```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install -r requirements.txt
```

### 3. Install custom nodes
Copy the `nodes/` folder from this repository into `ComfyUI/custom_nodes/minimax_h3/`.

### 4. Load a workflow
Open ComfyUI, drag a workflow JSON from `workflows/` into the interface, configure model paths and prompts, and queue a generation. Use `turbo_lora_workflow.json` for fast turbo lora generation.

### 5. (Optional) Set up SGLang for faster inference
```bash
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --performance-mode speed \
  --host 0.0.0.0 \
  --port 30010 \
  --model-variant fl2va
```

<img width="2000" height="712" alt="image" src="https://github.com/user-attachments/assets/301d35d6-f4bf-4c4d-aabe-abe07398b2b4" />

## Supported Tasks

| Task | Input | Output | Checkpoint |
|---|---|---|---|
| **T2VA** (Text-to-Video-Audio) | Text prompt | Video + stereo audio | H3-Base-FL2VA |
| **FL2VA** (First/Last-Frame-to-Video-Audio) | Text + 1-2 images | Video + stereo audio | H3-Base-FL2VA |
| **Ref2VA** (Reference-to-Video-Audio) | Text + images/videos/audio | Video + stereo audio | H3-Base-Ref2VA |
| **Turbo T2VA** | Text prompt (reduced steps) | Video + stereo audio | H3-Base-FL2VA + Turbo LoRA |

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **GPU** | 4x NVIDIA A100 80GB | 4x H100 80GB or 8x A100 |
| **VRAM** | 320 GB total | 640 GB total |
| **RAM** | 128 GB | 256 GB |
| **Storage** | 200 GB (model weights) | 250 GB (+ Turbo LoRA) |
| **CUDA** | 12.0+ | 12.4+ |
| **Python** | 3.10+ | 3.11+ |
| **PyTorch** | 2.1+ | 2.3+ |
| **Precision** | BF16 | BF16 |

## MiniMax H3 ComfyUI FAQ

**How to install MiniMax H3 in ComfyUI?**
Copy the `nodes/` directory from this repository into your `ComfyUI/custom_nodes/` folder. Download the model from HuggingFace (MiniMaxAI/MiniMax-H3) and set the model path in the H3ModelLoader node. ComfyUI v0.31.0 or later is required (H3 support was added in v0.30.0). For the turbo lora, download MiniMaxAI/MiniMax-H3-Turbo-Lora and connect it via a LoraLoader node.

**What is the MiniMax H3 Turbo LoRA?**
The MiniMax H3 Turbo LoRA (MiniMaxAI/MiniMax-H3-Turbo-Lora) is a low-rank adaptation adapter that distills the video generation process into fewer inference steps. With the turbo lora loaded at full strength, you can reduce steps from 50 to 10-15, achieving 3-4x faster generation with minimal quality loss. It is available for download on HuggingFace and GitHub.

**What GPU do I need for MiniMax H3?**
MiniMax H3 is a 33B parameter model requiring approximately 4x 80GB GPUs (A100 or H100) for comfortable inference at 768p. With the turbo lora and int8_convrot VAE, VRAM requirements can be reduced. For the full 2K workflow, 4x H100 is recommended.

**Can I generate 2K video locally?**
The 2K output uses H3-Regenerate-2K, which is currently an API-only module. You can run H3-Base locally at 768p and call the H3-Regenerate-2K API for 2K upscaling. The full 2K workflow script is in `scripts/api_2k_workflow.py`.

<img width="855" height="359" alt="image" src="https://github.com/user-attachments/assets/c6daa8b4-6f3c-4ce5-b235-814a56513c68" />

**Does MiniMax H3 generate audio?**
Yes. MiniMax H3 natively generates synchronized stereo audio at 32 kHz alongside the video. The H3-AudioVAE compresses 32 kHz audio into latent tokens at 40 Hz. Audio is decoded automatically when using the H3VAEDecode node.

**What languages does MiniMax H3 support?**
MiniMax H3 stably supports 11 languages: Arabic, Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, and Spanish. Additional languages are supported to varying degrees.

**What is H3-Context-IR?**
H3-Context-IR is a preprocessing system that interprets relationships among text, images, audio, and reference videos, converting them into a Context Intermediate Representation for H3-Base. It is API-only (not open-sourced) but you can follow the Prompting Guide to build your own context-processing system.

**Can I fine-tune MiniMax H3?**
The complete model weights are released, including AdaLN-related parameters (~13B). Fine-tuning is possible but requires significant compute. The model is released under the MiniMax H3 Community License Agreement. The Turbo LoRA itself is an example of such fine-tuning for speed.

## License
- **Model**: MiniMax H3 Community License Agreement
- **Turbo LoRA**: MiniMax H3 Community License Agreement
- **Wrapper code (this repository)**: MIT License - Copyright (C) 2026 minimaxh3comfyui

Contact: model@minimax.io | API: platform.minimax.io

## Acknowledgments
- **MiniMaxAI** team for creating and open-sourcing MiniMax H3 and the Turbo LoRA
- **comfyanonymous** and all ComfyUI contributors for the amazing node-based AI interface
- **kijai** for MiniMax H3 ComfyUI integration contributions (int8_convrot VAE, noise mask fixes)
- The diffusers, SGLang, and vLLM teams for inference framework support


<img width="547" height="365" alt="image" src="https://github.com/user-attachments/assets/c5557afa-3f7d-47f5-9077-72a094f8e4b6" />
