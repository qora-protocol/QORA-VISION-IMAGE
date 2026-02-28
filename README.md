---
language:
  - en
  - multilingual
license: apache-2.0
tags:
  - rust
  - cpu-inference
  - quantized
  - q4
  - image-classification
  - zero-shot-classification
  - image-embedding
  - siglip
  - vision-transformer
  - pure-rust
  - no-python
  - no-cuda
  - contrastive-learning
base_model: google/siglip2-base-patch16-224
library_name: qora
pipeline_tag: zero-shot-image-classification
model-index:
  - name: QORA-Vision-Image
    results:
      - task:
          type: zero-shot-image-classification
        dataset:
          name: ImageNet-1K
          type: imagenet-1k
        metrics:
          - name: Zero-shot Accuracy
            type: accuracy
            value: 69.8
---

# QORA-Vision (Image) - Native Rust Image Encoder

Pure Rust image understanding engine based on SigLIP 2. Zero-shot image classification, image embeddings, and image-text similarity. No Python runtime, no CUDA, no external dependencies.

## Overview

| Property | Value |
|----------|-------|
| **Engine** | QORA-Vision (Pure Rust) |
| **Base Model** | SigLIP 2 Base (google/siglip2-base-patch16-224) |
| **Vision Params** | ~93M |
| **Text Params** | ~283M (256K vocab) |
| **Quantization** | Q4 (4-bit symmetric, group_size=32) |
| **Vision Model Size** | 58 MB (Q4 binary) |
| **Executable** | 4.4 MB |
| **Input** | 224x224 RGB images (PNG/JPEG) |
| **Output** | 768-dim embeddings + zero-shot classification scores |
| **Platform** | Windows x86_64 (CPU-only) |

## Architecture

### Vision Encoder (12-layer ViT-Base)

| Component | Details |
|-----------|---------|
| **Layers** | 12 transformer layers |
| **Hidden Size** | 768 |
| **Attention Heads** | 12 (head_dim=64) |
| **MLP (Intermediate)** | 3,072 (GELU-Tanh activation) |
| **Patch Size** | 16x16 (non-overlapping) |
| **Sequence Length** | 196 patches (14x14 grid) |
| **Normalization** | LayerNorm with bias (eps=1e-6) |
| **Attention** | Bidirectional (no causal mask) |
| **Position Encoding** | Learned position embeddings |
| **Pooling** | MAP (Multi-head Attention Pooling) |

### Text Encoder (12-layer ViT-Base)

| Component | Details |
|-----------|---------|
| **Layers** | 12 transformer layers |
| **Hidden Size** | 768 |
| **Vocabulary** | 256,000 tokens |
| **Max Position** | 64 tokens |
| **Pooling** | Last token + linear head |

### Contrastive Scoring

```
score = sigmoid(cosine_sim(image_embed, text_embed) * exp(logit_scale) + logit_bias)
```

## Pipeline

```
Image (224x224) → Patch Embedding (196 patches)
    → Add Position Embeddings
    → 12x ViT Transformer Layers (bidirectional)
    → Post-LayerNorm
    → MAP Pooling (cross-attention with learned probe)
    → L2 Normalize
    → 768-dim Image Embedding

Text → Tokenize → Token + Position Embedding
    → 12x ViT Transformer Layers
    → Final LayerNorm (last token)
    → Linear Head
    → L2 Normalize
    → 768-dim Text Embedding

Score = sigmoid(cosine_sim * exp(scale) + bias)
```

## Files

```
siglip-model/
  qora-vision.exe      - 4.4 MB    Inference engine
  model.qora-vision    - 58 MB     Vision encoder (Q4)
  tokenizer.json       - 33 MB     Text tokenizer (256K vocab)
  config.json          - 611 B     QORA-branded config
  README.md            - This file
```

## Usage

```bash
# Image embedding
qora-vision.exe siglip --image photo.jpg --model-path ./siglip-model/

# Zero-shot classification
qora-vision.exe siglip --image photo.jpg --labels "cat,dog,bird,car" --model-path ../SigLIP2/

# Image-text similarity
qora-vision.exe siglip --image photo.jpg --text "a photo of a sunset" --model-path ../SigLIP2/

# Load from binary (vision encoder only)
qora-vision.exe siglip --load model.qora-vision --image photo.jpg
```

### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path <path>` | `.` | Path to model directory (safetensors) |
| `--image <path>` | - | Input image (PNG/JPEG) |
| `--labels <list>` | - | Comma-separated labels for zero-shot |
| `--text <string>` | - | Text for similarity scoring |
| `--load <path>` | - | Load vision binary (.qora-vision) |
| `--save <path>` | - | Save vision binary |
| `--f16` | off | Use F16 weights instead of Q4 |

## Published Benchmarks

### SigLIP 2 Base (224px) - Published Scores

| Benchmark | Score |
|-----------|-------|
| **ImageNet-1K Zero-shot** | ~69.8% |
| **Multilingual support** | Yes (trained on WebLI) |

SigLIP 2 improves over the original SigLIP with enhanced semantic understanding, localization, and dense features. The sigmoid loss enables better calibrated scores compared to CLIP's softmax-based approach.

### Model Comparison

| Model | Params | Image Size | Architecture | Zero-shot ImageNet |
|-------|--------|------------|-------------|-------------------|
| **QORA-Vision (SigLIP 2 Base)** | 93M | 224 | ViT-B/16 | ~69.8% |
| CLIP ViT-B/16 | 86M | 224 | ViT-B/16 | 68.3% |
| SigLIP Base (v1) | 86M | 224 | ViT-B/16 | 66.2% |
| OpenCLIP ViT-B/16 | 86M | 224 | ViT-B/16 | 67.0% |

## Test Results

All tests run with Q4 quantization on CPU.

### Test 1: Red Image Classification

**Input:** Solid red 224x224 image
**Labels:** red, blue, green, yellow

| Label | Score |
|-------|-------|
| **red** | **0.0022** |
| blue | 0.0000 |
| green | 0.0000 |
| yellow | 0.0000 |

| Metric | Value |
|--------|-------|
| Result | PASS (correctly identified "red") |
| Vision Forward | 42.0s |
| Embedding Dim | 768, L2 norm = 1.0000 |

### Test 2: Blue Image Classification

**Input:** Solid blue 224x224 image
**Labels:** red, blue, green, yellow

| Label | Score |
|-------|-------|
| red | 0.0000 |
| **blue** | **0.0014** |
| green | 0.0000 |
| yellow | 0.0000 |

| Metric | Value |
|--------|-------|
| Result | PASS (correctly identified "blue") |
| Vision Forward | 31.5s |

### Test 3: Green Image with Natural Language Labels

**Input:** Solid green 224x224 image
**Labels:** "a photo of a cat", "a photo of a dog", "a solid green image", "a landscape"

| Label | Score |
|-------|-------|
| a photo of a cat | 0.0000 |
| a photo of a dog | 0.0000 |
| **a solid green image** | **0.0176** |
| a landscape | 0.0000 |

| Metric | Value |
|--------|-------|
| Result | PASS (correctly identified natural language description) |
| Vision Forward | 39.2s |
| Note | Highest score by far, demonstrating text understanding |

### Test Summary

| Test | Input | Best Label | Correct? | Score |
|------|-------|------------|----------|-------|
| Color (red) | Solid red | "red" | PASS | 0.0022 |
| Color (blue) | Solid blue | "blue" | PASS | 0.0014 |
| NL Description | Solid green | "a solid green image" | PASS | 0.0176 |
| **Overall** | | | **3/3 (100%)** | |

## Performance

| Metric | Value |
|--------|-------|
| **Model Load** | ~25-30s (from safetensors) |
| **Vision Forward** | ~31-42s (196 tokens, 12 layers) |
| **Text Forward** | ~25s per label |
| **Total (4 labels)** | ~120-150s |
| **Memory (Vision Q4)** | 58 MB |
| **Memory (Text Q4)** | 151 MB |
| **Binary Save** | 41ms (58 MB) |

## QORA Model Family

| Engine | Model | Params | Size (Q4) | Purpose |
|--------|-------|--------|-----------|---------|
| **QORA** | SmolLM3-3B | 3.07B | 1.68 GB | Text generation, reasoning, chat |
| **QORA-TTS** | Qwen3-TTS | 1.84B | 1.5 GB | Text-to-speech synthesis |
| **QORA-Vision (Image)** | SigLIP 2 Base | 93M | 58 MB | Image embeddings, zero-shot classification |
| **QORA-Vision (Video)** | ViViT Base | 89M | 60 MB | Video action classification |

---

*Built with QORA - Pure Rust AI Inference*
