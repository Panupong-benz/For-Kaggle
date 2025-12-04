# SAM3-LoRA: Efficient Fine-tuning for Segment Anything Model 3

<div align="center">

**AI Research Group**
**King Mongkut's University of Technology Thonburi (KMUTT)**

---

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-SAM-green.svg)](LICENSE)

*LoRA fine-tuning integrated with Meta's official SAM3 training pipeline*

</div>

---

## ⚠️ Important Note

This implementation **integrates LoRA with the official SAM3 training pipeline** from Meta AI. It is **NOT** a standalone training script.

**What this means:**
- ✅ Uses official SAM3 training script (`sam3/train/train.py`)
- ✅ Uses Hydra configuration system
- ✅ Requires COCO JSON format data
- ✅ Uses official SAM3 loss functions and transforms
- ✅ Wraps SAM3 model with LoRA layers

**What this is NOT:**
- ❌ Not a standalone training script
- ❌ Not a simplified training pipeline
- ❌ Not compatible with custom data formats without conversion

---

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## 🎯 Overview

SAM3-LoRA enables efficient fine-tuning of Meta's Segment Anything Model 3 (848M parameters) by applying Low-Rank Adaptation (LoRA) to specific model components. Train with **less than 1% of the parameters** while maintaining performance.

### Why SAM3-LoRA?

- **Memory Efficient**: Train on 16GB+ GPUs (vs 80GB for full fine-tuning)
- **Fast Training**: 60-300x smaller checkpoints (10-50MB vs 3GB)
- **Selective Adaptation**: Apply LoRA to specific components (vision encoder, text encoder, DETR, etc.)
- **Official Integration**: Built on Meta's official training pipeline
- **Production Ready**: Uses proven SAM3 training methodology

---

## 🔧 Installation

### Prerequisites

- **Python**: 3.12+
- **PyTorch**: 2.7+ with CUDA 12.6+
- **GPU**: 16GB+ VRAM (24GB+ recommended)
- **Storage**: 50GB+ free space
- **HuggingFace Account**: For model access

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/Sompote/sam3_lora.git
cd sam3_lora

# Install dependencies (including SAM3)
pip install -r requirements.txt
# Or install SAM3 manually if needed:
# pip install -e "sam3/[train]"
```

### Step 2: Download Required Assets

```bash
# Create assets directory if it doesn't exist
mkdir -p sam3/assets

# Download BPE vocabulary file (required for text encoder)
wget https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz \
  -P sam3/assets/
```

### Step 3: Login to HuggingFace

```bash
# Required for SAM3 model access
huggingface-cli login
# Enter your token when prompted
```

---

## 📊 Data Preparation

### Required Format: COCO JSON

SAM3 requires **COCO JSON format** with segmentation annotations.

#### Directory Structure

```
your_dataset/
├── train/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── _annotations.coco.json    ← Required filename!
├── valid/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── _annotations.coco.json
```

**Note:** The configuration files currently expect the annotation file to be named `_annotations.coco.json`.

---

## 🚀 Training

### Step 1: Configure Paths

You can configure paths by either editing the YAML files or passing environment variables (if supported by your custom config setup), but direct editing of `sam3_lora_configs/lora_base.yaml` is recommended.

Edit `sam3_lora_configs/lora_base.yaml`:

```yaml
paths:
  dataset_root: /path/to/your_dataset        # Your dataset directory (must contain train/ and valid/)
  experiment_log_dir: /path/to/experiments   # Where to save outputs
```

### Step 2: Run Training

Use one of the pre-configured LoRA strategies. **Crucially, add the current directory to your `PYTHONPATH`**.

#### Minimal Configuration (Fastest)

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python sam3/train/train.py -c lora_minimal.yaml
```

**Specifications:**
- Trainable: ~500K params
- Best For: Quick experiments, proof of concept
- Targets: DETR Decoder only

#### Balanced Configuration (Recommended)

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python sam3/train/train.py -c lora_base.yaml
```

**Specifications:**
- Trainable: ~4M params
- Best For: General fine-tuning
- Targets: Vision Encoder, Text Encoder, DETR Encoder, DETR Decoder

#### Full Configuration

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python sam3/train/train.py -c lora_full.yaml
```

---

## 🔍 Inference

After training, use the provided `inference.py` script or `example_usage.py` to run your model.

```bash
python inference.py \
  --model_path outputs/sam3_lora_minimal/checkpoints/checkpoint_last.pt \
  --image_path test_image.jpg \
  --prompt "a photo of a cat"
```

---

## ⚙️ Configuration

### Key Configuration Files

- `sam3_lora_configs/lora_base.yaml`: The foundational configuration.
- `sam3_lora_configs/lora_minimal.yaml`: Extends `lora_base.yaml` for minimal training.
- `sam3_lora_configs/lora_full.yaml`: Extends `lora_base.yaml` for full training.

### Important Settings in `lora_base.yaml`

```yaml
training:
  max_epochs: 20
  batch_size: 1
  gradient_accumulation_steps: 4
  multiplier: 1  # REQUIRED: Required by Sam3ImageDataset
```

---

## 🐛 Troubleshooting

### Common Issues

1.  **`ModuleNotFoundError: No module named 'sam3_lora_wrapper'`**
    *   **Solution:** Ensure you set `export PYTHONPATH=$PYTHONPATH:$(pwd)` before running the script from the project root.

2.  **`TypeError: Sam3ImageDataset.__init__() missing 1 required positional argument: 'multiplier'`**
    *   **Solution:** Ensure your YAML configuration (e.g., `lora_base.yaml`) includes `multiplier: 1` under `training` -> `data` -> `train` -> `dataset` and `val` -> `dataset`.

3.  **`AttributeError: 'PredictionDumper' object has no attribute 'items'`**
    *   **Solution:** This indicates a mismatch in meter configuration or an issue in `trainer.py`. Ensure you are using the patched `trainer.py` provided in this repository or that your `lora_base.yaml` uses `dict_key: detection` for validation if using standard meters.

4.  **`AssertionError: please provide valid annotation file`**
    *   **Solution:** Verify `dataset_root` in your config points to a folder containing `train` and `valid` subfolders, and that those subfolders contain `_annotations.coco.json`.

---

## 📞 Contact & Support

**AI Research Group**
King Mongkut's University of Technology Thonburi (KMUTT)

- 🌐 Website: [https://ai.kmutt.ac.th](https://ai.kmutt.ac.th)
- 📧 Email: ai-research@kmutt.ac.th

---

<div align="center">

**Built with ❤️ by AI Research Group, KMUTT**

</div>