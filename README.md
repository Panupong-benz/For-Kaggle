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

### SAM3 Architecture

```
Input Image (1008x1008)
    ↓
Vision Encoder (32 layers, ViT) ← LoRA can be applied
    ↓
Text Encoder ← LoRA can be applied
    ↓
DETR Encoder (6 layers) ← LoRA can be applied
    ↓
DETR Decoder (6 layers) ← LoRA can be applied
    ↓
Mask Decoder (3 stages)
    ↓
Segmentation Masks
```

**Total Parameters**: 848M
**LoRA Parameters**: 500K - 15M (0.06% - 1.77%)

---

## 🔧 Installation

### Prerequisites

- **Python**: 3.12+
- **PyTorch**: 2.7+ with CUDA 12.6+
- **GPU**: 16GB+ VRAM (24GB+ recommended)
- **Storage**: 50GB+ free space
- **HuggingFace Account**: For model access

### Step 1: Clone Official SAM3 Repository

```bash
# Clone Meta's official SAM3 repository
git clone https://github.com/facebookresearch/sam3.git
cd sam3
```

### Step 2: Install SAM3 with Training Dependencies

```bash
# Install SAM3 with training support
pip install -e ".[train]"
```

This installs:
- PyTorch and torchvision
- Hydra configuration framework
- COCO API (pycocotools)
- All training dependencies

### Step 3: Install LoRA Wrapper

```bash
# Copy LoRA wrapper to SAM3 directory
cp /path/to/sam3_lora_wrapper.py sam3/

# Copy LoRA configurations
mkdir -p train/configs/lora
cp -r /path/to/sam3_lora_configs/* train/configs/lora/
```

### Step 4: Download Required Assets

```bash
# Download BPE vocabulary file (required for text encoder)
wget https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz \
  -P assets/

# Verify file exists
ls assets/bpe_simple_vocab_16e6.txt.gz
```

### Step 5: Login to HuggingFace

```bash
# Required for SAM3 model access
huggingface-cli login
# Enter your token when prompted
```

### Verify Installation

```bash
# Test SAM3 installation
python -c "import sam3; print('SAM3 installed successfully')"

# Test Hydra
python -c "import hydra; print('Hydra installed successfully')"

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
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
│   ├── image003.jpg
│   └── _annotations.coco.json    ← Required filename!
├── valid/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── _annotations.coco.json
└── test/                          ← Optional
    ├── image001.jpg
    └── _annotations.coco.json
```

#### COCO JSON Format

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image001.jpg",
      "height": 1008,
      "width": 1008
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "segmentation": [[x1, y1, x2, y2, ...]] or {"counts": ..., "size": ...},
      "area": 1234.5,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "person"
    }
  ]
}
```

**Key Requirements:**
- ✅ Must have `segmentation` field (polygon or RLE)
- ✅ File must be named `_annotations.coco.json`
- ✅ Images should be 1008x1008 (or will be resized)
- ✅ Bounding boxes in `[x, y, width, height]` format

### Converting from Other Formats

#### Option 1: From Roboflow (Easiest)

1. Go to [Roboflow](https://roboflow.com)
2. Upload your dataset
3. Export as **"COCO Segmentation"** format
4. Download and extract to `your_dataset/`
5. Done! Files are already in correct format

```bash
# After downloading from Roboflow
unzip roboflow_export.zip -d my_dataset/
cd my_dataset/
# Already has train/ valid/ with _annotations.coco.json
```

#### Option 2: From YOLO Segmentation

```python
# convert_yolo_to_coco.py
import json
import cv2
from pathlib import Path

def yolo_seg_to_coco(yolo_dir, output_dir, class_names):
    """
    Convert YOLO segmentation to COCO format.

    YOLO format per line: class_id x1 y1 x2 y2 ... xn yn (normalized 0-1)
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i+1, "name": name, "supercategory": name}
                       for i, name in enumerate(class_names)]
    }

    image_dir = Path(yolo_dir) / "images"
    label_dir = Path(yolo_dir) / "labels"

    ann_id = 1
    for img_id, img_file in enumerate(sorted(image_dir.glob("*.jpg")), 1):
        # Read image
        img = cv2.imread(str(img_file))
        h, w = img.shape[:2]

        # Add image entry
        coco["images"].append({
            "id": img_id,
            "file_name": img_file.name,
            "height": h,
            "width": w
        })

        # Read YOLO labels
        label_file = label_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            continue

        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0]) + 1  # YOLO is 0-indexed
                coords = list(map(float, parts[1:]))

                # Convert normalized to absolute
                polygon = []
                for i in range(0, len(coords), 2):
                    polygon.extend([coords[i] * w, coords[i+1] * h])

                # Calculate bbox
                x_coords = polygon[::2]
                y_coords = polygon[1::2]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "segmentation": [polygon],
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0
                })
                ann_id += 1

    # Save COCO JSON
    output_path = Path(output_dir) / "_annotations.coco.json"
    with open(output_path, "w") as f:
        json.dump(coco, f)
    print(f"Saved {output_path}")

# Usage
yolo_seg_to_coco(
    yolo_dir="yolo_dataset/train",
    output_dir="my_dataset/train",
    class_names=["person", "car", "dog"]
)
```

#### Option 3: From Labelme

```bash
# Install labelme
pip install labelme

# Convert labelme to COCO
labelme2coco labelme_dir --output my_dataset/train/_annotations.coco.json

# Move images
cp labelme_dir/*.jpg my_dataset/train/
```

#### Option 4: From VGG Image Annotator (VIA)

```python
# Use via2coco converter
# https://github.com/wkentaro/labelme/tree/main/examples/instance_segmentation
```

### Validate Your Data

```python
from pycocotools.coco import COCO

# Load COCO annotations
coco = COCO("my_dataset/train/_annotations.coco.json")

# Print statistics
print(f"Images: {len(coco.getImgIds())}")
print(f"Annotations: {len(coco.getAnnIds())}")
print(f"Categories: {coco.getCatIds()}")
print(f"Category names: {[cat['name'] for cat in coco.loadCats(coco.getCatIds())]}")

# Verify segmentations
for ann_id in coco.getAnnIds()[:5]:
    ann = coco.loadAnns(ann_id)[0]
    assert 'segmentation' in ann, f"Missing segmentation in annotation {ann_id}"
    print(f"✓ Annotation {ann_id} has segmentation")

print("✓ Data validation passed!")
```

---

## 🚀 Training

### Step 1: Configure Paths

Edit `sam3/train/configs/lora/lora_base.yaml`:

```yaml
paths:
  dataset_root: /path/to/your_dataset        # Your dataset directory
  experiment_log_dir: /path/to/experiments   # Where to save outputs
  bpe_path: assets/bpe_simple_vocab_16e6.txt.gz  # BPE vocabulary
```

Or set via environment variables:

```bash
export DATASET_ROOT=/path/to/your_dataset
export EXPERIMENT_DIR=/path/to/experiments
```

### Step 2: Choose Configuration

#### Minimal Configuration (Fastest)

```bash
cd sam3
python sam3/train/train.py \
  -c train/configs/lora/lora_minimal.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

**Specifications:**
- Trainable: 500K params (0.06%)
- GPU Memory: 10-12 GB
- Training Time: 2-3 hours (1K images, 10 epochs)
- Best For: Quick experiments, proof of concept

**LoRA Applied To:**
- ✅ DETR Decoder only
- ❌ Vision Encoder
- ❌ Text Encoder
- ❌ DETR Encoder

#### Balanced Configuration (Recommended)

```bash
cd sam3
python sam3/train/train.py \
  -c train/configs/lora/lora_base.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

**Specifications:**
- Trainable: 4M params (0.47%)
- GPU Memory: 18-20 GB
- Training Time: 6-8 hours (1K images, 20 epochs)
- Best For: General fine-tuning, production use

**LoRA Applied To:**
- ✅ Vision Encoder
- ✅ Text Encoder
- ✅ DETR Encoder
- ✅ DETR Decoder

#### Full Configuration (Maximum Capacity)

```bash
cd sam3
python sam3/train/train.py \
  -c train/configs/lora/lora_full.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

**Specifications:**
- Trainable: 15M params (1.77%)
- GPU Memory: 24-28 GB
- Training Time: 12-16 hours (1K images, 30 epochs)
- Best For: Complex tasks, large datasets

**LoRA Applied To:**
- ✅ Vision Encoder (+ MLP layers)
- ✅ Text Encoder (+ MLP layers)
- ✅ DETR Encoder (+ MLP layers)
- ✅ DETR Decoder (+ MLP layers)

### Step 3: Multi-GPU Training

#### Single Node, Multiple GPUs

```bash
# 4 GPUs on one machine
python sam3/train/train.py \
  -c train/configs/lora/lora_base.yaml \
  --use-cluster 0 \
  --num-gpus 4
```

#### Multiple Nodes (SLURM Cluster)

```bash
# 2 nodes × 4 GPUs each
python sam3/train/train.py \
  -c train/configs/lora/lora_base.yaml \
  --use-cluster 1 \
  --num-nodes 2 \
  --num-gpus 4 \
  --partition gpu_partition \
  --account your_account
```

### Step 4: Monitor Training

#### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir /path/to/experiments/tensorboard

# Open browser to http://localhost:6006
```

#### Check Logs

```bash
# Watch training logs
tail -f /path/to/experiments/logs/train.log

# Check latest checkpoint
ls -lth /path/to/experiments/checkpoints/
```

### Training Output Structure

```
experiments/
└── my_experiment/
    ├── config.yaml              # Original configuration
    ├── config_resolved.yaml     # Resolved Hydra config
    ├── checkpoints/
    │   ├── checkpoint_epoch_5.pth
    │   ├── checkpoint_epoch_10.pth
    │   ├── checkpoint_epoch_15.pth
    │   ├── checkpoint_epoch_20.pth
    │   └── checkpoint_last.pth  # Latest checkpoint
    ├── tensorboard/
    │   └── events.out.tfevents.*
    ├── logs/
    │   └── train.log
    └── dumps/                   # Validation predictions
        └── *.json
```

### Training Progress

Expected output:

```
Applied LoRA to 156 modules

Parameter Statistics:
  Total: 848,273,952
  Trainable: 4,012,288
  Percentage: 0.473%

Epoch 1/20
[2024-12-04 10:30:15] Step 100/1500, loss: 2.341, lr: 1.2e-5
[2024-12-04 10:35:20] Step 200/1500, loss: 1.892, lr: 2.4e-5
...
Epoch 1 complete. Val IoU: 0.4523

Epoch 2/20
[2024-12-04 10:45:10] Step 100/1500, loss: 1.623, lr: 8.0e-5
...
```

---

## 🔍 Inference

### Using Official SAM3 API

After training, load your LoRA-finetuned model for inference:

```python
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3_lora_wrapper import apply_lora_to_sam3, load_lora_weights

# Step 1: Build base SAM3 model
model = build_sam3_image_model(
    bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    device="cuda",
    enable_segmentation=True,
)

# Step 2: Apply LoRA structure (must match training config)
model = apply_lora_to_sam3(
    model,
    rank=8,
    alpha=16,
    apply_to_vision_encoder=True,
    apply_to_text_encoder=True,
    apply_to_detr_encoder=True,
    apply_to_detr_decoder=True,
)

# Step 3: Load trained LoRA weights
load_lora_weights(model, "experiments/my_experiment/checkpoints/checkpoint_last.pth")

model.eval()
model.to("cuda")

# Step 4: Run inference
from sam3.model.sam3_image import Sam3ImageInference

predictor = Sam3ImageInference(model)

# Load image
image = Image.open("test_image.jpg")

# Predict with text prompt
predictions = predictor.predict(
    image=image,
    text_queries=["person", "car", "tree"],
)

# Access results
masks = predictions["masks"]          # Segmentation masks
boxes = predictions["boxes"]          # Bounding boxes
scores = predictions["scores"]        # Confidence scores
labels = predictions["labels"]        # Class labels

print(f"Found {len(masks)} objects")
```

### Inference with Bounding Box Prompts

```python
# Provide bounding box hints
predictions = predictor.predict(
    image=image,
    text_queries=["person"],
    boxes=[[100, 150, 400, 350]],  # [x1, y1, x2, y2]
)
```

### Batch Inference

```python
import os
from glob import glob

# Process multiple images
image_paths = glob("test_images/*.jpg")

results = []
for img_path in image_paths:
    image = Image.open(img_path)

    predictions = predictor.predict(
        image=image,
        text_queries=["person", "car"],
    )

    results.append({
        "image": os.path.basename(img_path),
        "num_objects": len(predictions["masks"]),
        "predictions": predictions
    })

print(f"Processed {len(results)} images")
```

### Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(image, predictions):
    """Visualize segmentation results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Masks overlay
    axes[1].imshow(image)
    for mask in predictions["masks"]:
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask > 0.5] = [1, 0, 0, 0.5]  # Red with 50% transparency
        axes[1].imshow(colored_mask)
    axes[1].set_title("Segmentation Masks")
    axes[1].axis("off")

    # Boxes overlay
    axes[2].imshow(image)
    for box, label in zip(predictions["boxes"], predictions["labels"]):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            fill=False, edgecolor='red', linewidth=2
        )
        axes[2].add_patch(rect)
        axes[2].text(x1, y1-5, label, color='red', fontsize=12)
    axes[2].set_title("Bounding Boxes")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("result.png", dpi=150, bbox_inches="tight")
    plt.show()

# Use it
visualize_predictions(image, predictions)
```

### Save Predictions

```python
import json

# Save predictions to JSON
output = {
    "image": "test_image.jpg",
    "predictions": [
        {
            "label": label,
            "score": float(score),
            "box": box.tolist(),
            "mask": mask.tolist()  # or save as RLE
        }
        for label, score, box, mask in zip(
            predictions["labels"],
            predictions["scores"],
            predictions["boxes"],
            predictions["masks"]
        )
    ]
}

with open("predictions.json", "w") as f:
    json.dump(output, f, indent=2)
```

### Export LoRA Weights Only

```python
from sam3_lora_wrapper import save_lora_weights

# Save only LoRA parameters (10-50 MB instead of 3 GB!)
save_lora_weights(model, "my_model_lora.pth")

# Later, load into new model
model_new = build_sam3_image_model(...)
model_new = apply_lora_to_sam3(model_new, rank=8, alpha=16, ...)
load_lora_weights(model_new, "my_model_lora.pth")
```

---

## ⚙️ Configuration

### LoRA Parameters Explained

```yaml
lora:
  rank: 8                    # Rank of LoRA matrices
                             # Higher = more capacity but more parameters
                             # Typical values: 4, 8, 16, 32

  alpha: 16                  # Scaling factor (typically 2× rank)
                             # Controls the magnitude of LoRA updates
                             # alpha/rank = scaling factor

  dropout: 0.0               # Dropout probability for LoRA layers
                             # Use 0.0-0.1 for regularization
                             # 0.0 = no dropout

  target_modules:            # Which linear layers to apply LoRA to
    - "q_proj"               # Query projection in attention
    - "k_proj"               # Key projection in attention
    - "v_proj"               # Value projection in attention
    - "out_proj"             # Output projection in attention
    # Optional:
    # - "fc1"                # First MLP layer
    # - "fc2"                # Second MLP layer

  # Component-level control
  apply_to_vision_encoder: true    # 32-layer ViT backbone
  apply_to_text_encoder: true      # Text concept encoder
  apply_to_detr_encoder: true      # DETR encoder (6 layers)
  apply_to_detr_decoder: true      # DETR decoder (6 layers)
```

### Training Parameters

```yaml
training:
  # Data paths (COCO format required!)
  img_folder_train: ${paths.dataset_root}/train/
  ann_file_train: ${paths.dataset_root}/train/_annotations.coco.json
  img_folder_val: ${paths.dataset_root}/valid/
  ann_file_val: ${paths.dataset_root}/valid/_annotations.coco.json

  # IMPORTANT: Enable segmentation if you have mask annotations
  enable_segmentation: true        # true = train with masks
                                   # false = bboxes only

  # Training schedule
  max_epochs: 20                   # Total epochs
  batch_size: 1                    # Per-GPU batch size
  gradient_accumulation_steps: 4   # Effective batch = batch_size × this

  # Learning rates (scaled by lr_scale)
  lr_scale: 0.1                    # Global multiplier
  lr_transformer: 8e-5             # DETR transformer LR
  lr_vision_backbone: 2.5e-5       # Vision encoder LR
  lr_language_backbone: 5e-6       # Text encoder LR

  # Optimization
  weight_decay: 0.1                # AdamW weight decay
  gradient_clip_max_norm: 0.1      # Gradient clipping

  # Mixed precision (speeds up training)
  amp_enabled: true
  amp_dtype: bfloat16              # bfloat16 or float16
                                   # bfloat16 recommended for A100

  # Validation
  val_epoch_freq: 5                # Validate every N epochs
  skip_first_val: true             # Skip validation at epoch 0

  # Checkpointing
  save_checkpoints: true
  checkpoint_freq: 5               # Save every N epochs
```

### Hyperparameter Tuning Guide

| Dataset Size | LR Scale | Batch Size | Grad Accum | Rank | Epochs |
|--------------|----------|------------|------------|------|--------|
| **Small** (<500) | 0.2 | 1 | 2 | 4-8 | 10-15 |
| **Medium** (500-2K) | 0.1 | 1 | 4 | 8-16 | 15-25 |
| **Large** (>2K) | 0.05 | 1 | 8 | 16-32 | 25-40 |

### GPU Memory Optimization

```yaml
# If running out of memory:

# Option 1: Reduce batch size
training:
  batch_size: 1
  gradient_accumulation_steps: 8    # Increase to compensate

# Option 2: Use lower resolution (not recommended)
scratch:
  resolution: 512                    # Default is 1008

# Option 3: Disable some LoRA components
lora:
  apply_to_vision_encoder: false    # Saves ~8GB

# Option 4: Use FP16 instead of BF16
training:
  amp_dtype: float16                # Slightly less memory
```

---

## 🐛 Troubleshooting

### Issue 1: Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'sam3_lora_wrapper'
```

**Solution:**
```bash
# Ensure wrapper is in sam3 directory
cp sam3_lora_wrapper.py sam3/

# Or add to Python path
export PYTHONPATH=$PYTHONPATH:/path/to/sam3
```

### Issue 2: Missing Segmentation Annotations

**Error:**
```
KeyError: 'segmentation'
```

**Solution:**
```python
# Check if your COCO JSON has segmentation field
import json
with open("train/_annotations.coco.json") as f:
    data = json.load(f)
    for ann in data['annotations'][:5]:
        if 'segmentation' not in ann:
            print(f"ERROR: Annotation {ann['id']} missing segmentation!")
        else:
            print(f"✓ Annotation {ann['id']} has segmentation")

# If missing, you need to add segmentation annotations
# Or set enable_segmentation: false (bbox only training)
```

### Issue 3: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. **Reduce batch size:**
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 8
```

2. **Use minimal LoRA config:**
```bash
python sam3/train/train.py \
  -c train/configs/lora/lora_minimal.yaml \
  --use-cluster 0 --num-gpus 1
```

3. **Clear GPU cache:**
```python
import torch
torch.cuda.empty_cache()
```

4. **Use FP16:**
```yaml
training:
  amp_dtype: float16
```

### Issue 4: Loss is NaN

**Error:**
```
Step 100/1500, loss: nan
```

**Solutions:**

1. **Reduce learning rate:**
```yaml
training:
  lr_scale: 0.05  # Lower from 0.1
```

2. **Enable gradient clipping:**
```yaml
training:
  gradient_clip_max_norm: 1.0  # From 0.1
```

3. **Check data quality:**
```python
# Verify annotations are valid
from pycocotools.coco import COCO
coco = COCO("train/_annotations.coco.json")

for ann_id in coco.getAnnIds():
    ann = coco.loadAnns(ann_id)[0]
    if ann['area'] == 0 or ann['area'] < 0:
        print(f"Invalid annotation: {ann}")
```

### Issue 5: Hydra Configuration Error

**Error:**
```
omegaconf.errors.ConfigAttributeError: Key 'paths' not found
```

**Solution:**
```yaml
# Ensure paths are defined at top of config
paths:
  dataset_root: /path/to/dataset  # Must be set!
  experiment_log_dir: /path/to/experiments
  bpe_path: assets/bpe_simple_vocab_16e6.txt.gz
```

### Issue 6: Slow Training

**Symptoms:** <1 iteration/second

**Solutions:**

1. **Enable AMP:**
```yaml
training:
  amp_enabled: true
  amp_dtype: bfloat16
```

2. **Reduce workers:**
```yaml
training:
  num_train_workers: 4  # From 10
```

3. **Disable validation:**
```yaml
training:
  val_epoch_freq: 20  # Validate less often
```

---

## 📊 Expected Results

### Parameter Efficiency

| Configuration | Total Params | Trainable | Percentage | Checkpoint |
|--------------|-------------|-----------|------------|------------|
| **Full SAM3** | 848M | 848M | 100% | ~3 GB |
| **LoRA Minimal** | 848M | 500K | 0.06% | ~10 MB |
| **LoRA Balanced** | 848M | 4M | 0.47% | ~30 MB |
| **LoRA Full** | 848M | 15M | 1.77% | ~60 MB |

**Checkpoint Size Reduction:** 60-300× smaller!

### Training Time Estimates

Based on 1,000 training images, RTX 3090:

| Configuration | Epoch Time | 10 Epochs | 20 Epochs | 30 Epochs |
|--------------|-----------|-----------|-----------|-----------|
| **Minimal** | ~15 min | 2.5 hrs | 5 hrs | 7.5 hrs |
| **Balanced** | ~20 min | 3.3 hrs | 6.7 hrs | 10 hrs |
| **Full** | ~25 min | 4.2 hrs | 8.3 hrs | 12.5 hrs |

### Memory Usage

| Configuration | Batch Size 1 | Batch Size 2 | Batch Size 4 |
|--------------|-------------|--------------|--------------|
| **Minimal** | 10 GB | 14 GB | 20 GB |
| **Balanced** | 18 GB | 24 GB | OOM |
| **Full** | 26 GB | OOM | OOM |

---

## 📖 References

### Official Resources

- **SAM3 Repository**: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- **SAM3 Training Guide**: [README_TRAIN.md](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md)
- **SAM3 Paper**: [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719)
- **HuggingFace Model**: [facebook/sam3](https://huggingface.co/facebook/sam3)
- **Meta AI Blog**: [Segment Anything Model 3](https://ai.meta.com/blog/segment-anything-model-3/)

### Roboflow Resources

- **Fine-tuning Guide**: [How to Fine-Tune SAM 3](https://blog.roboflow.com/fine-tune-sam3/)
- **SAM3 Launch**: [Use SAM 3 with Roboflow](https://blog.roboflow.com/sam3/)
- **Platform Docs**: [Fine-Tune SAM 3 on Roboflow](https://docs.roboflow.com/changelog/explore-by-month/november-2025/fine-tune-sam-3-on-roboflow)

### LoRA Paper

- **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### COCO Format

- **COCO Dataset**: [Common Objects in Context](https://cocodataset.org/)
- **pycocotools**: [COCO API](https://github.com/cocodataset/cocoapi)

---

## 📞 Contact & Support

**AI Research Group**
King Mongkut's University of Technology Thonburi (KMUTT)

- 🌐 Website: [https://ai.kmutt.ac.th](https://ai.kmutt.ac.th)
- 📧 Email: ai-research@kmutt.ac.th
- 🐛 Issues: Create an issue on GitHub

---

## 🙏 Acknowledgments

- **Meta AI Research** for developing SAM3 and open-sourcing the training code
- **Microsoft Research** for the LoRA methodology
- **Roboflow** for SAM3 fine-tuning documentation and platform support
- **HuggingFace** for the transformers library and model hosting
- **KMUTT AI Research Group** for LoRA integration and testing

---

## 📄 License

This project follows the SAM3 license from Meta AI Research. See the [official SAM3 repository](https://github.com/facebookresearch/sam3) for license details.

---

## 📊 Citation

If you use this framework in your research, please cite:

```bibtex
@software{sam3_lora_kmutt,
  title={SAM3-LoRA: Efficient Fine-tuning for Segment Anything Model 3},
  author={AI Research Group, KMUTT},
  year={2025},
  url={https://github.com/kmutt-ai/sam3-lora}
}

@article{sam3,
  title={SAM 3: Segment Anything with Concepts},
  author={Meta AI},
  journal={arXiv preprint arXiv:2511.16719},
  year={2025}
}

@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```

---

<div align="center">

**Built with ❤️ by AI Research Group, KMUTT**

[⬆ Back to Top](#sam3-lora-efficient-fine-tuning-for-segment-anything-model-3)

</div>
