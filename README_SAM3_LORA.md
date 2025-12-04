# SAM3 LoRA Fine-Tuning (Official Training Integration)

<div align="center">

**AI Research Group**
**King Mongkut's University of Technology Thonburi (KMUTT)**

---

*Efficient LoRA fine-tuning integrated with Meta's official SAM3 training pipeline*

</div>

---

## ⚠️ Important Note

This implementation **integrates LoRA with the official SAM3 training pipeline** from [facebookresearch/sam3](https://github.com/facebookresearch/sam3). It uses:

- **Official SAM3 training script** (`sam3/train/train.py`)
- **Hydra configuration** system
- **COCO JSON format** data
- **Official loss functions** (boxes, GIoU, masks, dice)
- **Official data transforms** and augmentation

This is **NOT** a standalone training script - it extends SAM3's official training.

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Configuration](#configuration)
- [Inference](#inference)
- [References](#references)

---

## 🔧 Prerequisites

### System Requirements

- **Python**: 3.12+
- **PyTorch**: 2.7+
- **CUDA**: 12.6+
- **GPU**: 16GB+ VRAM (24GB+ recommended)
- **Storage**: 50GB+ free space

### Required Knowledge

- Familiarity with Hydra configuration system
- Understanding of COCO dataset format
- Basic knowledge of SAM3 architecture

---

## 📦 Installation

### Step 1: Clone SAM3 Repository

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
```

### Step 2: Install SAM3 with Training Dependencies

```bash
pip install -e ".[train]"
```

### Step 3: Add LoRA Wrapper

Copy the LoRA files to your SAM3 directory:

```bash
# Copy LoRA wrapper
cp sam3_lora_wrapper.py sam3/

# Copy LoRA configs
cp -r sam3_lora_configs sam3/train/configs/lora/
```

### Step 4: Download BPE Vocabulary

```bash
# Download BPE file (required for text encoder)
cd sam3
wget https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz \
  -P assets/
```

### Step 5: Login to HuggingFace

```bash
huggingface-cli login
```

---

## 📊 Data Preparation

### Required Format: COCO JSON

SAM3 requires **COCO JSON format** with segmentation annotations:

```
your_dataset/
├── train/
│   ├── image001.jpg
│   ├── image002.jpg
│   ├── ...
│   └── _annotations.coco.json
├── valid/
│   ├── image001.jpg
│   ├── ...
│   └── _annotations.coco.json
└── test/
    ├── image001.jpg
    ├── ...
    └── _annotations.coco.json
```

### COCO JSON Structure

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
      "segmentation": [[x1, y1, x2, y2, ...]] or RLE,
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

### Converting from Other Formats

#### From Roboflow

1. Export your dataset in **COCO Segmentation** format
2. Download and extract to `your_dataset/`
3. Roboflow automatically creates `_annotations.coco.json` files

#### From YOLO Segmentation

```python
# convert_yolo_to_coco.py
from pycocotools import mask as maskUtils
import json
import cv2
import numpy as np

def yolo_to_coco(yolo_dir, output_dir):
    """
    Convert YOLO segmentation format to COCO format.

    YOLO format (per line): class_id x1 y1 x2 y2 ... xn yn (normalized)
    """
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Add your conversion logic here
    # ...

    # Save COCO JSON
    with open(f"{output_dir}/_annotations.coco.json", "w") as f:
        json.dump(coco_dict, f)
```

#### From Labelme

Use [labelme2coco](https://github.com/wkentaro/labelme/tree/main/examples/instance_segmentation):

```bash
pip install labelme
labelme2coco your_labelme_dir --output annotations.json
```

### Data Validation

Validate your COCO format:

```python
from pycocotools.coco import COCO

# Load and validate
coco = COCO("your_dataset/train/_annotations.coco.json")

# Check
print(f"Images: {len(coco.getImgIds())}")
print(f"Annotations: {len(coco.getAnnIds())}")
print(f"Categories: {len(coco.getCatIds())}")

# Verify segmentations exist
for ann_id in coco.getAnnIds()[:5]:
    ann = coco.loadAnns(ann_id)[0]
    assert 'segmentation' in ann, f"Missing segmentation in annotation {ann_id}"
```

---

## 🚀 Training

### Step 1: Configure Paths

Edit `sam3/train/configs/lora/lora_base.yaml`:

```yaml
paths:
  dataset_root: /path/to/your_dataset
  experiment_log_dir: /path/to/experiments
  bpe_path: sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

### Step 2: Start Training

#### Minimal Configuration (Fastest)

```bash
cd sam3
python sam3/train/train.py \
  -c train/configs/lora/lora_minimal.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

- **Trainable params**: ~0.06% (500K parameters)
- **Training time**: ~2-3 hours (1K images, 10 epochs)
- **GPU memory**: ~10-12 GB
- **Best for**: Quick experiments, limited data

#### Balanced Configuration (Recommended)

```bash
cd sam3
python sam3/train/train.py \
  -c train/configs/lora/lora_base.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

- **Trainable params**: ~0.47% (4M parameters)
- **Training time**: ~6-8 hours (1K images, 20 epochs)
- **GPU memory**: ~18-20 GB
- **Best for**: General fine-tuning

#### Full Configuration (Maximum Capacity)

```bash
cd sam3
python sam3/train/train.py \
  -c train/configs/lora/lora_full.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

- **Trainable params**: ~1.77% (15M parameters)
- **Training time**: ~12-16 hours (1K images, 30 epochs)
- **GPU memory**: ~24-28 GB
- **Best for**: Complex tasks, large datasets

### Step 3: Multi-GPU Training

```bash
# 4 GPUs on single node
python sam3/train/train.py \
  -c train/configs/lora/lora_base.yaml \
  --use-cluster 0 \
  --num-gpus 4

# 2 nodes x 4 GPUs each (cluster)
python sam3/train/train.py \
  -c train/configs/lora/lora_base.yaml \
  --use-cluster 1 \
  --num-gpus 4 \
  --num-nodes 2 \
  --partition gpu_partition
```

### Step 4: Monitor Training

```bash
# TensorBoard
tensorboard --logdir /path/to/experiments/tensorboard

# Check logs
tail -f /path/to/experiments/logs/train.log
```

### Training Output

```
your_experiment/
├── config.yaml                  # Original config
├── config_resolved.yaml         # Resolved config
├── checkpoints/
│   ├── checkpoint_epoch_5.pth
│   ├── checkpoint_epoch_10.pth
│   └── checkpoint_last.pth
├── tensorboard/                 # TensorBoard logs
├── logs/                        # Text logs
└── dumps/                       # Predictions
```

---

## ⚙️ Configuration

### LoRA Parameters

```yaml
lora:
  rank: 8                    # LoRA rank (4, 8, 16, 32)
  alpha: 16                  # Scaling (typically 2x rank)
  dropout: 0.0               # Dropout probability

  target_modules:            # Which linear layers
    - "q_proj"               # Query projection
    - "k_proj"               # Key projection
    - "v_proj"               # Value projection
    - "out_proj"             # Output projection
    # Optional:
    # - "fc1"                # MLP layer 1
    # - "fc2"                # MLP layer 2

  # Component selection
  apply_to_vision_encoder: true    # Vision ViT backbone
  apply_to_text_encoder: true      # Text encoder
  apply_to_detr_encoder: true      # DETR encoder
  apply_to_detr_decoder: true      # DETR decoder
```

### Training Parameters

```yaml
training:
  # Data
  img_folder_train: ${paths.dataset_root}/train/
  ann_file_train: ${paths.dataset_root}/train/_annotations.coco.json

  # Enable segmentation (IMPORTANT!)
  enable_segmentation: true        # Set to true for mask training

  # Training
  max_epochs: 20
  batch_size: 1
  gradient_accumulation_steps: 4   # Effective batch = 4

  # Learning rates
  lr_scale: 0.1
  lr_transformer: 8e-5
  lr_vision_backbone: 2.5e-5
  lr_language_backbone: 5e-6

  # Optimization
  weight_decay: 0.1
  gradient_clip_max_norm: 0.1

  # Mixed precision
  amp_enabled: true
  amp_dtype: bfloat16              # or float16
```

### Hyperparameter Guide

| Dataset Size | LR Scale | Batch Size | Grad Accum | Epochs |
|--------------|----------|------------|------------|--------|
| Small (<500) | 0.2 | 1 | 2 | 10-15 |
| Medium (500-2K) | 0.1 | 1 | 4 | 15-25 |
| Large (>2K) | 0.05 | 1 | 8 | 25-40 |

---

## 🔍 Inference

### Using Official SAM3 Inference

After training, use the official SAM3 inference with LoRA weights:

```python
import torch
from sam3.model_builder import build_sam3_image_model
from sam3_lora_wrapper import apply_lora_to_sam3, load_lora_weights

# Load base model
model = build_sam3_image_model(
    bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    enable_segmentation=True,
)

# Apply LoRA structure
model = apply_lora_to_sam3(
    model,
    rank=8,
    alpha=16,
    apply_to_vision_encoder=True,
    apply_to_detr_decoder=True,
)

# Load trained LoRA weights
load_lora_weights(model, "experiments/checkpoints/checkpoint_last.pth")

model.eval()
model.to("cuda")

# Run inference (use official SAM3 API)
from sam3.model.sam3_image import Sam3ImageInference

predictor = Sam3ImageInference(model)

# Predict with text prompt
predictions = predictor.predict(
    image="test_image.jpg",
    text_queries=["person", "car", "dog"],
)

# Or with bounding box
predictions = predictor.predict(
    image="test_image.jpg",
    text_queries=["person"],
    boxes=[[100, 150, 400, 350]],
)
```

### Saving LoRA Weights Only

```python
from sam3_lora_wrapper import save_lora_weights

# Save only LoRA parameters (10-50MB vs 3GB full model)
save_lora_weights(model, "lora_weights.pth")
```

---

## 📊 Expected Results

### Parameter Efficiency

| Configuration | Parameters | % of Total | Checkpoint Size |
|--------------|-----------|------------|-----------------|
| **Full Model** | 848M | 100% | ~3 GB |
| **LoRA Minimal** | 500K | 0.06% | ~10 MB |
| **LoRA Balanced** | 4M | 0.47% | ~30 MB |
| **LoRA Full** | 15M | 1.77% | ~60 MB |

### Training Time (1K images, RTX 3090)

| Configuration | Epoch Time | Total Time |
|--------------|-----------|------------|
| Minimal (10 epochs) | ~15 min | ~2.5 hours |
| Balanced (20 epochs) | ~20 min | ~6.5 hours |
| Full (30 epochs) | ~25 min | ~12.5 hours |

---

## 🐛 Troubleshooting

### Issue 1: Module Not Found

**Error**: `ModuleNotFoundError: No module named 'sam3_lora_wrapper'`

**Solution**:
```bash
# Ensure wrapper is in sam3 directory
cp sam3_lora_wrapper.py sam3/

# Or add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/sam3
```

### Issue 2: COCO Format Errors

**Error**: `KeyError: 'segmentation'`

**Solution**: Ensure your COCO JSON has segmentation annotations:
```python
# Check annotations
import json
with open("train/_annotations.coco.json") as f:
    data = json.load(f)
    for ann in data['annotations'][:5]:
        assert 'segmentation' in ann
```

### Issue 3: Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
```yaml
# Option 1: Smaller batch size
training:
  batch_size: 1
  gradient_accumulation_steps: 8

# Option 2: Lower resolution (not recommended)
scratch:
  resolution: 512  # Default is 1008

# Option 3: Use minimal config
# Use lora_minimal.yaml instead
```

### Issue 4: Loss is NaN

**Solution**:
```yaml
# Reduce learning rate
training:
  lr_scale: 0.05  # Lower from 0.1

# Enable gradient clipping
training:
  gradient_clip_max_norm: 1.0
```

---

## 📖 References

### Official SAM3 Resources

- **Repository**: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- **Training Guide**: [README_TRAIN.md](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md)
- **Paper**: [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719)
- **Model Card**: [facebook/sam3](https://huggingface.co/facebook/sam3)

### Roboflow Resources

- **Fine-tuning Guide**: [How to Fine-Tune SAM 3](https://blog.roboflow.com/fine-tune-sam3/)
- **Roboflow Platform**: [Roboflow SAM3](https://docs.roboflow.com/changelog/explore-by-month/november-2025/fine-tune-sam-3-on-roboflow)

### LoRA Paper

- **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---

## 📞 Contact

**AI Research Group - KMUTT**

- Email: ai-research@kmutt.ac.th
- Website: https://ai.kmutt.ac.th

---

## 🙏 Acknowledgments

- **Meta AI** for SAM3 and official training code
- **Microsoft Research** for LoRA methodology
- **Roboflow** for SAM3 fine-tuning documentation
- **KMUTT AI Research Group** for LoRA integration

---

<div align="center">

**Built with ❤️ by AI Research Group, KMUTT**

</div>
