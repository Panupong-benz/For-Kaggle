import os
import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

from sam3.model_builder import build_sam3_image_model
from sam3.train.datasets.coco_seg_dataset import COCOSegmentDataset
from sam3.utils.mask_utils import masks_to_polygons


# =========================
# CONFIG
# =========================
CONFIG_PATH = "configs/full_lora_config.yaml"
LORA_WEIGHTS = "outputs/sam3_lora_full/best_lora_weights.pt"
VAL_DIR = "/kaggle/working/For-Kaggle/data"
OUTPUT_DIR = "vis_results"
NUM_SAMPLES = 5
DEVICE = "cuda"


# =========================
# Utils
# =========================
def overlay_mask(img, mask, color):
    overlay = img.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, 0.5, img, 0.5, 0)


# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model (image-only)...")
    model = build_sam3_image_model(
        apply_text_encoder=False  # 🔴 สำคัญมาก
    )

    ckpt = torch.load(LORA_WEIGHTS, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)

    model.to(DEVICE)
    model.eval()

    print("Loading validation dataset...")
    dataset = COCOSegmentDataset(VAL_DIR, split="valid")

    for idx in range(min(NUM_SAMPLES, len(dataset))):
        sample = dataset[idx]

        image = sample["image"].to(DEVICE).unsqueeze(0)
        gt_masks = sample["masks"].cpu().numpy()

        with torch.no_grad():
            outputs = model(image)

        pred_masks = outputs[0]["pred_masks"].sigmoid() > 0.5
        pred_masks = pred_masks.cpu().numpy()

        img_np = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # =========================
        # Visualization
        # =========================
        vis = img_np.copy()

        for m in gt_masks:
            vis = overlay_mask(vis, m, (0, 255, 0))  # GT = green

        for m in pred_masks:
            vis = overlay_mask(vis, m, (255, 0, 0))  # Pred = red

        save_path = f"{OUTPUT_DIR}/val_{idx}.png"
        cv2.imwrite(save_path, vis[:, :, ::-1])

        print(f"Saved: {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
