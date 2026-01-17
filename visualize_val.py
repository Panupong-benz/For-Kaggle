import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sam3.model_builder import build_sam3_image_model

# =====================
# PATH CONFIG
# =====================
VAL_DIR = "/kaggle/working/For-Kaggle/data/valid"
ANN_FILE = os.path.join(VAL_DIR, "_annotations.coco.json")
WEIGHTS = "outputs/sam3_lora_full/best_lora_weights.pt"

DEVICE = "cuda"

# =====================
# LOAD COCO
# =====================
with open(ANN_FILE, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

# image_id -> anns
img_to_anns = {}
for ann in annotations:
    img_to_anns.setdefault(ann["image_id"], []).append(ann)

# =====================
# LOAD MODEL
# =====================
model = build_sam3_image_model()
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE), strict=False)
model.to(DEVICE)
model.eval()

# =====================
# HELPER
# =====================
def draw_gt(ax, anns):
    for ann in anns:
        if "segmentation" in ann:
            for seg in ann["segmentation"]:
                xs = seg[0::2]
                ys = seg[1::2]
                ax.plot(xs, ys, color="lime", linewidth=2)

def draw_pred(ax, masks):
    for m in masks:
        m = m.cpu().numpy()
        ys, xs = np.where(m > 0.5)
        ax.scatter(xs, ys, s=1, c="red")

# =====================
# VISUALIZE 5 IMAGES
# =====================
for img_meta in images[:5]:
    img_path = os.path.join(VAL_DIR, img_meta["file_name"])
    img = Image.open(img_path).convert("RGB")

    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)

    pred_masks = outputs[0]["masks"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    ax.set_title(img_meta["file_name"])

    # GT = สีเขียว
    draw_gt(ax, img_to_anns.get(img_meta["id"], []))

    # Prediction = สีแดง
    draw_pred(ax, pred_masks)

    ax.axis("off")
    plt.show()
