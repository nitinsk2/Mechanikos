"""Step 4e: check every prepared base photo with Mask R-CNN and save its car mask.
Run from the project root after 04_prepare_bases.py:
    python scripts/05_check_bases.py

Outputs
  masks/car/base_0001.png ...      binary mask (255 = car) of the largest vehicle, 512x512
  manifests/base_car_check.csv     base_id, status, label, score, car_frac, n_vehicles
Status
  ok     largest vehicle covers >= 25% of the image: room for 3-15% edit blobs
  small  12-25%: usable, but only small edits (3-6%)
  fail   no vehicle, or < 12%: drop it or place the edit mask by hand
These car masks are the input to the edit-mask step (G1: masks to B and C by hour 3).
"""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_V2_Weights,
                                          maskrcnn_resnet50_fpn_v2)
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

BASES = Path("staging/real_base")
MASKS = Path("masks/car")
OUT = Path("manifests/base_car_check.csv")
VEHICLES = ("car", "truck", "bus")

dev = "cuda" if torch.cuda.is_available() else "cpu"
weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5).to(dev).eval()
cats = weights.meta["categories"]
MASKS.mkdir(parents=True, exist_ok=True)

rows = []
for p in tqdm(sorted(BASES.glob("base_*.png"))):
    im = Image.open(p).convert("RGB")
    with torch.inference_mode():
        out = model([to_tensor(im).to(dev)])[0]
    best, n = None, 0
    for i, lab in enumerate(out["labels"].tolist()):
        if cats[lab] not in VEHICLES:
            continue
        m = (out["masks"][i, 0] > 0.5).cpu().numpy()
        n += 1
        if best is None or m.sum() > best[0].sum():
            best = (m, cats[lab], float(out["scores"][i]))
    if best is None:
        rows.append(dict(base_id=p.stem, status="fail", label="", score=0, car_frac=0,
                         n_vehicles=0))
        continue
    m, lab, score = best
    frac = float(m.mean())
    status = "ok" if frac >= 0.25 else "small" if frac >= 0.12 else "fail"
    Image.fromarray((m * 255).astype(np.uint8)).save(MASKS / f"{p.stem}.png")
    rows.append(dict(base_id=p.stem, status=status, label=lab, score=round(score, 3),
                     car_frac=round(frac, 3), n_vehicles=n))

with open(OUT, "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=["base_id", "status", "label", "score", "car_frac",
                                       "n_vehicles"])
    wr.writeheader()
    wr.writerows(rows)
print(dict(Counter(r["status"] for r in rows)), "->", OUT)
print("Open a few masks/car/*.png next to their base photo to confirm the masks look right.")
