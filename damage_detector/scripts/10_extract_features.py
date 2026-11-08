"""Step 10: CLIP features for 9 crops of every image, plus a label for every crop.
Run from the project root after 09_normalize.py (re-run any time: images already done are cached):
    python scripts/10_extract_features.py
    python scripts/10_extract_features.py --dry-run     # random features, tests the wiring in seconds

Each 512 x 512 image is cut into a 3 x 3 grid of 224 px crops (stride 144). CLIP ViT-L/14 with
OpenAI weights embeds each crop (768 numbers). Crop labels:
  real photos (real_damaged, base_original)  -> 0 for every crop
  whole_edit, full_gen                       -> 1 for every crop
  paste_back  -> 1 if at least 10% of the crop is inside the edit mask,
                 0 if the crop doesn't touch the (feather-grown) mask at all,
                -1 (ignored) in between
  round_trip  -> -1 for every crop (never trained on; scored only as a diagnostic)
Outputs
  features/cache/<id>.npy     per-image features, so re-runs only do new images
  features/feats.npy          (N, 9, 768) float16
  features/crop_labels.npy    (N, 9) int8
  features/index.csv          the N rows of manifests/dataset.csv, same order
"""
import argparse
import csv
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

M, F = Path("manifests"), Path("features")
CACHE = F / "cache"
EDIT = Path("masks/edit")
STARTS = (0, 144, 288)
BOXES = [(x, y, x + 224, y + 224) for y in STARTS for x in STARTS]   # row-major, 3 x 3
AI_FRAC = 0.10


def crops_of(im):
    return [im.crop(b) for b in BOXES]


def crop_labels(row):
    label, kind = int(row["label"]), row["kind"]
    if label == -1:
        return [-1] * 9
    if label == 0:
        return [0] * 9
    if kind != "paste_back":
        return [1] * 9
    mp = Path(row["mask_file"])
    m = np.asarray(Image.open(mp if mp.exists() else EDIT / mp).convert("L")) > 127
    f = float(row.get("feather_px") or 0)
    rad = int(math.ceil(1.5 * f)) + 2
    grown = cv2.dilate(m.astype(np.uint8), np.ones((2 * rad + 1, 2 * rad + 1), np.uint8)) > 0
    out = []
    for x0, y0, x1, y1 in BOXES:
        if m[y0:y1, x0:x1].mean() >= AI_FRAC:
            out.append(1)
        elif not grown[y0:y1, x0:x1].any():
            out.append(0)
        else:
            out.append(-1)
    return out


def load_clip():
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", precision="fp16", device="cuda", force_quick_gelu=True)
    model.eval()

    def embed(crop_lists):
        batch = torch.stack([preprocess(c) for cl in crop_lists for c in cl]).to("cuda", torch.float16)
        with torch.inference_mode():
            f = model.encode_image(batch)
        return f.float().cpu().numpy().reshape(len(crop_lists), 9, -1)
    return embed


_PROJ = np.random.default_rng(0).normal(size=(16 * 16 * 3, 768)).astype(np.float32)


def fake_embed(crop_lists):
    """--dry-run only: a fixed random projection of tiny crops, to test the wiring without CLIP."""
    return np.stack([np.stack([np.asarray(c.resize((16, 16)), np.float32).ravel() / 255.0
                               for c in cl]) @ _PROJ for cl in crop_lists])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=4, help="images per GPU batch (9 crops each)")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(M / "dataset.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    key = lambda r: hashlib.sha1(r["file"].encode()).hexdigest()[:16]
    todo = [r for r in rows if not (CACHE / f"{key(r)}.npy").exists()]
    print(f"{len(rows)} images, {len(todo)} need features")
    embed = fake_embed if args.dry_run else load_clip()
    for i in tqdm(range(0, len(todo), args.batch)):
        chunk = todo[i:i + args.batch]
        crops = [crops_of(Image.open(r["file"]).convert("RGB")) for r in chunk]
        feats = embed(crops)
        for r, fe in zip(chunk, feats):
            np.save(CACHE / f"{key(r)}.npy", fe.astype(np.float16))

    feats = np.stack([np.load(CACHE / f"{key(r)}.npy") for r in rows])
    labels = np.array([crop_labels(r) for r in rows], dtype=np.int8)
    np.save(F / "feats.npy", feats)
    np.save(F / "crop_labels.npy", labels)
    with open(F / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"features {feats.shape}, crop labels: AI {int((labels == 1).sum())}, "
          f"real {int((labels == 0).sum())}, ignored {int((labels == -1).sum())}")


if __name__ == "__main__":
    main()
