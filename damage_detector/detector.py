"""The detector the demo calls. Put this file in the project root (next to scripts/ and models/).

    from detector import score
    result = score(PIL.Image.open("photo.jpg"))
    # {"prob": 0.87, "flagged": True, "crop_probs": 3x3 array, "normalized": 512x512 PIL image}

It applies exactly the training-time processing: upright, LANCZOS short side 512, center crop,
JPEG quality 90 in memory, 3 x 3 grid of 224 px crops, CLIP ViT-L/14 (OpenAI weights,
QuickGELU), the main model's crop classifier, max over crops, calibration, threshold.
Set DETECTOR_MODEL=models/sd15_only.joblib (etc.) to demo another experiment.
"""
import io
import os
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
STARTS = (0, 144, 288)
BOXES = [(x, y, x + 224, y + 224) for y in STARTS for x in STARTS]
_state = {}


def _load():
    if not _state:
        import open_clip
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", precision="fp16" if dev == "cuda" else "fp32",
            device=dev, force_quick_gelu=True)
        model.eval()
        path = Path(os.environ.get("DETECTOR_MODEL", ROOT / "models" / "main.joblib"))
        _state.update(torch=torch, dev=dev, model=model, pre=preprocess, m=joblib.load(path))
    return _state


def normalize(img):
    im = ImageOps.exif_transpose(img).convert("RGB")
    w, h = im.size
    s = 512 / min(w, h)
    im = im.resize((max(512, round(w * s)), max(512, round(h * s))), Image.LANCZOS)
    w, h = im.size
    l, t = (w - 512) // 2, (h - 512) // 2
    im = im.crop((l, t, l + 512, t + 512))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def score(img):
    st = _load()
    torch = st["torch"]
    im = normalize(img)
    batch = torch.stack([st["pre"](im.crop(b)) for b in BOXES]).to(st["dev"])
    if st["dev"] == "cuda":
        batch = batch.half()
    with torch.inference_mode():
        f = st["model"].encode_image(batch).float().cpu().numpy()
    f /= np.linalg.norm(f, axis=-1, keepdims=True) + 1e-8
    m = st["m"]
    crop_p = m["clf"].predict_proba(f)[:, 1]
    prob = float(m["cal"].predict_proba([[crop_p.max()]])[0, 1])
    return {"prob": prob, "flagged": prob > m["threshold"],
            "crop_probs": crop_p.reshape(3, 3), "normalized": im}


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        r = score(Image.open(p))
        print(f"{p}: prob={r['prob']:.3f} flagged={r['flagged']}")
