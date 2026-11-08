"""Step 2: download CLIP ViT-L/14 (OpenAI weights) and COCO Mask R-CNN, then smoke-test both
on one real car photo. Run from the project root:
    python scripts/01_models_smoke.py path/to/any_car_photo.jpg

First run downloads ~1.7 GB (CLIP) + ~180 MB (Mask R-CNN) into your user cache:
  CLIP        -> ~/.cache/huggingface/hub        (Windows: C:\\Users\\<you>\\.cache\\huggingface\\hub)
  Mask R-CNN  -> ~/.cache/torch/hub/checkpoints
Later runs load from cache and work offline.
"""
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageOps

if len(sys.argv) != 2:
    sys.exit("usage: python scripts/01_models_smoke.py path/to/car_photo.jpg")

DEV = "cuda"
assert torch.cuda.is_available(), "CUDA not available - run 00_check_env.py first"
img = ImageOps.exif_transpose(Image.open(sys.argv[1])).convert("RGB")


def center512(im):
    w, h = im.size
    s = 512 / min(w, h)
    im = im.resize((max(512, round(w * s)), max(512, round(h * s))), Image.LANCZOS)
    w, h = im.size
    l, t = (w - 512) // 2, (h - 512) // 2
    return im.crop((l, t, l + 512, t + 512))


# ---------------------------------------------------------------- CLIP ViT-L/14
import open_clip  # noqa: E402

t0 = time.time()
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai", precision="fp16", device=DEV, force_quick_gelu=True)
model.eval()
tok = open_clip.get_tokenizer("ViT-L-14")
print(f"CLIP loaded in {time.time() - t0:.0f}s")

prompts = ["a photo of a car", "a photo of a dog", "a photo of a plate of food",
           "a photo of a building"]
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    im_t = preprocess(img).unsqueeze(0).to(DEV, torch.float16)
    f = model.encode_image(im_t)
    t = model.encode_text(tok(prompts).to(DEV))
    f, t = f / f.norm(dim=-1, keepdim=True), t / t.norm(dim=-1, keepdim=True)
    probs = (100 * f @ t.T).softmax(-1)[0].float().cpu().numpy()
for p, q in zip(prompts, probs):
    print(f"   {q:.3f}  {p}")
assert probs.argmax() == 0, "CLIP did not pick 'car' - weights may be wrong or image is not a car"

# the exact feature shape the pipeline uses: 9 overlapping 224 crops of a 512x512 image
im512 = center512(img)
crops = [im512.crop((x, y, x + 224, y + 224)) for y in (0, 144, 288) for x in (0, 144, 288)]
batch = torch.stack([preprocess(c) for c in crops]).to(DEV, torch.float16)
with torch.inference_mode():
    torch.cuda.synchronize(); t0 = time.time()
    feats = model.encode_image(batch)
    torch.cuda.synchronize()
print(f"9-crop features: {tuple(feats.shape)} {feats.dtype}, {1000 * (time.time() - t0):.0f} ms, "
      f"finite={bool(torch.isfinite(feats).all())}")
assert feats.shape == (9, 768), "expected (9, 768)"
print(f"CLIP peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
del model, batch, feats
torch.cuda.empty_cache()

# ---------------------------------------------------------------- Mask R-CNN (COCO)
from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_V2_Weights,  # noqa: E402
                                          maskrcnn_resnet50_fpn_v2)
from torchvision.transforms.functional import to_tensor  # noqa: E402

weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
det = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5).to(DEV).eval()
cats = weights.meta["categories"]
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    out = det([to_tensor(im512).to(DEV)])[0]
vehicles = [i for i, l in enumerate(out["labels"].tolist()) if cats[l] in ("car", "truck", "bus")]
print(f"Mask R-CNN: {len(out['labels'])} detections, {len(vehicles)} vehicles")
for i in vehicles:
    m = (out["masks"][i, 0] > 0.5).float().mean().item()
    print(f"   {cats[out['labels'][i]]:6s} score={out['scores'][i]:.2f} mask covers {100 * m:.0f}% of image")
print(f"Mask R-CNN peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
assert vehicles, "no vehicle found - try a clearer car photo"
print("\nBOTH MODELS OK")
