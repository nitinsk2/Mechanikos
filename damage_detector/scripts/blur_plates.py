"""Mark licence plates (or faces) to blur, before base-photo prep.
Run from the project root:
    python scripts/blur_plates.py raw/phone raw/wikimedia
For each image a window opens:
    drag a box over a readable plate -> press SPACE or ENTER to keep it
    repeat for more boxes             -> press ESC when done with this image
    no plate?                         -> just press ESC
Boxes are saved after every image to manifests/plate_blur.json (re-running skips images
already done), in the coordinates of the upright original. 04_prepare_bases.py applies them.
Needs the GUI build of OpenCV (opencv-python, not opencv-python-headless).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
OUT = Path("manifests/plate_blur.json")
MAX_SIDE = 1100

done = json.loads(OUT.read_text()) if OUT.exists() else {}
OUT.parent.mkdir(parents=True, exist_ok=True)
files = sorted(p for d in sys.argv[1:] for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)
todo = [p for p in files if p.as_posix() not in done]
print(f"{len(files)} images, {len(todo)} still to check")

for i, p in enumerate(todo, 1):
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    scale = min(1.0, MAX_SIDE / max(im.size))
    show = im.resize((round(im.width * scale), round(im.height * scale)))
    bgr = cv2.cvtColor(np.asarray(show), cv2.COLOR_RGB2BGR)
    title = f"[{i}/{len(todo)}] {p.name}  - box + SPACE per plate, ESC = next image"
    rois = cv2.selectROIs(title, bgr, showCrosshair=False)
    cv2.destroyAllWindows()
    boxes = [[round(x / scale), round(y / scale), round(w / scale), round(h / scale)]
             for x, y, w, h in (rois if len(rois) else []) if w > 2 and h > 2]
    done[p.as_posix()] = boxes
    OUT.write_text(json.dumps(done, indent=1))
    if boxes:
        print(f"   {p.name}: {len(boxes)} box(es)")
print("saved", OUT)
