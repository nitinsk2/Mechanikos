# Hand-off from A: base photos and masks for the edits

## What's in `handoff_A.zip`

| Path | What it is |
| --- | --- |
| `real_base/base_XXXX.png` | Undamaged real car photos, 512 × 512 RGB PNG. These are the **originals**: never overwrite them |
| `masks/edit/base_XXXX_eK.png` | Edit masks, 512 × 512, white (255) = the region to repaint with damage. One irregular blob inside the car, 3–15% of the image |
| `masks/car/base_XXXX.png` | Car masks (white = car). For reference; you don't need them to edit |
| `edit_masks.csv` | **Your work list.** One row per edit: `mask_file, base_id, edit_k, size_bucket, mask_frac, prompt, feather_px` |
| `base_photos.csv` | Source, license and author of every base photo |
| `base_car_check.csv` | Mask R-CNN check per base (only `ok` / `small` bases have edit masks) |

## The rules (both generators, identical)

1. **Work from `edit_masks.csv`.** For every row, inpaint `real_base/<base_id>.png` inside `masks/edit/<mask_file>` using that row's `prompt`. Both of you use **the same mask, prompt and feather** for every row. That's what makes the cross-generator test a matched pair.
2. **Paste back.** Only the masked region may change. Inpainting pipelines alter every pixel slightly (the VAE round trip), so always paste the generated region back onto the **original** 512 base, with that row's `feather_px` (code below).
3. **512 × 512 in, 512 × 512 out.** If your model wants 1024, upscale the base and mask to 1024 for generation, resize the result back to 512 with LANCZOS, *then* paste back onto the original 512 base.
4. **PNG only, never JPEG.** The single JPEG save happens later in A's normalize script, for every image. A JPEG save on your side would let the model learn "double compressed = AI".
5. **Names and folders.** Generator A writes to `ai_A/`, generator B to `ai_B/`:
   - paste-back edit: `<base_id>_e<K>_<A|B>.png`, e.g. `base_0012_e1_A.png`
   - whole-image img2img edit: `<base_id>_whole_<A|B>.png`
   - round-trip diagnostic (through the edit pipeline, nothing changed): `<base_id>_rt_<A|B>.png`
   - fully generated: `gen_<A|B>_<0000>.png`
6. **Log every output** in `ai_<A|B>/outputs.csv` with columns
   `file, kind, base_id, mask_file, prompt, feather_px, model, seed, color, lighting`.
   `kind` is one of `paste_back`, `whole_edit`, `round_trip`, `full_gen`. For `full_gen`, fill `color` and `lighting` from the prompt lists, and leave `base_id` and `mask_file` empty.

## Paste-back code

```python
import numpy as np
from PIL import Image, ImageFilter

def paste_back(base_path, mask_path, generated, feather_px, out_path):
    base = Image.open(base_path).convert("RGB")                       # original 512x512
    gen = generated.convert("RGB").resize(base.size, Image.LANCZOS)   # model output
    m = Image.open(mask_path).convert("L")
    if feather_px > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather_px / 2))       # soft edge
    a = np.asarray(m, np.float32)[..., None] / 255.0
    out = np.asarray(gen, np.float32) * a + np.asarray(base, np.float32) * (1 - a)
    Image.fromarray(out.round().clip(0, 255).astype(np.uint8)).save(out_path)  # .png
```

## Colab setup (sessions disconnect, so write straight to Drive)

```python
from google.colab import drive
drive.mount("/content/drive")
!unzip -q -o "/content/drive/MyDrive/damage_detector/handoff_A.zip" -d /content/handoff

import csv, os
ROOT = "/content/handoff"
OUT = "/content/drive/MyDrive/damage_detector/ai_A"    # ai_B for generator B
os.makedirs(OUT, exist_ok=True)
rows = list(csv.DictReader(open(f"{ROOT}/edit_masks.csv")))
todo = [r for r in rows
        if not os.path.exists(f"{OUT}/{r['mask_file'].replace('.png', '')}_A.png")]  # resume
print(len(todo), "of", len(rows), "edits left")
```

- Inputs: `f"{ROOT}/real_base/{r['base_id']}.png"` and `f"{ROOT}/masks/edit/{r['mask_file']}"`.
- Save each image, and append its `outputs.csv` row, right after it's made, so a disconnect loses nothing.
- Download every model checkpoint once at the start of the session.
- Run 10 rows first and send A two or three results to eyeball before running everything.
