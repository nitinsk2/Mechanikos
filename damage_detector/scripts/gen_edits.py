"""Add AI damage to A's real base photos, inside A's edit masks, and paste it back.
Shared by B and C. Runs on a Colab T4 (or Kaggle). Example:

    python gen_edits.py --gen sd15 --out /content/drive/MyDrive/damage_detector/ai_sd15 \
        --zoom 2.5 --guidance 10 --steps 40 --whole-every 3 --rt-every 4
    python gen_edits.py --gen sdxl --out .../ai_sdxl --splits test --rt-every 2
    python gen_edits.py --gen flux --out .../ai_flux --max-minutes 300
    python gen_edits.py --gen sd15 --out /content/test_out --limit 3 --dry-run   # no model: tests I/O

What it does for every row of edit_masks.csv (in the order test, val, train):
  1. inpaint the masked region of real_base/<base_id>.png with the row's prompt and a seed made
     from the mask file name (so all generators get the same seed),
  2. paste only the masked region back onto the ORIGINAL 512 x 512 base with the row's feather,
  3. save <mask stem>_<gen>.png and append a row to outputs_<gen>_s<shard>.csv.
Optionally every Nth row also saves the raw generator output (whole_edit, nothing pasted back)
and a VAE round trip of the base (round_trip: through the model's autoencoder, no edit).
Files that already exist are skipped, so after a disconnect just run the same command again.
Everything is PNG. Never save as JPEG here.
"""
import argparse
import csv
import hashlib
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

CFG = {
    "sd15": dict(repo="stable-diffusion-v1-5/stable-diffusion-inpainting", size=512, steps=30, guidance=7.5),
    "sdxl": dict(repo="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", size=1024, steps=25, guidance=8.0),
    "flux": dict(repo="black-forest-labs/FLUX.1-Fill-dev", nf4="diffusers/FLUX.1-Fill-dev-nf4",
                 size=768, steps=24, guidance=30.0),
}
NEG = "cartoon, illustration, render, 3d, painting, drawing, text, watermark, blurry, deformed, low quality"
NEG_CLEAN = "clean undamaged paint, pristine, brand new car, smooth glossy panel, perfect condition"
BOOST = ("severe collision damage, crumpled and bent metal, deep gouges, scraped and cracked paint, "
         "exposed primer, dented panel")
COLS = ["file", "kind", "base_id", "mask_file", "prompt", "feather_px", "model", "seed",
        "color", "lighting", "generator", "split"]

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True, choices=list(CFG))
ap.add_argument("--root", default="/content/handoff", help="unzipped handoff_A.zip")
ap.add_argument("--out", required=True, help="output folder, ideally on Google Drive")
ap.add_argument("--splits", default="train,val,test")
ap.add_argument("--order", default="test,val,train")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0, help="stop after this many rows (0 = all)")
ap.add_argument("--reverse", action="store_true",
                help="work from the end of the list (a second person helping on the same folder)")
ap.add_argument("--max-minutes", type=float, default=0, help="stop cleanly after this long")
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--size", type=int, default=None, help="working resolution (multiple of 64)")
ap.add_argument("--guidance", type=float, default=None)
ap.add_argument("--whole-every", type=int, default=0, help="also save raw output every Nth row")
ap.add_argument("--rt-every", type=int, default=0, help="also save a VAE round trip every Nth row")
ap.add_argument("--zoom", type=float, default=0,
                help="paint at higher zoom: crop a square of ZOOM x the mask's size around it, "
                     "enlarge it to the working size, inpaint, shrink back (0 = whole image)")
ap.add_argument("--damage-boost", action="store_true",
                help="add strong damage words to the prompt the model sees (the CSV keeps the row's prompt)")
ap.add_argument("--no-neg-clean", action="store_true",
                help="SD only: don't add 'clean undamaged paint' etc. to the negative prompt")
ap.add_argument("--seed-damage", default="",
                help="folder of real damage patches (from 13_make_damage_donors.py): paste one into the "
                     "mask first, then let the model repaint it at --strength, so it keeps a damage shape")
ap.add_argument("--strength", type=float, default=None,
                help="how much the model repaints the masked area (1 = everything). Default 0.65 with "
                     "--seed-damage, otherwise the model's normal setting")
ap.add_argument("--no-nf4", action="store_true",
                help="FLUX only: load the full-precision model (A100 with High-RAM); faster, more memory")
ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"], help="FLUX only")
ap.add_argument("--dry-run", action="store_true", help="no model: fake damage, to test files and CSVs")
args = ap.parse_args()
cfg = dict(CFG[args.gen])
for k in ("steps", "size", "guidance"):
    if getattr(args, k) is not None:
        cfg[k] = getattr(args, k)
ROOT, OUT = Path(args.root), Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
G = args.gen


def load_pipe():
    import torch
    if G == "flux":
        from diffusers import FluxFillPipeline, FluxTransformer2DModel
        from transformers import T5EncoderModel
        dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        if args.no_nf4:
            pipe = FluxFillPipeline.from_pretrained(cfg["repo"], torch_dtype=dt)
        else:
            tr = FluxTransformer2DModel.from_pretrained(cfg["nf4"], subfolder="transformer", torch_dtype=dt)
            te = T5EncoderModel.from_pretrained(cfg["nf4"], subfolder="text_encoder_2", torch_dtype=dt)
            pipe = FluxFillPipeline.from_pretrained(cfg["repo"], transformer=tr, text_encoder_2=te,
                                                    torch_dtype=dt)
        pipe.enable_model_cpu_offload()
    else:
        from diffusers import AutoPipelineForInpainting
        try:
            pipe = AutoPipelineForInpainting.from_pretrained(cfg["repo"], torch_dtype=torch.float16,
                                                             variant="fp16")
        except Exception:
            pipe = AutoPipelineForInpainting.from_pretrained(cfg["repo"], torch_dtype=torch.float16)
        pipe.to("cuda")
        if getattr(pipe, "safety_checker", None) is not None:
            pipe.safety_checker = None          # it blacks out harmless car photos at random
    pipe.set_progress_bar_config(disable=True)
    return pipe


pipe = None if args.dry_run else load_pipe()
STRENGTH = args.strength if args.strength is not None else (0.65 if args.seed_damage else None)
DONORS = sorted(Path(args.seed_damage).glob("*.png")) if args.seed_damage else []
if args.seed_damage and not DONORS:
    raise SystemExit(f"no .png damage patches in {args.seed_damage}")


def seed_patch(base, mask, seed):
    """Paste a real damage patch into the mask (color shifted halfway toward the panel), so the
    model has a damage shape to keep. The model then repaints it; nothing pasted survives as-is."""
    m = np.asarray(mask) > 127
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    donor = Image.open(DONORS[seed % len(DONORS)]).convert("RGB").resize((x1 - x0, y1 - y0), Image.LANCZOS)
    d = np.asarray(donor, np.float32)
    b = np.asarray(base, np.float32)
    d = d + 0.5 * (b[y0:y1, x0:x1][m[y0:y1, x0:x1]].mean(0) - d.reshape(-1, 3).mean(0))
    canvas = b.copy()
    canvas[y0:y1, x0:x1] = d
    soft = np.asarray(mask.filter(ImageFilter.GaussianBlur(2)), np.float32)[..., None] / 255.0
    out = canvas * soft + b * (1 - soft)
    return Image.fromarray(out.round().clip(0, 255).astype(np.uint8))


def run_model(img, m, prompt, seed):
    """One generator call on a square image + mask of any size; returns the same size."""
    n = img.size[0]
    if args.dry_run:
        rng = np.random.default_rng(seed)
        a = np.asarray(img, np.float32) * 0.6 + rng.uniform(0, 255, (n, n, 3)) * 0.4
        return Image.fromarray(a.clip(0, 255).astype(np.uint8))
    import torch
    S = cfg["size"]
    img_s = img if n == S else img.resize((S, S), Image.LANCZOS)
    m_s = m if n == S else m.resize((S, S), Image.NEAREST)
    g = torch.Generator("cpu").manual_seed(seed)
    if args.damage_boost:
        prompt = f"{BOOST}, {prompt}"
    if G == "flux":
        kw = dict(prompt=prompt, image=img_s, mask_image=m_s, height=S, width=S,
                  guidance_scale=cfg["guidance"], num_inference_steps=cfg["steps"],
                  max_sequence_length=512, generator=g)
        if STRENGTH is not None:
            kw["strength"] = STRENGTH
        try:
            out = pipe(**kw).images[0]
        except TypeError as e:
            raise SystemExit(f"{e}\nThis diffusers version's FLUX Fill has no 'strength'. Run "
                             "'pip install -U diffusers', restart, or use --gen sdxl for seeded damage.")
    else:
        neg = NEG if args.no_neg_clean else f"{NEG}, {NEG_CLEAN}"
        kw = dict(prompt=prompt, negative_prompt=neg, image=img_s, mask_image=m_s, height=S, width=S,
                  guidance_scale=cfg["guidance"], num_inference_steps=cfg["steps"], generator=g)
        if STRENGTH is not None:
            kw["strength"] = STRENGTH
        elif G == "sdxl":
            kw["strength"] = 0.99
        out = pipe(**kw).images[0]
    out = out.convert("RGB")
    return out if out.size == (n, n) else out.resize((n, n), Image.LANCZOS)


def zoom_box(mask):
    """Square around the mask, ZOOM times its larger side, kept inside the 512 x 512 photo."""
    ys, xs = np.nonzero(np.asarray(mask) > 127)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * args.zoom)
    side = max(160, min(512, side))
    left = int(np.clip((x0 + x1) // 2 - side // 2, 0, 512 - side))
    top = int(np.clip((y0 + y1) // 2 - side // 2, 0, 512 - side))
    return (left, top, left + side, top + side)


def inpaint(base, mask, prompt, seed, whole=False):
    """Returns the generator's output at 512 x 512 (before paste-back).
    With --zoom, only a square around the mask is regenerated, at much higher detail, and the
    rest of the returned image is the original base. whole=True always regenerates everything."""
    if not args.zoom or whole:
        return run_model(base, mask, prompt, seed)
    box = zoom_box(mask)
    out = base.copy()
    out.paste(run_model(base.crop(box), mask.crop(box), prompt, seed), box[:2])
    return out


def paste_back(base, gen, mask, feather_px):
    """Exactly the hand-off rule: only the (feathered) mask region comes from the generator."""
    m = mask
    if feather_px > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather_px / 2))
    a = np.asarray(m, np.float32)[..., None] / 255.0
    out = np.asarray(gen, np.float32) * a + np.asarray(base, np.float32) * (1 - a)
    return Image.fromarray(out.round().clip(0, 255).astype(np.uint8))


def round_trip(base):
    """The base through the model's autoencoder only: AI-processed, nothing added."""
    if args.dry_run:
        a = np.asarray(base, np.int16) + np.random.default_rng(0).integers(-3, 4, (512, 512, 3))
        return Image.fromarray(a.clip(0, 255).astype(np.uint8))
    import torch
    vae = pipe.vae
    dt = vae.dtype
    vae.to(dtype=torch.float32)                 # SDXL's VAE overflows in fp16
    x = torch.from_numpy(np.asarray(base, np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to("cuda")
    with torch.no_grad():
        y = vae.decode(vae.encode(x).latent_dist.mode()).sample
    vae.to(dtype=dt)
    y = ((y[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).round().byte().cpu().numpy()
    return Image.fromarray(y)


with open(ROOT / "edit_masks.csv", encoding="utf-8") as f:
    plan = list(csv.DictReader(f))
want = args.splits.split(",")
rank = {s: i for i, s in enumerate(args.order.split(","))}
plan = [r for r in plan if r.get("split", "train") in want]
plan.sort(key=lambda r: (rank.get(r.get("split", "train"), 9), r["mask_file"]))
mine = [(i, r) for i, r in enumerate(plan) if i % args.nshards == args.shard]
if args.reverse:
    mine = mine[::-1]
if args.limit:
    mine = mine[:args.limit]

log_path = OUT / f"outputs_{G}_s{args.shard}{'_rev' if args.reverse else ''}.csv"
new = not log_path.exists()
log = open(log_path, "a", newline="", encoding="utf-8")
w = csv.DictWriter(log, fieldnames=COLS)
if new:
    w.writeheader()
model_name = "dry-run" if args.dry_run else cfg["repo"]
if args.seed_damage:
    model_name += f" +seeded-damage strength {STRENGTH}"


def record(fname, kind, r, prompt="", feather="", seed="", mask_file=""):
    w.writerow(dict(file=fname, kind=kind, base_id=r["base_id"], mask_file=mask_file, prompt=prompt,
                    feather_px=feather, model=model_name, seed=seed, color=r.get("color", ""),
                    lighting=r.get("lighting", ""), generator=G, split=r.get("split", "")))
    log.flush()


t_start, done, times = time.time(), 0, []
print(f"{G}: {len(mine)} rows for shard {args.shard}/{args.nshards}, splits {want}, "
      f"size {cfg['size']}, steps {cfg['steps']}, guidance {cfg['guidance']}, zoom {args.zoom or 'off'}, "
      f"damage boost {'on' if args.damage_boost else 'off'}, "
      f"seeded damage {'on, strength ' + str(STRENGTH) if DONORS else 'off'}")
for i, r in mine:
    if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
        print("time budget reached, stopping cleanly")
        break
    stem = r["mask_file"][:-4]
    f_pb = f"{stem}_{G}.png"
    if (OUT / f_pb).exists():
        continue
    t0 = time.time()
    base = Image.open(ROOT / "real_base" / f"{r['base_id']}.png").convert("RGB")
    mask = Image.open(ROOT / "masks" / "edit" / r["mask_file"]).convert("L")
    seed = int(hashlib.md5(r["mask_file"].encode()).hexdigest()[:8], 16) % (2 ** 31)
    feather = int(float(r["feather_px"] or 0))
    src = seed_patch(base, mask, seed) if DONORS else base
    gen = inpaint(src, mask, r["prompt"], seed)
    if np.asarray(gen).max() < 10 or not np.isfinite(np.asarray(gen, np.float32)).all():
        print(f"   {stem}: blank output, skipped (with FLUX try the other --dtype)")
        continue
    f_we = f"{stem}_whole_{G}.png"
    if args.whole_every and i % args.whole_every == 0 and not (OUT / f_we).exists():
        raw = inpaint(src, mask, r["prompt"], seed, whole=True) if args.zoom else gen
        raw.save(OUT / f_we)
        record(f_we, "whole_edit", r, r["prompt"], "", seed, r["mask_file"])
    if args.rt_every and i % args.rt_every == 0 and G != "flux":
        f_rt = f"{r['base_id']}_rt_{G}.png"
        if not (OUT / f_rt).exists():
            round_trip(base).save(OUT / f_rt)
            record(f_rt, "round_trip", r)
    paste_back(base, gen, mask, feather).save(OUT / f_pb)        # saved last = row complete
    record(f_pb, "paste_back", r, r["prompt"], feather, seed, r["mask_file"])
    done += 1
    times.append(time.time() - t0)
    left = len(mine) - done
    avg = float(np.mean(times[-20:]))
    print(f"   {stem} [{r.get('split', '')}] {times[-1]:.1f}s  done {done}, "
          f"~{left * avg / 60:.0f} min left at this speed")
print(f"finished: {done} new paste-backs in {(time.time() - t_start) / 60:.1f} min -> {OUT}")
