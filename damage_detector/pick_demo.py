"""Pick the demo photos from the test split (never from training data).
Run in A's project root after 12_test.py:
    python pick_demo.py
Copies candidates into demo/ and prints why each was picked:
  1. a base original the main model scored low, whose SDXL (held-out) or FLUX paste-back it flagged
  2. a real damaged photo it scored low (shows it doesn't just flag damage)
  3. the same flagged edit after JPEG quality 50 (shown honestly, whatever the score)
"""
import shutil
from pathlib import Path

import pandas as pd

p = pd.read_csv("predictions/test_all.csv", keep_default_na=False)
p["flagged"] = p.flagged.astype(str).str.lower().isin(["true", "1"])
p["prob"] = p.prob.astype(float)
m = p[(p.experiment == "main") & (p.variant == "clean")]
out = Path("demo")
out.mkdir(exist_ok=True)

bases = m[(m.kind == "base_original") & (~m.flagged)].set_index("base_id")
pairs = []
for g in ("sdxl", "kontext", "flux"):
    e = m[(m.kind == "paste_back") & (m.generator == g) & (m.flagged) & (m.base_id.isin(bases.index))]
    for _, r in e.sort_values("prob", ascending=False).head(5).iterrows():
        pairs.append((g, bases.loc[r.base_id], r))
if not pairs:
    print("no base photo was scored clean with its edit flagged; pick a pair by hand from test_all.csv")
for k, (g, b, r) in enumerate(pairs[:5]):
    shutil.copy(b.file, out / f"pair{k}_original.jpg")
    shutil.copy(r.file, out / f"pair{k}_edited_{g}.jpg")
    jp = p[(p.experiment == "main") & (p.file == r.file) & (p.variant == "jpeg50")]
    j50 = f"{jp.prob.iloc[0]:.0%}" if len(jp) else "n/a"
    print(f"pair{k}: original {b.prob:.0%} -> {g} edit {r.prob:.0%} (after JPEG q50: {j50})  {r.file}")

real = m[(m.kind == "real_damaged") & (~m.flagged)].sort_values("prob").head(5)
for k, (_, r) in enumerate(real.iterrows()):
    shutil.copy(r.file, out / f"real_damaged{k}.jpg")
    print(f"real_damaged{k}: {r.prob:.0%}  {r.file}")
print("Look at the candidates in demo/ and pick the clearest pair: same car, one visible dent.")
