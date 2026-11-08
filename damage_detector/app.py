"""Gradio demo: upload a damage photo, get the probability it was AI-edited and where to look.
Put app.py in A's project root (next to detector.py and models/) and run:
    python app.py
Until A's model exists it runs on a stub that returns random scores, so you can build and style
it now. As soon as detector.py and models/main.joblib are present, it uses the real model.
"""
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw

try:
    from detector import score                       # A's real model
    MODE = "model"
except Exception:                                    # stub until the model is ready
    MODE = "stub"

    def score(img):
        im = img.convert("RGB")
        s = min(im.size)
        im = im.crop(((im.width - s) // 2, (im.height - s) // 2,
                      (im.width + s) // 2, (im.height + s) // 2)).resize((512, 512))
        cp = np.random.default_rng(int(np.asarray(im).sum()) % 2 ** 32).uniform(0, 1, (3, 3))
        return {"prob": float(cp.max()), "flagged": bool(cp.max() > 0.7), "crop_probs": cp,
                "normalized": im}

STARTS = (0, 144, 288)          # 3 x 3 grid of 224 px crops on the 512 x 512 photo


def run(img):
    if img is None:
        return None, "Upload a photo of a damaged car."
    r = score(img)
    view = r["normalized"].convert("RGB").copy()
    d = ImageDraw.Draw(view)
    i, j = np.unravel_index(int(np.argmax(r["crop_probs"])), (3, 3))
    x, y = STARTS[j], STARTS[i]
    color = (230, 60, 60) if r["flagged"] else (60, 160, 90)
    d.rectangle([x, y, x + 223, y + 223], outline=color, width=5)
    if r["flagged"]:
        verdict = "### Flagged for review\nThis photo shows signs of AI generation or editing."
    else:
        verdict = "### No AI signs found by this model"
    md = (f"{verdict}\n\n**Probability of AI generation or editing: {r['prob']:.0%}**\n\n"
          "The box marks the part of the photo that drove the score. It is a hint where to "
          "look, not a precise outline of an edit. A person makes the final decision.")
    if MODE == "stub":
        md += "\n\n*(Demo stub: random scores until the model is connected.)*"
    return view, md


demo = gr.Interface(
    fn=run,
    inputs=gr.Image(type="pil", label="Damage photo"),
    outputs=[gr.Image(label="Where the model looked"), gr.Markdown()],
    title="AI-Edited Damage Photo Check",
    description=("Scores how likely a car damage photo was generated or edited by AI, so an "
                 "arbitrator knows which photos deserve a closer look. It flags for review only; "
                 "it never approves or denies a claim."),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
