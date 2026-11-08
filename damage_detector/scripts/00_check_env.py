"""Step 1 check: is the environment ready? Run from the project root:
    python scripts/00_check_env.py
Every line should say OK. Any FAIL line tells you what to fix.
"""
import importlib
import platform
import sys

ok = True


def report(name, good, detail=""):
    global ok
    ok &= good
    print(f"[{'OK  ' if good else 'FAIL'}] {name:22s} {detail}")


report("python", sys.version_info >= (3, 10), f"{sys.version.split()[0]} on {platform.platform()}")

try:
    import torch
    report("torch", True, f"{torch.__version__} (built for CUDA {torch.version.cuda})")
    cuda = torch.cuda.is_available()
    report("cuda available", cuda,
           "" if cuda else "-> CPU-only torch wheel or old driver; see Troubleshooting")
    if cuda:
        p = torch.cuda.get_device_properties(0)
        report("gpu", True, f"{p.name}, {p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}")
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        y = (x @ x).float().abs().mean().item()
        torch.cuda.synchronize()
        report("fp16 matmul on gpu", y == y, f"mean={y:.2f}")
except Exception as e:  # noqa: BLE001
    report("torch", False, repr(e))

try:
    import torch
    import torchvision
    from torchvision.ops import nms
    b = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32, device="cuda")
    s = torch.tensor([0.9, 0.8], device="cuda")
    nms(b, s, 0.5)
    report("torchvision cuda ops", True, torchvision.__version__)
except Exception as e:  # noqa: BLE001
    report("torchvision cuda ops", False,
           f"{e!r} -> torch and torchvision came from different indexes; reinstall both together")

for mod in ["open_clip", "sklearn", "numpy", "PIL", "pillow_heif", "cv2", "imagehash",
            "pandas", "requests", "tqdm", "gradio"]:
    try:
        m = importlib.import_module(mod)
        report(mod, True, getattr(m, "__version__", ""))
    except Exception as e:  # noqa: BLE001
        report(mod, False, repr(e))

try:
    import cv2
    gui = "GTK" in cv2.getBuildInformation() or "Win32 UI" in cv2.getBuildInformation() \
        or "Cocoa" in cv2.getBuildInformation()
    report("opencv gui (for blur tool)", gui,
           "" if gui else "-> headless OpenCV; pip uninstall -y opencv-python-headless && "
                          "pip install --force-reinstall opencv-python")
except Exception:
    pass

print("\nALL OK" if ok else "\nFix the FAIL lines before moving on.")
