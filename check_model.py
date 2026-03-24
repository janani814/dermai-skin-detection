# ============================================================
# check_model.py — Run this FIRST to diagnose your model
# Command: python check_model.py
# ============================================================

print("\n" + "="*55)
print("   MODEL DIAGNOSTIC TOOL")
print("="*55)

import os
import numpy as np
from PIL import Image

# ── STEP 1: Check if best.pt exists ───────────────────────
print("\n[1] Checking best.pt file...")
if os.path.exists("best.pt"):
    size_mb = os.path.getsize("best.pt") / (1024 * 1024)
    print(f"    ✅ best.pt found — Size: {size_mb:.1f} MB")
else:
    print("    ❌ best.pt NOT found! Make sure you are in the right folder.")
    exit()

# ── STEP 2: Load model and check type ─────────────────────
print("\n[2] Loading model...")
try:
    from ultralytics import YOLO
    model = YOLO("best.pt")
    print(f"    ✅ Model loaded with Ultralytics")
    print(f"    Task type   : {model.task}")
    print(f"    Class names : {model.names}")
    print(f"    Num classes : {len(model.names)}")

    # ── STEP 3: Test prediction on a dummy image ───────────
    print("\n[3] Testing prediction on dummy images...")
    results_summary = []

    # Create 5 different test images with different colors
    test_images = [
        ("Dark Brown (mole-like)",  np.full((224,224,3), [80,50,30],   dtype=np.uint8)),
        ("Red Lesion",              np.full((224,224,3), [200,60,60],  dtype=np.uint8)),
        ("Light Pink (fair skin)",  np.full((224,224,3), [230,180,160],dtype=np.uint8)),
        ("Scaly/Rough texture",     np.array([[[i%100+80, i%80+60, i%60+40]
                                    for i in range(224)] for _ in range(224)],dtype=np.uint8)),
        ("Purple/Blue lesion",      np.full((224,224,3), [100,80,200], dtype=np.uint8)),
    ]

    for name, arr in test_images:
        img = Image.fromarray(arr)
        results = model.predict(source=img, verbose=False)
        r = results[0]

        if r.probs is not None:
            idx   = int(r.probs.top1)
            conf  = float(r.probs.top1conf)*100
            cls   = model.names.get(idx, str(idx))
            print(f"    [{name}] → Predicted: {cls} ({conf:.1f}%)")
            results_summary.append(cls)
        elif r.boxes is not None and len(r.boxes) > 0:
            idx  = int(r.boxes.cls[0])
            conf = float(r.boxes.conf[0])*100
            cls  = model.names.get(idx, str(idx))
            print(f"    [{name}] → Detected: {cls} ({conf:.1f}%) — DETECTION MODEL")
            results_summary.append(cls)
        else:
            print(f"    [{name}] → No prediction (empty result)")
            results_summary.append("none")

    # ── STEP 4: Diagnose the problem ──────────────────────
    print("\n[4] DIAGNOSIS:")
    unique = set(results_summary)
    if len(unique) == 1:
        only = list(unique)[0]
        print(f"    ⚠️  ALL images predicted as '{only}' — MODEL BIAS DETECTED")
        print(f"    This means the model was not trained for classification")
        print(f"    OR it is heavily biased toward '{only}' class")
        print(f"\n    SOLUTION: Use the fixed app.py with --force-variety flag")
    else:
        print(f"    ✅ Different images give different results: {unique}")
        print(f"    Your model is working correctly!")
        print(f"    The issue was in how app.py called the model.")

    print(f"\n[5] Model task is: '{model.task}'")
    if model.task == 'classify':
        print("    ✅ CLASSIFICATION model — correct type for this project")
    elif model.task == 'detect':
        print("    ⚠️  DETECTION model — needs different prediction approach")
    elif model.task == 'segment':
        print("    ⚠️  SEGMENTATION model — needs different prediction approach")

except Exception as e:
    print(f"    ❌ Error: {e}")
    print("\n    Trying PyTorch direct load...")
    try:
        import torch
        ckpt = torch.load("best.pt", map_location="cpu")
        print(f"    Loaded as PyTorch. Keys: {list(ckpt.keys()) if isinstance(ckpt,dict) else type(ckpt)}")
        if isinstance(ckpt, dict) and 'names' in ckpt:
            print(f"    Class names: {ckpt['names']}")
    except Exception as e2:
        print(f"    ❌ PyTorch also failed: {e2}")

print("\n" + "="*55)
print("   Copy the output above and share it!")
print("="*55 + "\n")