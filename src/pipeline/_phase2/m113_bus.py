"""M-113: BUS-UCLM breast ultrasound lesion segmentation.

Layout: _extracted/M-113_BUS-UCLM/.../BUS-UCLM/images/<case>.png + masks/<case>.png
Case D single-image, loop 4s, lesion red overlay.
"""
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np
from common import DATA_ROOT, write_task, COLORS, fit_square, overlay_mask

PID = "M-113"; TASK_NAME = "bus_uclm_breast_lesion_seg"; FPS = 8
PROMPT = ("This is a breast ultrasound image from the BUS-UCLM dataset. "
          "Segment breast lesions (benign or malignant) with a red mask overlay.")

def loop_frames(f, n): return [f.copy() for _ in range(n)]

def process_case(img_p, mask_p, idx):
    img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None: return None
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    img_r = fit_square(img, 512)
    mask_r = cv2.resize((mask > 0).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
    annot = overlay_mask(img_r, mask_r, color=COLORS["red"], alpha=0.5)
    n = FPS * 4
    meta = {"task": "BUS-UCLM breast lesion segmentation", "dataset": "BUS-UCLM",
            "case_id": img_p.stem, "modality": "breast ultrasound",
            "classes": ["lesion"], "colors": {"lesion": "red"},
            "fps": FPS, "frames_per_video": n, "case_type": "D_single_image_loop",
            "lesion_area_px": int(mask_r.sum())}
    return write_task(PID, TASK_NAME, idx, img_r, annot,
                      loop_frames(img_r, n), loop_frames(annot, n), loop_frames(annot, n),
                      PROMPT, meta, FPS)

def main():
    root = DATA_ROOT / "_extracted" / "M-113_BUS-UCLM"
    # images/masks 在深层嵌套目录中（filename 里有空格），用 rglob
    images = list(root.rglob("images/*.png"))
    masks = {p.name: p for p in root.rglob("masks/*.png")}
    print(f"  {len(images)} BUS-UCLM images, {len(masks)} masks")
    i = 0
    for img in sorted(images):
        mask = masks.get(img.name)
        if not mask: continue
        d = process_case(img, mask, i)
        if d: print(f"  wrote {d}"); i += 1

if __name__ == "__main__":
    main()
