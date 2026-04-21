import sys
import shutil
from pathlib import Path
import cv2
import numpy as np


INPUT_FOLDER  = "./data/output/images_raw"
OUTPUT_FOLDER = "./data/output/images"

# --- Resize ---
SCALE         = 0.4       # 0.1 → 1.0  (1.0 = keep original)

# --- Color quantization ---
N_COLORS      = 16

# --- PNG compression level: 0 → 9 ---
PNG_COMPRESS  = 9

#  Copy if file is small
KEEP_SMALLER  = True
# ============================================================

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def format_size(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} GB"


def quantize_colors(img_bgr: np.ndarray, n_colors: int) -> np.ndarray:
    """Color quantization K-Means clustering."""
    h, w = img_bgr.shape[:2]
    has_alpha = img_bgr.shape[2] == 4 if img_bgr.ndim == 3 else False

    # Split alpha
    if has_alpha:
        bgr   = img_bgr[:, :, :3]
        alpha = img_bgr[:, :, 3]
    else:
        bgr = img_bgr

    # Reshape to pixels
    pixels = bgr.reshape(-1, 3).astype(np.float32)

    # K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, n_colors, None, criteria,
        attempts=3, flags=cv2.KMEANS_PP_CENTERS
    )

    # Map pixel
    quantized = centers[labels.flatten()].reshape(h, w, 3).astype(np.uint8)

    if has_alpha:
        quantized = np.dstack([quantized, alpha])

    return quantized


def compress_one(src: Path, dst: Path) -> tuple[int, int, str]:
    original_size = src.stat().st_size
    dst = dst.with_suffix(".png")
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Read img
    img = cv2.imdecode(
        np.frombuffer(src.read_bytes(), np.uint8),
        cv2.IMREAD_UNCHANGED
    )
    if img is None:
        raise ValueError("Cant read img")

    # Ensure 3 channel at least
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 1. Resize
    if SCALE != 1.0:
        h, w = img.shape[:2]
        new_w = max(1, int(w * SCALE))
        new_h = max(1, int(h * SCALE))
        # INTER_AREA
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2. Reduce color
    if N_COLORS:
        img = quantize_colors(img, N_COLORS)

    # 3. Save PNG
    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESS]
    success, buf = cv2.imencode(".png", img, encode_params)
    if not success:
        raise ValueError("Encode PNG fail")

    compressed_size = len(buf)

    # Keep small file
    if KEEP_SMALLER and compressed_size >= original_size:
        shutil.copy2(src, dst.with_suffix(src.suffix))
        return original_size, original_size, "kept_original"

    dst.write_bytes(buf)
    return original_size, compressed_size, "compressed"


def main():
    input_dir  = Path(INPUT_FOLDER)
    output_dir = Path(OUTPUT_FOLDER)

    if not input_dir.exists():
        print(f"Not found: {input_dir.resolve()}")
        sys.exit(1)

    images = [p for p in input_dir.rglob("*") if p.suffix.lower() in SUPPORTED]
    if not images:
        print("No Image")
        sys.exit(0)

    print(f"\nRoot      : {input_dir.resolve()}")
    print(f"Output	: {output_dir.resolve()}")
    print(f"Images	: {len(images)}")
    print(f"Scale	: {SCALE}x  |  Màu: {N_COLORS or 'full'}  |  PNG compress: {PNG_COMPRESS}")
    print("-" * 65)

    total_before = total_after = 0
    n_compressed = n_kept = n_err = 0
    errors = []

    for i, src in enumerate(images, 1):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        try:
            before, after, status = compress_one(src, dst)
            total_before += before
            total_after  += after

            if status == "compressed":
                n_compressed += 1
                tag = f"-{(1 - after/before)*100:.0f}%"
            else:
                n_kept += 1
                tag = "keep original"

            print(f"[{i:>4}/{len(images)}] {rel}  "
                  f"{format_size(before)} → {format_size(after)}  ({tag})")
        except Exception as e:
            n_err += 1
            errors.append((src, str(e)))
            print(f"[{i:>4}/{len(images)}] Error: {rel} — {e}")

    saved = total_before - total_after
    pct   = saved / total_before * 100 if total_before else 0

    print("-" * 65)
    print(f"Completed !")
    print(f"Before	: {format_size(total_before)}")
    print(f"After	: {format_size(total_after)}")
    print(f"Reduce	: {format_size(saved)}  ({pct:.1f}%)")
    print(f"Compressed	: {n_compressed} images  |  No change: {n_kept} images")
    if errors:
        print(f"\n {n_err} error:")
        for p, msg in errors:
            print(f"   • {p.name}: {msg}")


if __name__ == "__main__":
    main()
