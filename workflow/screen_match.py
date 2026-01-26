from __future__ import annotations

import base64
import io
from pathlib import Path


def _dhash(img, hash_size: int = 8) -> int:
    from PIL import Image

    img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = list(img.getdata())
    result = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            bit = 1 if left > right else 0
            result = (result << 1) | bit
    return result


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dhash_similarity(img_a, img_b, *, hash_size: int = 8) -> float:
    """
    Return similarity in [0,1] using dHash Hamming distance.
    """
    ha = _dhash(img_a, hash_size=hash_size)
    hb = _dhash(img_b, hash_size=hash_size)
    bits = hash_size * hash_size
    dist = _hamming(ha, hb)
    return 1.0 - (dist / bits)


def similarity_file_vs_base64_jpeg(image_path: str | Path, image_base64: str) -> float:
    """
    Compare a reference image file vs a base64-encoded JPEG/PNG (data is just base64 body).
    """
    from PIL import Image

    ref = Image.open(Path(image_path))
    raw = base64.b64decode(image_base64)
    cur = Image.open(io.BytesIO(raw))
    return dhash_similarity(ref, cur)

