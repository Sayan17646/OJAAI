"""
preprocessor.py — Image cleaning pipeline for OJAAI.

Accepts JPEG, PNG, or PDF (first page only).

TWO SEPARATE PIPELINES:
  preprocess_for_tesseract() — heavy binarisation, CLAHE, perspective correction.
                               Returns grayscale binary numpy array for Tesseract.
  preprocess_for_gemini()    — light touch: EXIF rotation, mild sharpening, resize.
                               Returns JPEG bytes preserving colour for Gemini Vision.

Processing order for Tesseract (per TRD Section 1 + Priority-1 upgrade):
  1. Grayscale conversion
  2. CLAHE (Contrast Limited Adaptive Histogram Equalisation)
  3. Dark-background inversion
  4. Deskew (only if angle > 0.5°)
  5. Perspective correction (document boundary detection)
  6. Denoising
  7. Adaptive thresholding
  8. Morphological cleanup
  9. Upscale if width < 1000px

Processing order for Gemini:
  1. EXIF / orientation correction (phone photos)
  2. Mild unsharp-mask sharpening
  3. Resize if longest side > 2000px
  4. Output as JPEG quality 90
"""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_image_from_bytes(data: bytes, filename: str) -> np.ndarray:
    """
    Load raw file bytes into an OpenCV BGR numpy array.
    Supports JPEG, PNG, and PDF (first page only via pdf2image).
    Raises ValueError if the file cannot be read.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _load_pdf_first_page(data)

    # JPEG / PNG path
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2.imdecode could not read image: {filename!r}")
    return img


def _load_pdf_first_page(data: bytes) -> np.ndarray:
    """Convert first page of a PDF to a BGR numpy array at 200 DPI."""
    try:
        from pdf2image import convert_from_bytes  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pdf2image is required for PDF support. "
            "Run: pip install pdf2image and install Poppler."
        ) from exc

    pages = convert_from_bytes(data, dpi=200, first_page=1, last_page=1)
    if not pages:
        raise ValueError("pdf2image returned no pages from PDF.")

    pil_img = pages[0].convert("RGB")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return bgr


# ---------------------------------------------------------------------------
# Tesseract preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_for_tesseract(img_bgr: np.ndarray) -> np.ndarray:
    """
    Full heavy-cleaning pipeline for Tesseract OCR.
    Returns a binary (thresholded) grayscale uint8 numpy array.

    Improvements over original preprocess():
      - CLAHE replaces simple brightness inversion for better local contrast
      - Perspective correction straightens phone-camera angles
    """
    # Step 1: Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2: CLAHE — Contrast Limited Adaptive Histogram Equalisation
    # Much better than a global brightness check for uneven lighting from phones.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Step 3: Dark background detection → invert so text is dark on white
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 127:
        logger.debug("Dark background detected — inverting image.")
        gray = cv2.bitwise_not(gray)

    # Step 4: Deskew (only if tilt > 0.5°)
    gray = _deskew(gray)

    # Step 5: Perspective correction — straighten document from phone angles
    gray = _perspective_correct(gray)

    # Step 6: Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Step 7: Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    # Step 8: Morphological cleanup — close tiny noise holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Step 9: Upscale if narrower than 1000px
    h, w = cleaned.shape
    if w < 1000:
        scale = 1000.0 / w
        new_w = 1000
        new_h = int(h * scale)
        cleaned = cv2.resize(cleaned, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.debug("Upscaled image from %dx%d to %dx%d.", w, h, new_w, new_h)

    return cleaned


def preprocess_from_bytes(data: bytes, filename: str) -> np.ndarray:
    """
    Convenience wrapper: bytes + filename → cleaned grayscale numpy array for Tesseract.
    This is the only function the Tesseract pipeline path needs to call.
    """
    img_bgr = load_image_from_bytes(data, filename)
    return preprocess_for_tesseract(img_bgr)


# ---------------------------------------------------------------------------
# Gemini preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_for_gemini(image_bytes: bytes, filename: str) -> bytes:
    """
    Light preprocessing for Gemini Vision — preserves colour and natural detail.

    Steps:
      1. EXIF orientation correction (phone photos are often rotated 90/270°)
      2. Mild unsharp-mask sharpening for handwriting clarity
      3. Resize so the longest side ≤ 2000px (preserving aspect ratio)
      4. Output as JPEG quality 90

    Does NOT binarise, threshold, or convert to grayscale — Gemini needs
    the full colour image for best vision performance.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        try:
            bgr = load_image_from_bytes(image_bytes, filename)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
        except Exception:
            return image_bytes
    else:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            # Step 1: EXIF orientation correction
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception:
            return image_bytes

    # Step 2: Mild unsharp-mask sharpening (helps handwriting legibility)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_cv = _unsharp_mask(img_cv, kernel_size=3, sigma=1.0, strength=0.6)
    img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # Step 3: Resize if longest side > 2000px
    w, h = img.size
    max_side = max(w, h)
    if max_side > 2000:
        scale = 2000.0 / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        logger.debug("Gemini: resized image from %dx%d to %dx%d.", w, h, new_w, new_h)

    # Step 4: Encode as JPEG quality 90
    out_buf = io.BytesIO()
    img.save(out_buf, format="JPEG", quality=90)
    return out_buf.getvalue()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Detect and correct rotation using minAreaRect on white pixels.
    Only applies correction if tilt angle is more than 0.5°.
    """
    inverted = cv2.bitwise_not(gray)
    coords = cv2.findNonZero(inverted)

    if coords is None or len(coords) < 100:
        return gray

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Normalise angle: convert to skew angle relative to horizontal
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) <= 0.5:
        return gray  # negligible tilt — skip

    logger.debug("Deskewing by %.2f degrees.", angle)
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return deskewed


def _perspective_correct(gray: np.ndarray) -> np.ndarray:
    """
    Detect the document boundary and apply perspective correction.

    Strategy:
      1. Blur + Canny edge detection
      2. Find the largest 4-sided contour (the prescription slip)
      3. If a reliable quadrilateral is found, warpPerspective to a flat rectangle
      4. If detection fails (borderless, crumpled), return original unchanged

    This corrects keystoning from phone photos taken at an angle.
    """
    h, w = gray.shape

    # Work on a blurred copy for better edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Dilate edges to close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray

    # Sort by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    doc_quad = None
    for cnt in contours[:5]:  # only check top 5 largest
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            # Must cover at least 20% of the image area to be the document
            if area > 0.20 * h * w:
                doc_quad = approx
                break

    if doc_quad is None:
        logger.debug("Perspective correction: no reliable quad found — skipping.")
        return gray

    # Order the 4 corners: top-left, top-right, bottom-right, bottom-left
    pts = doc_quad.reshape(4, 2).astype(np.float32)
    ordered = _order_corners(pts)

    # Compute output dimensions from the ordered corners
    tl, tr, br, bl = ordered
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    out_w = int(max(width_top, width_bot))
    out_h = int(max(height_left, height_right))

    if out_w < 100 or out_h < 100:
        return gray  # sanity check

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(gray, M, (out_w, out_h), flags=cv2.INTER_CUBIC)
    logger.debug("Perspective correction applied: %dx%d → %dx%d.", w, h, out_w, out_h)
    return warped


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 corner points as: [top-left, top-right, bottom-right, bottom-left].
    """
    # Sum and diff to find corners
    s = pts.sum(axis=1)       # tl has smallest sum, br has largest
    d = np.diff(pts, axis=1)  # tr has smallest diff, bl has largest

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    ordered[1] = pts[np.argmin(d)]   # top-right
    ordered[3] = pts[np.argmax(d)]   # bottom-left
    return ordered


def _unsharp_mask(
    img: np.ndarray,
    kernel_size: int = 3,
    sigma: float = 1.0,
    strength: float = 0.6,
) -> np.ndarray:
    """
    Apply mild unsharp masking to sharpen handwriting details.
    strength controls how much sharpening is applied (0.0 = none, 1.0 = full).
    """
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return sharpened


# ---------------------------------------------------------------------------
# Legacy alias — kept for backward compatibility
# ---------------------------------------------------------------------------

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for preprocess_for_tesseract()."""
    return preprocess_for_tesseract(img_bgr)
