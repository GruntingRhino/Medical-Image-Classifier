"""Luminance/ITA-based demographic proxies for skin images.

These proxies use only the image itself (no demographic labels), mirroring the
original study's setting. We compute:

- ITA (Individual Typology Angle): arctan((L*-50)/b*) in CIE Lab; the
  dermatology-standard image-derived skin-tone measure (Del Bino & Bernerd
  2013). Computed on the central crop (lesion/skin-heavy) and on the border
  ring (background-heavy) separately: the contrast between them is our test
  of whether the proxy measures *skin* or *lighting/background*.
- Simple grayscale percentiles as robust alternatives.

Higher ITA = lighter skin (FST I ~ very light, ITA > 55; FST VI < -30).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# Del Bino & Bernerd 2013 ITA boundaries (degrees):
#   >55: I; 41-55: II; 28-41: III; 10-28: IV; -30-10: V; <-30: VI
ITA_BOUNDS = np.array([55.0, 41.0, 28.0, 10.0, -30.0])


def _rgb_to_lab(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sRGB [0,255] -> CIE L*, a*, b* (D65)."""
    c = rgb.astype(np.float64) / 255.0
    # linearize (sRGB EOTF)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]]
    )
    xyz = lin @ M.T
    white = np.array([0.95047, 1.0, 1.08883])

    def f(t):
        eps = 216 / 24389
        kap = 24389 / 27
        return np.where(t > eps, np.cbrt(t), (kap * t + 16) / 116)

    fx, fy, fz = f(xyz[..., 0] / white[0]), f(xyz[..., 1] / white[1]), f(xyz[..., 2] / white[2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b


def ita_from_region(rgb: np.ndarray) -> float:
    """Median-based ITA of an RGB region (H,W,3) uint8."""
    L, a, b = _rgb_to_lab(rgb)
    Lm = float(np.median(L))
    bm = float(np.median(b))
    if abs(bm) < 1e-6:
        bm = 1e-6
    return float(np.degrees(np.arctan2(Lm - 50.0, bm)))


def region_masks(h: int, w: int, border_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (center, border-ring) for an h x w image."""
    yy, xx = np.mgrid[0:h, 0:w]
    inner_h, inner_w = int(h * (1 - 2 * border_frac)), int(w * (1 - 2 * border_frac))
    y0, x0 = (h - inner_h) // 2, (w - inner_w) // 2
    center = np.zeros((h, w), dtype=bool)
    center[y0 : y0 + inner_h, x0 : x0 + inner_w] = True
    return center, ~center


def luminance_proxies(img: Image.Image) -> dict[str, float]:
    """Compute all luminance-family proxies for a PIL image."""
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    # downscale for speed if large
    scale = int(np.ceil(max(h, w) / 512))
    if scale > 1:
        rgb = rgb[::scale, ::scale]
        h, w = rgb.shape[:2]
    center_m, border_m = region_masks(h, w)
    center = rgb[center_m]
    border = rgb[border_m]
    gray_c = 0.2126 * center[:, 0] + 0.7152 * center[:, 1] + 0.0722 * center[:, 2]
    gray_b = 0.2126 * border[:, 0] + 0.7152 * border[:, 1] + 0.0722 * border[:, 2]
    return {
        "ita_center": ita_from_region(center),
        "ita_border": ita_from_region(border) if border.size else np.nan,
        "lum_med_center": float(np.median(gray_c)),
        "lum_p10_center": float(np.percentile(gray_c, 10)),
        "lum_med_border": float(np.median(gray_b)) if border.size else np.nan,
        "sat_center": float(np.mean(center.max(axis=1).astype(float) - center.min(axis=1))),
    }


def proxy_from_array(arr: np.ndarray, name: str = "ita_center") -> float:
    d = luminance_proxies(Image.fromarray(arr))
    return d[name]
