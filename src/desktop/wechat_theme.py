"""WeChat theme detection and validation.

Aligned with original app.desktop.wechat_theme - detects WeChat UI theme
(light/dark) by sampling known pixel positions and checking color values.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


LIGHT_THEME = "light"
DARK_THEME = "dark"
DEFAULT_THEME = "light"


def required_wechat_theme(
    config: dict[str, Any] | None = None,
) -> str:
    """Get the required WeChat theme from config."""
    if config is None:
        return LIGHT_THEME
    return config.get("wechat_theme", LIGHT_THEME)


def validate_wechat_theme(
    image_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate that WeChat is using the correct theme."""
    detected = detect_wechat_theme(image_path)
    required = required_wechat_theme(config)
    matches = detected.get("theme") == required
    return {
        "detected": detected.get("theme", "unknown"),
        "required": required,
        "matches": matches,
        "samples": detected.get("samples", []),
    }


def detect_wechat_theme(
    image_path: str | Path | None = None,
) -> dict[str, Any]:
    """Detect WeChat theme from screenshot or file path."""
    result = {"theme": "unknown", "samples": []}

    if image_path is None:
        return result

    try:
        from PIL import Image
        if isinstance(image_path, (str, Path)):
            img = Image.open(str(image_path))
        else:
            img = image_path

        if hasattr(img, "convert"):
            img = img.convert("RGB")
            arr = np.array(img)
        elif isinstance(image_path, np.ndarray):
            arr = image_path
        else:
            return result

        samples = _sample_wechat_ui_pixels(arr)
        if not samples:
            return result

        light_count = sum(
            1 for r, g, b in samples
            if r > 200 and g > 200 and b > 200)
        dark_count = sum(
            1 for r, g, b in samples
            if r < 60 and g < 60 and b < 60)

        if light_count > dark_count:
            result["theme"] = LIGHT_THEME
        elif dark_count > light_count:
            result["theme"] = DARK_THEME

        result["samples"] = samples
        result["light_count"] = light_count
        result["dark_count"] = dark_count

    except Exception:
        pass

    return result


def _sample_wechat_ui_pixels(
    image: Any,
) -> list[tuple[int, int, int]]:
    """Sample known WeChat UI pixel positions for theme detection."""
    samples = []
    try:
        h, w = image.shape[:2] if hasattr(image, "shape") else (0, 0)
        if h < 100 or w < 100:
            return samples

        positions = [
            (5, 5),
            (w - 5, 5),
            (5, 30),
            (w // 2, 5),
            (5, h - 5),
        ]

        for px, py in positions:
            if 0 <= px < w and 0 <= py < h:
                if hasattr(image, "shape"):
                    pixel = tuple(image[py, px].tolist())
                else:
                    pixel = (0, 0, 0)
                samples.append(pixel)
    except Exception:
        pass
    return samples