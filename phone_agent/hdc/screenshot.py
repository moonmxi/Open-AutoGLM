"""Screenshot utilities for capturing HarmonyOS device screen."""

import base64
import io
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Tuple

from PIL import Image
from phone_agent.hdc.connection import _run_hdc_command

# Cache auto-detected device id for commands that require `-t <device>`.
_AUTO_DEVICE_ID: str | None = None


@dataclass
class Screenshot:
    """Represents a captured screenshot."""

    base64_data: str
    width: int
    height: int
    is_sensitive: bool = False
    mean_luma: float | None = None
    std_luma: float | None = None


def get_screenshot(device_id: str | None = None, timeout: int = 10) -> Screenshot:
    """
    Capture a screenshot from the connected HarmonyOS device.

    Args:
        device_id: Optional HDC device ID for multi-device setups.
        timeout: Timeout in seconds for screenshot operations.

    Returns:
        Screenshot object containing base64 data and dimensions.

    Note:
        If the screenshot fails (e.g., on sensitive screens like payment pages),
        a black fallback image is returned with is_sensitive=True.
    """
    temp_path = os.path.join(tempfile.gettempdir(), f"screenshot_{uuid.uuid4()}.png")
    hdc_prefix = _get_hdc_prefix(device_id)

    try:
        # Use a unique remote path each time to avoid stale captures
        remote_path = f"/data/local/tmp/tmp_screenshot_{uuid.uuid4().hex}.jpeg"

        def _cmd_failed(result) -> bool:
            out = ((result.stdout or "") + (result.stderr or "")).lower()
            return (
                result.returncode != 0
                or "fail" in out
                or "error" in out
                or "need connect-key" in out
                or "inaccessible" in out
                or "not found" in out
            )

        # Prefer snapshot_display (supported on this device), then fallback to screenshot.
        snap = _run_hdc_command(
            hdc_prefix + ["shell", "snapshot_display", "-f", remote_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if _cmd_failed(snap):
            shot = _run_hdc_command(
                hdc_prefix + ["shell", "screenshot", remote_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if _cmd_failed(shot):
                return _create_fallback_screenshot(is_sensitive=True)

        # Pull screenshot to local temp path
        # Note: remote file is JPEG, but PIL can open it regardless of local extension
        recv = _run_hdc_command(
            hdc_prefix + ["file", "recv", remote_path, temp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if recv.returncode != 0 or not os.path.exists(temp_path) or os.path.getsize(temp_path) <= 0:
            return _create_fallback_screenshot(is_sensitive=True)

        # PIL automatically detects the image format from file content.
        # IMPORTANT: Keep width/height as the real device resolution for coordinate mapping,
        # but compress the image payload to reduce model input token usage.
        img = Image.open(temp_path)
        width, height = img.size

        img_for_model = img.convert("RGB")
        # Zhipu/OpenAI-compatible gateways may count base64 in the context window.
        # Keep the payload small enough to avoid 400 "maximum context length".
        max_side = int(os.getenv("PHONE_AGENT_SCREENSHOT_MAX_SIDE", "640"))
        jpeg_quality = int(os.getenv("PHONE_AGENT_SCREENSHOT_JPEG_QUALITY", "55"))
        w, h = img_for_model.size
        if max(w, h) > max_side:
            scale = max(w, h) / float(max_side)
            img_for_model = img_for_model.resize((int(w / scale), int(h / scale)))

        # Detect "black-like" screenshots (some sensitive screens return a black frame without an explicit error).
        gray_probe = img_for_model.convert("L").resize((64, 64))
        pixels = list(gray_probe.getdata())
        mean_luma = sum(pixels) / max(1, len(pixels))
        var = sum((p - mean_luma) ** 2 for p in pixels) / max(1, len(pixels))
        std_luma = var**0.5
        is_black_like = mean_luma < 8 and std_luma < 6

        buf = io.BytesIO()
        img_for_model.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        base64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        return Screenshot(
            base64_data=base64_data,
            width=width,
            height=height,
            # Treat black-like frames as "sensitive" so the agent can avoid hallucinating based on an unusable screenshot.
            is_sensitive=is_black_like,
            mean_luma=float(mean_luma),
            std_luma=float(std_luma),
        )

    except Exception as e:
        print(f"Screenshot error: {e}")
        return _create_fallback_screenshot(is_sensitive=True)
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # Best-effort cleanup on device
            _run_hdc_command(_get_hdc_prefix(device_id) + ["shell", "rm", "-f", remote_path], timeout=3)
        except Exception:
            pass


def _get_hdc_prefix(device_id: str | None) -> list:
    """Get HDC command prefix with optional device specifier."""
    if device_id:
        return ["hdc", "-t", device_id]

    global _AUTO_DEVICE_ID
    if _AUTO_DEVICE_ID is None:
        try:
            result = _run_hdc_command(
                ["hdc", "list", "targets"], capture_output=True, text=True, timeout=5
            )
            lines = (result.stdout or "").strip().splitlines()
            candidates = [l.strip() for l in lines if l.strip()]
            _AUTO_DEVICE_ID = candidates[0] if candidates else None
        except Exception:
            _AUTO_DEVICE_ID = None

    if _AUTO_DEVICE_ID:
        return ["hdc", "-t", _AUTO_DEVICE_ID]
    return ["hdc"]


def _create_fallback_screenshot(is_sensitive: bool) -> Screenshot:
    """Create a black fallback image when screenshot fails."""
    default_width, default_height = 1080, 2400

    black_img = Image.new("RGB", (default_width, default_height), color="black")
    buffered = BytesIO()
    black_img.save(buffered, format="JPEG", quality=70, optimize=True)
    base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return Screenshot(
        base64_data=base64_data,
        width=default_width,
        height=default_height,
        is_sensitive=is_sensitive,
        mean_luma=0.0,
        std_luma=0.0,
    )
