"""VDD health check - USBMMVID virtual display driver diagnostics.

Aligned with original app.desktop.vdd_health - reads Windows registry
to detect VDD driver status, device nodes, and health metrics.
"""
from __future__ import annotations

import winreg
from typing import Any


VDD_KEYWORDS = ["mttvdd", "virtual display driver", "usbmmidd", "usbmmvid"]
VDD_CLASS_GUID = "{4d36e968-e325-11ce-bfc1-08002be10318}"
VDD_MAX_DEVICES = 8
VDD_MIN_WIDTH = 1024
VDD_ENUM_FLAGS = 14


def vdd_health_report() -> dict[str, Any]:
    """Generate a VDD health report."""
    devices = summarize_vdd_devices()
    return {
        "vdd_detected": len(devices.get("devices", [])) > 0,
        "device_count": len(devices.get("devices", [])),
        "devices": devices.get("devices", []),
        "healthy": len(devices.get("devices", [])) > 0,
    }


def summarize_vdd_devices(
    devices: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize VDD device status."""
    if devices is None:
        devices = _registry_vdd_devices()
    return {
        "devices": devices,
        "count": len(devices),
        "all_ok": all(
            d.get("status", "") == "OK" for d in devices),
    }


def _registry_vdd_devices() -> list[dict[str, Any]]:
    """Scan registry for VDD devices."""
    results = []
    try:
        key_path = f"SYSTEM\\CurrentControlSet\\Enum\\DISPLAY"
        hkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
        i = 0
        while True:
            try:
                subkey_name = winreg.EnumKey(hkey, i)
                i += 1
                if any(
                    kw.lower() in subkey_name.lower()
                    for kw in VDD_KEYWORDS
                ):
                    results.append({
                        "name": subkey_name,
                        "status": _device_node_status(subkey_name),
                        "friendly": _friendly_name(subkey_name),
                    })
            except OSError:
                break
        winreg.CloseKey(hkey)
    except OSError:
        pass
    return results


def _device_node_status(instance_id: str) -> str:
    """Get device node status from registry."""
    try:
        key_path = (
            f"SYSTEM\\CurrentControlSet\\Enum\\DISPLAY\\{instance_id}")
        hkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
        try:
            status, _ = winreg.QueryValueEx(hkey, "ConfigFlags")
            return "OK" if status == 0 else f"ConfigFlags={status}"
        except FileNotFoundError:
            return "UNKNOWN"
        finally:
            winreg.CloseKey(hkey)
    except OSError:
        return "ERROR"


def _enum_subkeys(key_path: str) -> list[str]:
    """Enumerate subkeys under a registry path."""
    result = []
    try:
        hkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
        i = 0
        while True:
            try:
                result.append(winreg.EnumKey(hkey, i))
                i += 1
            except OSError:
                break
        winreg.CloseKey(hkey)
    except OSError:
        pass
    return result


def _read_registry_values(key_path: str) -> dict[str, Any]:
    """Read all values from a registry key."""
    result = {}
    try:
        hkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
        i = 0
        while True:
            try:
                name, value, vtype = winreg.EnumValue(hkey, i)
                result[name] = value
                i += 1
            except OSError:
                break
        winreg.CloseKey(hkey)
    except OSError:
        pass
    return result


def _value_strings(values: dict[str, Any]) -> list[str]:
    """Extract string values from registry dict."""
    return [str(v) for v in values.values() if v is not None]


def _as_string_list(value: Any) -> list[str]:
    """Convert value to string list."""
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)] if value is not None else []


def _friendly_name(instance_id: str) -> str:
    """Get friendly name for a device."""
    try:
        key_path = (
            f"SYSTEM\\CurrentControlSet\\Enum\\DISPLAY\\{instance_id}"
            "\\Device Parameters")
        values = _read_registry_values(key_path)
        return values.get("FriendlyName", instance_id)
    except Exception:
        return instance_id