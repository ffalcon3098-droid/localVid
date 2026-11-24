#!/usr/bin/env python3
"""
localVid Hardware Diagnostic Probe (CUDA + Vulkan + System RAM).
Reads real GPU VRAM budget, Vulkan maxMemoryAllocationSize, driver capabilities,
and sd-cli --list-devices output, then emits a structured JSON report.

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_cmd(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return -1, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -2, "", f"command timed out after {timeout}s"


def probe_system() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "cpu_threads": os.cpu_count(),
        "ram_total_bytes": None,
        "ram_total_gib": None,
    }
    if platform.system() == "Windows":
        rc, out, _ = run_cmd(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ]
        )
        if rc == 0 and out.strip().isdigit():
            b = int(out.strip())
            info["ram_total_bytes"] = b
            info["ram_total_gib"] = round(b / (1024**3), 2)
    else:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["ram_total_bytes"] = kb * 1024
                        info["ram_total_gib"] = round(kb / (1024**2), 2)
                        break
        except OSError:
            pass
    return info


def probe_nvidia_cuda() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "gpus": []}
    rc, out, err = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,memory.used,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0:
        return {"available": False, "error": err.strip(), "gpus": []}

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "driver_version": parts[2],
                    "vram_total_mib": int(parts[3]),
                    "vram_free_mib": int(parts[4]),
                    "vram_used_mib": int(parts[5]),
                    "compute_capability": parts[6],
                }
            )
    return {"available": len(gpus) > 0, "gpus": gpus}


def probe_vulkan() -> dict[str, Any]:
    exe = shutil.which("vulkaninfo")
    if not exe:
        return {"available": False, "error": "vulkaninfo not found in PATH", "devices": []}

    rc, out, err = run_cmd([exe], timeout=30)
    if not out:
        return {"available": False, "error": err.strip(), "devices": []}

    devices: list[dict[str, Any]] = []
    # Split per GPU block (GPU0:, GPU1:, ...)
    gpu_blocks = re.split(r"(?=^GPU\d+:)", out, flags=re.MULTILINE)
    for block in gpu_blocks:
        m_idx = re.match(r"^GPU(\d+):", block)
        if not m_idx:
            continue
        dev: dict[str, Any] = {"gpu_index": int(m_idx.group(1))}

        def extract_str(pattern: str) -> str | None:
            m = re.search(pattern, block)
            return m.group(1).strip() if m else None

        dev["name"] = extract_str(r"deviceName\s*=\s*(.+)")
        dev["device_type"] = extract_str(r"deviceType\s*=\s*(PHYSICAL_DEVICE_TYPE_\w+)")
        dev["vendor_id"] = extract_str(r"vendorID\s*=\s*(0x[0-9a-fA-F]+)")
        dev["device_id"] = extract_str(r"deviceID\s*=\s*(0x[0-9a-fA-F]+)")
        dev["api_version"] = extract_str(r"apiVersion\s*=\s*([^\r\n(]+)")
        dev["driver_name"] = extract_str(r"driverName\s*=\s*(.+)")
        dev["driver_info"] = extract_str(r"driverInfo\s*=\s*(.+)")

        # maxMemoryAllocationSize (hex or decimal)
        m_alloc = re.search(r"maxMemoryAllocationSize\s*=\s*(0x[0-9a-fA-F]+|\d+)", block)
        if m_alloc:
            val = int(m_alloc.group(1), 0)
            dev["max_memory_allocation_bytes"] = val
            dev["max_memory_allocation_gib"] = round(val / (1024**3), 3)

        m_ssbo = re.search(r"maxStorageBufferRange\s*=\s*(0x[0-9a-fA-F]+|\d+)", block)
        if m_ssbo:
            val = int(m_ssbo.group(1), 0)
            dev["max_storage_buffer_range_bytes"] = val

        # Key extensions & features
        dev["extensions"] = {
            "VK_EXT_memory_budget": "VK_EXT_memory_budget" in block,
            "VK_KHR_cooperative_matrix": "VK_KHR_cooperative_matrix" in block,
        }
        dev["features"] = {
            "shaderFloat16": bool(re.search(r"shaderFloat16\s*=\s*true", block)),
            "shaderInt8": bool(re.search(r"shaderInt8\s*=\s*true", block)),
            "storageBuffer16BitAccess": bool(re.search(r"storageBuffer16BitAccess\s*=\s*true", block)),
        }

        # Parse memoryHeaps[i]
        heaps = []
        heap_pattern = re.compile(
            r"memoryHeaps\[(\d+)\]:\s*"
            r"size\s*=\s*(\d+)[^\n]*\n"
            r"(?:\s*budget\s*=\s*(\d+)[^\n]*\n)?"
            r"(?:\s*usage\s*=\s*(\d+)[^\n]*\n)?"
            r"(?:\s*flags:\s*count\s*=\s*\d+\s*\n((?:\s+MEMORY_HEAP_\w+\n)*))?",
            re.MULTILINE,
        )
        for hm in heap_pattern.finditer(block):
            h_idx = int(hm.group(1))
            size_b = int(hm.group(2))
            budget_b = int(hm.group(3)) if hm.group(3) else None
            usage_b = int(hm.group(4)) if hm.group(4) else None
            flags_raw = hm.group(5) or ""
            flags = re.findall(r"MEMORY_HEAP_\w+", flags_raw)
            heaps.append(
                {
                    "index": h_idx,
                    "size_bytes": size_b,
                    "size_gib": round(size_b / (1024**3), 3),
                    "budget_bytes": budget_b,
                    "budget_gib": round(budget_b / (1024**3), 3) if budget_b is not None else None,
                    "usage_bytes": usage_b,
                    "usage_mib": round(usage_b / (1024**2), 2) if usage_b is not None else None,
                    "flags": flags,
                    "device_local": "MEMORY_HEAP_DEVICE_LOCAL_BIT" in flags,
                }
            )
        dev["memory_heaps"] = heaps

        dl_heaps = [h for h in heaps if h["device_local"]]
        dev["device_local_total_gib"] = round(sum(h["size_bytes"] for h in dl_heaps) / (1024**3), 3)
        if dl_heaps and all(h["budget_bytes"] is not None for h in dl_heaps):
            dev["device_local_budget_gib"] = round(
                sum(h["budget_bytes"] for h in dl_heaps if h["budget_bytes"] is not None) / (1024**3), 3
            )
            dev["primary_device_local_budget_gib"] = dl_heaps[0]["budget_gib"]
        else:
            dev["device_local_budget_gib"] = None
            dev["primary_device_local_budget_gib"] = None

        devices.append(dev)

    return {"available": len(devices) > 0, "devices": devices}


def probe_sd_cli(sd_cli_path: Path | None) -> dict[str, Any]:
    if sd_cli_path is None or not sd_cli_path.exists():
        return {"available": False, "path": str(sd_cli_path) if sd_cli_path else None}
    rc, out, err = run_cmd([str(sd_cli_path), "--list-devices"], timeout=20)
    combined = (out + "\n" + err).strip()
    vk_line = None
    devices = []
    for line in combined.splitlines():
        if "ggml_vulkan: 0 =" in line:
            vk_line = line.strip()
        if line.startswith(("Vulkan", "CUDA", "ROCm", "CPU")):
            devices.append(line.strip())
    return {
        "available": rc == 0,
        "path": str(sd_cli_path),
        "ggml_vulkan_summary": vk_line,
        "listed_devices": devices,
        "raw_output": combined,
    }


def build_recommendations(report: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    vk = report.get("vulkan", {})
    for d in vk.get("devices", []):
        max_alloc = d.get("max_memory_allocation_gib")
        if max_alloc is not None and max_alloc <= 2.05:
            recs.append(
                f"Vulkan GPU '{d.get('name')}' has maxMemoryAllocationSize={max_alloc} GiB. "
                "Any single tensor/compute/im2col buffer > 2.0 GiB will fail with VK_ERROR_OUT_OF_DEVICE_MEMORY."
            )
        if d.get("device_type") == "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU":
            budget = d.get("primary_device_local_budget_gib")
            if budget is not None and budget < 6.0:
                recs.append(
                    f"Primary DEVICE_LOCAL heap budget is {budget} GiB (discrete GPU). "
                    "Use --max-vram <= "
                    f"{max(2.5, round(budget - 0.35, 2))} GiB to avoid sd.cpp model_manager.cpp:2477 Vulkan budget underflow (available_device_bytes=0)."
                )
    return recs


def main() -> int:
    parser = argparse.ArgumentParser(description="localVid cross-vendor GPU/VRAM diagnostic probe")
    parser.add_argument(
        "--sd-cli",
        type=Path,
        default=Path("bin/sdcpp-vulkan/sd-cli.exe"),
        help="Path to sd-cli binary (optional)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/hardware_profile.json"),
        help="Output JSON report path",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "system": probe_system(),
        "cuda": probe_nvidia_cuda(),
        "vulkan": probe_vulkan(),
        "sd_cli": probe_sd_cli(args.sd_cli),
    }
    report["recommendations"] = build_recommendations(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote hardware profile to {args.out}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
