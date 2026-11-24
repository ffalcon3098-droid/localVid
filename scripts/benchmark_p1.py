#!/usr/bin/env python3
"""
localVid P1 Benchmark:
Compares patched build/sdcpp-vulkan/bin/sd-cli.exe with P1 env flags:
  - SD_LOADER_FAST_POLL=1
  - SD_EVICT_ON_END=1
  - SD_EVICT_PAST=1
  - SD_DROP_QUEUED=1
  - SD_STAGING_POOL_MB=512
against stock 480x320 x 17f (--steps 1, --max-vram 1.5, TAEHV).
Also measures --flash-attn auto (normal attention on RDNA 1, 16.7x faster per probe).
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

PS_VRAM_CMD = (
    "$d = (Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue)"
    ".CounterSamples | Measure-Object -Property CookedValue -Maximum | Select-Object -ExpandProperty Maximum; "
    "$s = (Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage' -ErrorAction SilentlyContinue)"
    ".CounterSamples | Measure-Object -Property CookedValue -Maximum | Select-Object -ExpandProperty Maximum; "
    "Write-Output \"$d,$s\""
)


def sample_vram_mib() -> tuple[float | None, float | None]:
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", PS_VRAM_CMD],
            text=True,
            timeout=4,
        ).strip()
        d_str, s_str = out.split(",")
        return float(d_str) / (1024.0 * 1024.0), float(s_str) / (1024.0 * 1024.0)
    except Exception:
        return None, None


def run_case(label: str, width: int, height: int, max_vram: float, flash_attn: str, extra_env: dict[str, str]) -> dict:
    run_dir = pathlib.Path("benchmarks/runs/p1_bench")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{label}.log"
    out_avi = run_dir / f"{label}.avi"

    cmd = [
        "build/sdcpp-vulkan/bin/sd-cli.exe",
        "-M", "vid_gen",
        "--diffusion-model", "models/diffusion_models/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf",
        "--t5xxl", "models/text_encoders/umt5-xxl-encoder-Q4_K_M.gguf",
        "--taesd", "models/vae/taew2_1.safetensors",
        "-p", "A cinematic wide shot of ocean waves crashing onto dark volcanic rocks at golden hour",
        "-W", str(width),
        "-H", str(height),
        "--video-frames", "17",
        "--fps", "16",
        "--steps", "1",
        "--cfg-scale", "1.0",
        "--seed", "42",
        "--offload-to-cpu",
        "--mmap",
        "--backend", "te=cpu",
        "--flash-attn", flash_attn,
        "--max-vram", str(max_vram),
        "-v",
        "-o", str(out_avi),
    ]

    env = os.environ.copy()
    env.update(extra_env)

    stop = threading.Event()
    peaks = {"ded_mib": 0.0, "shr_mib": 0.0, "ded_dit_mib": 0.0, "shr_dit_mib": 0.0}
    in_vae = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            d, s = sample_vram_mib()
            if d is not None and s is not None:
                peaks["ded_mib"] = max(peaks["ded_mib"], d)
                peaks["shr_mib"] = max(peaks["shr_mib"], s)
                if not in_vae.is_set():
                    peaks["ded_dit_mib"] = max(peaks["ded_dit_mib"], d)
                    peaks["shr_dit_mib"] = max(peaks["shr_dit_mib"], s)
            stop.wait(1.0)

    print(f"\n=== RUNNING {label} ({width}x{height}, mv={max_vram}, fa={flash_attn}, env={extra_env}) ===", flush=True)
    t = threading.Thread(target=poll, daemon=True)
    t0 = time.monotonic()
    t.start()
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            lines.append(line)
            if "sampling completed" in line:
                in_vae.set()
        proc.wait()

    stop.set()
    t.join(timeout=5)
    wall = round(time.monotonic() - t0, 2)
    txt = "".join(lines)

    m_samp = re.search(r"sampling completed, taking ([\d.]+)s", txt)
    m_vae = re.search(r"computing vae decode graph completed, taking ([\d.]+)s", txt)
    m_buf = re.findall(r"(Wan2\.x-T2V-14B|taehv) compute buffer size:\s*([\d.]+) MB\(([^)]+)\)", txt)
    errs = [l.strip() for l in lines if "[E]" in l or "ErrorDeviceLost" in l][-6:]

    res = {
        "label": label,
        "width": width,
        "height": height,
        "max_vram_gib": max_vram,
        "flash_attn": flash_attn,
        "env": extra_env,
        "returncode": proc.returncode,
        "wall_s": wall,
        "sampling_s": float(m_samp.group(1)) if m_samp else None,
        "vae_s": float(m_vae.group(1)) if m_vae else None,
        "peak_dedicated_mib": round(peaks["ded_mib"], 1),
        "peak_shared_mib": round(peaks["shr_mib"], 1),
        "peak_dit_dedicated_mib": round(peaks["ded_dit_mib"], 1),
        "peak_dit_shared_mib": round(peaks["shr_dit_mib"], 1),
        "compute_buffers": m_buf,
        "errors": errs,
        "log_file": str(log_path),
    }
    print(json.dumps(res, indent=2), flush=True)
    return res


def main() -> int:
    p1_env = {
        "SD_LOADER_FAST_POLL": "1",
        "SD_EVICT_ON_END": "1",
        "SD_EVICT_PAST": "1",
        "SD_DROP_QUEUED": "1",
        "SD_STAGING_POOL_MB": "512",
    }
    results = []
    # P1-A: Apples-to-apples vs stock 137.45s (flash-attn=off,diffusion=on)
    results.append(run_case("p1a_480x320_mv1.5_fa_on", 480, 320, 1.5, "off,diffusion=on", p1_env))
    # P1-B: Same P1 flags + flash-attn=auto (RDNA 1 normal attention)
    results.append(run_case("p1b_480x320_mv1.5_fa_auto", 480, 320, 1.5, "auto", p1_env))

    out_json = pathlib.Path("benchmarks/runs/p1_bench/summary.json")
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nP1_BENCH_DONE -> {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
