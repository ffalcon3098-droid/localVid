#!/usr/bin/env python3
"""
localVid P2 Benchmark:
1. 640x384 x 17f TDR survival + blocks.0 sub-segment timing breakdown (Self-Attn / Cross-Attn / FFN)
2. 640x384 frame ladder (33 -> 49 -> 81 frames) with --max-vram 1.5, --steps 1, TAEHV.
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


def run_rung(frames: int, env_flags: dict[str, str], width: int = 640, height: int = 384, max_vram: float = 1.5) -> dict:
    run_dir = pathlib.Path("benchmarks/runs/p2_bench")
    run_dir.mkdir(parents=True, exist_ok=True)
    label = f"p2_{width}x{height}_{frames}f"
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
        "--video-frames", str(frames),
        "--fps", "16",
        "--steps", "1",
        "--cfg-scale", "1.0",
        "--seed", "42",
        "--offload-to-cpu",
        "--mmap",
        "--backend", "te=cpu",
        "--flash-attn", "off,diffusion=on",
        "--max-vram", str(max_vram),
        "-v",
        "-o", str(out_avi),
    ]

    env = os.environ.copy()
    env.update(env_flags)

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

    print(f"\n=== RUNNING {label} (max-vram={max_vram}) ===", flush=True)
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
    seg_timings = [l.strip() for l in lines if "seg_timing:" in l][:20]
    staged_sample = [l.strip() for l in lines if "model manager staged compute params" in l][:12]
    errs = [l.strip() for l in lines if "[E]" in l or "ErrorDeviceLost" in l][-8:]

    res = {
        "label": label,
        "frames": frames,
        "returncode": proc.returncode,
        "wall_s": wall,
        "sampling_s": float(m_samp.group(1)) if m_samp else None,
        "vae_s": float(m_vae.group(1)) if m_vae else None,
        "peak_dit_dedicated_mib": round(peaks["ded_dit_mib"], 1),
        "peak_dit_shared_mib": round(peaks["shr_dit_mib"], 1),
        "peak_total_dedicated_mib": round(peaks["ded_mib"], 1),
        "compute_buffers": m_buf,
        "seg_timings": seg_timings,
        "staged_sample": staged_sample,
        "errors": errs,
        "log_file": str(log_path),
    }
    print(json.dumps(res, indent=2), flush=True)
    return res


def main() -> int:
    p2_env = {
        "SD_LOADER_FAST_POLL": "1",
        "SD_EVICT_ON_END": "1",
        "SD_EVICT_PAST": "1",
        "SD_DROP_QUEUED": "1",
        "SD_STAGING_POOL_MB": "512",
        "SD_SEG_TIMING": "1",
        "SD_WAN_SPLIT_BLOCK": "1",
        "SD_WAN_ATTN_HEAD_CHUNK": "8",
        "SD_WAN_ATTN_HEAD_CUT": "1",
        "SD_WAN_MLP_CHUNK_MB": "128",
    }
    results = []
    for f in [17, 33, 49, 81]:
        r = run_rung(f, p2_env)
        results.append(r)
        if r["returncode"] != 0:
            print(f"Stopping ladder at {f} frames (rc={r['returncode']}).", flush=True)
            break

    out_json = pathlib.Path("benchmarks/runs/p2_bench/summary.json")
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nP2_BENCH_DONE -> {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
