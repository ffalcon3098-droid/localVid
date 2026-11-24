#!/usr/bin/env python3
"""
localVid 640x384 Fast Attention + Q-Chunk Frame Ladder (33 -> 49 -> 81 frames).
Tests --flash-attn off + SD_WAN_SPLIT_BLOCK=1 + SD_WAN_ATTN_HEAD_CUT=1 + SD_WAN_ATTN_Q_CHUNK
with careful --max-vram control (starts at 2.0 GiB, retries up to 2.5 GiB only on budget gate).
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


def run_rung(frames: int, head_chunk: int, q_chunk: int, max_vram: float) -> dict:
    run_dir = pathlib.Path("benchmarks/runs/ladder_640_fast")
    run_dir.mkdir(parents=True, exist_ok=True)
    label = f"640x384_{frames}f_h{head_chunk}_q{q_chunk}_mv{max_vram}"
    log_path = run_dir / f"{label}.log"
    out_avi = run_dir / f"{label}.avi"

    cmd = [
        "build/sdcpp-vulkan/bin/sd-cli.exe",
        "-M", "vid_gen",
        "--diffusion-model", "models/diffusion_models/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf",
        "--t5xxl", "models/text_encoders/umt5-xxl-encoder-Q4_K_M.gguf",
        "--taesd", "models/vae/taew2_1.safetensors",
        "-p", "A cinematic wide shot of ocean waves crashing onto dark volcanic rocks at golden hour",
        "-W", "640",
        "-H", "384",
        "--video-frames", str(frames),
        "--fps", "16",
        "--steps", "1",
        "--cfg-scale", "1.0",
        "--seed", "42",
        "--offload-to-cpu",
        "--mmap",
        "--backend", "te=cpu",
        "--flash-attn", "off",
        "--max-vram", str(max_vram),
        "-v",
        "-o", str(out_avi),
    ]

    env = os.environ.copy()
    env.update(
        {
            "SD_LOADER_FAST_POLL": "1",
            "SD_EVICT_ON_END": "1",
            "SD_EVICT_PAST": "1",
            "SD_DROP_QUEUED": "1",
            "SD_STAGING_POOL_MB": "512",
            "SD_SEG_TIMING": "1",
            "SD_WAN_SPLIT_BLOCK": "1",
            "SD_WAN_ATTN_HEAD_CHUNK": str(head_chunk),
            "SD_WAN_ATTN_HEAD_CUT": "1",
            "SD_WAN_ATTN_Q_CHUNK": str(q_chunk),
            "SD_WAN_MLP_CHUNK_MB": "128",
        }
    )

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

    print(f"\n=== RUNNING {label} ===", flush=True)
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
    sa_h_ms = [float(m.group(1)) for m in re.finditer(r"wan\.blocks\.0\.sa_h\d+\):\s*([\d.]+)\s*ms", txt)]
    n_groups = (40 + head_chunk - 1) // head_chunk
    sa_h_avg = round(sum(sa_h_ms) / len(sa_h_ms), 1) if sa_h_ms else None
    sa_h_block_est = round(sa_h_avg * n_groups, 1) if sa_h_avg is not None else None
    errs = [l.strip() for l in lines if "[E]" in l or "ErrorDeviceLost" in l or "cannot make enough memory" in l][-8:]

    res = {
        "label": label,
        "frames": frames,
        "head_chunk": head_chunk,
        "q_chunk": q_chunk,
        "max_vram_gib": max_vram,
        "returncode": proc.returncode,
        "wall_s": wall,
        "sampling_s": float(m_samp.group(1)) if m_samp else None,
        "vae_s": float(m_vae.group(1)) if m_vae else None,
        "sa_h_avg_ms": sa_h_avg,
        "sa_h_block_est_ms": sa_h_block_est,
        "peak_dit_dedicated_mib": round(peaks["ded_dit_mib"], 1),
        "peak_dit_shared_mib": round(peaks["shr_dit_mib"], 1),
        "compute_buffers": m_buf,
        "errors": errs,
        "log_file": str(log_path),
    }
    print(json.dumps(res, indent=2), flush=True)
    return res


def main() -> int:
    plan = [
        (33, 4, 2048, [2.0, 2.5]),
        (49, 4, 1024, [2.0, 2.5]),
        (81, 4, 1024, [2.5, 3.2]),
    ]
    results = []
    for frames, h_chunk, q_chunk, mv_list in plan:
        passed = False
        for mv in mv_list:
            r = run_rung(frames, h_chunk, q_chunk, mv)
            results.append(r)
            if r["returncode"] == 0:
                passed = True
                break
            if not any("cannot make enough memory" in e for e in r["errors"]):
                # Hard crash (TDR/OOM/assert) — stop trying higher max-vram
                break
        if not passed:
            print(f"Stopping ladder at {frames} frames.", flush=True)
            break

    out_json = pathlib.Path("benchmarks/runs/ladder_640_fast/summary.json")
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nLADDER_640_FAST_DONE -> {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
