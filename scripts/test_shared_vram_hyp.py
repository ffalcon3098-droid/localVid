#!/usr/bin/env python3
"""
localVid Hypothesis Test Runner:
1. Shared VRAM = 0 hypothesis on 480x320 x 17f (--steps 1, TAEHV, --max-vram 2.0 / 1.5)
2. Automatic branching:
   - If Shared VRAM ~0 and step time drops significantly (<180s): run 640x384 x 17f TDR survival test.
   - If Shared VRAM ~0 and step time stays >=200s: run 480x320 --flash-attn off comparison.
"""

from __future__ import annotations

import json
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


def run_case(label: str, width: int, height: int, max_vram: float, flash_attn: str) -> dict:
    run_dir = pathlib.Path("benchmarks/runs/hyp_test")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{label}.log"
    out_avi = run_dir / f"{label}.avi"

    cmd = [
        "bin/sdcpp-vulkan/sd-cli.exe",
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

    stop = threading.Event()
    peaks = {"ded_mib": 0.0, "shr_mib": 0.0, "ded_dit_mib": 0.0, "shr_dit_mib": 0.0, "samples": 0}
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
                peaks["samples"] += 1
            stop.wait(1.0)

    print(f"\n=== RUNNING {label} ({width}x{height}, max-vram={max_vram}, flash-attn={flash_attn}) ===", flush=True)
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
    results = []
    # 1. Test --max-vram 2.0 on 480x320
    r1 = run_case("1a_480x320_mv2.0_fa_on", 480, 320, 2.0, "off,diffusion=on")
    results.append(r1)

    best_mv = 2.0
    best_r = r1

    # If Shared VRAM during DiT still spilled (>400 MiB), try --max-vram 1.5
    if r1["peak_dit_shared_mib"] > 400.0:
        r1b = run_case("1b_480x320_mv1.5_fa_on", 480, 320, 1.5, "off,diffusion=on")
        results.append(r1b)
        best_mv = 1.5
        best_r = r1b

    # Critical distinction: if DiT step time is still >= 200s, test --flash-attn off
    if best_r["sampling_s"] is not None and best_r["sampling_s"] >= 200.0:
        r_fa = run_case(f"1c_480x320_mv{best_mv}_fa_off", 480, 320, best_mv, "off")
        results.append(r_fa)

    # 2. If speed improved significantly (<200s) OR Shared dropped, run 640x384 TDR survival test
    if best_r["sampling_s"] is not None and best_r["sampling_s"] < 200.0:
        r2 = run_case(f"2_640x384_mv{best_mv}_fa_on", 640, 384, best_mv, "off,diffusion=on")
        results.append(r2)
    else:
        # Still test 640x384 with best_mv (or fa_off if fa_off was much faster)
        fa_choice = "off,diffusion=on"
        if len(results) >= 2 and results[-1]["label"].endswith("fa_off"):
            if results[-1]["sampling_s"] and best_r["sampling_s"] and results[-1]["sampling_s"] < best_r["sampling_s"] * 0.7:
                fa_choice = "off"
        r2 = run_case(f"2_640x384_mv{best_mv}_fa_{'off' if fa_choice == 'off' else 'on'}", 640, 384, best_mv, fa_choice)
        results.append(r2)

    out_json = pathlib.Path("benchmarks/runs/hyp_test/summary.json")
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nALL_HYP_TESTS_DONE -> {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
