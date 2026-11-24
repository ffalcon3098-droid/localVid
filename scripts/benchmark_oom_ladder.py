#!/usr/bin/env python3
"""
localVid Phase 1-2 OOM Ladder & Bottleneck Profiler for Wan 2.2 A14B MoE.

Runs an automated frame ladder (default: 17 -> 33 -> 49 -> 65 -> 81 frames)
using stock or patched stable-diffusion.cpp (Vulkan or CUDA), polls live GPU
Dedicated VRAM, parses DiT vs VAE compute buffer peaks and timings, and
classifies the exact OOM failure mode:
  - Class A: Weight residency / blocks.0 capacity gate / Vulkan budget underflow
  - Class B: Single allocation > maxMemoryAllocationSize (2.0 GiB) or DiT MLP/Attn peak
  - Class C: Graph-cut boundary cache accumulation
  - Class D: Wan VAE 3D im2col / decode OOM or forced tiling fallback

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
import threading
import time
from pathlib import Path
from typing import Any


class VramMonitor:
    """Background VRAM poller supporting NVIDIA (nvidia-smi) and Windows AMD/Intel (PerfCounter)."""

    def __init__(self, interval_sec: float = 1.0) -> None:
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples_mib: list[tuple[float, float]] = []
        self.t0 = 0.0
        self.has_nvidia = shutil.which("nvidia-smi") is not None

    def _sample_once_mib(self) -> float | None:
        if self.has_nvidia:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=3,
                )
                first = out.strip().splitlines()[0].strip()
                return float(first)
            except Exception:
                return None

        if platform.system() == "Windows":
            try:
                cmd = [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples | Measure-Object -Property CookedValue -Maximum | Select-Object -ExpandProperty Maximum",
                ]
                out = subprocess.check_output(cmd, text=True, timeout=4).strip()
                if out:
                    return float(out) / (1024.0 * 1024.0)
            except Exception:
                return None
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            val = self._sample_once_mib()
            if val is not None:
                self.samples_mib.append((round(time.monotonic() - self.t0, 2), round(val, 1)))
            self._stop.wait(self.interval_sec)

    def start(self) -> None:
        self.samples_mib = []
        self.t0 = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if not self.samples_mib:
            return None
        return max(v for _, v in self.samples_mib)


def parse_sd_log(log_text: str, returncode: int) -> dict[str, Any]:
    """Extract DiT/VAE compute buffer sizes, timings, memory reports, and OOM class."""
    compute_buffers = []
    for m in re.finditer(
        r"([a-zA-Z0-9_.\-]+)\s+compute buffer size:\s*([\d.]+)\s*MB\(([^)]+)\)\s+on\s+(\S+)(?:\s+\(peak across\s+(\d+)\s+segment)?",
        log_text,
    ):
        compute_buffers.append(
            {
                "runner": m.group(1),
                "size_mb": float(m.group(2)),
                "mem_type": m.group(3),
                "backend": m.group(4),
                "segments": int(m.group(5)) if m.group(5) else 1,
            }
        )

    mem_reports = []
    for m in re.finditer(
        r"model manager memory on (\S+): reported free ([\d.]+) MB / total ([\d.]+) MB, tracked weights ([\d.]+) MB / other runtime ([\d.]+) MB / current runtime ([\d.]+) MB",
        log_text,
    ):
        mem_reports.append(
            {
                "backend": m.group(1),
                "free_mb": float(m.group(2)),
                "total_mb": float(m.group(3)),
                "tracked_weights_mb": float(m.group(4)),
                "other_runtime_mb": float(m.group(5)),
                "current_runtime_mb": float(m.group(6)),
            }
        )

    # Timings
    def extract_time(pattern: str) -> float | None:
        m = re.search(pattern, log_text)
        return float(m.group(1)) if m else None

    te_sec = extract_time(r"get_learned_condition completed, taking ([\d.]+)s")
    if te_sec is None:
        te_ms = extract_time(r"computing condition graph completed, taking ([\d.]+) ms")
        if te_ms is not None:
            te_sec = round(te_ms / 1000.0, 2)
    sampling_sec = extract_time(r"sampling completed, taking ([\d.]+)s")
    vae_decode_sec = extract_time(r"(?:decode_first_stage|computing vae decode graph) completed, taking ([\d.]+)s")
    total_gen_sec = extract_time(r"generate_video completed in ([\d.]+)s")

    # Per-step timing hints from progress bar or step logs
    step_rates = re.findall(r"(\d+/\d+)\s*-\s*([\d.]+)(s/it|it/s)", log_text)

    # Capture key ERROR / WARN lines for concise root-cause reporting
    error_lines = [
        line.strip()
        for line in log_text.splitlines()
        if any(tag in line for tag in ("[ERROR]", "VK_ERROR", "failed to", "retrying with", "impossible free", "insufficient"))
    ][-15:]

    # Tiling fallback or OOM detection
    vae_tiling_fallback = bool(
        re.search(r"retrying with (?:temporal|spatial) tiling|VAE decode failed.*tiling", log_text, re.IGNORECASE)
    )
    budget_underflow = bool(re.search(r"impossible free memory report|available_device_bytes\s*=\s*0", log_text))

    # Classify failure
    oom_class = "OK"
    oom_detail = ""
    if returncode != 0 or vae_tiling_fallback:
        lower = log_text.lower()
        vae_started = ("decode_first_stage" in lower) or ("wan_vae" in lower) or ("vae decode" in lower)
        im2col_hit = "im2col" in lower
        max_alloc_hit = any(
            k in lower
            for k in [
                "vk_error_out_of_device_memory",
                "maxmemoryallocationsize",
                "exceeds max",
                "ggml_gallocr_reserve_n: failed to allocate",
            ]
        )
        cut_cache_hit = any(k in lower for k in ["cut_cache", "graph_cut", "runner_cache"])
        weight_gate_hit = any(
            k in lower
            for k in [
                "capacity check",
                "insufficient memory",
                "failed to prepare",
                "failed to reserve",
                "evict",
                "blocks.0",
            ]
        )

        if vae_tiling_fallback or (vae_started and (returncode != 0 or im2col_hit)):
            oom_class = "D_VAE_DECODE_OR_IM2COL"
            oom_detail = "VAE decode OOM / 3D im2col >2 GiB or forced tiling fallback."
        elif max_alloc_hit or any(b["size_mb"] >= 2000.0 for b in compute_buffers):
            oom_class = "B_SINGLE_BUFFER_OR_ACT_PEAK"
            oom_detail = "Single buffer > 2.0 GiB (maxMemoryAllocationSize) or DiT activation peak OOM."
        elif cut_cache_hit:
            oom_class = "C_CUT_CACHE_ACCUMULATION"
            oom_detail = "Graph-cut boundary tensor cache exceeded VRAM."
        elif weight_gate_hit or budget_underflow:
            oom_class = "A_WEIGHT_GATE_OR_BUDGET_UNDERFLOW"
            oom_detail = "blocks.0 / segment weight residency gate failed or Vulkan budget underflowed."
        else:
            oom_class = "UNKNOWN_FAILURE"
            oom_detail = f"Non-zero exit code ({returncode}); inspect raw log."

    return {
        "returncode": returncode,
        "oom_class": oom_class,
        "oom_detail": oom_detail,
        "vae_tiling_fallback": vae_tiling_fallback,
        "vulkan_budget_underflow_detected": budget_underflow,
        "compute_buffers": compute_buffers,
        "memory_reports_count": len(mem_reports),
        "memory_reports_sample": mem_reports[:5],
        "timings_sec": {
            "text_encoder": te_sec,
            "sampling": sampling_sec,
            "vae_decode": vae_decode_sec,
            "total_generate_video": total_gen_sec,
        },
        "step_rates": step_rates,
        "error_lines": error_lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="localVid Wan 2.2 A14B Phase 1-2 OOM Ladder Profiler")
    parser.add_argument("--config", type=Path, default=Path("configs/wan22_a14b_minimal.json"))
    parser.add_argument("--sd-cli", type=Path, default=Path("bin/sdcpp-vulkan/sd-cli.exe"))
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--frames", type=int, nargs="+", default=[17, 33, 49, 65, 81])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-vram", type=float, default=None, help="Explicit --max-vram GiB limit (omit for stock behavior)")
    parser.add_argument("--use-high-noise", action="store_true", help="Include HighNoise expert if present")
    parser.add_argument("--use-taehv", action="store_true", help="Use TAEHV instead of full Wan 2.1 VAE")
    parser.add_argument("--vae-tiling", action="store_true", help="Enable --vae-tiling")
    parser.add_argument("--stop-on-oom", action="store_true", help="Stop ladder after first hard OOM")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    files_by_role = {f["role"]: Path(f["filename"]) for f in cfg["download_sets"]["files"]}

    required_roles = ["diffusion_model", "t5xxl", "taesd" if args.use_taehv else "vae"]
    if args.use_high_noise:
        required_roles.append("high_noise_diffusion_model")

    missing = [str(files_by_role[r]) for r in required_roles if not files_by_role[r].exists()]
    if missing and not args.dry_run:
        print("ERROR: Missing required model files:")
        for m in missing:
            print(f"  - {m}")
        print("Download the approved model set first, or pass --dry-run to preview commands.")
        return 2

    if not args.sd_cli.exists() and not args.dry_run:
        print(f"ERROR: sd-cli not found at {args.sd_cli}")
        return 2

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("benchmarks/runs") / f"oom_ladder_{args.width}x{args.height}_{stamp}"
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    prompt = cfg["oom_ladder_defaults"]["prompt"]

    for n_frames in args.frames:
        out_video = run_dir / f"wan22_{args.width}x{args.height}_{n_frames}f.avi"
        cmd = [
            str(args.sd_cli),
            "-M", "vid_gen",
            "--diffusion-model", str(files_by_role["diffusion_model"]),
            "--t5xxl", str(files_by_role["t5xxl"]),
            "--vae" if not args.use_taehv else "--taesd",
            str(files_by_role["vae" if not args.use_taehv else "taesd"]),
            "-p", prompt,
            "-W", str(args.width),
            "-H", str(args.height),
            "--video-frames", str(n_frames),
            "--fps", str(cfg["oom_ladder_defaults"]["fps"]),
            "--steps", str(args.steps),
            "--cfg-scale", str(args.cfg_scale),
            "--seed", str(cfg["oom_ladder_defaults"]["seed"]),
            "--offload-to-cpu",
            "--mmap",
            "--backend", "te=cpu",
            "--flash-attn", "off,diffusion=on",
            "-v",
            "-o", str(out_video),
        ]
        if args.max_vram is not None:
            cmd.extend(["--max-vram", str(args.max_vram)])
        if args.use_high_noise:
            cmd.extend(["--high-noise-diffusion-model", str(files_by_role["high_noise_diffusion_model"])])
        if args.vae_tiling:
            cmd.append("--vae-tiling")

        print(f"\n=== Rung: {args.width}x{args.height} x {n_frames} frames ({args.steps} steps, max-vram={args.max_vram} GiB) ===")
        if args.dry_run:
            print("DRY-RUN:", " ".join(cmd))
            continue

        log_path = run_dir / f"rung_{n_frames}f.log"
        monitor = VramMonitor(interval_sec=1.0)
        t_start = time.monotonic()
        monitor.start()
        log_chunks: list[str] = []
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
                log_chunks.append(line)
            proc.wait()
        peak_vram_mib = monitor.stop()
        wall_sec = round(time.monotonic() - t_start, 2)
        full_log = "".join(log_chunks)

        parsed = parse_sd_log(full_log, proc.returncode)
        entry = {
            "width": args.width,
            "height": args.height,
            "frames": n_frames,
            "steps": args.steps,
            "max_vram_gib": args.max_vram,
            "wall_sec": wall_sec,
            "peak_dedicated_vram_mib": peak_vram_mib,
            "log_file": str(log_path),
            **parsed,
        }
        results.append(entry)
        print(
            f"Result: {entry['oom_class']} | wall={wall_sec}s | peak_vram={peak_vram_mib} MiB | "
            f"DiT/VAE buffers={[(b['runner'], b['size_mb']) for b in entry['compute_buffers']]}"
        )

        if args.stop_on_oom and entry["oom_class"] != "OK":
            print("Stopping ladder on first OOM as requested.")
            break

    if not args.dry_run:
        summary_json = run_dir / "summary.json"
        summary_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        md_lines = [
            f"# Wan 2.2 A14B OOM Ladder Report ({args.width}x{args.height})",
            "",
            "| Frames | Result | Wall (s) | Peak VRAM (MiB) | Sampling (s) | VAE Decode (s) | Compute Buffers (MB) |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
        for r in results:
            t = r["timings_sec"]
            bufs = ", ".join(f"{b['runner']}:{b['size_mb']:.0f}" for b in r["compute_buffers"]) or "-"
            md_lines.append(
                f"| {r['frames']} | `{r['oom_class']}` | {r['wall_sec']} | {r['peak_dedicated_vram_mib'] or '-'} | "
                f"{t['sampling'] or '-'} | {t['vae_decode'] or '-'} | {bufs} |"
            )
        (run_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(f"\nSaved benchmark report to {run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
