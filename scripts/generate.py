#!/usr/bin/env python3
"""
localVid Smart One-Command Video Generator (CUDA + Vulkan, 4-6 GB+ VRAM).

Auto-detects GPU vendor, backend (CUDA / Vulkan), real VRAM budget, and
maxMemoryAllocationSize, selects the verified optimal SD_WAN_* + VRAM profile
for the requested frame count (17..81 frames), runs the two-process pipeline
(Process 1: TE + DiT -> float32 latent; Process 2: Standalone Wan 2.1 VAE with
SD_VIDEO_LATENT_DIT_TO_VAE=1 & conv_direct), and packages a playable .mp4.

Supports:
- Interactive wizard mode when run without --prompt (or via localvid.cmd)
- Animated Rich live dashboard with 4 stages, smooth DiT block progress, and
  live GPU Dedicated/Shared VRAM bar (safe plain-text fallback on non-TTY / CI)
- --ui-demo 5.5-second dashboard self-test without running sd-cli

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

# Reuse hardware detection and Rich TUI from sibling scripts
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import probe_hardware  # noqa: E402
import ui_console  # noqa: E402


def measure_live_dedicated_used_gib() -> float | None:
    """Measure current GPU Dedicated VRAM usage (GiB) to guard against WDDM Shared spill."""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
                timeout=3,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                return round(float(out.splitlines()[0].strip()) / 1024.0, 2)
        except Exception:
            return None

    if platform.system() == "Windows":
        ps_cmd = (
            "$d = (Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue)"
            ".CounterSamples | Measure-Object -Property CookedValue -Maximum | "
            "Select-Object -ExpandProperty Maximum; "
            "if ($null -ne $d) { [long][math]::Round([double]$d) }"
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                text=True,
                timeout=5,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            ).strip()
            if out and out.isdigit():
                return round(int(out) / (1024.0 ** 3), 2)
        except Exception:
            return None
    return None


def detect_hardware(simulate_profile: str | None = None, live_used_gib_override: float | None = None) -> dict[str, Any]:
    if simulate_profile:
        presets: dict[str, dict[str, Any]] = {
            "cuda-4gb": {
                "vendor": "NVIDIA",
                "gpu_name": "Simulated NVIDIA 4GB (e.g. GTX 1050 Ti / RTX 3050 Laptop)",
                "backend": "cuda",
                "vram_total_gib": 4.0,
                "vram_budget_gib": 3.5,
                "max_alloc_gib": 4.0,
                "vram_class": "4GB",
            },
            "cuda-6gb": {
                "vendor": "NVIDIA",
                "gpu_name": "Simulated NVIDIA 6GB (e.g. RTX 2060 / RTX 3060 Laptop)",
                "backend": "cuda",
                "vram_total_gib": 6.0,
                "vram_budget_gib": 5.2,
                "max_alloc_gib": 6.0,
                "vram_class": "6GB",
            },
            "cuda-8gb": {
                "vendor": "NVIDIA",
                "gpu_name": "Simulated NVIDIA 8GB (e.g. RTX 3060 Ti / RTX 4060)",
                "backend": "cuda",
                "vram_total_gib": 8.0,
                "vram_budget_gib": 7.0,
                "max_alloc_gib": 8.0,
                "vram_class": "8GB+",
            },
            "vulkan-4gb": {
                "vendor": "AMD",
                "gpu_name": "Simulated AMD/Intel 4GB Vulkan",
                "backend": "vulkan",
                "vram_total_gib": 4.0,
                "vram_budget_gib": 3.4,
                "max_alloc_gib": 2.0,
                "vram_class": "4GB",
            },
            "vulkan-6gb": {
                "vendor": "AMD",
                "gpu_name": "Simulated AMD RX 5600 XT 6GB Vulkan",
                "backend": "vulkan",
                "vram_total_gib": 6.0,
                "vram_budget_gib": 4.34,
                "max_alloc_gib": 2.0,
                "vram_class": "6GB",
            },
            "vulkan-8gb": {
                "vendor": "AMD",
                "gpu_name": "Simulated AMD 8GB Vulkan (e.g. RX 6600 / RX 7600)",
                "backend": "vulkan",
                "vram_total_gib": 8.0,
                "vram_budget_gib": 7.0,
                "max_alloc_gib": 4.0,
                "vram_class": "8GB+",
            },
        }
        hw = dict(presets[simulate_profile])
        if live_used_gib_override is not None:
            hw["live_used_ded_gib"] = round(live_used_gib_override, 2)
            hw["live_free_ded_gib"] = round(max(0.0, hw["vram_total_gib"] - live_used_gib_override), 2)
        return hw

    hw_info: dict[str, Any]
    cuda = probe_hardware.probe_nvidia_cuda()
    if cuda.get("available") and cuda.get("gpus"):
        gpu = cuda["gpus"][0]
        total_gib = round(gpu["vram_total_mib"] / 1024.0, 2)
        free_gib = round(gpu["vram_free_mib"] / 1024.0, 2)
        vclass = "4GB" if total_gib <= 4.5 else ("6GB" if total_gib <= 6.5 else "8GB+")
        hw_info = {
            "vendor": "NVIDIA",
            "gpu_name": gpu["name"],
            "backend": "cuda",
            "vram_total_gib": total_gib,
            "vram_budget_gib": free_gib,
            "max_alloc_gib": total_gib,
            "vram_class": vclass,
            "live_used_ded_gib": round(max(0.0, total_gib - free_gib), 2),
            "live_free_ded_gib": free_gib,
        }
        return hw_info

    vk = probe_hardware.probe_vulkan()
    if vk.get("available") and vk.get("devices"):
        dev = vk["devices"][0]
        name = dev.get("name") or "Unknown Vulkan GPU"
        vendor = "AMD" if dev.get("vendor_id") == "0x1002" else ("Intel" if dev.get("vendor_id") == "0x8086" else "Vulkan")
        total_gib = dev.get("device_local_total_gib") or 6.0
        budget_gib = dev.get("primary_device_local_budget_gib") or total_gib
        max_alloc_gib = dev.get("max_memory_allocation_gib") or 2.0
        vclass = "4GB" if total_gib <= 4.5 else ("6GB" if total_gib <= 6.5 else "8GB+")
        hw_info = {
            "vendor": vendor,
            "gpu_name": name,
            "backend": "vulkan",
            "vram_total_gib": total_gib,
            "vram_budget_gib": budget_gib,
            "max_alloc_gib": max_alloc_gib,
            "vram_class": vclass,
        }
    else:
        hw_info = {
            "vendor": "Unknown",
            "gpu_name": "Fallback (no GPU probe)",
            "backend": "vulkan",
            "vram_total_gib": 6.0,
            "vram_budget_gib": 4.0,
            "max_alloc_gib": 2.0,
            "vram_class": "6GB",
        }

    used_gib = live_used_gib_override if live_used_gib_override is not None else measure_live_dedicated_used_gib()
    if used_gib is not None:
        hw_info["live_used_ded_gib"] = used_gib
        hw_info["live_free_ded_gib"] = round(max(0.0, hw_info["vram_total_gib"] - used_gib), 2)
    return hw_info


def compute_dit_tokens(width: int, height: int, frames: int) -> int:
    """Compute Wan 2.2 DiT sequence length N = (W/16) * (H/16) * ((F-1)/4 + 1)."""
    t_lat = max(1, (frames - 1) // 4 + 1)
    return max(1, (width // 16) * (height // 16) * t_lat)


def select_profile(
    hw: dict[str, Any],
    frames: int,
    fast_preview: bool,
    width: int = 640,
    height: int = 384,
) -> dict[str, Any]:
    """Select verified optimal environment flags and CLI arguments based on DiT token count."""
    backend = hw["backend"]
    vclass = hw["vram_class"]
    tokens = compute_dit_tokens(width, height, frames)
    long_seq = tokens > 16000  # e.g. 640x384 x 65f/81f (16.3k/20.2k), 832x480 x 49f/81f (20.3k/32.8k)

    # Common P1 backend-agnostic memory hygiene flags (always beneficial on <=8GB)
    env_p1: dict[str, str] = {
        "SD_LOADER_FAST_POLL": "1",
        "SD_EVICT_ON_END": "1",
        "SD_EVICT_PAST": "1",
        "SD_DROP_QUEUED": "1",
        "SD_STAGING_POOL_MB": "512" if vclass != "4GB" else "256",
        "SD_WAN_SPLIT_BLOCK": "1",
        "SD_WAN_ATTN_HEAD_CUT": "1",
        "SD_WAN_MLP_CHUNK_MB": "128",
    }

    # Attention backend choice:
    # - AMD/Vulkan RDNA 1: normal attention with HEAD_CHUNK=4 is 37.3x faster than Vulkan FA!
    # - NVIDIA CUDA: Flash Attention is fast; use FA=on for diffusion unless Q-chunked normal attn is needed.
    if backend == "vulkan":
        flash_attn = "off"
        head_chunk = 4 if vclass != "4GB" else 2
    else:
        flash_attn = "off,diffusion=on"
        head_chunk = 8 if vclass == "8GB+" else 4

    env_p1["SD_WAN_ATTN_HEAD_CHUNK"] = str(head_chunk)

    # Token-aware profile selection (handles 480x320, 640x384, native 480p 832x480, etc.):
    # - N <= 5,200   (e.g. 640x384 x 17f [4,800]): fast, no Q_CHUNK, max-vram=1.6 (6GB) / 1.5 (4GB)
    # - 5,200..9,600 (e.g. 640x384 x 33f [8,640], 832x480 x 17f [7,800]): fast, Q_CHUNK=2048, mv=2.0
    # - 9,600..16,000(e.g. 640x384 x 49f [12,480], 832x480 x 33f [14,040]): fast, Q_CHUNK=1024, mv=2.5
    # - N > 16,000   (e.g. 640x384 x 65f/81f, 832x480 x 49f/81f): fit_long (QKV_SPLIT=1 + CUT_HOST=32)
    if long_seq:
        profile_name = f"{backend}_{vclass}_fit_long"
        env_p1["SD_WAN_QKV_SPLIT"] = "1"
        env_p1["SD_CUT_CACHE_HOST_MB"] = "32"
        env_p1["SD_WAN_ATTN_Q_CHUNK"] = "1024"
        max_vram = 2.0 if vclass == "4GB" else (2.5 if vclass == "6GB" else 4.0)
    else:
        profile_name = f"{backend}_{vclass}_fast"
        if tokens > 9600:
            env_p1["SD_WAN_ATTN_Q_CHUNK"] = "1024"
            if vclass == "4GB" and tokens > 12800:
                env_p1["SD_WAN_QKV_SPLIT"] = "1"
                env_p1["SD_CUT_CACHE_HOST_MB"] = "32"
            max_vram = 2.0 if vclass == "4GB" else (2.5 if vclass == "6GB" else 4.0)
        elif tokens > 5200:
            env_p1["SD_WAN_ATTN_Q_CHUNK"] = "2048"
            max_vram = 1.6 if vclass == "4GB" else (2.0 if vclass == "6GB" else 3.5)
        else:
            max_vram = 1.5 if vclass == "4GB" else (1.6 if vclass == "6GB" else 3.0)

    # Clamp max_vram safely below hardware budget
    safe_cap = max(1.5, round(hw["vram_budget_gib"] - 0.35, 2))
    max_vram = min(max_vram, safe_cap)

    # Dynamic live free-VRAM guard: prevents WDDM Shared VRAM spill (2x+ slowdown)
    # when background apps (browser/IDE/desktop) already occupy ~2.8-3.2+ GiB.
    vram_warning: str | None = None
    clamped_from: float | None = None
    live_free = hw.get("live_free_ded_gib")
    live_used = hw.get("live_used_ded_gib")
    if live_free is not None and live_used is not None and live_free < 3.2:
        # Leave ~1.45 GiB WDDM/heap-split headroom; never drop below compute-buffer floor
        floor_gib = 1.5 if tokens <= 9600 else 2.0
        dyn_cap = max(floor_gib, round(live_free - 1.45, 2))
        if dyn_cap < max_vram:
            clamped_from = max_vram
            max_vram = dyn_cap
        vram_warning = (
            f"Uyarı: Arka plan uygulamaları {live_used:.2f} GiB VRAM kullanıyor "
            f"(boş: {live_free:.2f} GiB); tarayıcı/IDE kapatmak hızı ~2× artırır."
        )
        if clamped_from is not None:
            vram_warning += f" (--max-vram otomatik {clamped_from:.2f} -> {max_vram:.2f} GiB çekildi)"

    env_p2: dict[str, str] = {
        "SD_LOADER_FAST_POLL": "1",
        "SD_VIDEO_LATENT_DIT_TO_VAE": "1",
    }
    if backend == "vulkan":
        env_p2["SD_CONV_DIRECT_TOL"] = "0.005"

    return {
        "profile_name": profile_name,
        "dit_tokens": tokens,
        "backend": backend,
        "flash_attn": flash_attn,
        "max_vram_gib": max_vram,
        "max_vram_clamped_from": clamped_from,
        "vram_warning": vram_warning,
        "two_process_vae": not fast_preview,
        "vae_conv_direct_mode": "on",
        "vae_flash_attn_mode": "off" if backend == "vulkan" else "auto",
        "env_p1": env_p1,
        "env_p2": env_p2,
    }


def resolve_project_path(p: pathlib.Path) -> pathlib.Path:
    """Resolve relative model/build paths from CWD first, then REPO_ROOT."""
    if p.is_absolute() or p.exists():
        return p
    alt = REPO_ROOT / p
    if alt.exists():
        return alt
    return p


def resolve_sd_cli(backend: str, explicit: pathlib.Path | None) -> pathlib.Path:
    if explicit is not None:
        return resolve_project_path(explicit)
    candidates = [
        pathlib.Path(f"build/sdcpp-{backend}/bin/sd-cli.exe"),
        pathlib.Path(f"build/sdcpp-{backend}/bin/sd-cli"),
        pathlib.Path("build/sdcpp-vulkan/bin/sd-cli.exe"),
    ]
    for c in candidates:
        resolved = resolve_project_path(c)
        if resolved.exists():
            return resolved
    return candidates[0]


def resolve_ffmpeg() -> tuple[str | None, str]:
    """Return (ffmpeg_exe, label): prefer system ffmpeg, then imageio-ffmpeg."""
    sys_ff = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if sys_ff:
        return sys_ff, "ffmpeg_h264"
    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and pathlib.Path(bundled).exists():
            return bundled, "imageio_ffmpeg_h264"
    except Exception:
        pass
    return None, ""


def convert_avi_to_mp4(avi_path: pathlib.Path, mp4_path: pathlib.Path) -> str:
    ffmpeg, ff_label = resolve_ffmpeg()
    if ffmpeg:
        cmd = [
            ffmpeg, "-y", "-i", str(avi_path),
            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(mp4_path),
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode == 0 and mp4_path.exists():
            return ff_label

    try:
        import cv2
        cap = cv2.VideoCapture(str(avi_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            writer.write(fr)
        cap.release()
        writer.release()
        if mp4_path.exists():
            return "opencv_mp4v"
    except Exception:
        pass
    return "avi_only"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="localVid: Run Wan 2.2 14B on 4-6 GB VRAM GPUs (CUDA + Vulkan)"
    )
    parser.add_argument("-p", "--prompt", default=None, help="Text prompt (omit to open interactive wizard)")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("output.mp4"), help="Output .mp4 path")
    parser.add_argument("-W", "--width", type=int, default=640)
    parser.add_argument("-H", "--height", type=int, default=384)
    parser.add_argument("-f", "--frames", type=int, default=33, help="Video frames (17, 33, 49, 65, 81)")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--sampler", default="euler_a")
    parser.add_argument("--scheduler", default="beta")
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion-model", type=pathlib.Path, default=pathlib.Path("models/diffusion_models/wan2.2-t2v-rapid-aio-v10-Q4_K.gguf"))
    parser.add_argument("--t5xxl", type=pathlib.Path, default=pathlib.Path("models/text_encoders/umt5-xxl-encoder-Q4_K_M.gguf"))
    parser.add_argument("--vae", type=pathlib.Path, default=pathlib.Path("models/vae/wan_2.1_vae.safetensors"))
    parser.add_argument("--taesd", type=pathlib.Path, default=pathlib.Path("models/vae/taew2_1.safetensors"))
    parser.add_argument("--sd-cli", type=pathlib.Path, default=None)
    parser.add_argument("--fast-preview", action="store_true", help="Decode directly with TAEHV (~2-6s) instead of separate-process Wan 2.1 VAE")
    parser.add_argument("--keep-latent", action="store_true", help="Keep intermediate float32 .lvlt latent file")
    parser.add_argument("--simulate-profile", choices=["cuda-4gb", "cuda-6gb", "cuda-8gb", "vulkan-4gb", "vulkan-6gb", "vulkan-8gb"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print detected hardware, profile, and commands without running")
    parser.add_argument("--raw-logs", action="store_true", help="Disable Rich dashboard and stream raw sd-cli logs directly")
    parser.add_argument("--verbose", action="store_true", help="Stream raw sd-cli logs to terminal (disables Rich Live panel)")
    parser.add_argument("--ui-demo", action="store_true", help="Run a 5.5-second animated dashboard demo without invoking sd-cli")
    return parser


def main() -> int:
    ui_console.configure_console_utf8()
    parser = build_arg_parser()
    args = parser.parse_args()

    hw = detect_hardware(args.simulate_profile)

    # Quick 5.5s UI/terminal self-test mode
    if args.ui_demo:
        prof = select_profile(hw, args.frames, args.fast_preview, args.width, args.height)
        return ui_console.run_ui_demo(hw, prof, raw_logs=args.raw_logs or args.verbose)

    # Interactive wizard when launched without --prompt (except --dry-run)
    interactive_mode = False
    if args.prompt is None:
        if args.dry_run:
            args.prompt = ui_console.DEFAULT_SAMPLE_PROMPT
        else:
            interactive_mode = True
            try:
                wiz = ui_console.run_interactive_wizard(hw, select_profile)
            except KeyboardInterrupt:
                print("\n[localVid] İptal edildi.")
                return 130
            args.prompt = wiz["prompt"]
            args.frames = wiz["frames"]
            args.width = wiz["width"]
            args.height = wiz["height"]
            args.fast_preview = wiz["fast_preview"]
            args.seed = wiz["seed"]
            args.output = wiz["output"]

    prof = select_profile(hw, args.frames, args.fast_preview, args.width, args.height)
    sd_cli = resolve_sd_cli(prof["backend"], args.sd_cli)
    diffusion_model = resolve_project_path(args.diffusion_model)
    t5xxl = resolve_project_path(args.t5xxl)
    vae_path = resolve_project_path(args.vae)
    taesd_path = resolve_project_path(args.taesd)

    out_mp4 = args.output.resolve()
    if not args.dry_run:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
    work_dir = pathlib.Path(tempfile.mkdtemp(prefix="localvid_")) if not args.dry_run else out_mp4.parent
    lat_path = out_mp4.with_suffix(".lvlt") if args.keep_latent else (work_dir / "latent.lvlt")
    raw_avi = out_mp4.with_suffix(".avi")
    dummy_avi = work_dir / "dit_dummy.avi"
    log_path = out_mp4.with_suffix(".log")

    # Build Process 1 command (TE + DiT)
    use_tae_in_p1 = args.fast_preview or taesd_path.exists() or args.dry_run
    env_p1 = dict(prof["env_p1"])
    cmd_p1 = [
        str(sd_cli),
        "-M", "vid_gen",
        "--diffusion-model", str(diffusion_model),
        "--t5xxl", str(t5xxl),
        "--taesd" if use_tae_in_p1 else "--vae",
        str(taesd_path if use_tae_in_p1 else vae_path),
        "-p", args.prompt,
        "-W", str(args.width),
        "-H", str(args.height),
        "--video-frames", str(args.frames),
        "--fps", str(args.fps),
        "--steps", str(args.steps),
        "--cfg-scale", str(args.cfg_scale),
        "--sampling-method", args.sampler,
        "--scheduler", args.scheduler,
        "--flow-shift", str(args.flow_shift),
        "--seed", str(args.seed),
        "--offload-to-cpu",
        "--mmap",
        "--backend", "te=cpu",
        "--flash-attn", prof["flash_attn"],
        "--max-vram", str(prof["max_vram_gib"]),
        "-o", str(raw_avi if args.fast_preview else dummy_avi),
    ]
    if not args.fast_preview:
        env_p1["SD_VIDEO_LATENT_SAVE"] = str(lat_path)
        env_p1["SD_VIDEO_LATENT_SAVE_ONLY"] = "1"

    # Build Process 2 command (Standalone Wan 2.1 VAE)
    dgl_path = work_dir / "decode_wan_vae.dgl"
    env_p2 = dict(prof["env_p2"])
    env_p2["SD_VIDEO_LATENT_LOAD"] = str(lat_path)
    if not use_tae_in_p1:
        # Process 1 already ran Wan VAE's dit_to_vae_latents before saving
        env_p2.pop("SD_VIDEO_LATENT_DIT_TO_VAE", None)
    cmd_p2 = [str(sd_cli), "-M", "dgl", "--script", str(dgl_path)]

    plan = {
        "hardware": hw,
        "selected_profile": prof["profile_name"],
        "dit_tokens": prof["dit_tokens"],
        "vram_warning": prof["vram_warning"],
        "two_process_vae": prof["two_process_vae"],
        "vae_conv_direct_mode": prof["vae_conv_direct_mode"],
        "mp4_encoder": resolve_ffmpeg()[1] or "opencv_mp4v",
        "process_1_dit": {"env": env_p1, "cmd": cmd_p1},
        "process_2_vae": None if args.fast_preview else {"env": env_p2, "cmd": cmd_p2},
        "output_mp4": str(out_mp4),
    }

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    # Verify required binaries and models exist
    for label, p in [("sd-cli", sd_cli), ("diffusion-model", diffusion_model), ("t5xxl", t5xxl)]:
        if not p.exists():
            print(f"HATA: Eksik {label}: {p}", file=sys.stderr)
            return 2
    if args.fast_preview and not taesd_path.exists():
        print(f"HATA: --fast-preview için TAEHV gerekli: {taesd_path}", file=sys.stderr)
        return 2
    if not args.fast_preview and not vae_path.exists():
        print(f"HATA: Eksik Wan 2.1 VAE: {vae_path}", file=sys.stderr)
        return 2

    # Enable smooth per-block progress events for the live dashboard
    use_rich = ui_console.should_use_rich(raw_logs=args.raw_logs, verbose=args.verbose)
    echo_raw = args.raw_logs or args.verbose
    if not echo_raw:
        env_p1.setdefault("SD_SEG_TIMING", "2")

    vram_sampler = ui_console.LiveVramSampler(total_vram_gib=hw["vram_total_gib"], interval_sec=1.0)
    dash_cfg = ui_console.DashboardConfig(
        gpu_name=hw["gpu_name"],
        backend=hw["backend"],
        vram_class=hw["vram_class"],
        vram_total_gib=hw["vram_total_gib"],
        vram_budget_gib=hw["vram_budget_gib"],
        profile_name=prof["profile_name"],
        width=args.width,
        height=args.height,
        frames=args.frames,
        fps=args.fps,
        steps=args.steps,
        vae_label="TAEHV" if args.fast_preview else "Wan 2.1 VAE",
        log_path=log_path,
    )
    dashboard: ui_console.BaseDashboard = (
        ui_console.RichDashboard(dash_cfg, vram_sampler)
        if use_rich
        else ui_console.PlainDashboard(dash_cfg, vram_sampler)
    )

    if prof.get("vram_warning") and not interactive_mode:
        print(f"[localVid] {prof['vram_warning']}", file=sys.stderr, flush=True)

    vram_sampler.start()
    dashboard.start()
    final_output: pathlib.Path | None = None
    encoder_label = ""
    err_msg = ""
    rc = 0

    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as lf:
            lf.write(f"=== localVid v0.1.0 Run Log ===\n")
            lf.write(json.dumps(plan, indent=2, ensure_ascii=False) + "\n\n")
            lf.write("=== PROCESS 1: TE + DiT ===\n")
            lf.flush()

            p1_full_env = os.environ.copy()
            p1_full_env.update(env_p1)
            rc = ui_console.run_process_monitored(
                cmd=cmd_p1,
                env=p1_full_env,
                log_file=lf,
                dashboard=dashboard,
                phase="process1",
                total_steps=args.steps,
                fast_preview=args.fast_preview,
                echo_raw=echo_raw,
            )
            if rc != 0:
                err_msg = f"Süreç 1 (DiT) çıkış kodu {rc} ile başarısız oldu"
                dashboard.fail_stage("dit", detail=f"Çıkış kodu {rc}")
                return rc

            if not args.fast_preview:
                dgl_path.write_text(
                    f"""# version: 0.0.0
var {{
    vae_path = "{vae_path.resolve().as_posix()}"
    output_path = "{raw_avi.as_posix()}"
    fps = {args.fps}
}}
fn main() {{
    vae = load_vae(vae_path, family = "wan2", params_backend = "cpu", conv_direct_mode = "{prof['vae_conv_direct_mode']}", flash_attn_mode = "{prof['vae_flash_attn_mode']}")
    dummy_latent = tensor::zeros([8, 8, 1, 16, 1])
    output_frames = vae_decode(vae, dummy_latent)
    video::save(output_path, {{frames = output_frames, fps_num = fps, fps_den = 1}})
    return {{video = {{frames = output_frames, fps_num = fps, fps_den = 1}}}}
}}
""",
                    encoding="utf-8",
                )
                lf.write("\n=== PROCESS 2: Standalone Wan 2.1 VAE ===\n")
                lf.flush()
                p2_full_env = os.environ.copy()
                p2_full_env.update(env_p2)
                rc = ui_console.run_process_monitored(
                    cmd=cmd_p2,
                    env=p2_full_env,
                    log_file=lf,
                    dashboard=dashboard,
                    phase="process2",
                    total_steps=args.steps,
                    fast_preview=False,
                    echo_raw=echo_raw,
                )
                if rc != 0:
                    err_msg = f"Süreç 2 (Wan 2.1 VAE) çıkış kodu {rc} ile başarısız oldu"
                    dashboard.fail_stage("vae", detail=f"Çıkış kodu {rc}")
                    return rc

        # Stage 4: MP4 Packaging
        dashboard.start_stage("mp4", detail="AVI -> MP4 dönüştürülüyor...", determinate=True)
        dashboard.set_status("MP4 paketleme başlıyor...")
        if raw_avi.exists():
            dashboard.update_stage("mp4", pct=40.0, detail="Video akışı kodlanıyor...")
            encoder_label = convert_avi_to_mp4(raw_avi, out_mp4)
            if encoder_label != "avi_only":
                raw_avi.unlink(missing_ok=True)
                final_output = out_mp4
                dashboard.finish_stage("mp4", detail=f"{out_mp4.name} ({encoder_label})")
                dashboard.set_status(f"MP4 hazır: {out_mp4.name}")
            else:
                final_output = raw_avi
                dashboard.finish_stage("mp4", detail=f"{raw_avi.name} (AVI)")
                dashboard.set_status(f"AVI kaydedildi (MP4 için ffmpeg/opencv-python kurun): {raw_avi.name}")
        else:
            err_msg = f"Beklenen ara AVI dosyası oluşmadı: {raw_avi}"
            dashboard.fail_stage("mp4", detail="AVI bulunamadı")
            rc = 3
            return rc

        return 0
    except KeyboardInterrupt:
        err_msg = "Kullanıcı tarafından iptal edildi (Ctrl+C)"
        rc = 130
        return rc
    finally:
        dashboard.stop()
        vram_sampler.stop()
        shutil.rmtree(work_dir, ignore_errors=True)
        dashboard.render_summary(
            success=(rc == 0 and final_output is not None),
            output_path=final_output,
            encoder_label=encoder_label,
            error_msg=err_msg,
        )
        if interactive_mode and rc == 0 and final_output is not None:
            ui_console.ask_open_video(final_output, use_rich=use_rich)


if __name__ == "__main__":
    sys.exit(main())
