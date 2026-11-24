#!/usr/bin/env python3
"""
PicoVid v0.2.0 Smart Video & Studio Generator (CUDA + Vulkan, 4-6 GB+ VRAM).

Modes:
  - t2v    : Text -> Video (Wan 2.2 14B Rapid-AIO v10, 4 steps)
  - i2v    : Image -> Video (Wan 2.2 14B LightX2V 4-Step MoE High+Low, GPU-tiled vae_encode + CPU fallback)
  - t2i    : Text -> Image (Z-Image-Turbo 6B + Qwen3-4B + ae.safetensors, Apache 2.0)
  - studio : Text -> Image (Z-Image-Turbo) -> Interactive Preview/Approve -> Video (Wan 2.2 I2V)

Features:
  - Centralized configs/models_registry.json model catalog & interactive missing-model auto-download
  - First-run missing sd-cli.exe auto-build offer (apply_patches.ps1 + build.ps1 -Backend auto)
  - Auto hardware/VRAM detection + live free-VRAM guard against WDDM Shared spill
  - Token-aware SD_WAN_* profile selection (480x320, 640x384, native 480p 832x480; 17f..81f)
  - Two-process Wan 2.1 VAE with deterministic conv_direct_mode="on" and automatic
    SD_VIDEO_LATENT_DIT_TO_VAE handling (1 for T2V --taesd, 0 for I2V --vae)
  - High-quality H.264 MP4 (libx264 -crf 16) via system ffmpeg or imageio-ffmpeg
  - Animated alt-screen Rich Live dashboard with live Dedicated/Shared VRAM bars

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

# Reuse hardware detection and Rich TUI from sibling scripts
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
REGISTRY_PATH = REPO_ROOT / "configs" / "models_registry.json"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import download_models  # noqa: E402
import postprocess  # noqa: E402
import probe_hardware  # noqa: E402
import ui_console  # noqa: E402


def prompt_yes_no(question: str, default: bool = True) -> bool:
    if ui_console.HAS_RICH and ui_console.is_interactive_tty():
        from rich.prompt import Prompt
        console = ui_console.make_console()
        ans = Prompt.ask(question, default="E" if default else "h", console=console).strip().lower()
        return ans in ("", "e", "evet", "y", "yes") if default else ans in ("e", "evet", "y", "yes")
    hint = "[E/h]" if default else "[e/H]"
    ans = input(f"{question} {hint}: ").strip().lower()
    return ans in ("", "e", "evet", "y", "yes") if default else ans in ("e", "evet", "y", "yes")


def ensure_sd_cli_available(sd_cli: pathlib.Path, backend: str, interactive: bool) -> pathlib.Path | None:
    """If compiled sd-cli.exe is missing, offer automatic patch + build for detected GPU backend."""
    if sd_cli.exists():
        return sd_cli
    print(f"\n[PicoVid] Derlenmiş motor bulunamadı: {sd_cli}", file=sys.stderr)
    print(f"          Algılanan backend: {backend.upper()}", file=sys.stderr)
    if not (interactive or ui_console.is_interactive_tty()):
        print(
            f"Derlemek için:\n"
            f"  powershell -ExecutionPolicy Bypass -File scripts/apply_patches.ps1\n"
            f"  powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Backend {backend}",
            file=sys.stderr,
        )
        return None

    if not prompt_yes_no(
        f"Yamaları uygulayıp {backend.upper()} motorunu şimdi otomatik derleyelim mi? (apply_patches.ps1 + build.ps1)",
        default=True,
    ):
        return None

    src_dir = REPO_ROOT / "third_party" / "stable-diffusion.cpp"
    if not src_dir.exists():
        print("[PicoVid] stable-diffusion.cpp (3b12781) klonlanıyor...")
        r_clone = subprocess.run(
            [
                "git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules",
                "https://github.com/leejet/stable-diffusion.cpp.git", str(src_dir),
            ],
            cwd=str(REPO_ROOT),
        )
        if r_clone.returncode != 0:
            print("HATA: git clone başarısız oldu.", file=sys.stderr)
            return None
        subprocess.run(["git", "-C", str(src_dir), "checkout", "3b12781"], cwd=str(REPO_ROOT))

    # Apply patches only if not already applied
    chk = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(REPO_ROOT / "scripts" / "apply_patches.ps1"), "-CheckOnly"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if chk.returncode == 0:
        r_patch = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(REPO_ROOT / "scripts" / "apply_patches.ps1")],
            cwd=str(REPO_ROOT),
        )
        if r_patch.returncode != 0:
            print("HATA: apply_patches.ps1 başarısız oldu.", file=sys.stderr)
            return None

    r_build = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(REPO_ROOT / "scripts" / "build.ps1"), "-Backend", backend],
        cwd=str(REPO_ROOT),
    )
    if r_build.returncode != 0:
        print(f"HATA: build.ps1 -Backend {backend} başarısız oldu.", file=sys.stderr)
        return None

    re_resolved = resolve_sd_cli(backend, None)
    return re_resolved if re_resolved.exists() else None


def get_missing_mode_files(
    mode: str,
    reg: dict[str, Any],
    i2v_variant: str,
    fast_preview: bool,
    engine: str = "14b",
) -> tuple[list[str], int]:
    """Return (missing_file_ids, total_missing_bytes) for the requested PicoVid mode and engine."""
    files_map = reg.get("files", {})
    req_ids: list[str] = []
    if mode in ("t2i", "studio"):
        req_ids += ["z_image_turbo_q4_k", "qwen3_4b_q4_k_m", "flux_ae"]
    if engine == "5b" and mode in ("t2v", "i2v", "studio"):
        req_ids += ["wan22_ti2v_5b_turbo_q4", "umt5_xxl_q4_k_m", "wan22_vae"]
        if fast_preview:
            req_ids.append("taew2_2")
    elif mode == "t2v":
        req_ids += ["wan22_t2v_rapid_v10", "umt5_xxl_q4_k_m"]
        req_ids.append("taew2_1" if fast_preview else "wan21_vae")
    elif mode in ("i2v", "studio"):
        if i2v_variant == "q3":
            req_ids += ["wan22_i2v_high_q3", "wan22_i2v_low_q3"]
        elif i2v_variant == "q4":
            req_ids += ["wan22_i2v_high_q4", "wan22_i2v_low_q4"]
        else:
            # auto: if compact Q3 pair is already present, don't ask for Q4
            q3_ok = all(download_models.check_file(files_map[k])[0] for k in ("wan22_i2v_high_q3", "wan22_i2v_low_q3"))
            req_ids += ["wan22_i2v_high_q3", "wan22_i2v_low_q3"] if q3_ok else ["wan22_i2v_high_q4", "wan22_i2v_low_q4"]
        req_ids += ["umt5_xxl_q4_k_m", "wan21_vae"]

    missing_ids: list[str] = []
    missing_bytes = 0
    for fid in dict.fromkeys(req_ids):
        if fid not in files_map:
            continue
        ok, _, rem = download_models.check_file(files_map[fid])
        if not ok:
            missing_ids.append(fid)
            missing_bytes += rem
    return missing_ids, missing_bytes


def ensure_mode_models_available(
    mode: str,
    reg: dict[str, Any],
    i2v_variant: str,
    fast_preview: bool,
    interactive: bool,
    engine: str = "14b",
) -> bool:
    """If models for the selected mode are missing, show exact size and offer inline download."""
    missing_ids, missing_bytes = get_missing_mode_files(mode, reg, i2v_variant, fast_preview, engine=engine)
    if not missing_ids:
        return True

    files_map = reg["files"]
    print(f"\n[PicoVid] '{mode.upper()} ({engine.upper()})' modu için {len(missing_ids)} eksik model bulundu (toplam {download_models.format_gb(missing_bytes)}):")
    for fid in missing_ids:
        item = files_map[fid]
        print(f"  - {item['path']} ({download_models.format_gb(int(item['expected_bytes']))})")

    if not (interactive or ui_console.is_interactive_tty()):
        prof_hint = "fast-5b" if engine == "5b" else {"t2v": "t2v", "t2i": "t2i", "i2v": "i2v", "studio": "studio"}.get(mode, "t2v")
        print(f"İndirmek için: py scripts/download_models.py --profile {prof_hint}", file=sys.stderr)
        return False

    if not prompt_yes_no(
        f"Eksik {len(missing_ids)} model ({download_models.format_gb(missing_bytes)}) şimdi indirilsin mi?",
        default=True,
    ):
        print("[PicoVid] İndirme iptal edildi.")
        return False

    return download_models.download_file_ids(missing_ids, reg)


def load_models_registry(path: pathlib.Path = REGISTRY_PATH) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"files": {}, "pipelines": {}, "download_profiles": {}}


def reg_file_path(reg: dict[str, Any], file_id: str, fallback: str) -> pathlib.Path:
    entry = reg.get("files", {}).get(file_id)
    rel = entry["path"] if entry else fallback
    return resolve_project_path(pathlib.Path(rel))


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

    cuda = probe_hardware.probe_nvidia_cuda()
    if cuda.get("available") and cuda.get("gpus"):
        gpu = cuda["gpus"][0]
        total_gib = round(gpu["vram_total_mib"] / 1024.0, 2)
        free_gib = round(gpu["vram_free_mib"] / 1024.0, 2)
        vclass = "4GB" if total_gib <= 4.5 else ("6GB" if total_gib <= 6.5 else "8GB+")
        return {
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

    vk = probe_hardware.probe_vulkan()
    if vk.get("available") and vk.get("devices"):
        dev = vk["devices"][0]
        name = dev.get("name") or "Unknown Vulkan GPU"
        vendor = "AMD" if dev.get("vendor_id") == "0x1002" else ("Intel" if dev.get("vendor_id") == "0x8086" else "Vulkan")
        total_gib = dev.get("device_local_total_gib") or 6.0
        budget_gib = dev.get("primary_device_local_budget_gib") or total_gib
        max_alloc_gib = dev.get("max_memory_allocation_gib") or 2.0
        vclass = "4GB" if total_gib <= 4.5 else ("6GB" if total_gib <= 6.5 else "8GB+")
        hw_info: dict[str, Any] = {
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


def compute_dit_tokens(width: int, height: int, frames: int, engine: str = "14b") -> int:
    """Compute Wan 2.2 DiT sequence length.

    - 14B (Wan 2.1 VAE 8x8 + patch 2x2): stride 16 -> (W/16) * (H/16) * ((F-1)/4 + 1)
    - 5B  (Wan 2.2 VAE 16x16 + patch 2x2): stride 32 -> (W/32) * (H/32) * ((F-1)/4 + 1)
    """
    stride = 32 if engine == "5b" else 16
    t_lat = max(1, (frames - 1) // 4 + 1)
    return max(1, (width // stride) * (height // stride) * t_lat)


def start_file_prewarm(paths: list[pathlib.Path], stop_event: threading.Event) -> threading.Thread | None:
    """Sequentially warm OS file cache for large GGUF models while TE / VAE encode run.

    Only runs on hosts with >=20 GiB system RAM so 12-16 GB laptops are never pressured.
    Eliminates +50-80s of synchronous cold-mmap page faults on DiT Step 1 & Step 3.
    """
    try:
        sys_info = probe_hardware.probe_system(REPO_ROOT / "models")
        if (sys_info.get("ram_total_gib") or 0.0) < 20.0:
            return None
    except Exception:
        return None

    valid_paths = [p for p in paths if p is not None and p.exists()]
    if not valid_paths:
        return None

    def _worker() -> None:
        chunk = 8 * 1024 * 1024  # 8 MiB sequential reads
        for p in valid_paths:
            if stop_event.is_set():
                return
            try:
                with p.open("rb") as f:
                    while not stop_event.is_set():
                        buf = f.read(chunk)
                        if not buf:
                            break
            except OSError:
                continue

    t = threading.Thread(target=_worker, name="picovid_nvme_prewarm", daemon=True)
    t.start()
    return t


def select_profile(
    hw: dict[str, Any],
    frames: int,
    fast_preview: bool,
    width: int = 640,
    height: int = 384,
    engine: str = "14b",
) -> dict[str, Any]:
    """Select verified optimal environment flags and CLI arguments based on DiT token count & engine."""
    backend = hw["backend"]
    vclass = hw["vram_class"]
    tokens = compute_dit_tokens(width, height, frames, engine=engine)
    long_seq = tokens > 13000  # >=832x480x33f (14,040) & >=640x384x65f (16,320) need QKV_SPLIT + CUT_HOST
    is_5b = engine == "5b"

    env_p1: dict[str, str] = {
        "SD_LOADER_FAST_POLL": "1",
        "SD_EVICT_ON_END": "1",
        "SD_EVICT_PAST": "1",
        "SD_DROP_QUEUED": "1",
        "SD_STAGING_POOL_MB": "512" if vclass != "4GB" else "256",
        "SD_WAN_SPLIT_BLOCK": "1",
        "SD_WAN_ATTN_HEAD_CUT": "1",
        "SD_WAN_MLP_CHUNK_MB": "256" if (is_5b and vclass != "4GB" and tokens <= 5200) else "128",
    }

    if backend == "vulkan":
        flash_attn = "off"
        # HEAD_CHUNK=12 (2 chunks/block) on 5B <=5200 tokens (e.g. 640x384x81f=5040) halves attention submits
        # while staying ~1.2 GiB (< 2 GiB); >5200 tokens uses HEAD_CHUNK=4 to stay under dynamic VRAM floors.
        if vclass == "4GB":
            head_chunk = 4 if is_5b else 2
        elif is_5b:
            head_chunk = 12 if tokens <= 5200 else 4
        elif not long_seq and tokens <= 9600:
            head_chunk = 8
        else:
            head_chunk = 4
    else:
        flash_attn = "off,diffusion=on"
        head_chunk = 8 if (vclass == "8GB+" or is_5b) else 4

    env_p1["SD_WAN_ATTN_HEAD_CHUNK"] = str(head_chunk)
    eng_tag = "_5b" if is_5b else ""

    if long_seq:
        profile_name = f"{backend}_{vclass}{eng_tag}_fit_long"
        env_p1["SD_WAN_QKV_SPLIT"] = "1"
        env_p1["SD_CUT_CACHE_HOST_MB"] = "32"
        env_p1["SD_WAN_ATTN_Q_CHUNK"] = "1024"
        env_p1["SD_WAN_CROSS_Q_CHUNK"] = "4096"
        # QKV_SPLIT + CUT_HOST shrinks peak compute buffer to ~1.1-1.4 GiB (14k-16k tokens)
        # or ~1.89 GiB (20k tokens / 81f). Keeping max-vram <= 1.85 GiB prevents WDDM Shared spill!
        if tokens <= 17000:
            max_vram = 1.65 if vclass == "4GB" else (1.85 if vclass == "6GB" else 3.5)
        else:
            max_vram = 2.0 if vclass == "4GB" else (2.15 if vclass == "6GB" else 4.0)
    else:
        profile_name = f"{backend}_{vclass}{eng_tag}_fast"
        if tokens > 9600:
            env_p1["SD_WAN_ATTN_Q_CHUNK"] = "1024"
            env_p1["SD_WAN_CROSS_Q_CHUNK"] = "4096"
            if vclass == "4GB" and tokens > 12800:
                env_p1["SD_WAN_QKV_SPLIT"] = "1"
                env_p1["SD_CUT_CACHE_HOST_MB"] = "32"
            max_vram = 1.85 if vclass == "4GB" else (1.95 if is_5b else (2.15 if vclass == "6GB" else 4.0))
        elif tokens > 5200:
            env_p1["SD_WAN_ATTN_Q_CHUNK"] = "2048"
            env_p1["SD_WAN_CROSS_Q_CHUNK"] = "4096"
            max_vram = 1.6 if vclass == "4GB" else (1.95 if vclass == "6GB" else 3.5)
        else:
            max_vram = 1.5 if vclass == "4GB" else (1.95 if is_5b else (1.6 if vclass == "6GB" else 3.0))

    safe_cap = max(1.5, round(hw["vram_budget_gib"] - 0.35, 2))
    max_vram = min(max_vram, safe_cap)

    # Dynamic live free-VRAM guard against WDDM Shared spill (prevents crossing heap[0] 4.98 GiB cliff)
    vram_warning: str | None = None
    clamped_from: float | None = None
    live_free = hw.get("live_free_ded_gib")
    live_used = hw.get("live_used_ded_gib")
    if live_free is not None and live_used is not None and live_free < 4.0:
        floor_gib = 1.55 if tokens <= 5200 else (1.65 if tokens <= 9600 else (1.75 if tokens <= 17000 else 2.05))
        dyn_cap = max(floor_gib, round(live_free - 1.55, 2))
        if dyn_cap < max_vram:
            clamped_from = max_vram
            max_vram = dyn_cap
        if live_free < 3.2:
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


def resolve_i2v_models(
    reg: dict[str, Any],
    variant: str,
    explicit_high: pathlib.Path | None,
    explicit_low: pathlib.Path | None,
) -> tuple[pathlib.Path, pathlib.Path, str]:
    """Resolve HighNoise and LowNoise Wan 2.2 I2V GGUF paths (supports Q4_K_M & Q3_K_M compact)."""
    if explicit_high and explicit_low:
        return resolve_project_path(explicit_high), resolve_project_path(explicit_low), "custom"

    q4_h = reg_file_path(reg, "wan22_i2v_high_q4", "models/diffusion_models/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412-Q4_K_M.gguf")
    q4_l = reg_file_path(reg, "wan22_i2v_low_q4", "models/diffusion_models/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412-Q4_K_M.gguf")
    q3_h = reg_file_path(reg, "wan22_i2v_high_q3", "models/diffusion_models/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412-Q3_K_M.gguf")
    q3_l = reg_file_path(reg, "wan22_i2v_low_q3", "models/diffusion_models/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412-Q3_K_M.gguf")

    if variant == "q3":
        return q3_h, q3_l, "q3_compact"
    if variant == "q4":
        return q4_h, q4_l, "q4_default"
    # auto: prefer Q4_K_M if present, else Q3_K_M if present, else default Q4_K_M
    if q4_h.exists() and q4_l.exists():
        return q4_h, q4_l, "q4_default"
    if q3_h.exists() and q3_l.exists():
        return q3_h, q3_l, "q3_compact"
    return q4_h, q4_l, "q4_default"


def resolve_ffmpeg() -> tuple[str | None, str]:
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


def build_t2i_plan(
    hw: dict[str, Any],
    reg: dict[str, Any],
    sd_cli: pathlib.Path,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    out_png: pathlib.Path,
) -> dict[str, Any]:
    backend = hw["backend"]
    vclass = hw["vram_class"]
    z_dit = reg_file_path(reg, "z_image_turbo_q4_k", "models/diffusion_models/z_image_turbo-Q4_K.gguf")
    z_llm = reg_file_path(reg, "qwen3_4b_q4_k_m", "models/text_encoders/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    z_vae = reg_file_path(reg, "flux_ae", "models/vae/ae.safetensors")

    max_vram = 1.6 if vclass == "4GB" else (2.0 if vclass == "6GB" else 4.0)
    live_free = hw.get("live_free_ded_gib")
    if live_free is not None and live_free < 3.2:
        max_vram = min(max_vram, max(1.5, round(live_free - 1.45, 2)))

    flash_attn = "off" if backend == "vulkan" else "off,diffusion=on"
    env_t2i = {
        "SD_LOADER_FAST_POLL": "1",
        "SD_EVICT_ON_END": "1",
        "SD_EVICT_PAST": "1",
        "SD_DROP_QUEUED": "1",
        "SD_STAGING_POOL_MB": "512" if vclass != "4GB" else "256",
    }
    cmd_t2i = [
        str(sd_cli),
        "-M", "img_gen",
        "--diffusion-model", str(z_dit),
        "--llm", str(z_llm),
        "--vae", str(z_vae),
        "-p", prompt,
        "-W", str(width),
        "-H", str(height),
        "--steps", str(steps),
        "--cfg-scale", "1.0",
        "--sampling-method", "euler",
        "--scheduler", "discrete",
        "--seed", str(seed),
        "--offload-to-cpu",
        "--mmap",
        "--backend", "te=cpu",
        "--flash-attn", flash_attn,
        "--conv-direct", "on" if backend == "vulkan" else "auto",
        "--max-vram", str(max_vram),
        "-o", str(out_png),
    ]
    return {
        "models": {"diffusion_model": z_dit, "llm": z_llm, "vae": z_vae},
        "env": env_t2i,
        "cmd": cmd_t2i,
        "output_png": str(out_png),
    }


def run_t2i_first_frame(
    t2i_plan: dict[str, Any],
    hw: dict[str, Any],
    log_path: pathlib.Path,
    echo_raw: bool = False,
) -> tuple[int, float, float | None, float | None]:
    """Execute Z-Image-Turbo T2I with live VRAM sampling and clean console feedback."""
    env = os.environ.copy()
    env.update(t2i_plan["env"])
    sampler = ui_console.LiveVramSampler(total_vram_gib=hw["vram_total_gib"], interval_sec=0.8)
    sampler.start()
    t0 = time.monotonic()

    use_rich = ui_console.should_use_rich(raw_logs=echo_raw, verbose=echo_raw)
    console = ui_console.make_console() if use_rich else None

    if use_rich and console is not None:
        status_cm = console.status("[bold cyan]Z-Image-Turbo ilk kare üretiliyor (Qwen3-4B CPU -> 6B DiT)...[/bold cyan]", spinner="dots")
    else:
        print("[PicoVid] Z-Image-Turbo ilk kare üretiliyor...", flush=True)
        status_cm = None

    rc = 1
    try:
        if status_cm is not None:
            status_cm.__enter__()
        with log_path.open("w", encoding="utf-8", errors="replace") as lf:
            proc = subprocess.Popen(
                t2i_plan["cmd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                bufsize=0,
            )
            assert proc.stdout is not None
            for line in ui_console.iter_stream_lines(proc.stdout):
                lf.write(line + "\n")
                lf.flush()
                if echo_raw:
                    print(line, flush=True)
                if status_cm is not None:
                    clean = ui_console.RE_ANSI.sub("", line).strip()
                    m_pb = ui_console.RE_PROGRESS_BAR.search(clean)
                    if m_pb and "=" in line:
                        status_cm.update(
                            f"[bold cyan]Z-Image-Turbo örnekleme: adım {m_pb.group(1)}/{m_pb.group(2)} "
                            f"({m_pb.group(3)} {m_pb.group(4)})...[/bold cyan]"
                        )
                    elif "computing condition graph completed" in clean:
                        status_cm.update("[bold cyan]Z-Image-Turbo: metin kodlandı, 6B DiT çalışıyor...[/bold cyan]")
                    elif "decoding" in clean.lower() or "vae_decode" in clean.lower():
                        status_cm.update("[bold cyan]Z-Image-Turbo: VAE görsel çözümleniyor...[/bold cyan]")
            rc = proc.wait()
    finally:
        if status_cm is not None:
            status_cm.__exit__(None, None, None)
        elapsed = time.monotonic() - t0
        snap = sampler.snapshot()
        sampler.stop()

    return rc, elapsed, snap.get("peak_ded_mib"), snap.get("peak_shr_mib")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PicoVid v0.3.0: Wan 2.2 TI2V-5B (Fast 5s) + 14B T2V/I2V + Z-Image-Turbo + Cinema HD (4-6 GB VRAM, CUDA + Vulkan)"
    )
    parser.add_argument("--engine", choices=["5b", "14b"], default="14b", help="Video engine: 5b (Wan 2.2 TI2V-5B Turbo, fast 5s 24fps) or 14b (Wan 2.2 14B)")
    parser.add_argument("--mode", choices=["t2v", "i2v", "t2i", "studio"], default=None, help="Generation mode (default: t2v, or i2v if --init-img given)")
    parser.add_argument("-p", "--prompt", default=None, help="Text prompt (omit to open interactive wizard)")
    parser.add_argument("-i", "--init-img", type=pathlib.Path, default=None, help="Input first-frame image for I2V")
    parser.add_argument("--end-img", type=pathlib.Path, default=None, help="Optional end-frame image (FLF2V)")
    parser.add_argument("--motion-prompt", default=None, help="Optional separate motion prompt for Studio Mode I2V stage")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("output.mp4"), help="Output .mp4 (or .png for --mode t2i)")
    parser.add_argument("-W", "--width", type=int, default=640)
    parser.add_argument("-H", "--height", type=int, default=384)
    parser.add_argument("-f", "--frames", type=int, default=None, help="Video frames per segment (5b default: 81=5.1s@16fps; 14b default: 33=2.1s@16fps)")
    parser.add_argument("--fps", type=int, default=None, help="Output FPS (default: 16; use 24 for native 24fps 5B)")
    parser.add_argument("--segments", type=int, default=1, help="Number of chained I2V segments (e.g. 2 = ~10s, 3 = ~15s via automatic last-frame handoff)")
    parser.add_argument("--segment-prompts", default=None, help="Optional pipe-separated per-segment prompts (e.g. 'scene 1 | camera rises | finale')")
    parser.add_argument("--extend-video", type=pathlib.Path, default=None, help="Continue an existing MP4 video seamlessly from its last frame and append new segment(s)")
    parser.add_argument("--steps", type=int, default=4, help="Total diffusion steps (default: 4)")
    parser.add_argument("--high-noise-steps", type=int, default=2, help="HighNoise expert steps for dual-expert I2V (default: 2 of 4 total)")
    parser.add_argument("--t2i-steps", type=int, default=6, help="Z-Image-Turbo steps for T2I / Studio first frame (default: 6)")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--sampler", default=None, help="Sampling method (default: euler_a for T2V, euler for I2V)")
    parser.add_argument("--scheduler", default=None, help="Sigma scheduler (default: beta for T2V, discrete for I2V)")
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--i2v-variant", choices=["auto", "q4", "q3"], default="auto", help="Wan 2.2 I2V quant variant (q4=18GB, q3=13.4GB compact)")
    parser.add_argument("--diffusion-model", type=pathlib.Path, default=None)
    parser.add_argument("--high-noise-model", type=pathlib.Path, default=None)
    parser.add_argument("--t5xxl", type=pathlib.Path, default=None)
    parser.add_argument("--vae", type=pathlib.Path, default=None)
    parser.add_argument("--taesd", type=pathlib.Path, default=None)
    parser.add_argument("--sd-cli", type=pathlib.Path, default=None)
    parser.add_argument("--fast-preview", action="store_true", help="T2V only: decode directly with TAEHV (~2-6s)")
    parser.add_argument("--cinema-hd", action="store_true", help="Cinema HD post-process: 2x Vulkan AI Upscale (1280x768) + 2x Vulkan RIFE (32fps) + H.264 CRF 15")
    parser.add_argument("--upscale", type=int, choices=[1, 2], default=1, help="Vulkan AI upscale factor (1 or 2)")
    parser.add_argument("--rife", type=int, choices=[1, 2], default=1, help="Vulkan RIFE frame interpolation factor (1 or 2)")
    parser.add_argument("--upscale-model", choices=["realesrgan-x4plus", "realesr-animevideov3"], default="realesrgan-x4plus", help="Real-ESRGAN Vulkan model for 2x HD upscale (default: realesrgan-x4plus photorealistic)")
    parser.add_argument("--keep-latent", action="store_true", help="Keep intermediate float32 .lvlt latent file")
    parser.add_argument("--auto-approve", action="store_true", help="Studio Mode: auto-approve first frame without interactive prompt")
    parser.add_argument("--simulate-profile", choices=["cuda-4gb", "cuda-6gb", "cuda-8gb", "vulkan-4gb", "vulkan-6gb", "vulkan-8gb"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print detected hardware, profile, and commands without running")
    parser.add_argument("--raw-logs", action="store_true", help="Disable Rich dashboard and stream raw sd-cli logs directly")
    parser.add_argument("--verbose", action="store_true", help="Stream raw sd-cli logs to terminal (disables Rich Live panel)")
    parser.add_argument("--ui-demo", action="store_true", help="Run a 5.5-second animated dashboard demo without invoking sd-cli")
    return parser


def main() -> int:
    ui_console.configure_console_utf8()
    reg = load_models_registry()
    parser = build_arg_parser()
    args = parser.parse_args()

    hw = detect_hardware(args.simulate_profile)

    if args.ui_demo:
        demo_frames = args.frames or 33
        prof = select_profile(hw, demo_frames, args.fast_preview, args.width, args.height, engine=args.engine)
        return ui_console.run_ui_demo(hw, prof, raw_logs=args.raw_logs or args.verbose)

    # Determine mode or open interactive 3-mode wizard
    interactive_mode = False
    if args.prompt is None and args.init_img is None and args.extend_video is None:
        if args.dry_run:
            args.prompt = ui_console.DEFAULT_SAMPLE_PROMPT
            args.mode = args.mode or "t2v"
        else:
            interactive_mode = True
            try:
                wiz = ui_console.run_interactive_wizard(hw, select_profile)
            except KeyboardInterrupt:
                print("\n[PicoVid] İptal edildi.")
                return 130
            args.engine = wiz.get("engine", args.engine)
            args.mode = wiz["mode"]
            args.init_img = wiz["init_img"]
            args.prompt = wiz["prompt"]
            args.motion_prompt = wiz["motion_prompt"]
            args.frames = wiz["frames"]
            args.fps = wiz.get("fps", 16)
            args.segments = wiz.get("segments", args.segments)
            args.width = wiz["width"]
            args.height = wiz["height"]
            args.fast_preview = wiz["fast_preview"]
            args.cinema_hd = wiz.get("cinema_hd", False)
            args.seed = wiz["seed"]
            args.output = wiz["output"]
    else:
        if args.mode is None:
            args.mode = "i2v" if args.init_img is not None else "t2v"
        if args.prompt is None:
            args.prompt = "Gentle cinematic camera motion, natural realistic movement, smooth lighting, 35mm film"

    engine: str = args.engine
    is_5b = engine == "5b"
    if args.frames is None:
        args.frames = 81 if is_5b else 33
    if args.fps is None:
        args.fps = 16
    if is_5b:
        # Wan 2.2 TI2V-5B requires 32-aligned spatial dimensions (16x VAE * 2x patch)
        w32 = max(64, (args.width // 32) * 32)
        h32 = max(64, (args.height // 32) * 32)
        if (w32, h32) != (args.width, args.height):
            print(f"[PicoVid] 5B için çözünürlük 32'nin katına yuvarlandı: {args.width}x{args.height} -> {w32}x{h32}", file=sys.stderr)
            args.width, args.height = w32, h32

    if args.cinema_hd:
        # Native 480p+ (832x480) already has crisp photorealistic detail: default to 32fps AOBMC (11s)
        # unless user explicitly requested --upscale 2. For <=640x384, upscale 2x -> 1280x768 + 32fps!
        if args.width <= 640 and args.upscale == 1:
            args.upscale = 2
        if args.rife == 1 and args.fps < 24:
            args.rife = 2
    use_postprocess = (not args.fast_preview) and (args.upscale >= 2 or args.rife >= 2)

    mode: str = args.mode
    if mode in ("i2v", "studio") and not is_5b:
        args.fast_preview = False  # 14B I2V requires real Wan 2.1 VAE for init_img encode + clean decode

    sd_cli = resolve_sd_cli(hw["backend"], args.sd_cli)
    echo_raw = args.raw_logs or args.verbose
    use_rich = ui_console.should_use_rich(raw_logs=args.raw_logs, verbose=args.verbose)

    # Extend an existing MP4 seamlessly from its last frame (--extend-video existing.mp4)
    if args.extend_video is not None:
        base_vid = args.extend_video.resolve()
        if not base_vid.exists() and not args.dry_run:
            print(f"HATA: Uzatılacak video bulunamadı: {base_vid}", file=sys.stderr)
            return 2
        if args.dry_run:
            print(json.dumps({
                "action": "extend_video",
                "base_video": str(base_vid),
                "engine": engine,
                "new_segments": args.segments,
                "frames_per_segment": args.frames,
                "resolution": f"{args.width}x{args.height}",
                "output_mp4": str(args.output.resolve()),
            }, indent=2, ensure_ascii=False))
            return 0

        import cv2
        cap_b = cv2.VideoCapture(str(base_vid))
        base_fps = float(cap_b.get(cv2.CAP_PROP_FPS) or args.fps)
        base_w = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
        base_h = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
        cap_b.release()
        ext_dir = pathlib.Path(tempfile.mkdtemp(prefix="picovid_extend_"))
        try:
            tail_png = ext_dir / "base_last_frame.png"
            postprocess.extract_last_frame(base_vid, tail_png)
            new_seg_mp4 = ext_dir / "extension.mp4"
            sub_cmd = [
                sys.executable, str(pathlib.Path(__file__).resolve()),
                "--engine", engine,
                "--mode", "i2v",
                "-i", str(tail_png),
                "-p", args.prompt,
                "-W", str(args.width),
                "-H", str(args.height),
                "-f", str(args.frames),
                "--fps", str(args.fps),
                "--steps", str(args.steps),
                "--seed", str(args.seed),
                "--segments", str(args.segments),
                "--upscale", str(args.upscale if use_postprocess else 1),
                "--rife", str(args.rife if use_postprocess else 1),
                "--upscale-model", args.upscale_model,
                "-o", str(new_seg_mp4),
            ]
            if args.segment_prompts:
                sub_cmd += ["--segment-prompts", args.segment_prompts]
            if args.scheduler:
                sub_cmd += ["--scheduler", args.scheduler]
            if args.sampler:
                sub_cmd += ["--sampler", args.sampler]
            if args.fast_preview:
                sub_cmd.append("--fast-preview")
            if args.raw_logs:
                sub_cmd.append("--raw-logs")
            if args.verbose:
                sub_cmd.append("--verbose")
            print(f"[PicoVid Uzatma] '{base_vid.name}' ({base_w}x{base_h} @ {base_fps:.1f}fps) son karesinden uzatılıyor...", flush=True)
            p_ext = subprocess.run(sub_cmd)
            if p_ext.returncode != 0 or not new_seg_mp4.exists():
                return p_ext.returncode or 1
            out_mp4 = args.output.resolve()
            info = postprocess.stitch_video_segments([base_vid, new_seg_mp4], out_mp4, fps=base_fps, drop_overlap_frames=1, crf=15)
            print(f"[PicoVid Uzatma] TAMAMLANDI -> {out_mp4} ({info['total_frames']}f · {info['duration_sec']}s, Sınır PSNR: {info['seam_psnr_db']} dB)")
            return 0
        finally:
            shutil.rmtree(ext_dir, ignore_errors=True)

    # Multi-segment automatic I2V chaining (--segments >= 2) for 6-15+ second continuous videos!
    if args.segments > 1 and mode != "t2i":
        seg_count = int(args.segments)
        raw_seg_prompts = [s.strip() for s in args.segment_prompts.split("|") if s.strip()] if args.segment_prompts else []
        seg_prompts = [
            raw_seg_prompts[i] if i < len(raw_seg_prompts) else (args.motion_prompt if (i > 0 and args.motion_prompt) else args.prompt)
            for i in range(seg_count)
        ]
        total_raw_frames = seg_count * args.frames - (seg_count - 1)
        out_fps_est = args.fps * (args.rife if use_postprocess else 1)
        out_w_est = args.width * (args.upscale if use_postprocess else 1)
        out_h_est = args.height * (args.upscale if use_postprocess else 1)

        if args.dry_run:
            prof_seg = select_profile(hw, args.frames, args.fast_preview, args.width, args.height, engine=engine)
            print(json.dumps({
                "engine": engine,
                "mode": mode,
                "segments": seg_count,
                "frames_per_segment": args.frames,
                "total_raw_frames": total_raw_frames,
                "estimated_duration_sec": round(total_raw_frames / max(1, args.fps), 2),
                "target_output": f"{out_w_est}x{out_h_est} @ {out_fps_est}fps",
                "segment_prompts": seg_prompts,
                "selected_profile": prof_seg["profile_name"],
                "dit_tokens_per_segment": prof_seg["dit_tokens"],
                "cinema_hd": {"enabled": use_postprocess, "upscale": args.upscale, "rife": args.rife, "upscale_model": args.upscale_model},
                "output_mp4": str(args.output.resolve()),
            }, indent=2, ensure_ascii=False))
            return 0

        # Ensure models for both initial mode and chained I2V segments are present upfront
        if args.diffusion_model is None and args.high_noise_model is None:
            if not ensure_mode_models_available(mode, reg, args.i2v_variant, args.fast_preview, interactive=interactive_mode, engine=engine):
                return 2
            if mode == "t2v" and not is_5b:
                if not ensure_mode_models_available("i2v", reg, args.i2v_variant, False, interactive=interactive_mode, engine=engine):
                    return 2

        out_mp4 = args.output.resolve()
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        chain_dir = pathlib.Path(tempfile.mkdtemp(prefix="picovid_chain_"))
        seg_t0 = time.monotonic()
        try:
            # If Studio Mode, generate & approve the initial first frame ONCE before chaining
            cur_init_img: pathlib.Path | None = args.init_img.resolve() if args.init_img else None
            first_seg_mode = mode
            if mode == "studio":
                first_frame_png = out_mp4.with_name(out_mp4.stem + "_first_frame.png")
                studio_prompt = seg_prompts[0]
                studio_seed = args.seed
                while True:
                    studio_t2i_plan = build_t2i_plan(hw, reg, sd_cli, studio_prompt, args.width, args.height, args.t2i_steps, studio_seed, first_frame_png)
                    rc_t2i, el_t2i, _, _ = run_t2i_first_frame(studio_t2i_plan, hw, first_frame_png.with_suffix(".t2i.log"), echo_raw=echo_raw)
                    if rc_t2i != 0 or not first_frame_png.exists():
                        print(f"HATA: Z-Image-Turbo ilk kare üretimi başarısız (exit={rc_t2i})", file=sys.stderr)
                        return rc_t2i or 1
                    if args.auto_approve or not ui_console.is_interactive_tty():
                        print(f"[Stüdyo] İlk kare hazır ({el_t2i:.1f}s): {first_frame_png}", flush=True)
                        break
                    action, extra = ui_console.ask_studio_frame_decision(first_frame_png, studio_prompt, studio_seed, el_t2i, use_rich=use_rich)
                    if action == "proceed":
                        if extra and not raw_seg_prompts:
                            seg_prompts = [extra for _ in range(seg_count)]
                        break
                    if action == "reroll":
                        studio_seed = random.randint(1, 2_147_483_647)
                        continue
                    if action == "prompt" and extra:
                        studio_prompt = extra
                        if not raw_seg_prompts:
                            seg_prompts = [extra for _ in range(seg_count)]
                        studio_seed = random.randint(1, 2_147_483_647)
                        continue
                    print(f"[Stüdyo] İptal edildi. İlk kare korundu: {first_frame_png}")
                    return 0
                cur_init_img = first_frame_png
                first_seg_mode = "i2v"

            seg_outputs: list[pathlib.Path] = []
            for s_idx in range(seg_count):
                seg_num = s_idx + 1
                seg_mode = first_seg_mode if s_idx == 0 else "i2v"
                seg_out = chain_dir / f"segment_{seg_num:02d}.mp4"
                seg_seed = args.seed + s_idx
                print(
                    f"\n[PicoVid Zincir] Bölüm {seg_num}/{seg_count} başlıyor "
                    f"({engine.upper()} · {seg_mode.upper()} · {args.width}x{args.height} x {args.frames}f · seed={seg_seed})...",
                    flush=True,
                )
                sub_cmd = [
                    sys.executable, str(pathlib.Path(__file__).resolve()),
                    "--engine", engine,
                    "--mode", seg_mode,
                    "-p", seg_prompts[s_idx],
                    "-W", str(args.width),
                    "-H", str(args.height),
                    "-f", str(args.frames),
                    "--fps", str(args.fps),
                    "--steps", str(args.steps),
                    "--high-noise-steps", str(args.high_noise_steps),
                    "--cfg-scale", str(args.cfg_scale),
                    "--flow-shift", str(args.flow_shift),
                    "--seed", str(seg_seed),
                    "--i2v-variant", args.i2v_variant,
                    "--segments", "1",
                    "--upscale", "1",
                    "--rife", "1",
                    "-o", str(seg_out),
                ]
                if seg_mode == "i2v" and cur_init_img is not None:
                    sub_cmd += ["-i", str(cur_init_img)]
                if args.sampler:
                    sub_cmd += ["--sampler", args.sampler]
                if args.scheduler:
                    sub_cmd += ["--scheduler", args.scheduler]
                if args.fast_preview:
                    sub_cmd.append("--fast-preview")
                if args.raw_logs:
                    sub_cmd.append("--raw-logs")
                if args.verbose:
                    sub_cmd.append("--verbose")
                if args.sd_cli:
                    sub_cmd += ["--sd-cli", str(args.sd_cli)]
                if args.diffusion_model:
                    sub_cmd += ["--diffusion-model", str(args.diffusion_model)]
                if args.high_noise_model:
                    sub_cmd += ["--high-noise-model", str(args.high_noise_model)]
                if args.t5xxl:
                    sub_cmd += ["--t5xxl", str(args.t5xxl)]
                if args.vae:
                    sub_cmd += ["--vae", str(args.vae)]
                if args.taesd:
                    sub_cmd += ["--taesd", str(args.taesd)]

                p_seg = subprocess.run(sub_cmd)
                if p_seg.returncode != 0 or not seg_out.exists():
                    print(f"HATA: Bölüm {seg_num}/{seg_count} başarısız oldu (exit={p_seg.returncode}).", file=sys.stderr)
                    return p_seg.returncode or 1
                seg_outputs.append(seg_out)

                if seg_num < seg_count:
                    handoff_png = chain_dir / f"segment_{seg_num:02d}_last.png"
                    postprocess.extract_last_frame(seg_out, handoff_png)
                    # Gently anchor handoff frame Lab color toward Segment 1's opening frame to prevent multi-pass saturation drift
                    if s_idx == 0:
                        seg1_first_png = chain_dir / "segment_01_first.png"
                        import cv2
                        cap0 = cv2.VideoCapture(str(seg_out))
                        ok0, fr0 = cap0.read()
                        cap0.release()
                        if ok0 and fr0 is not None:
                            cv2.imwrite(str(seg1_first_png), fr0)
                    seg1_ref = chain_dir / "segment_01_first.png"
                    if seg1_ref.exists():
                        import cv2
                        ref_im = cv2.imread(str(seg1_ref))
                        ho_im = cv2.imread(str(handoff_png))
                        if ref_im is not None and ho_im is not None:
                            rm, rs = postprocess._compute_lab_stats(ref_im)
                            cv2.imwrite(str(handoff_png), postprocess._apply_lab_color_anchor(ho_im, rm, rs, strength=0.45))
                    cur_init_img = handoff_png
                    print(f"[PicoVid Zincir] Bölüm {seg_num} son karesi sonraki bölüme aktarıldı (Lab renk çıpası aktif): {handoff_png.name}", flush=True)

            stitched_mp4 = chain_dir / "stitched_raw.mp4" if use_postprocess else out_mp4
            print(f"\n[PicoVid Zincir] {seg_count} bölüm kesintisiz birleştiriliyor (çift sınır kareleri atılıyor)...", flush=True)
            stitch_info = postprocess.stitch_video_segments(seg_outputs, stitched_mp4, fps=float(args.fps), drop_overlap_frames=1, crf=15)

            if use_postprocess:
                ok_u, ok_r = postprocess.ensure_tools_installed(args.upscale >= 2, args.rife >= 2)
                if ok_u and ok_r:
                    print(f"[PicoVid Zincir] Tüm {stitch_info['duration_sec']}s film için tek seferde Sinema HD / 32fps uygulanıyor...", flush=True)
                    pp_res = postprocess.enhance_video(
                        stitched_mp4,
                        out_mp4,
                        upscale_factor=args.upscale,
                        rife_factor=args.rife,
                        upscale_model=args.upscale_model,
                        crf=15,
                    )
                    final_desc = f"{pp_res['out_res']} · {pp_res['out_frames']}f @ {pp_res['out_fps']}fps ({pp_res['total_sec']}s post)"
                else:
                    shutil.copyfile(stitched_mp4, out_mp4)
                    final_desc = f"{stitch_info['width']}x{stitch_info['height']} · {stitch_info['total_frames']}f @ {args.fps}fps"
            else:
                final_desc = f"{stitch_info['width']}x{stitch_info['height']} · {stitch_info['total_frames']}f @ {args.fps}fps"

            total_el = time.monotonic() - seg_t0
            print("-" * 64)
            print(f"[PicoVid Zincir] TAMAMLANDI ({total_el:.1f}s) · {seg_count} bölüm -> {stitch_info['duration_sec']}s kesintisiz video")
            print(f"  - Çıktı    : {out_mp4} ({final_desc}) [{ui_console.format_file_size(out_mp4)}]")
            if stitch_info.get("seam_psnr_db"):
                print(f"  - Sınır PSNR: {stitch_info['seam_psnr_db']} dB (kare-kilitli geçiş)")
            print("-" * 64)
            if interactive_mode and out_mp4.exists():
                ui_console.ask_open_video(out_mp4, use_rich=use_rich)
            return 0
        finally:
            shutil.rmtree(chain_dir, ignore_errors=True)

    # First-run guard: if not --dry-run, offer automatic build for missing sd-cli & inline download for missing models
    if not args.dry_run:
        verified_cli = ensure_sd_cli_available(sd_cli, hw["backend"], interactive=interactive_mode)
        if verified_cli is None:
            return 2
        sd_cli = verified_cli
        if args.diffusion_model is None and args.high_noise_model is None:
            if not ensure_mode_models_available(mode, reg, args.i2v_variant, args.fast_preview, interactive=interactive_mode, engine=engine):
                return 2

    # Standalone T2I mode (Z-Image-Turbo only)
    if mode == "t2i":
        out_png = args.output.resolve()
        if out_png.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            out_png = out_png.with_suffix(".png")
        t2i_plan = build_t2i_plan(hw, reg, sd_cli, args.prompt, args.width, args.height, args.t2i_steps, args.seed, out_png)
        if args.dry_run:
            print(json.dumps({
                "mode": "t2i",
                "hardware": hw,
                "t2i": {"env": t2i_plan["env"], "cmd": t2i_plan["cmd"]},
                "output_png": str(out_png),
            }, indent=2, ensure_ascii=False))
            return 0
        for label, p in [("sd-cli", sd_cli), *t2i_plan["models"].items()]:
            if not p.exists():
                print(f"HATA: Eksik {label}: {p}\nİndirmek için: py scripts/download_models.py --profile t2i", file=sys.stderr)
                return 2
        out_png.parent.mkdir(parents=True, exist_ok=True)
        rc, elapsed, p_ded, p_shr = run_t2i_first_frame(t2i_plan, hw, out_png.with_suffix(".log"), echo_raw=echo_raw)
        if rc != 0 or not out_png.exists():
            print(f"HATA: Z-Image-Turbo başarısız (exit={rc})", file=sys.stderr)
            return rc or 1
        vram_s = f" | Peak VRAM: Ded {p_ded:.0f} MiB, Shr {p_shr:.0f} MiB" if p_ded else ""
        print(f"[PicoVid] Z-Image-Turbo hazır ({elapsed:.1f}s{vram_s}): {out_png}")
        return 0

    # Studio Mode: Step 1 -> Generate & Approve First Frame with Z-Image-Turbo
    studio_t2i_plan: dict[str, Any] | None = None
    if mode == "studio":
        out_mp4_preview = args.output.resolve()
        first_frame_png = out_mp4_preview.with_name(out_mp4_preview.stem + "_first_frame.png")
        studio_prompt = args.prompt
        studio_seed = args.seed
        studio_t2i_plan = build_t2i_plan(hw, reg, sd_cli, studio_prompt, args.width, args.height, args.t2i_steps, studio_seed, first_frame_png)

        if not args.dry_run:
            for label, p in [("sd-cli", sd_cli), *studio_t2i_plan["models"].items()]:
                if not p.exists():
                    print(f"HATA: Stüdyo Modu için eksik {label}: {p}\nİndirmek için: py scripts/download_models.py --profile t2i", file=sys.stderr)
                    return 2
            first_frame_png.parent.mkdir(parents=True, exist_ok=True)
            while True:
                studio_t2i_plan = build_t2i_plan(hw, reg, sd_cli, studio_prompt, args.width, args.height, args.t2i_steps, studio_seed, first_frame_png)
                rc, elapsed, p_ded, p_shr = run_t2i_first_frame(
                    studio_t2i_plan, hw, first_frame_png.with_suffix(".t2i.log"), echo_raw=echo_raw
                )
                if rc != 0 or not first_frame_png.exists():
                    print(f"HATA: Z-Image-Turbo ilk kare üretimi başarısız (exit={rc})", file=sys.stderr)
                    return rc or 1

                if args.auto_approve or not ui_console.is_interactive_tty():
                    print(f"[Stüdyo] İlk kare hazır ({elapsed:.1f}s, seed={studio_seed}): {first_frame_png}")
                    break

                action, extra = ui_console.ask_studio_frame_decision(
                    first_frame_png, studio_prompt, studio_seed, elapsed, use_rich=use_rich
                )
                if action == "proceed":
                    if extra:
                        args.motion_prompt = extra
                    break
                if action == "reroll":
                    studio_seed = random.randint(1, 2_147_483_647)
                    continue
                if action == "prompt" and extra:
                    studio_prompt = extra
                    studio_seed = random.randint(1, 2_147_483_647)
                    continue
                print(f"[Stüdyo] İptal edildi. İlk kare korundu: {first_frame_png}")
                return 0

            args.init_img = first_frame_png
            args.prompt = args.motion_prompt or studio_prompt
        else:
            args.init_img = first_frame_png
            args.prompt = args.motion_prompt or studio_prompt

    # Video Stage (T2V or I2V / Studio Stage 2 — supports both Wan 2.2 TI2V-5B and 14B)
    is_i2v = mode in ("i2v", "studio")
    prof = select_profile(hw, args.frames, args.fast_preview, args.width, args.height, engine=engine)

    t5xxl = resolve_project_path(args.t5xxl) if args.t5xxl else reg_file_path(reg, "umt5_xxl_q4_k_m", "models/text_encoders/umt5-xxl-encoder-Q4_K_M.gguf")
    if is_5b:
        vae_path = resolve_project_path(args.vae) if args.vae else reg_file_path(reg, "wan22_vae", "models/vae/wan2.2_vae.safetensors")
        taesd_path = resolve_project_path(args.taesd) if args.taesd else reg_file_path(reg, "taew2_2", "models/vae/taew2_2.safetensors")
    else:
        vae_path = resolve_project_path(args.vae) if args.vae else reg_file_path(reg, "wan21_vae", "models/vae/wan_2.1_vae.safetensors")
        taesd_path = resolve_project_path(args.taesd) if args.taesd else reg_file_path(reg, "taew2_1", "models/vae/taew2_1.safetensors")

    high_model: pathlib.Path | None = None
    i2v_variant_used: str | None = None
    if is_5b:
        diffusion_model = (
            resolve_project_path(args.diffusion_model)
            if args.diffusion_model
            else reg_file_path(reg, "wan22_ti2v_5b_turbo_q4", "models/diffusion_models/Wan2_2-TI2V-5B-Turbo-Q4_K_M.gguf")
        )
    elif is_i2v:
        high_model, diffusion_model, i2v_variant_used = resolve_i2v_models(
            reg, args.i2v_variant, args.high_noise_model, args.diffusion_model
        )
    else:
        diffusion_model = (
            resolve_project_path(args.diffusion_model)
            if args.diffusion_model
            else reg_file_path(reg, "wan22_t2v_rapid_v10", "models/diffusion_models/wan2.2-t2v-rapid-aio-v10-Q4_K.gguf")
        )

    sampler_name = args.sampler or ("euler" if (is_i2v or is_5b) else "euler_a")
    scheduler_name = args.scheduler or ("beta" if (is_5b or not is_i2v) else "discrete")

    out_mp4 = args.output.resolve()
    if not args.dry_run:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
    work_dir = pathlib.Path(tempfile.mkdtemp(prefix="localvid_")) if not args.dry_run else out_mp4.parent
    lat_path = out_mp4.with_suffix(".lvlt") if args.keep_latent else (work_dir / "latent.lvlt")
    raw_avi = out_mp4.with_suffix(".avi")
    dummy_avi = work_dir / "dit_dummy.avi"
    log_path = out_mp4.with_suffix(".log")

    # CRITICAL NORMALIZATION RULE (verified via video_wan.dgl:203 & components.cpp:240-255):
    # - 14B T2V with --taesd: Stage 1 saves RAW DiT latents -> Stage 2 MUST set SD_VIDEO_LATENT_DIT_TO_VAE=1.
    # - 14B I2V or 5B TI2V with --vae: Stage 1 runs Wan VAE's dit_to_vae_latents BEFORE save hook -> Stage 2 MUST NOT set SD_VIDEO_LATENT_DIT_TO_VAE!
    use_tae_in_p1 = (not is_i2v) and (not is_5b) and (args.fast_preview or taesd_path.exists() or args.dry_run)
    env_p1 = dict(prof["env_p1"])

    # I2V Stage 1 VAE encode policy:
    # - 5B TI2V frozen-frame I2V only encodes 1 single frame -> fast on GPU for ANY frame count!
    # - 14B I2V 6GB/8GB+ with <=49f: GPU-tiled vae_encode (320x192, conv_direct vae=on)
    #   -> 11.2x faster (19.3s vs 216.3s), 39.35 dB PSNR, zero seams, cleanly evicted before DiT!
    # - Automatic runtime fallback to CPU vae_encode if GPU vae_encode ever fails before DiT starts.
    use_gpu_i2v_encode = is_i2v and (hw["vram_class"] != "4GB") and (is_5b or args.frames <= 49)

    def build_p1_cmd(gpu_vae_encode: bool) -> list[str]:
        cmd: list[str] = [str(sd_cli), "-M", "vid_gen"]
        if is_i2v and high_model is not None:
            # sd-cli --high-noise-steps >= 0 ADDS to --steps!
            # So for 4 total steps (2 High + 2 Low), pass --steps 2 --high-noise-steps 2.
            hi_steps = max(1, min(args.high_noise_steps, max(1, args.steps - 1)))
            lo_steps = max(1, args.steps - hi_steps)
            cmd += [
                "--high-noise-diffusion-model", str(high_model),
                "--diffusion-model", str(diffusion_model),
                "--high-noise-steps", str(hi_steps),
                "--steps", str(lo_steps),
            ]
        else:
            cmd += [
                "--diffusion-model", str(diffusion_model),
                "--steps", str(args.steps),
            ]

        cmd += [
            "--t5xxl", str(t5xxl),
            "--taesd" if use_tae_in_p1 else "--vae",
            str(taesd_path if use_tae_in_p1 else vae_path),
        ]
        if is_i2v and args.init_img is not None:
            cmd += ["--init-img", str(args.init_img.resolve())]
        if is_i2v and args.end_img is not None:
            cmd += ["--end-img", str(args.end_img.resolve())]

        p1_backend = "te=cpu" if (not is_i2v or gpu_vae_encode) else "te=cpu,vae=cpu"
        cmd += [
            "-p", args.prompt,
            "-W", str(args.width),
            "-H", str(args.height),
            "--video-frames", str(args.frames),
            "--fps", str(args.fps),
            "--cfg-scale", str(args.cfg_scale),
            "--sampling-method", sampler_name,
            "--scheduler", scheduler_name,
            "--flow-shift", str(args.flow_shift),
            "--seed", str(args.seed),
            "--offload-to-cpu",
            "--mmap",
            "--backend", p1_backend,
            "--flash-attn", prof["flash_attn"],
            "--max-vram", str(prof["max_vram_gib"]),
            "-o", str(raw_avi if (args.fast_preview and not is_5b) else dummy_avi),
        ]
        if is_i2v and gpu_vae_encode:
            cmd += [
                "--vae-tiling",
                "--vae-tile-size", "320x192",
                "--conv-direct", "auto,vae=on",
            ]
        return cmd

    # For 5B, ALWAYS use two-process decode (Stage 1 saves latent via real wan2.2_vae, Stage 2 decodes
    # with either real Wan 2.2 VAE or 1.9s TAEW2.2 when --fast-preview is set!).
    two_proc_vae = is_5b or (not args.fast_preview)
    cmd_p1 = build_p1_cmd(use_gpu_i2v_encode)
    if two_proc_vae:
        env_p1["SD_VIDEO_LATENT_SAVE"] = str(lat_path)
        env_p1["SD_VIDEO_LATENT_SAVE_ONLY"] = "1"

    # Build Process 2 command (Standalone Wan VAE or TAEW2.2)
    dgl_path = work_dir / "decode_wan_vae.dgl"
    env_p2 = dict(prof["env_p2"])
    env_p2["SD_VIDEO_LATENT_LOAD"] = str(lat_path)
    if not use_tae_in_p1:
        # Process 1 used real Wan VAE -> video_wan.dgl:203 already applied dit_to_vae_latents!
        env_p2.pop("SD_VIDEO_LATENT_DIT_TO_VAE", None)
    cmd_p2 = [str(sd_cli), "-M", "dgl", "--script", str(dgl_path)]

    plan: dict[str, Any] = {
        "engine": engine,
        "mode": mode,
        "hardware": hw,
        "selected_profile": prof["profile_name"],
        "dit_tokens": prof["dit_tokens"],
        "vram_warning": prof["vram_warning"],
        "two_process_vae": two_proc_vae,
        "vae_conv_direct_mode": prof["vae_conv_direct_mode"],
        "stage2_dit_to_vae": env_p2.get("SD_VIDEO_LATENT_DIT_TO_VAE", "0"),
        "cinema_hd": {"enabled": use_postprocess, "upscale": args.upscale, "rife": args.rife, "upscale_model": args.upscale_model},
        "mp4_encoder": resolve_ffmpeg()[1] or "opencv_mp4v",
        "process_1_dit": {"env": env_p1, "cmd": cmd_p1},
        "process_2_vae": {"env": env_p2, "cmd": cmd_p2} if two_proc_vae else None,
        "output_mp4": str(out_mp4),
    }
    if i2v_variant_used is not None:
        plan["i2v_variant"] = i2v_variant_used
        plan["i2v_vae_encode_mode"] = "gpu_tiled_320x192" if use_gpu_i2v_encode else "cpu_safe"
    if studio_t2i_plan is not None:
        plan["studio_t2i"] = {"env": studio_t2i_plan["env"], "cmd": studio_t2i_plan["cmd"], "output_png": studio_t2i_plan["output_png"]}

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    # Verify required binaries and models exist
    req_checks: list[tuple[str, pathlib.Path]] = [
        ("sd-cli", sd_cli),
        ("diffusion-model", diffusion_model),
        ("t5xxl", t5xxl),
    ]
    if is_i2v and high_model is not None:
        req_checks.append(("high-noise-model", high_model))
    if is_i2v:
        if args.init_img is None or not args.init_img.exists():
            print(f"HATA: I2V için geçerli --init-img gerekli: {args.init_img}", file=sys.stderr)
            return 2
    for label, p in req_checks:
        if not p.exists():
            hint = " --profile i2v" if is_i2v else " --profile t2v"
            print(f"HATA: Eksik {label}: {p}\nİndirmek için: py scripts/download_models.py{hint}", file=sys.stderr)
            return 2
    if args.fast_preview and not taesd_path.exists():
        print(f"HATA: --fast-preview için TAEHV gerekli: {taesd_path}", file=sys.stderr)
        return 2
    if not args.fast_preview and not vae_path.exists():
        print(f"HATA: Eksik Wan 2.1 VAE: {vae_path}", file=sys.stderr)
        return 2

    if not echo_raw:
        env_p1.setdefault("SD_SEG_TIMING", "2")

    vram_sampler = ui_console.LiveVramSampler(total_vram_gib=hw["vram_total_gib"], interval_sec=1.0)
    stage4_lbl = (
        f"4. Sinema HD ({args.width*args.upscale}x{args.height*args.upscale} · {args.fps*args.rife}fps)"
        if use_postprocess
        else "4. H.264 MP4 Paketleme"
    )
    dash_cfg = ui_console.DashboardConfig(
        gpu_name=hw["gpu_name"],
        backend=hw["backend"],
        vram_class=hw["vram_class"],
        vram_total_gib=hw["vram_total_gib"],
        vram_budget_gib=hw["vram_budget_gib"],
        profile_name=f"{prof['profile_name']} ({mode.upper()})",
        width=args.width,
        height=args.height,
        frames=args.frames,
        fps=args.fps,
        steps=args.steps,
        vae_label="TAEHV" if args.fast_preview else ("Wan 2.2 VAE" if is_5b else "Wan 2.1 VAE"),
        log_path=log_path,
        stage4_label=stage4_lbl,
        dit_label="Wan 2.2 TI2V-5B Turbo" if is_5b else "Wan 2.2 14B",
        total_blocks=30 if is_5b else 40,
    )
    dashboard: ui_console.BaseDashboard = (
        ui_console.RichDashboard(dash_cfg, vram_sampler)
        if use_rich
        else ui_console.PlainDashboard(dash_cfg, vram_sampler)
    )

    if prof.get("vram_warning") and not interactive_mode:
        print(f"[PicoVid] {prof['vram_warning']}", file=sys.stderr, flush=True)

    if not args.dry_run and use_postprocess:
        ok_u, ok_r = postprocess.ensure_tools_installed(args.upscale >= 2, args.rife >= 2)
        if not (ok_u and ok_r):
            print("[PicoVid] Uyarı: Sinema HD araçları kurulamadı; standart H.264 çıkışına düşülüyor.", file=sys.stderr)
            use_postprocess = False

    vram_sampler.start()
    dashboard.start()
    prewarm_stop = threading.Event()
    prewarm_targets = [high_model, diffusion_model] if (is_i2v and high_model is not None) else [diffusion_model]
    start_file_prewarm(prewarm_targets, prewarm_stop)
    final_output: pathlib.Path | None = None
    encoder_label = ""
    err_msg = ""
    rc = 0

    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as lf:
            lf.write("=== PicoVid v0.2.0 Run Log ===\n")
            lf.write(json.dumps(plan, indent=2, ensure_ascii=False) + "\n\n")
            lf.write("=== PROCESS 1: TE (+ I2V VAE Encode CPU) + DiT ===\n")
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
            # Automatic safe fallback: if GPU vae_encode failed before DiT sampling started,
            # retry Stage 1 cleanly with --backend te=cpu,vae=cpu!
            if rc != 0 and is_i2v and use_gpu_i2v_encode:
                lf.flush()
                log_so_far = log_path.read_text(encoding="utf-8", errors="replace")
                if "sampling started:" not in log_so_far:
                    dashboard.set_status("GPU vae_encode başarısız; güvenli CPU vae_encode ile yeniden deneniyor...")
                    dashboard.start_stage("te", detail="CPU vae_encode fallback...", determinate=False)
                    lf.write("\n=== FALLBACK: Retrying Process 1 with --backend te=cpu,vae=cpu ===\n")
                    lf.flush()
                    cmd_p1_cpu = build_p1_cmd(False)
                    rc = ui_console.run_process_monitored(
                        cmd=cmd_p1_cpu,
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

            if two_proc_vae:
                vae_ch = 48 if is_5b else 16
                vae_family = "wan2_2_ti2v" if is_5b else "wan2"
                if is_5b and args.fast_preview:
                    dgl_path.write_text(
                        f"""# version: 0.0.0
var {{
    tae_path = "{taesd_path.resolve().as_posix()}"
    output_path = "{raw_avi.as_posix()}"
    fps = {args.fps}
}}
fn main() {{
    vae = load_tae(tae_path, family = "wan2_2_ti2v", decode_only = true, conv_direct_mode = "{prof['vae_conv_direct_mode']}", flash_attn_mode = "{prof['vae_flash_attn_mode']}")
    dummy_latent = tensor::zeros([8, 8, 1, 48, 1])
    output_frames = vae_decode(vae, dummy_latent)
    video::save(output_path, {{frames = output_frames, fps_num = fps, fps_den = 1}})
    return {{video = {{frames = output_frames, fps_num = fps, fps_den = 1}}}}
}}
""",
                        encoding="utf-8",
                    )
                    lf.write("\n=== PROCESS 2: Standalone TAEW2.2 Fast Decode (1.9s) ===\n")
                else:
                    # For 1.31 GiB Wan 2.2 VAE on 4-6 GB GPUs, resolution-adaptive spatial+temporal tiling
                    # eliminates OOM retries and cuts decode time 3.6x (606s -> 167.8s on 640x384x81f,
                    # and 4x2=8 tiles instead of 4x3=12 tiles on 832x480x81f!).
                    if is_5b and hw["vram_class"] != "8GB+":
                        if hw["vram_class"] == "4GB":
                            tw, th, ov = 192, 192, 0.25
                        elif args.height > 384:
                            tw, th, ov = 224, 256, 0.08
                        else:
                            tw, th, ov = 224, 224, 0.08
                        decode_call = (
                            f"vae_decode(vae, dummy_latent, tiling = tiling(enabled = true, temporal_tiling = true, "
                            f"tile_size_w = {tw}, tile_size_h = {th}, target_overlap = {ov}))"
                        )
                    else:
                        decode_call = "vae_decode(vae, dummy_latent)"
                    dgl_path.write_text(
                        f"""# version: 0.0.0
var {{
    vae_path = "{vae_path.resolve().as_posix()}"
    output_path = "{raw_avi.as_posix()}"
    fps = {args.fps}
}}
fn main() {{
    vae = load_vae(vae_path, family = "{vae_family}", params_backend = "cpu", conv_direct_mode = "{prof['vae_conv_direct_mode']}", flash_attn_mode = "{prof['vae_flash_attn_mode']}")
    dummy_latent = tensor::zeros([8, 8, 1, {vae_ch}, 1])
    output_frames = {decode_call}
    video::save(output_path, {{frames = output_frames, fps_num = fps, fps_den = 1}})
    return {{video = {{frames = output_frames, fps_num = fps, fps_den = 1}}}}
}}
""",
                        encoding="utf-8",
                    )
                    lf.write(f"\n=== PROCESS 2: Standalone {'Wan 2.2' if is_5b else 'Wan 2.1'} VAE ===\n")
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

        dashboard.start_stage("mp4", detail="Sinema HD / H.264 başlıyor...", determinate=True)
        dashboard.set_status("Çıktı paketleniyor...")
        if raw_avi.exists():
            if use_postprocess:
                def _pp_cb(pct: float, msg: str) -> None:
                    dashboard.update_stage("mp4", pct=pct, detail=msg)
                    dashboard.set_status(msg)
                pp_res = postprocess.enhance_video(
                    raw_avi,
                    out_mp4,
                    upscale_factor=args.upscale,
                    rife_factor=args.rife,
                    upscale_model=args.upscale_model,
                    crf=15,
                    progress_cb=_pp_cb,
                )
                raw_avi.unlink(missing_ok=True)
                final_output = out_mp4
                encoder_label = f"CinemaHD {pp_res['out_res']}@{pp_res['out_fps']}fps CRF15"
                dashboard.finish_stage("mp4", detail=f"{pp_res['out_res']} · {pp_res['out_fps']}fps ({pp_res['total_sec']}s)")
                dashboard.set_status(f"Sinema HD hazır: {out_mp4.name} ({pp_res['out_res']} @ {pp_res['out_fps']}fps)")
            else:
                dashboard.update_stage("mp4", pct=40.0, detail="libx264 (CRF 16) kodlanıyor...")
                encoder_label = convert_avi_to_mp4(raw_avi, out_mp4)
                if encoder_label != "avi_only":
                    raw_avi.unlink(missing_ok=True)
                    final_output = out_mp4
                    dashboard.finish_stage("mp4", detail=f"{out_mp4.name} ({encoder_label})")
                    dashboard.set_status(f"MP4 hazır: {out_mp4.name}")
                else:
                    final_output = raw_avi
                    dashboard.finish_stage("mp4", detail=f"{raw_avi.name} (AVI)")
                    dashboard.set_status(f"AVI kaydedildi: {raw_avi.name}")
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
        prewarm_stop.set()
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
