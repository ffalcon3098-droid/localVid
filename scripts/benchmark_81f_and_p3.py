#!/usr/bin/env python3
"""
localVid 81f + P3 (Wan 2.1 VAE conv_direct + Separate-Process VAE) Benchmark Suite:
1. 640x384 x 17f: Process 1 (DiT -> save latent) + Process 2 (Wan 2.1 VAE with SD_CONV_DIRECT_TOL=0.005)
2. 640x384 x 49f: Process 1 (DiT with SD_WAN_QKV_SPLIT=1 + SD_CUT_CACHE_HOST_MB=32, mv=2.0) + Process 2 (Wan 2.1 VAE)
3. 640x384 x 81f: Process 1 (DiT with SD_WAN_QKV_SPLIT=1 + SD_CUT_CACHE_HOST_MB=32, mv=2.0) + Process 2 (Wan 2.1 VAE)
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


def run_proc(cmd: list[str], env_extra: dict[str, str], log_path: pathlib.Path) -> tuple[int, float, float, float, str]:
    env = os.environ.copy()
    env.update(env_extra)
    stop = threading.Event()
    peaks = {"ded": 0.0, "shr": 0.0}

    def poll() -> None:
        while not stop.is_set():
            d, s = sample_vram_mib()
            if d is not None and s is not None:
                peaks["ded"] = max(peaks["ded"], d)
                peaks["shr"] = max(peaks["shr"], s)
            stop.wait(1.0)

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
        proc.wait()
    stop.set()
    t.join(timeout=5)
    return proc.returncode, round(time.monotonic() - t0, 2), round(peaks["ded"], 1), round(peaks["shr"], 1), "".join(lines)


def run_full_case(frames: int, head_chunk: int, q_chunk: int, max_vram: float) -> dict:
    run_dir = pathlib.Path("benchmarks/runs/p3_81f")
    run_dir.mkdir(parents=True, exist_ok=True)
    lat_path = run_dir / f"wan22_640x384_{frames}f.lvlt"
    p1_log = run_dir / f"p1_dit_{frames}f.log"
    p2_log = run_dir / f"p2_vae_{frames}f.log"
    dummy_avi = run_dir / f"p1_dummy_{frames}f.avi"
    out_avi = run_dir / f"p2_decoded_{frames}f.avi"

    # Process 1: DiT only -> save float32 latent
    cmd_p1 = [
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
        "-o", str(dummy_avi),
    ]
    env_p1 = {
        "SD_LOADER_FAST_POLL": "1",
        "SD_EVICT_ON_END": "1",
        "SD_EVICT_PAST": "1",
        "SD_DROP_QUEUED": "1",
        "SD_STAGING_POOL_MB": "512",
        "SD_SEG_TIMING": "1",
        "SD_WAN_SPLIT_BLOCK": "1",
        "SD_WAN_QKV_SPLIT": "1",
        "SD_WAN_ATTN_HEAD_CHUNK": str(head_chunk),
        "SD_WAN_ATTN_HEAD_CUT": "1",
        "SD_WAN_MLP_CHUNK_MB": "128",
        "SD_CUT_CACHE_HOST_MB": "32",
        "SD_VIDEO_LATENT_SAVE": str(lat_path),
        "SD_VIDEO_LATENT_SAVE_ONLY": "1",
    }
    if q_chunk > 0:
        env_p1["SD_WAN_ATTN_Q_CHUNK"] = str(q_chunk)

    print(f"\n=== PROCESS 1 (DiT): 640x384 x {frames}f (h={head_chunk}, q={q_chunk}, mv={max_vram}, qkv_split=1, cut_host=32MB) ===", flush=True)
    rc1, wall1, ded1, shr1, txt1 = run_proc(cmd_p1, env_p1, p1_log)
    m_samp = re.search(r"sampling completed, taking ([\d.]+)s", txt1)
    dit_bufs = re.findall(r"(Wan2\.x-T2V-14B) compute buffer size:\s*([\d.]+) MB\(([^)]+)\)", txt1)
    cut_cache_lines = [l.strip() for l in txt1.splitlines() if "graph cut cache added" in l][:8]
    errs1 = [l.strip() for l in txt1.splitlines() if "[E]" in l or "ErrorDeviceLost" in l or "cannot make enough memory" in l][-6:]

    res: dict = {
        "frames": frames,
        "head_chunk": head_chunk,
        "q_chunk": q_chunk,
        "max_vram_gib": max_vram,
        "dit": {
            "rc": rc1,
            "wall_s": wall1,
            "sampling_s": float(m_samp.group(1)) if m_samp else None,
            "peak_ded_mib": ded1,
            "peak_shr_mib": shr1,
            "compute_buffers": dit_bufs,
            "cut_cache_sample": cut_cache_lines,
            "errors": errs1,
            "log": str(p1_log),
        },
        "vae": None,
    }

    # Process 2: Standalone Wan 2.1 VAE decode with SD_CONV_DIRECT_TOL=0.005
    if rc1 == 0 and lat_path.exists():
        dgl_script = run_dir / f"decode_{frames}f.dgl"
        dgl_script.write_text(
            f"""# version: 0.0.0
var {{
    vae_path = "models/vae/wan_2.1_vae.safetensors"
    output_path = "{out_avi.as_posix()}"
    fps = 16
}}
fn main() {{
    vae = load_vae(vae_path, family = "wan2", params_backend = "cpu")
    dummy_latent = tensor::zeros([8, 8, 1, 16, 1])
    output_frames = vae_decode(vae, dummy_latent)
    video::save(output_path, {{frames = output_frames, fps_num = fps, fps_den = 1}})
    return {{video = {{frames = output_frames, fps_num = fps, fps_den = 1}}}}
}}
""",
            encoding="utf-8",
        )
        cmd_p2 = [
            "build/sdcpp-vulkan/bin/sd-cli.exe",
            "-M", "dgl",
            "--script", str(dgl_script),
            "-v",
        ]
        env_p2 = {
            "SD_LOADER_FAST_POLL": "1",
            "SD_CONV_DIRECT_TOL": "0.005",
            "SD_VIDEO_LATENT_LOAD": str(lat_path),
        }
        print(f"=== PROCESS 2 (Wan 2.1 VAE, conv_direct tol=0.005): 640x384 x {frames}f ===", flush=True)
        rc2, wall2, ded2, shr2, txt2 = run_proc(cmd_p2, env_p2, p2_log)
        m_vae = re.search(r"computing vae decode graph completed, taking ([\d.]+)s", txt2)
        vae_bufs = re.findall(r"(wan_vae) compute buffer size:\s*([\d.]+) MB\(([^)]+)\)", txt2)
        conv_lines = [l.strip() for l in txt2.splitlines() if "conv2d direct" in l or "conv3d direct" in l]
        tiling_lines = [l.strip() for l in txt2.splitlines() if "tiling" in l.lower() or "cannot make enough memory" in l][:6]
        errs2 = [l.strip() for l in txt2.splitlines() if "[E]" in l or "ErrorDeviceLost" in l][-6:]
        res["vae"] = {
            "rc": rc2,
            "wall_s": wall2,
            "decode_s": float(m_vae.group(1)) if m_vae else None,
            "peak_ded_mib": ded2,
            "peak_shr_mib": shr2,
            "conv_direct": conv_lines,
            "tiling_notes": tiling_lines,
            "compute_buffers": vae_bufs[:6],
            "errors": errs2,
            "log": str(p2_log),
        }

    print(json.dumps(res, indent=2), flush=True)
    return res


def main() -> int:
    results = []
    # 1. 49f with mv=2.2 (needed 1988 MB budget)
    results.append(run_full_case(49, 4, 1024, 2.2))
    # 2. 81f (5-second clip!) with mv=2.5 (needed 2362 MB budget)
    results.append(run_full_case(81, 4, 1024, 2.5))

    out_json = pathlib.Path("benchmarks/runs/p3_81f/summary_49_81.json")
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nP3_81F_BENCH_DONE -> {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
