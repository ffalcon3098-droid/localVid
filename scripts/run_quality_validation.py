#!/usr/bin/env python3
"""
localVid First Real 4-Step Visual Quality & Patch Validation Suite:
- Model: wan2.2-t2v-rapid-aio-v10-Q4_K.gguf (pre-merged 4-step LightX2V + Lightning)
- Run 1: 640x384 x 17f, 4 steps, CFG=1.0, euler_a + beta, flow-shift 5.0, mv=1.5, Separate-Process Wan 2.1 VAE
- Run 2: 640x384 x 33f, 4 steps, CFG=1.0, euler_a + beta, flow-shift 5.0, Q_CHUNK=2048, mv=2.0, Separate-Process Wan 2.1 VAE
- Outputs: playable .avi + .mp4 (OpenCV) + 5 sample PNG frames per video + pixel artifact metrics.
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

import cv2
import numpy as np

PROMPT = (
    "A cinematic tracking shot of a vintage red sports car driving down a rain-slicked neon-lit city street "
    "at sunset, golden light reflecting off wet asphalt and puddles, soft mist, realistic motion, film grain, 35mm lens"
)

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


def package_video_and_frames(avi_path: pathlib.Path, out_dir: pathlib.Path, prefix: str) -> dict:
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        return {"error": f"Failed to open {avi_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_bgr: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
    cap.release()

    n = len(frames_bgr)
    if n == 0:
        return {"error": "Zero frames decoded"}

    mp4_path = out_dir / f"{prefix}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (width, height))
    for fr in frames_bgr:
        writer.write(fr)
    writer.release()

    # Extract 5 evenly spaced PNG frames
    indices = sorted(set([0, n // 4, n // 2, (3 * n) // 4, n - 1]))
    png_paths: list[str] = []
    for idx in indices:
        p = out_dir / f"{prefix}_frame_{idx:02d}.png"
        cv2.imwrite(str(p), frames_bgr[idx])
        png_paths.append(str(p))

    # Pixel health & motion stats
    means = [float(fr.mean()) for fr in frames_bgr]
    stds = [float(fr.std()) for fr in frames_bgr]
    black_frames = sum(1 for m in means if m < 3.0)
    motion_diffs = [
        float(np.abs(frames_bgr[i].astype(np.float32) - frames_bgr[i - 1].astype(np.float32)).mean())
        for i in range(1, n)
    ]

    return {
        "frame_count": n,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "avi_path": str(avi_path),
        "mp4_path": str(mp4_path),
        "sample_pngs": png_paths,
        "mean_luma_min_max": [round(min(means), 2), round(max(means), 2)],
        "std_min_max": [round(min(stds), 2), round(max(stds), 2)],
        "black_frames": black_frames,
        "mean_interframe_motion": round(float(np.mean(motion_diffs)), 2) if motion_diffs else 0.0,
    }


def run_quality_case(frames: int, q_chunk: int, max_vram: float) -> dict:
    samples_dir = pathlib.Path("benchmarks/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    work_dir = pathlib.Path("benchmarks/runs/quality_v10")
    work_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"wan22_rapid_v10_640x384_{frames}f_4step"
    lat_path = work_dir / f"{prefix}.lvlt"
    dummy_avi = work_dir / f"{prefix}_dummy.avi"
    out_avi = samples_dir / f"{prefix}.avi"
    p1_log = work_dir / f"{prefix}_dit.log"
    p2_log = work_dir / f"{prefix}_vae.log"

    cmd_p1 = [
        "build/sdcpp-vulkan/bin/sd-cli.exe",
        "-M", "vid_gen",
        "--diffusion-model", "models/diffusion_models/wan2.2-t2v-rapid-aio-v10-Q4_K.gguf",
        "--t5xxl", "models/text_encoders/umt5-xxl-encoder-Q4_K_M.gguf",
        "--taesd", "models/vae/taew2_1.safetensors",
        "-p", PROMPT,
        "-W", "640",
        "-H", "384",
        "--video-frames", str(frames),
        "--fps", "16",
        "--steps", "4",
        "--cfg-scale", "1.0",
        "--sampling-method", "euler_a",
        "--scheduler", "beta",
        "--flow-shift", "5.0",
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
        "SD_WAN_SPLIT_BLOCK": "1",
        "SD_WAN_ATTN_HEAD_CHUNK": "4",
        "SD_WAN_ATTN_HEAD_CUT": "1",
        "SD_WAN_MLP_CHUNK_MB": "128",
        "SD_VIDEO_LATENT_SAVE": str(lat_path),
        "SD_VIDEO_LATENT_SAVE_ONLY": "1",
    }
    if q_chunk > 0:
        env_p1["SD_WAN_ATTN_Q_CHUNK"] = str(q_chunk)

    print(f"\n=== RUN {prefix}: Process 1 (TE + 4-step DiT, q={q_chunk}, mv={max_vram}) ===", flush=True)
    rc1, wall1, ded1, shr1, txt1 = run_proc(cmd_p1, env_p1, p1_log)
    m_te = re.search(r"computing condition graph completed, taking ([\d.]+) ms", txt1)
    m_samp = re.search(r"sampling completed, taking ([\d.]+)s", txt1)
    dit_bufs = re.findall(r"(Wan2\.x-T2V-14B) compute buffer size:\s*([\d.]+) MB\(([^)]+)\)", txt1)
    errs1 = [l.strip() for l in txt1.splitlines() if "[E]" in l or "ErrorDeviceLost" in l][-6:]

    res: dict = {
        "prefix": prefix,
        "frames": frames,
        "q_chunk": q_chunk,
        "max_vram_gib": max_vram,
        "dit": {
            "rc": rc1,
            "wall_s": wall1,
            "te_s": round(float(m_te.group(1)) / 1000.0, 2) if m_te else None,
            "sampling_4step_s": float(m_samp.group(1)) if m_samp else None,
            "peak_ded_mib": ded1,
            "peak_shr_mib": shr1,
            "compute_buffers": dit_bufs,
            "errors": errs1,
            "log": str(p1_log),
        },
        "vae": None,
        "media": None,
    }

    if rc1 == 0 and lat_path.exists():
        dgl_script = work_dir / f"decode_{frames}f.dgl"
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
        print(f"=== RUN {prefix}: Process 2 (Wan 2.1 VAE, SD_CONV_DIRECT_TOL=0.005) ===", flush=True)
        rc2, wall2, ded2, shr2, txt2 = run_proc(cmd_p2, env_p2, p2_log)
        m_vae = re.search(r"computing vae decode graph completed, taking ([\d.]+)s", txt2)
        conv_lines = [l.strip() for l in txt2.splitlines() if "conv2d direct" in l or "conv3d direct" in l]
        errs2 = [l.strip() for l in txt2.splitlines() if "[E]" in l or "ErrorDeviceLost" in l][-6:]
        res["vae"] = {
            "rc": rc2,
            "wall_s": wall2,
            "decode_s": float(m_vae.group(1)) if m_vae else None,
            "peak_ded_mib": ded2,
            "peak_shr_mib": shr2,
            "conv_direct": conv_lines,
            "errors": errs2,
            "log": str(p2_log),
        }
        if rc2 == 0 and out_avi.exists():
            res["media"] = package_video_and_frames(out_avi, samples_dir, prefix)

    print(json.dumps(res, indent=2), flush=True)
    return res


def main() -> int:
    results = []
    # Run 1: 640x384 x 17f, 4 steps, q=off, mv=1.5
    results.append(run_quality_case(17, 0, 1.5))
    # Run 2: 640x384 x 33f, 4 steps, Q_CHUNK=2048, mv=2.0
    results.append(run_quality_case(33, 2048, 2.0))

    report_path = pathlib.Path("benchmarks/samples/quality_report.json")
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nQUALITY_VALIDATION_DONE -> {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
