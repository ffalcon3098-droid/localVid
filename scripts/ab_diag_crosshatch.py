#!/usr/bin/env python3
"""
A/B Diagnostic for Diagonal Cross-Hatch Artifact:
- Test A1: Decode saved 17f float32 latent with stock Wan 2.1 VAE (SD_CONV_DIRECT_TOL unset -> conv2d/3d direct OFF)
- Test A2: Decode saved 17f float32 latent with TAEHV (taew2_1.safetensors)
- Optional Test B (--run-test-b): Re-run 17f DiT with --sampling-method euler --scheduler discrete --flow-shift 5.0 (no "film grain")
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

import cv2


def extract_frame_12(avi_path: pathlib.Path, out_png: pathlib.Path) -> bool:
    cap = cv2.VideoCapture(str(avi_path))
    idx = 0
    saved = False
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx == 12:
            cv2.imwrite(str(out_png), fr)
            saved = True
            break
        idx += 1
    cap.release()
    return saved


def run_decode(
    name: str,
    loader: str,
    model_path: str,
    lat_path: pathlib.Path,
    ab_dir: pathlib.Path,
    conv_direct_tol: str | None = None,
    apply_dit_to_vae: bool = False,
) -> pathlib.Path | None:
    out_avi = ab_dir / f"{name}.avi"
    dgl = ab_dir / f"{name}.dgl"
    dgl.write_text(
        f"""# version: 0.0.0
var {{
    vae_path = "{model_path}"
    output_path = "{out_avi.as_posix()}"
    fps = 16
}}
fn main() {{
    vae = {loader}(vae_path, family = "wan2", params_backend = "cpu")
    dummy_latent = tensor::zeros([8, 8, 1, 16, 1])
    output_frames = vae_decode(vae, dummy_latent)
    video::save(output_path, {{frames = output_frames, fps_num = fps, fps_den = 1}})
    return {{video = {{frames = output_frames, fps_num = fps, fps_den = 1}}}}
}}
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    if conv_direct_tol:
        env["SD_CONV_DIRECT_TOL"] = conv_direct_tol
    else:
        env.pop("SD_CONV_DIRECT_TOL", None)
    if apply_dit_to_vae:
        env["SD_VIDEO_LATENT_DIT_TO_VAE"] = "1"
    else:
        env.pop("SD_VIDEO_LATENT_DIT_TO_VAE", None)
    env["SD_LOADER_FAST_POLL"] = "1"
    env["SD_VIDEO_LATENT_LOAD"] = str(lat_path)
    log = ab_dir / f"{name}.log"
    p = subprocess.run(
        ["build/sdcpp-vulkan/bin/sd-cli.exe", "-M", "dgl", "--script", str(dgl), "-v"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log.write_text(p.stdout, encoding="utf-8")
    print(f"=== {name} rc={p.returncode} ===", flush=True)
    for line in p.stdout.splitlines():
        if any(k in line for k in ("DIT_TO_VAE", "conv2d direct", "conv3d direct", "completed, taking", "[E]")):
            print(" ", line.strip(), flush=True)
    if p.returncode == 0 and out_avi.exists():
        png = ab_dir / f"frame_12_{name}.png"
        if extract_frame_12(out_avi, png):
            print(f"  Saved: {png}", flush=True)
        return out_avi
    return None


def run_test_b(ab_dir: pathlib.Path) -> None:
    prompt_clean = (
        "A cinematic tracking shot of a vintage red sports car driving down a rain-slicked neon-lit city street "
        "at sunset, golden light reflecting off wet asphalt and puddles, soft mist, realistic motion, sharp 35mm lens"
    )
    lat_path = ab_dir / "euler_discrete_17f.lvlt"
    dummy_avi = ab_dir / "euler_discrete_dummy.avi"
    cmd = [
        "build/sdcpp-vulkan/bin/sd-cli.exe",
        "-M", "vid_gen",
        "--diffusion-model", "models/diffusion_models/wan2.2-t2v-rapid-aio-v10-Q4_K.gguf",
        "--t5xxl", "models/text_encoders/umt5-xxl-encoder-Q4_K_M.gguf",
        "--taesd", "models/vae/taew2_1.safetensors",
        "-p", prompt_clean,
        "-W", "640", "-H", "384", "--video-frames", "17", "--fps", "16",
        "--steps", "4", "--cfg-scale", "1.0",
        "--sampling-method", "euler", "--scheduler", "discrete", "--flow-shift", "5.0",
        "--seed", "42", "--offload-to-cpu", "--mmap", "--backend", "te=cpu",
        "--flash-attn", "off", "--max-vram", "1.5", "-v", "-o", str(dummy_avi),
    ]
    env = os.environ.copy()
    env.update(
        {
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
    )
    log = ab_dir / "test_b_dit.log"
    print("\n=== TEST B (euler + discrete, no film grain): Process 1 DiT ===", flush=True)
    p = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
    )
    log.write_text(p.stdout, encoding="utf-8")
    print(f"Test B DiT rc={p.returncode}", flush=True)
    for line in p.stdout.splitlines():
        if any(k in line for k in ("sampling completed", "[E]")):
            print(" ", line.strip(), flush=True)
    if p.returncode == 0 and lat_path.exists():
        run_decode("euler_vae_im2col", "load_vae", "models/vae/wan_2.1_vae.safetensors", lat_path, ab_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-test-b", action="store_true")
    parser.add_argument("--fixed-norm", action="store_true")
    args = parser.parse_args()

    ab_dir = pathlib.Path("benchmarks/samples/ab_test")
    ab_dir.mkdir(parents=True, exist_ok=True)
    lat = pathlib.Path("benchmarks/runs/quality_v10/wan22_rapid_v10_640x384_17f_4step.lvlt")

    if args.fixed_norm:
        assert lat.exists(), f"Missing latent: {lat}"
        run_decode(
            "vae_fixed_norm_direct",
            "load_vae",
            "models/vae/wan_2.1_vae.safetensors",
            lat,
            ab_dir,
            conv_direct_tol="0.005",
            apply_dit_to_vae=True,
        )
    elif not args.run_test_b:
        assert lat.exists(), f"Missing latent: {lat}"
        run_decode("vae_im2col", "load_vae", "models/vae/wan_2.1_vae.safetensors", lat, ab_dir)
        run_decode("taehv", "load_tae", "models/vae/taew2_1.safetensors", lat, ab_dir)
    else:
        run_test_b(ab_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
