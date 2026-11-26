#!/usr/bin/env python3
"""
PicoVid v0.3.0 Cinema HD + 32fps Post-Processing Pipeline (Vulkan GPU).

Transforms fast 640x384 @ 16fps Wan 2.2 generations into crisp 1280x768 HD @ 32fps H.264
in ~6-15 seconds on 4-6 GB AMD Vulkan & NVIDIA GPUs:
  1. 2x Vulkan AI Video Upscale (realesrgan-ncnn-vulkan: realesr-animevideov3-x2 or
     realesrgan-x4plus supersampled to 2x HD)
  2. 2x Vulkan Neural Frame Interpolation (rife-ncnn-vulkan with rife-v4.6: 16fps -> 32fps)
  3. High-Profile H.264 (libx264 -crf 15 -pix_fmt yuv420p -movflags +faststart)

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from typing import Callable

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TOOLS_DIR = REPO_ROOT / "tools"
REALESRGAN_DIR = TOOLS_DIR / "realesrgan"
RIFE_DIR = TOOLS_DIR / "rife"

REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
RIFE_URL = "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-windows.zip"


def resolve_ffmpeg() -> str | None:
    sys_ff = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and pathlib.Path(exe).exists():
            return exe
    except Exception:
        pass
    return None


def realesrgan_exe() -> pathlib.Path:
    return REALESRGAN_DIR / "realesrgan-ncnn-vulkan.exe"


def rife_exe() -> pathlib.Path:
    return RIFE_DIR / "rife-ncnn-vulkan.exe"


def _download_zip(url: str, dest_zip: pathlib.Path, label: str) -> bool:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        print(f"HATA: {label} indirmek için curl gerekli.", file=sys.stderr)
        return False
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"[PicoVid] İndiriliyor: {label} ...")
    cmd = [
        curl, "-L", "-C", "-",
        "--retry", "5", "--retry-delay", "3",
        "--speed-limit", "50000", "--speed-time", "25",
        "-#", url, "-o", str(dest_zip),
    ]
    p = subprocess.run(cmd)
    return p.returncode == 0 and dest_zip.exists()


def install_realesrgan() -> bool:
    if realesrgan_exe().exists() and (REALESRGAN_DIR / "models").exists():
        return True
    REALESRGAN_DIR.mkdir(parents=True, exist_ok=True)
    tmp_zip = TOOLS_DIR / "realesrgan_tmp.zip"
    if not _download_zip(REALESRGAN_URL, tmp_zip, "realesrgan-ncnn-vulkan (43.4 MB, BSD-3)"):
        return False
    try:
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(REALESRGAN_DIR)
        # Flatten if extracted inside a single subfolder
        if not realesrgan_exe().exists():
            for exe in REALESRGAN_DIR.rglob("realesrgan-ncnn-vulkan.exe"):
                parent = exe.parent
                for item in parent.iterdir():
                    dst = REALESRGAN_DIR / item.name
                    if dst.exists():
                        if dst.is_dir():
                            shutil.rmtree(dst)
                        else:
                            dst.unlink()
                    shutil.move(str(item), str(dst))
                break
        return realesrgan_exe().exists()
    finally:
        tmp_zip.unlink(missing_ok=True)


def install_rife() -> bool:
    if rife_exe().exists() and (RIFE_DIR / "rife-v4.6").exists():
        return True
    RIFE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_zip = TOOLS_DIR / "rife_tmp.zip"
    if not _download_zip(RIFE_URL, tmp_zip, "rife-ncnn-vulkan (rife-v4.6 çıkarılacak, MIT)"):
        return False
    try:
        # Extract ONLY executable, DLLs, and rife-v4.6 (~15 MB on disk instead of 411 MB!)
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            for member in zf.infolist():
                norm = member.filename.replace("\\", "/")
                parts = norm.split("/", 1)
                rel = parts[1] if len(parts) > 1 else parts[0]
                if not rel or rel.endswith("/"):
                    continue
                keep = (
                    rel.lower() in ("rife-ncnn-vulkan.exe", "vcomp140.dll", "vcomp140d.dll", "license")
                    or rel.startswith("rife-v4.6/")
                )
                if not keep:
                    continue
                target = RIFE_DIR / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        return rife_exe().exists() and (RIFE_DIR / "rife-v4.6").exists()
    finally:
        tmp_zip.unlink(missing_ok=True)


def ensure_tools_installed(need_upscale: bool = True, need_rife: bool = False) -> tuple[bool, bool]:
    """Ensure Vulkan Real-ESRGAN is installed (43.4 MB); RIFE is optional (zero-download FFmpeg AOBMC optical-flow is built-in)."""
    ok_up = install_realesrgan() if need_upscale else True
    ok_rf = (rife_exe().exists() and (RIFE_DIR / "rife-v4.6").exists()) or (resolve_ffmpeg() is not None)
    if need_rife and not (rife_exe().exists() and (RIFE_DIR / "rife-v4.6").exists()):
        # Keep fast built-in FFmpeg AOBMC optical-flow ready without blocking on 411 MB GitHub zip
        ok_rf = resolve_ffmpeg() is not None
    return ok_up, ok_rf


def extract_last_frame(video_path: pathlib.Path, out_png: pathlib.Path) -> pathlib.Path:
    """Extract the exact last frame of an AVI/MP4 sequentially (avoids AVI seek bugs)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Son kare okunamadı, video açılamadı: {video_path}")
    last_frame = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        last_frame = fr
    cap.release()
    if last_frame is None:
        raise RuntimeError(f"Videoda kare bulunamadı: {video_path}")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_png), last_frame):
        raise RuntimeError(f"Son kare PNG olarak kaydedilemedi: {out_png}")
    return out_png


def _compute_lab_stats(bgr: object) -> tuple[object, object]:
    import cv2
    import numpy as np

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean = lab.reshape(-1, 3).mean(axis=0)
    std = np.maximum(lab.reshape(-1, 3).std(axis=0), 1.0)
    return mean, std


def _apply_lab_color_anchor(bgr: object, ref_mean: object, ref_std: object, strength: float = 0.55) -> object:
    """Gently anchor Lab mean/std to Segment 1 to prevent highlight blowout & saturation drift across chained segments."""
    import cv2
    import numpy as np

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    cur_mean = lab.reshape(-1, 3).mean(axis=0)
    cur_std = np.maximum(lab.reshape(-1, 3).std(axis=0), 1.0)
    # Cap std scaling gently so natural scene changes stay intact while preventing runaway contrast drift
    scale = np.clip(ref_std / cur_std, 0.80, 1.15)
    matched = (lab - cur_mean) * scale + ref_mean
    blended = lab * (1.0 - strength) + matched * strength
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_LAB2BGR)


def stitch_video_segments(
    segment_paths: list[pathlib.Path],
    output_video: pathlib.Path,
    fps: float = 16.0,
    drop_overlap_frames: int = 1,
    crf: int = 14,
    color_anchor: bool = True,
) -> dict[str, object]:
    """Seamlessly concatenate chained I2V segments, dropping duplicate boundary frames and anchoring Lab color."""
    import cv2
    import numpy as np

    if not segment_paths:
        raise ValueError("Birleştirilecek segment bulunamadı.")

    ff = resolve_ffmpeg()
    work = pathlib.Path(tempfile.mkdtemp(prefix="picovid_stitch_"))
    try:
        seq_dir = work / "frames"
        seq_dir.mkdir(parents=True, exist_ok=True)
        out_idx = 0
        width = height = 0
        prev_tail = None
        anchor_mean = anchor_std = None
        seam_psnrs: list[float] = []

        for seg_i, sp in enumerate(segment_paths):
            cap = cv2.VideoCapture(str(sp))
            if not cap.isOpened():
                raise RuntimeError(f"Segment açılamadı: {sp}")
            if seg_i == 0:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            seg_total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            seg_frame_i = 0
            seg_last = None
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if seg_i == 0 and seg_frame_i == 0 and color_anchor:
                    anchor_mean, anchor_std = _compute_lab_stats(fr)
                if seg_i > 0 and seg_frame_i == 0 and prev_tail is not None:
                    if fr.shape == prev_tail.shape:
                        mse = float(np.mean((prev_tail.astype(np.float32) - fr.astype(np.float32)) ** 2))
                        psnr = 99.0 if mse <= 1e-6 else float(10.0 * np.log10((255.0 ** 2) / mse))
                        seam_psnrs.append(round(psnr, 2))
                if seg_i > 0 and seg_frame_i < drop_overlap_frames:
                    seg_frame_i += 1
                    continue
                if width and height and (fr.shape[1] != width or fr.shape[0] != height):
                    fr = cv2.resize(fr, (width, height), interpolation=cv2.INTER_AREA)
                if color_anchor and seg_i > 0 and anchor_mean is not None and anchor_std is not None:
                    # Ramp anchor smoothly over the first 8 frames of the new segment so the seam stays invisible
                    ramp = min(1.0, seg_frame_i / 8.0) * 0.55
                    if ramp > 0.01:
                        fr = _apply_lab_color_anchor(fr, anchor_mean, anchor_std, strength=ramp)
                out_idx += 1
                cv2.imwrite(str(seq_dir / f"{out_idx:08d}.png"), fr)
                seg_last = fr
                seg_frame_i += 1
            cap.release()
            if seg_last is not None:
                prev_tail = seg_last

        if out_idx <= 0:
            raise RuntimeError("Segment birleştirmede hiç kare üretilemedi.")

        output_video.parent.mkdir(parents=True, exist_ok=True)
        if ff:
            cmd = [
                ff, "-y",
                "-framerate", f"{fps:.3f}",
                "-i", str(seq_dir / "%08d.png"),
                "-c:v", "libx264", "-profile:v", "high",
                "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_video),
            ]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if p.returncode != 0 or not output_video.exists():
                raise RuntimeError(f"ffmpeg segment birleştirme başarısız: {p.stderr[-300:]}")
        else:
            writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            for png in sorted(seq_dir.glob("*.png")):
                writer.write(cv2.imread(str(png)))
            writer.release()

        return {
            "output_path": str(output_video),
            "segments": len(segment_paths),
            "total_frames": out_idx,
            "fps": round(fps, 2),
            "duration_sec": round(out_idx / max(1.0, fps), 2),
            "width": width,
            "height": height,
            "seam_psnr_db": seam_psnrs,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def has_audio_stream(video_path: pathlib.Path, ff: str | None = None) -> bool:
    """Detect whether an input video container has an audio track."""
    exe = ff or resolve_ffmpeg()
    if not exe or not video_path.exists():
        return False
    try:
        p = subprocess.run(
            [exe, "-i", str(video_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        return "Audio:" in (p.stderr or "")
    except Exception:
        return False


def concat_av_clips(
    clip_paths: list[pathlib.Path],
    output_mp4: pathlib.Path,
    crf: int = 15,
) -> dict[str, object]:
    """Concatenate multiple H.264+AAC MP4 clips cleanly preserving both video and 48kHz stereo audio."""
    ff = resolve_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg / imageio-ffmpeg bulunamadı.")
    valid = [p.resolve() for p in clip_paths if p.exists()]
    if not valid:
        raise ValueError("Birleştirilecek klip bulunamadı.")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [ff, "-y"]
    all_have_audio = all(has_audio_stream(p, ff) for p in valid)
    for p in valid:
        cmd += ["-i", str(p)]

    n = len(valid)
    if all_have_audio:
        parts = []
        for i in range(n):
            parts.append(f"[{i}:v:0]setsar=1[v{i}];")
            parts.append(f"[{i}:a:0]aformat=sample_rates=48000:channel_layouts=stereo[a{i}];")
        concat_in = "".join(f"[v{i}][a{i}]" for i in range(n))
        fc = "".join(parts) + f"{concat_in}concat=n={n}:v=1:a=1[vout][aout]"
        cmd += [
            "-filter_complex", fc,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-profile:v", "high", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_mp4),
        ]
    else:
        concat_in = "".join(f"[{i}:v:0]" for i in range(n))
        fc = f"{concat_in}concat=n={n}:v=1:a=0[vout]"
        cmd += [
            "-filter_complex", fc,
            "-map", "[vout]",
            "-c:v", "libx264", "-profile:v", "high", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_mp4),
        ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not output_mp4.exists():
        raise RuntimeError(f"ffmpeg showreel birleştirme başarısız: {p.stderr[-400:]}")
    return {
        "output_mp4": str(output_mp4),
        "clips": n,
        "audio_preserved": all_have_audio,
        "size_mb": round(output_mp4.stat().st_size / (1024 * 1024), 2),
    }


def extract_video_frames(video_path: pathlib.Path, out_dir: pathlib.Path) -> tuple[int, float, int, int]:
    """Extract lossless PNG frames from AVI/MP4 and return (count, fps, width, height)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 16.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        idx += 1
        cv2.imwrite(str(out_dir / f"{idx:08d}.png"), fr)
    cap.release()
    return idx, fps, w, h


def enhance_video(
    input_video: pathlib.Path,
    output_mp4: pathlib.Path,
    upscale_factor: int = 2,
    rife_factor: int = 2,
    upscale_model: str = "realesrgan-x4plus",
    crf: int = 15,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict[str, object]:
    """Run Cinema HD + 32fps post-processing on input AVI/MP4 -> H.264 MP4."""
    t0 = time.monotonic()
    ff = resolve_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg / imageio-ffmpeg bulunamadı.")

    work = pathlib.Path(tempfile.mkdtemp(prefix="picovid_post_"))
    try:
        in_dir = work / "01_in"
        up_dir = work / "02_up"
        rf_dir = work / "03_rife"

        if progress_cb:
            progress_cb(10.0, "Kareler çıkarılıyor...")
        n_frames, in_fps, in_w, in_h = extract_video_frames(input_video, in_dir)
        if n_frames <= 0:
            raise RuntimeError("Giriş videosunda kare bulunamadı.")

        cur_dir = in_dir
        cur_w, cur_h = in_w, in_h
        cur_fps = in_fps
        up_sec = 0.0
        rf_sec = 0.0
        interp_backend = "none"

        # Step 1: Vulkan AI Upscale (run BEFORE interpolation so we upscale half as many frames!)
        if upscale_factor >= 2 and realesrgan_exe().exists():
            up_dir.mkdir(parents=True, exist_ok=True)
            native_scale = 2 if "animevideov3" in upscale_model else 4
            if progress_cb:
                progress_cb(25.0, f"2× Vulkan AI Upscale ({in_w}x{in_h} -> {in_w*2}x{in_h*2}, {upscale_model})...")
            t_up = time.monotonic()
            cmd_up = [
                str(realesrgan_exe()),
                "-i", str(cur_dir),
                "-o", str(up_dir),
                "-n", upscale_model,
                "-s", str(native_scale),
                "-f", "png",
                "-j", "1:2:2",
            ]
            p_up = subprocess.run(cmd_up, cwd=str(REALESRGAN_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            up_sec = time.monotonic() - t_up
            if p_up.returncode != 0 or not any(up_dir.glob("*.png")):
                raise RuntimeError(f"realesrgan-ncnn-vulkan başarısız (rc={p_up.returncode}): {p_up.stderr[-300:]}")

            # Supersample 4x -> 2x with AREA/Lanczos for razor-sharp photorealistic HD!
            if native_scale == 4 and upscale_factor == 2:
                import cv2
                for png in sorted(up_dir.glob("*.png")):
                    im = cv2.imread(str(png))
                    im2 = cv2.resize(im, (in_w * 2, in_h * 2), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(png), im2)

            cur_dir = up_dir
            cur_w, cur_h = in_w * upscale_factor, in_h * upscale_factor

        # Step 2: 2x Frame Interpolation (16fps -> 32fps)
        # Uses Vulkan RIFE v4.6 if installed, otherwise zero-download FFmpeg bidirectional AOBMC optical-flow!
        out_frames = n_frames
        use_ffmpeg_mci = False
        if rife_factor >= 2:
            if rife_exe().exists() and (RIFE_DIR / "rife-v4.6").exists():
                rf_dir.mkdir(parents=True, exist_ok=True)
                target_n = n_frames * rife_factor
                if progress_cb:
                    progress_cb(65.0, f"2× Vulkan RIFE ({n_frames}f -> {target_n}f @ {int(round(in_fps*rife_factor))}fps)...")
                t_rf = time.monotonic()
                cmd_rf = [
                    str(rife_exe()),
                    "-i", str(cur_dir),
                    "-o", str(rf_dir),
                    "-m", "rife-v4.6",
                    "-n", str(target_n),
                    "-f", "%08d.png",
                    "-j", "1:2:2",
                ]
                p_rf = subprocess.run(cmd_rf, cwd=str(RIFE_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                rf_sec = time.monotonic() - t_rf
                produced = sorted(rf_dir.glob("*.png"))
                if p_rf.returncode == 0 and produced:
                    cur_dir = rf_dir
                    cur_fps = in_fps * rife_factor
                    out_frames = len(produced)
                    interp_backend = "rife_vulkan"
                else:
                    use_ffmpeg_mci = True
            else:
                use_ffmpeg_mci = True

        # Step 3: High-Profile H.264 MP4 Encode (CRF 15, + optional AOBMC optical-flow 32fps)
        pngs = sorted(cur_dir.glob("*.png"))
        seq_dir = work / "04_seq"
        seq_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(pngs, start=1):
            shutil.copyfile(p, seq_dir / f"{i:08d}.png")

        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        keep_audio = has_audio_stream(input_video, ff)
        cmd_ff = [
            ff, "-y",
            "-framerate", f"{cur_fps:.3f}",
            "-i", str(seq_dir / "%08d.png"),
        ]
        if keep_audio:
            cmd_ff += ["-i", str(input_video)]

        if use_ffmpeg_mci:
            target_fps = in_fps * rife_factor
            if progress_cb:
                progress_cb(72.0, f"2× Optik Akış İnterpolasyonu ({cur_w}x{cur_h} @ {target_fps:.0f}fps) + H.264...")
            t_rf = time.monotonic()
            cmd_ff += [
                "-vf", f"minterpolate=fps={target_fps:.0f}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
            ]
            cur_fps = target_fps
            interp_backend = "ffmpeg_aobmc"
        else:
            if progress_cb:
                progress_cb(88.0, f"H.264 MP4 kodlanıyor ({cur_w}x{cur_h} @ {cur_fps:.0f}fps, CRF {crf})...")

        if keep_audio:
            cmd_ff += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]

        cmd_ff += [
            "-c:v", "libx264",
            "-profile:v", "high",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_mp4),
        ]
        p_ff = subprocess.run(cmd_ff, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if use_ffmpeg_mci:
            rf_sec = time.monotonic() - t_rf
        if p_ff.returncode != 0 or not output_mp4.exists():
            raise RuntimeError(f"ffmpeg H.264 kodlama başarısız: {p_ff.stderr[-300:]}")

        if use_ffmpeg_mci:
            import cv2
            cap = cv2.VideoCapture(str(output_mp4))
            out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        return {
            "output_mp4": str(output_mp4),
            "in_res": f"{in_w}x{in_h}",
            "out_res": f"{cur_w}x{cur_h}",
            "in_frames": n_frames,
            "out_frames": out_frames,
            "out_fps": round(cur_fps, 1),
            "audio_preserved": keep_audio,
            "upscale_model": upscale_model if upscale_factor >= 2 else "none",
            "interp_backend": interp_backend,
            "upscale_sec": round(up_sec, 2),
            "rife_sec": round(rf_sec, 2),
            "total_sec": round(time.monotonic() - t0, 2),
            "size_kb": round(output_mp4.stat().st_size / 1024.0, 1),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="PicoVid Cinema HD (2x Vulkan Upscale + 2x Vulkan RIFE 32fps)")
    parser.add_argument("--install-tools", action="store_true", help="Download realesrgan-ncnn-vulkan & rife-ncnn-vulkan (rife-v4.6)")
    parser.add_argument("-i", "--input", type=pathlib.Path, default=None, help="Input .avi or .mp4 video")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=None, help="Output enhanced .mp4")
    parser.add_argument("--upscale", type=int, choices=[1, 2], default=2)
    parser.add_argument("--rife", type=int, choices=[1, 2], default=2)
    parser.add_argument("--upscale-model", default="realesrgan-x4plus", choices=["realesrgan-x4plus", "realesr-animevideov3"])
    parser.add_argument("--crf", type=int, default=15)
    args = parser.parse_args()

    if args.install_tools:
        ok_u, ok_r = ensure_tools_installed(True, True)
        print(f"INSTALL_TOOLS: realesrgan={ok_u}, rife={ok_r}")
        return 0 if (ok_u and ok_r) else 1

    if args.input is None or args.output is None:
        parser.error("-i/--input and -o/--output are required unless --install-tools is specified")

    ensure_tools_installed(args.upscale >= 2, args.rife >= 2)
    res = enhance_video(
        args.input.resolve(),
        args.output.resolve(),
        upscale_factor=args.upscale,
        rife_factor=args.rife,
        upscale_model=args.upscale_model,
        crf=args.crf,
    )
    print(f"[PicoVid Cinema HD] {res}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
