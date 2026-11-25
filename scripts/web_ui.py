#!/usr/bin/env python3
"""
PicoVid v0.4.0 Local Web Studio (Zero external web dependencies — pure Python stdlib).

Launches a modern dark Cinema Studio web app on http://127.0.0.1:7860 featuring:
  - Live GPU Telemetry (Dedicated + Shared VRAM bars, backend, hardware class)
  - 3 Modes: Studio (Z-Image-Turbo First Frame -> Preview/Reroll -> Animate), T2V, and Drag-and-Drop I2V
  - Dual Engine selector: Wan 2.2 TI2V-5B Turbo (fast 5s-15s) & Wan 2.2 14B (ultra detail)
  - Continuous Multi-Segment Chaining (--segments 1..3 + per-segment director prompts)
  - Cinema HD 32fps & Ultra-Fast TAEW2.2 Preview toggles + 1-click Motion Presets
  - Live 4-stage progress stream, Stop/Cancel control, and built-in Output Video Gallery

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
UPLOADS_DIR = OUTPUTS_DIR / "_uploads"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import download_models  # noqa: E402
import generate  # noqa: E402
import ui_console  # noqa: E402


def get_models_status() -> dict[str, Any]:
    """Check installed vs missing status for key PicoVid profiles."""
    reg = download_models.load_registry()
    files_map = reg.get("files", {})
    profiles = reg.get("download_profiles", {})
    out: dict[str, Any] = {}
    for pname in ("fast-5b", "t2i", "t2v", "i2v", "studio"):
        if pname not in profiles:
            continue
        req = profiles[pname]["required"]
        rem = sum(download_models.check_file(files_map[fid])[2] for fid in req if fid in files_map)
        tot = sum(int(files_map[fid]["expected_bytes"]) for fid in req if fid in files_map)
        out[pname] = {
            "ready": rem == 0,
            "remaining_gib": round(rem / (1024**3), 2),
            "total_gib": round(tot / (1024**3), 2),
        }
    return out


class StudioState:
    """Thread-safe job runner and live status tracker for PicoVid Web Studio."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.hw: dict[str, Any] = generate.detect_hardware()
        self.vram_sampler = ui_console.LiveVramSampler(total_vram_gib=self.hw["vram_total_gib"], interval_sec=1.5)
        self.vram_sampler.start()
        self.job_running: bool = False
        self.job_kind: str = ""  # "t2i" | "video"
        self.job_id: str = ""
        self.started_at: float = 0.0
        self.stage: str = "Hazır"
        self.progress_pct: float = 0.0
        self.detail: str = "Yeni üretim için hazır."
        self.logs: list[str] = []
        self.last_t2i_png: str | None = None
        self.last_t2i_seed: int | None = None
        self.last_video_rel: str | None = None
        self.last_error: str | None = None
        self.proc: subprocess.Popen[str] | None = None
        self._cancel_requested: bool = False

    def shutdown(self) -> None:
        self.cancel_job()
        self.vram_sampler.stop()

    def snapshot(self) -> dict[str, Any]:
        snap_vram = self.vram_sampler.snapshot()
        with self.lock:
            elapsed = round(time.monotonic() - self.started_at, 1) if self.job_running and self.started_at else 0.0
            return {
                "running": self.job_running,
                "kind": self.job_kind,
                "job_id": self.job_id,
                "elapsed_sec": elapsed,
                "stage": self.stage,
                "progress_pct": round(self.progress_pct, 1),
                "detail": self.detail,
                "logs": self.logs[-40:],
                "last_t2i_url": f"/files/{urllib.parse.quote(self.last_t2i_png)}" if self.last_t2i_png else None,
                "last_t2i_path": self.last_t2i_png,
                "last_t2i_seed": self.last_t2i_seed,
                "last_video_url": f"/files/{urllib.parse.quote(self.last_video_rel)}" if self.last_video_rel else None,
                "last_video_path": self.last_video_rel,
                "last_error": self.last_error,
                "vram": {
                    "ded_mib": snap_vram.get("current_ded_mib"),
                    "shr_mib": snap_vram.get("current_shr_mib"),
                    "peak_ded_mib": snap_vram.get("peak_ded_mib"),
                    "peak_shr_mib": snap_vram.get("peak_shr_mib"),
                    "total_mib": snap_vram.get("total_vram_mib") or round(self.hw["vram_total_gib"] * 1024.0, 1),
                },
            }

    def cancel_job(self) -> bool:
        with self.lock:
            if not self.job_running:
                return False
            self._cancel_requested = True
            proc = self.proc
            self.stage = "İptal ediliyor..."
            self.detail = "Çalışan süreç durduruluyor..."
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        if os.name == "nt":
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-Process -Name 'sd-cli' -ErrorAction SilentlyContinue | Stop-Process -Force"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return True

    def _append_log(self, line: str) -> None:
        clean = ui_console.RE_ANSI.sub("", line).strip()
        if not clean:
            return
        with self.lock:
            self.logs.append(clean)
            if len(self.logs) > 250:
                self.logs = self.logs[-250:]

            # Lightweight live stage & progress parser
            if "[PicoVid Zincir] Bölüm" in clean:
                self.stage = clean.replace("[PicoVid Zincir]", "").strip()
                self.detail = clean
            elif "Metin Kodlama" in clean or "computing condition graph" in clean:
                self.stage = "1/4 Metin & Görsel Kodlama"
                self.progress_pct = max(self.progress_pct, 12.0)
                self.detail = clean
            elif "sampling started:" in clean or "DiT Örnekleme" in clean:
                self.stage = "2/4 DiT Video Üretimi"
                self.progress_pct = max(self.progress_pct, 22.0)
                self.detail = clean
            elif "VAE Decode" in clean or "SD_VIDEO_LATENT_LOAD" in clean:
                self.stage = "3/4 Wan VAE Decode"
                self.progress_pct = max(self.progress_pct, 58.0)
                self.detail = clean
            elif "Sinema HD" in clean or "birleştiriliyor" in clean or "MP4 Paketleme" in clean:
                self.stage = "4/4 Sinema HD & H.264 Paketleme"
                self.progress_pct = max(self.progress_pct, 88.0)
                self.detail = clean

            m_pb = ui_console.RE_PROGRESS_BAR.search(clean)
            if m_pb and "=" in line:
                cur_i, tot_i = int(m_pb.group(1)), max(1, int(m_pb.group(2)))
                frac = cur_i / tot_i
                if "2/4" in self.stage:
                    self.progress_pct = 22.0 + frac * 35.0
                    self.detail = f"DiT Adım {cur_i}/{tot_i} ({m_pb.group(3)} {m_pb.group(4)})"
                elif "3/4" in self.stage:
                    self.progress_pct = 58.0 + frac * 30.0
                    self.detail = f"VAE Karo {cur_i}/{tot_i} ({m_pb.group(3)} {m_pb.group(4)})"
                elif self.job_kind == "t2i":
                    self.progress_pct = 15.0 + frac * 80.0
                    self.detail = f"Z-Image-Turbo Adım {cur_i}/{tot_i} ({m_pb.group(3)} {m_pb.group(4)})"

    def start_job(self, kind: str, cmd: list[str], out_rel: str, seed: int | None = None) -> tuple[bool, str]:
        with self.lock:
            if self.job_running:
                return False, "Zaten aktif bir üretim çalışıyor. Önce durdurun veya bitmesini bekleyin."
            self.job_running = True
            self.job_kind = kind
            self.job_id = datetime.now().strftime("%H%M%S")
            self.started_at = time.monotonic()
            self.stage = "İlk Kare Üretiliyor (Z-Image-Turbo)..." if kind == "t2i" else "Video Motoru Başlatılıyor..."
            self.progress_pct = 5.0
            self.detail = "Alt süreç başlatılıyor..."
            self.logs = [f"$ {' '.join(cmd)}"]
            self.last_error = None
            self._cancel_requested = False

        def _runner() -> None:
            rc = 1
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=0,
                )
                with self.lock:
                    self.proc = proc
                assert proc.stdout is not None
                for line in ui_console.iter_stream_lines(proc.stdout):
                    self._append_log(line)
                rc = proc.wait()
            except Exception as exc:
                self._append_log(f"HATA: {exc}")
                rc = 1
            finally:
                with self.lock:
                    was_cancel = self._cancel_requested
                    self.proc = None
                    self.job_running = False
                    out_abs = REPO_ROOT / out_rel
                    if was_cancel:
                        self.stage = "İptal Edildi"
                        self.detail = "Üretim kullanıcı tarafından durduruldu."
                    elif rc == 0 and out_abs.exists():
                        self.progress_pct = 100.0
                        self.stage = "Tamamlandı"
                        el = time.monotonic() - self.started_at
                        self.detail = f"Başarıyla tamamlandı ({el:.1f} sn): {out_abs.name}"
                        if kind == "t2i":
                            self.last_t2i_png = out_rel.replace("\\", "/")
                            self.last_t2i_seed = seed
                        else:
                            self.last_video_rel = out_rel.replace("\\", "/")
                    else:
                        self.stage = "Hata"
                        self.last_error = f"Süreç çıkış kodu {rc} ile sonlandı."
                        self.detail = self.last_error

        threading.Thread(target=_runner, name="picovid_web_job", daemon=True).start()
        return True, "Başlatıldı"


def list_gallery_items() -> list[dict[str, Any]]:
    """Return recent generated MP4/PNG items from outputs/ and docs/media/."""
    items: list[dict[str, Any]] = []
    search_dirs = [OUTPUTS_DIR, REPO_ROOT / "benchmarks" / "runs" / "v03_5b"]
    for base in search_dirs:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or "_uploads" in p.parts:
                continue
            ext = p.suffix.lower()
            if ext not in (".mp4", ".png"):
                continue
            if p.name.startswith(("frame_", "chain_f", "test_")):
                continue
            rel = p.relative_to(REPO_ROOT).as_posix()
            st = p.stat()
            items.append({
                "name": p.name,
                "rel": rel,
                "url": f"/files/{urllib.parse.quote(rel)}",
                "kind": "video" if ext == ".mp4" else "image",
                "size_mb": round(st.st_size / (1024 * 1024), 2),
                "mtime": st.st_mtime,
            })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items[:24]


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PicoVid v0.4.0 · AI Video Studio</title>
<style>
:root{--bg:#090d16;--panel:#111827;--panel2:#162033;--border:#24324a;--accent:#38bdf8;--accent2:#818cf8;--ok:#34d399;--warn:#fbbf24;--err:#f87171;--text:#f1f5f9;--muted:#94a3b8}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0%,#132038,#090d16 55%);color:var(--text);font-family:Inter,Segoe UI,system-ui,sans-serif;min-height:100vh}
header{display:flex;align-items:center;justify-content:space-between;padding:14px 24px;border-bottom:1px solid var(--border);background:rgba(17,24,39,.85);backdrop-filter:blur(10px);position:sticky;top:0;z-index:10;gap:16px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.logo{font-size:22px;font-weight:800;letter-spacing:-.5px;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.badge{font-size:11px;padding:3px 8px;border-radius:999px;background:rgba(56,189,248,.14);color:var(--accent);border:1px solid rgba(56,189,248,.3)}
.mbadge{font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid var(--border);color:var(--muted);background:#0f172a;cursor:pointer}
.mbadge.ok{color:var(--ok);border-color:rgba(52,211,153,.35);background:rgba(52,211,153,.1);cursor:default}
.mbadge.miss{color:var(--warn);border-color:rgba(251,191,36,.4);background:rgba(251,191,36,.1)}
.gfoot{display:flex;align-items:center;justify-content:space-between;gap:6px;padding:6px 8px}
.gfoot .gmeta{padding:0;flex:1;min-width:0}
.ext-btn{border:1px solid rgba(56,189,248,.4);background:rgba(56,189,248,.12);color:var(--accent);border-radius:6px;padding:3px 7px;font-size:10px;font-weight:700;cursor:pointer;white-space:nowrap}
.ext-btn:hover{background:rgba(56,189,248,.25);color:#fff}
.vram-strip{display:flex;align-items:center;gap:14px;font-size:12px;color:var(--muted)}
.vbar{width:110px;height:8px;background:#1e293b;border-radius:999px;overflow:hidden}.vbar-fill{height:100%;background:linear-gradient(90deg,var(--ok),var(--accent));transition:width .4s}
main{display:grid;grid-template-columns:430px 1fr;gap:20px;padding:20px 24px;max-width:1480px;margin:0 auto}
@media(max-width:980px){main{grid-template-columns:1fr}}
.card{background:rgba(17,24,39,.9);border:1px solid var(--border);border-radius:16px;padding:18px;box-shadow:0 12px 32px rgba(0,0,0,.3)}
.tabs{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;background:#0b1220;padding:5px;border-radius:12px;margin-bottom:16px}
.tab{border:0;background:transparent;color:var(--muted);padding:9px 8px;border-radius:8px;font-weight:600;font-size:13px;cursor:pointer;transition:.15s}
.tab.active{background:linear-gradient(135deg,rgba(56,189,248,.22),rgba(129,140,248,.22));color:#fff;box-shadow:inset 0 0 0 1px rgba(56,189,248,.45)}
label{display:block;font-size:12px;font-weight:600;color:var(--muted);margin:12px 0 6px}
textarea,input,select{width:100%;background:#0b1220;border:1px solid var(--border);color:var(--text);border-radius:10px;padding:10px 12px;font-size:13px;font-family:inherit}
textarea:focus,input:focus,select:focus{outline:none;border-color:var(--accent)}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.chip{font-size:11px;padding:5px 10px;border-radius:999px;background:#0f172a;border:1px solid var(--border);color:var(--muted);cursor:pointer;transition:.15s}
.chip:hover,.chip.active{border-color:var(--accent);color:#fff;background:rgba(56,189,248,.14)}
.checks{display:flex;gap:14px;flex-wrap:wrap;margin-top:12px}.chk{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text);cursor:pointer}.chk input{width:auto}
.btn{width:100%;border:0;border-radius:11px;padding:12px 16px;font-weight:700;font-size:14px;cursor:pointer;margin-top:14px;transition:.15s}
.btn-primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#08101c}.btn-primary:hover{filter:brightness(1.08)}
.btn-secondary{background:#1e293b;color:#fff;border:1px solid var(--border)}.btn-danger{background:rgba(248,113,113,.16);color:var(--err);border:1px solid rgba(248,113,113,.4)}
.btn:disabled{opacity:.45;cursor:not-allowed}
.dropzone{border:1.5px dashed var(--border);border-radius:12px;padding:14px;text-align:center;color:var(--muted);font-size:12px;cursor:pointer;background:#0b1220}
.dropzone img{max-height:130px;border-radius:8px;margin-top:8px;max-width:100%}
.stage-box{display:flex;flex-direction:column;gap:16px}
.preview-frame{background:#05080f;border:1px solid var(--border);border-radius:14px;min-height:360px;display:flex;align-items:center;justify-content:center;overflow:hidden;position:relative}
.preview-frame video,.preview-frame img{max-width:100%;max-height:480px;display:block}
.empty-hint{color:var(--muted);text-align:center;padding:30px;font-size:14px}
.prog-wrap{background:#0b1220;border:1px solid var(--border);border-radius:12px;padding:14px}
.prog-head{display:flex;justify-content:space-between;font-size:13px;font-weight:600;margin-bottom:8px}
.pbar{height:10px;background:#1e293b;border-radius:999px;overflow:hidden}.pbar-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--ok));width:0;transition:width .35s}
.logbox{margin-top:10px;background:#05080f;border:1px solid #1a2438;border-radius:8px;padding:8px 10px;font-family:Consolas,monospace;font-size:11px;color:#93c5fd;max-height:120px;overflow-y:auto;white-space:pre-wrap}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:10px;margin-top:10px}
.gitem{background:#0b1220;border:1px solid var(--border);border-radius:10px;overflow:hidden;cursor:pointer;transition:.15s}.gitem:hover{border-color:var(--accent);transform:translateY(-2px)}
.gitem video,.gitem img{width:100%;height:105px;object-fit:cover;display:block;background:#000}
.gmeta{padding:7px 9px;font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.studio-actions{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
.hidden{display:none!important}
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="logo">PicoVid Studio</div>
    <span class="badge">v0.4.1 · 4-6 GB VRAM Engine</span>
    <span id="gpuName" style="font-size:12px;color:var(--muted)">GPU okunuyor...</span>
    <span id="modelBadges" style="display:flex;gap:6px;flex-wrap:wrap"></span>
  </div>
  <div class="vram-strip">
    <span>Dedicated: <strong id="vramDed">-</strong></span>
    <div class="vbar"><div class="vbar-fill" id="vramBar"></div></div>
    <span>Shared: <strong id="vramShr">-</strong></span>
  </div>
</header>
<main>
  <section class="card">
    <div class="tabs">
      <button class="tab active" data-mode="studio" onclick="setMode('studio')">Stüdyo (T2I→V)</button>
      <button class="tab" data-mode="t2v" onclick="setMode('t2v')">Metinden Video</button>
      <button class="tab" data-mode="i2v" onclick="setMode('i2v')">Görselden Video</button>
    </div>

    <div id="i2vUploadWrap" class="hidden">
      <label>Giriş Görseli (--init-img)</label>
      <div class="dropzone" onclick="document.getElementById('fileIn').click()">
        <div id="dropText">Görsel seçmek için tıklayın (PNG / JPG / WEBP)</div>
        <img id="uploadPreview" class="hidden">
      </div>
      <input id="fileIn" type="file" accept="image/*" class="hidden" onchange="uploadImage(this.files[0])">
    </div>

    <label>Sahne / Video Prompt'u (İngilizce önerilir)</label>
    <textarea id="prompt" rows="3" placeholder="A cinematic wide shot of the Bosphorus at golden hour, historic ferry gliding across sparkling water, 35mm film">A cinematic wide shot of the Bosphorus at golden hour, historic ferry gliding across sparkling water, seagulls flying, 35mm film</textarea>

    <div class="chips" id="motionChips">
      <span class="chip" data-add="slow smooth cinematic camera push-in, gentle atmospheric motion">+ Yavaş Zoom</span>
      <span class="chip" data-add="sweeping aerial drone glide forward, natural wind and water motion">+ Drone Pan</span>
      <span class="chip" data-add="dynamic cinematic tracking shot, dramatic realistic motion">+ Aksiyon Takip</span>
      <span class="chip" data-add="steady tripod shot, subtle lifelike natural motion">+ Sabit Kamera</span>
    </div>

    <div class="row2">
      <div>
        <label>Video Motoru</label>
        <select id="engine" onchange="onEngineChange()">
          <option value="5b" selected>Wan 2.2 TI2V-5B Turbo (Hızlı 5-15sn)</option>
          <option value="14b">Wan 2.2 14B (Ultra Detay 1-6sn)</option>
        </select>
      </div>
      <div>
        <label>Çözünürlük</label>
        <select id="res">
          <option value="640x384" selected>640×384 (En Hızlı · HD→1280×768)</option>
          <option value="832x480">832×480 (Doğal Fotogerçekçi 480p)</option>
          <option value="480x320">480×320 (Ultra Hızlı Taslak)</option>
        </select>
      </div>
    </div>

    <div class="row2">
      <div>
        <label>Süre / Bölüm Zinciri</label>
        <select id="durationPreset">
          <option value="81:1:16" selected>5.1 sn (81f · ~4.8 dk) ★</option>
          <option value="49:1:16">3.1 sn (49f · ~3.0 dk)</option>
          <option value="81:2:16">10.1 sn (2×81f Zincir · ~9.6 dk)</option>
          <option value="81:3:16">15.1 sn (3×81f Zincir · ~14.4 dk)</option>
          <option value="121:1:24">5.0 sn (121f @24fps native)</option>
        </select>
      </div>
      <div>
        <label>Seed (boş = rastgele)</label>
        <input id="seed" type="text" placeholder="Rastgele">
      </div>
    </div>

    <div id="segPromptsWrap" class="hidden">
      <label>Bölüm Bazlı Yönetmen Prompt'ları (opsiyonel, | ile ayırın)</label>
      <input id="segPrompts" type="text" placeholder="Bölüm 1 sahnesi | Bölüm 2 kamera açısı | Bölüm 3 final">
    </div>

    <div class="checks">
      <label class="chk"><input type="checkbox" id="cinemaHd" checked> Sinema HD + 32fps</label>
      <label class="chk"><input type="checkbox" id="fastPreview"> Ultra-Hızlı Önizleme (TAEW ~2sn)</label>
    </div>

    <div id="studioBtnGroup">
      <button class="btn btn-primary" id="btnGenFrame" onclick="runT2I()">1. İlk Kareyi Üret (Z-Image-Turbo ~22sn)</button>
      <div class="studio-actions hidden" id="studioApproveRow">
        <button class="btn btn-secondary" onclick="runT2I(true)">Yeni Seed</button>
        <button class="btn btn-primary" onclick="runVideo(true)">2. Videoya Çevir ▶</button>
      </div>
    </div>
    <button class="btn btn-primary hidden" id="btnGenVideo" onclick="runVideo(false)">Video Üret ▶</button>
    <button class="btn btn-danger hidden" id="btnCancel" onclick="cancelJob()">■ Üretimi Durdur</button>
  </section>

  <section class="stage-box">
    <div class="card">
      <div class="preview-frame" id="previewBox">
        <div class="empty-hint">Burada üretilen ilk kare veya bitmiş H.264 videonuz görünecek.<br><small>Stüdyo Modu ile 22 saniyede ilk kareyi onaylayıp sonra 5–15 sn videoya çevirebilirsiniz.</small></div>
      </div>
      <div class="prog-wrap" style="margin-top:14px">
        <div class="prog-head"><span id="stTitle">Hazır</span><span id="stPct">0%</span></div>
        <div class="pbar"><div class="pbar-fill" id="stBar"></div></div>
        <div id="stDetail" style="font-size:12px;color:var(--muted);margin-top:6px">Yeni üretim için hazır.</div>
        <div class="logbox" id="logBox">PicoVid Web Studio hazır.</div>
      </div>
    </div>

    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <strong style="font-size:14px">Son Çıktılar (Galeri)</strong>
        <button class="chip" onclick="loadGallery()">↻ Yenile</button>
      </div>
      <div class="gallery" id="gallery"></div>
    </div>
  </section>
</main>

<script>
let curMode = 'studio';
let uploadedPath = null;
let studioFramePath = null;

function setMode(m) {
  curMode = m;
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.mode === m));
  document.getElementById('i2vUploadWrap').classList.toggle('hidden', m !== 'i2v');
  document.getElementById('studioBtnGroup').classList.toggle('hidden', m !== 'studio');
  document.getElementById('btnGenVideo').classList.toggle('hidden', m === 'studio');
}

function onEngineChange() {
  const eng = document.getElementById('engine').value;
  const sel = document.getElementById('durationPreset');
  if (eng === '5b') {
    sel.innerHTML = `
      <option value="81:1:16" selected>5.1 sn (81f · ~4.8 dk) ★</option>
      <option value="49:1:16">3.1 sn (49f · ~3.0 dk)</option>
      <option value="81:2:16">10.1 sn (2×81f Zincir · ~9.6 dk)</option>
      <option value="81:3:16">15.1 sn (3×81f Zincir · ~14.4 dk)</option>
      <option value="121:1:24">5.0 sn (121f @24fps native)</option>`;
  } else {
    sel.innerHTML = `
      <option value="17:1:16" selected>1.1 sn (17f · ~4.5 dk)</option>
      <option value="33:1:16">2.1 sn (33f · ~8.5 dk)</option>
      <option value="49:1:16">3.1 sn (49f · ~14 dk)</option>
      <option value="33:2:16">4.1 sn (2×33f Zincir · ~17 dk)</option>`;
  }
  toggleSegPrompts();
}

function toggleSegPrompts() {
  const parts = document.getElementById('durationPreset').value.split(':');
  document.getElementById('segPromptsWrap').classList.toggle('hidden', parseInt(parts[1]) <= 1);
}
document.getElementById('durationPreset').addEventListener('change', toggleSegPrompts);

document.querySelectorAll('#motionChips .chip').forEach(ch => {
  ch.addEventListener('click', () => {
    const p = document.getElementById('prompt');
    if (!p.value.includes(ch.dataset.add)) p.value = p.value.trim().replace(/,\s*$/, '') + ', ' + ch.dataset.add;
  });
});

async function uploadImage(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async () => {
    const r = await fetch('/api/upload', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:file.name, data_url:reader.result})});
    const d = await r.json();
    if (d.ok) {
      uploadedPath = d.rel_path;
      document.getElementById('dropText').textContent = 'Yüklendi: ' + file.name;
      const img = document.getElementById('uploadPreview');
      img.src = d.url; img.classList.remove('hidden');
    }
  };
  reader.readAsDataURL(file);
}

async function runT2I(reroll=false) {
  const [w, h] = document.getElementById('res').value.split('x').map(Number);
  const seedRaw = reroll ? '' : document.getElementById('seed').value.trim();
  const r = await fetch('/api/t2i', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({prompt: document.getElementById('prompt').value.trim(), width:w, height:h, seed: seedRaw ? parseInt(seedRaw) : null})
  });
  const d = await r.json();
  if (!d.ok) alert(d.error || 'Başlatılamadı');
}

async function runVideo(fromStudio) {
  const [w, h] = document.getElementById('res').value.split('x').map(Number);
  const [frames, segments, fps] = document.getElementById('durationPreset').value.split(':').map(Number);
  const seedRaw = document.getElementById('seed').value.trim();
  const initPath = fromStudio ? studioFramePath : (curMode === 'i2v' ? uploadedPath : null);
  if ((fromStudio || curMode === 'i2v') && !initPath) { alert('Önce bir giriş görseli üretin veya yükleyin.'); return; }
  const r = await fetch('/api/generate', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      engine: document.getElementById('engine').value,
      mode: (fromStudio || curMode === 'i2v') ? 'i2v' : 't2v',
      init_img: initPath,
      prompt: document.getElementById('prompt').value.trim(),
      segment_prompts: document.getElementById('segPrompts').value.trim() || null,
      width: w, height: h, frames, segments, fps,
      cinema_hd: document.getElementById('cinemaHd').checked,
      fast_preview: document.getElementById('fastPreview').checked,
      seed: seedRaw ? parseInt(seedRaw) : null
    })
  });
  const d = await r.json();
  if (!d.ok) alert(d.error || 'Başlatılamadı');
}

async function cancelJob() {
  await fetch('/api/cancel', {method:'POST'});
}

function showInPreview(url, kind) {
  const box = document.getElementById('previewBox');
  if (kind === 'video') box.innerHTML = `<video src="${url}" controls autoplay loop playsinline></video>`;
  else box.innerHTML = `<img src="${url}">`;
}

async function loadModelsStatus() {
  try {
    const r = await fetch('/api/models');
    const m = await r.json();
    const defs = [['fast-5b','5B Turbo'], ['t2i','T2I'], ['t2v','14B T2V'], ['i2v','14B I2V']];
    document.getElementById('modelBadges').innerHTML = defs.map(([k, lbl]) => {
      const info = m[k]; if (!info) return '';
      if (info.ready) return `<span class="mbadge ok" title="${lbl} kurulu">${lbl} ✓</span>`;
      return `<span class="mbadge miss" title="İndirmek için tıklayın (${info.remaining_gib} GiB)" onclick="downloadProfile('${k}', ${info.remaining_gib})">${lbl} ↓ ${info.remaining_gib}G</span>`;
    }).join('');
  } catch(e) {}
}

async function downloadProfile(profile, gib) {
  if (!confirm(`'${profile}' modeli (~${gib} GiB) şimdi indirilsin mi?`)) return;
  const r = await fetch('/api/download-model', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({profile})});
  const d = await r.json();
  if (!d.ok) alert(d.error || 'İndirme başlatılamadı');
}

async function extendVideo(relPath, ev) {
  if (ev) ev.stopPropagation();
  const [w, h] = document.getElementById('res').value.split('x').map(Number);
  const eng = document.getElementById('engine').value;
  const frames = eng === '5b' ? 81 : 33;
  const promptTxt = document.getElementById('prompt').value.trim() || 'Continue smooth natural cinematic camera motion, 35mm film';
  if (!confirm(`"${relPath.split('/').pop()}" videosu son karesinden +${(frames/16).toFixed(1)} sn uzatılsın mı?`)) return;
  const r = await fetch('/api/extend', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      base_video: relPath, engine: eng, prompt: promptTxt,
      width: w, height: h, frames, fps: 16,
      cinema_hd: document.getElementById('cinemaHd').checked,
      fast_preview: document.getElementById('fastPreview').checked
    })
  });
  const d = await r.json();
  if (!d.ok) alert(d.error || 'Uzatma başlatılamadı');
}

async function loadGallery() {
  const r = await fetch('/api/gallery');
  const d = await r.json();
  const g = document.getElementById('gallery');
  g.innerHTML = d.items.map(it => `
    <div class="gitem" onclick="showInPreview('${it.url}','${it.kind}')">
      ${it.kind === 'video' ? `<video src="${it.url}" muted preload="metadata"></video>` : `<img src="${it.url}">`}
      <div class="gfoot">
        <div class="gmeta" title="${it.name}">${it.name} · ${it.size_mb}M</div>
        ${it.kind === 'video' ? `<button class="ext-btn" title="Son karesinden +5 sn uzat" onclick="extendVideo('${it.rel}', event)">+5s Uzat</button>` : ''}
      </div>
    </div>`).join('') || '<div style="color:var(--muted);font-size:12px">Henüz çıktı yok.</div>';
}

let lastShownT2I = null, lastShownVid = null;
async function pollStatus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    document.getElementById('stTitle').textContent = s.running ? `${s.stage} (${s.elapsed_sec}s)` : s.stage;
    document.getElementById('stPct').textContent = `${s.progress_pct}%`;
    document.getElementById('stBar').style.width = `${s.progress_pct}%`;
    document.getElementById('stDetail').textContent = s.detail || '';
    const lb = document.getElementById('logBox');
    lb.textContent = (s.logs || []).slice(-18).join('\n') || 'Hazır.';
    lb.scrollTop = lb.scrollHeight;

    document.getElementById('btnCancel').classList.toggle('hidden', !s.running);
    document.getElementById('btnGenFrame').disabled = s.running;
    document.getElementById('btnGenVideo').disabled = s.running;

    if (s.vram && s.vram.ded_mib != null) {
      const tot = s.vram.total_mib || 6128;
      document.getElementById('vramDed').textContent = `${(s.vram.ded_mib/1024).toFixed(2)}/${(tot/1024).toFixed(1)} GiB`;
      document.getElementById('vramBar').style.width = `${Math.min(100, (s.vram.ded_mib/tot)*100).toFixed(0)}%`;
      document.getElementById('vramShr').textContent = s.vram.shr_mib != null ? `${(s.vram.shr_mib/1024).toFixed(2)} GiB` : '-';
    }
    if (s.last_t2i_url && s.last_t2i_url !== lastShownT2I) {
      lastShownT2I = s.last_t2i_url;
      studioFramePath = s.last_t2i_path;
      showInPreview(s.last_t2i_url + '?t=' + Date.now(), 'image');
      document.getElementById('studioApproveRow').classList.remove('hidden');
      loadGallery();
    }
    if (s.last_video_url && s.last_video_url !== lastShownVid) {
      lastShownVid = s.last_video_url;
      showInPreview(s.last_video_url + '?t=' + Date.now(), 'video');
      loadGallery();
    }
  } catch (e) {}
}

fetch('/api/hardware').then(r => r.json()).then(hw => {
  document.getElementById('gpuName').textContent = `${hw.gpu_name} · ${hw.backend.toUpperCase()} (${hw.vram_class})`;
});
loadGallery();
loadModelsStatus();
setInterval(pollStatus, 1200);
pollStatus();
</script>
</body>
</html>
"""


def make_handler(state: StudioState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(self, obj: Any, status: int = 200) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n) if n > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/hardware":
                self._send_json(state.hw)
                return
            if path == "/api/status":
                self._send_json(state.snapshot())
                return
            if path == "/api/gallery":
                self._send_json({"items": list_gallery_items()})
                return
            if path == "/api/models":
                self._send_json(get_models_status())
                return
            if path.startswith("/files/"):
                rel = urllib.parse.unquote(path[len("/files/"):]).lstrip("/")
                target = (REPO_ROOT / rel).resolve()
                if not str(target).startswith(str(REPO_ROOT.resolve())) or not target.exists() or not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urllib.parse.urlparse(self.path).path
            try:
                body = self._read_json()
            except Exception as exc:
                self._send_json({"ok": False, "error": f"Geçersiz JSON: {exc}"}, 400)
                return

            if path == "/api/cancel":
                stopped = state.cancel_job()
                self._send_json({"ok": True, "stopped": stopped})
                return

            if path == "/api/upload":
                data_url = str(body.get("data_url", ""))
                name = re.sub(r"[^a-zA-Z0-9._-]", "_", str(body.get("name", "upload.png")))
                if "," not in data_url:
                    self._send_json({"ok": False, "error": "Geçersiz data_url"}, 400)
                    return
                b64 = data_url.split(",", 1)[1]
                UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest = UPLOADS_DIR / f"{ts}_{name}"
                dest.write_bytes(base64.b64decode(b64))
                rel = dest.relative_to(REPO_ROOT).as_posix()
                self._send_json({"ok": True, "rel_path": rel, "url": f"/files/{urllib.parse.quote(rel)}"})
                return

            if path == "/api/download-model":
                profile = str(body.get("profile", "fast-5b"))
                cmd = [sys.executable, str(SCRIPT_DIR / "download_models.py"), "--profile", profile]
                ok, msg = state.start_job("download", cmd, "configs/models_registry.json")
                self._send_json({"ok": ok, "message": msg, "profile": profile}, 200 if ok else 409)
                return

            if path == "/api/extend":
                base_rel = str(body.get("base_video", "")).lstrip("/")
                base_abs = (REPO_ROOT / base_rel).resolve()
                if not str(base_abs).startswith(str(REPO_ROOT.resolve())) or not base_abs.exists():
                    self._send_json({"ok": False, "error": "Uzatılacak video bulunamadı."}, 404)
                    return
                engine = str(body.get("engine", "5b"))
                prompt = str(body.get("prompt") or "Continue smooth natural cinematic camera motion, 35mm film").strip()
                w = int(body.get("width", 640))
                h = int(body.get("height", 384))
                frames = int(body.get("frames", 81 if engine == "5b" else 33))
                fps = int(body.get("fps", 16))
                cinema_hd = bool(body.get("cinema_hd", True))
                fast_preview = bool(body.get("fast_preview", False))
                OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_rel = f"outputs/extended_{base_abs.stem}_{ts}.mp4"
                cmd = [
                    sys.executable, str(SCRIPT_DIR / "generate.py"),
                    "--engine", engine, "--extend-video", str(base_abs),
                    "-p", prompt, "-W", str(w), "-H", str(h),
                    "-f", str(frames), "--fps", str(fps),
                    "-o", out_rel, "--raw-logs",
                ]
                if cinema_hd and not fast_preview:
                    cmd.append("--cinema-hd")
                if fast_preview:
                    cmd.append("--fast-preview")
                ok, msg = state.start_job("video", cmd, out_rel)
                self._send_json({"ok": ok, "message": msg, "out_rel": out_rel}, 200 if ok else 409)
                return

            if path == "/api/t2i":
                prompt = str(body.get("prompt") or ui_console.DEFAULT_SAMPLE_PROMPT).strip()
                w = int(body.get("width", 640))
                h = int(body.get("height", 384))
                seed = int(body["seed"]) if body.get("seed") is not None else random.randint(1, 2_147_483_647)
                OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_rel = f"outputs/studio_frame_{ts}_{w}x{h}_s{seed}.png"
                cmd = [
                    sys.executable, str(SCRIPT_DIR / "generate.py"),
                    "--mode", "t2i", "-p", prompt,
                    "-W", str(w), "-H", str(h), "--seed", str(seed),
                    "-o", out_rel, "--raw-logs",
                ]
                ok, msg = state.start_job("t2i", cmd, out_rel, seed=seed)
                self._send_json({"ok": ok, "message": msg, "out_rel": out_rel, "seed": seed}, 200 if ok else 409)
                return

            if path == "/api/generate":
                engine = str(body.get("engine", "5b"))
                mode = str(body.get("mode", "t2v"))
                prompt = str(body.get("prompt") or ui_console.DEFAULT_SAMPLE_PROMPT).strip()
                w = int(body.get("width", 640))
                h = int(body.get("height", 384))
                frames = int(body.get("frames", 81 if engine == "5b" else 33))
                segments = max(1, min(6, int(body.get("segments", 1))))
                fps = int(body.get("fps", 16))
                seed = int(body["seed"]) if body.get("seed") is not None else random.randint(1, 2_147_483_647)
                cinema_hd = bool(body.get("cinema_hd", True))
                fast_preview = bool(body.get("fast_preview", False))
                init_img = body.get("init_img")
                seg_prompts = body.get("segment_prompts")

                OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                tag = f"{segments}x{frames}f" if segments > 1 else f"{frames}f"
                out_rel = f"outputs/web_{engine}_{mode}_{ts}_{w}x{h}_{tag}.mp4"
                cmd = [
                    sys.executable, str(SCRIPT_DIR / "generate.py"),
                    "--engine", engine, "--mode", mode,
                    "-p", prompt,
                    "-W", str(w), "-H", str(h),
                    "-f", str(frames), "--fps", str(fps),
                    "--segments", str(segments),
                    "--seed", str(seed),
                    "-o", out_rel, "--raw-logs",
                ]
                if mode == "i2v" and init_img:
                    cmd += ["-i", str(init_img)]
                if seg_prompts:
                    cmd += ["--segment-prompts", str(seg_prompts)]
                if cinema_hd and not fast_preview:
                    cmd.append("--cinema-hd")
                if fast_preview:
                    cmd.append("--fast-preview")
                ok, msg = state.start_job("video", cmd, out_rel, seed=seed)
                self._send_json({"ok": ok, "message": msg, "out_rel": out_rel, "seed": seed}, 200 if ok else 409)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


def main() -> int:
    ui_console.configure_console_utf8()
    parser = argparse.ArgumentParser(description="PicoVid v0.4.0 Local Web Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    state = StudioState()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    url = f"http://{args.host}:{args.port}"
    print(f"[PicoVid Web Studio] Hazır -> {url}  (GPU: {state.hw['gpu_name']} · {state.hw['backend'].upper()})", flush=True)
    print("[PicoVid Web Studio] Kapatmak için Ctrl+C basın.", flush=True)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[PicoVid Web Studio] Kapatılıyor...")
    finally:
        server.server_close()
        state.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
