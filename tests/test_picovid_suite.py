#!/usr/bin/env python3
"""
PicoVid v0.4.0 Automated Regression & Integration Test Suite.

Covers:
  - configs/models_registry.json integrity & download_models hardware recommendations
  - generate.py token math, VRAM profile rules (4GB/6GB/8GB, CUDA+Vulkan, 5B+14B), and CLI dry-runs
  - postprocess.py frame extraction, Lab color anchoring, and multi-segment stitching
  - web_ui.py live HTTP API smoke tests (/, /api/hardware, /api/status, /api/gallery, /api/cancel)

Run: py -m unittest tests/test_picovid_suite.py -v
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import download_models  # noqa: E402
import generate  # noqa: E402
import postprocess  # noqa: E402
import web_ui  # noqa: E402


class TestRegistryAndDownloader(unittest.TestCase):
    def test_registry_profiles_valid(self) -> None:
        reg = download_models.load_registry()
        files = reg["files"]
        for prof_name, pdef in reg["download_profiles"].items():
            for fid in pdef["required"] + pdef.get("optional", []):
                self.assertIn(fid, files, f"Missing file id {fid} in profile {prof_name}")
                self.assertGreater(int(files[fid]["expected_bytes"]), 1000)

    def test_hardware_recommendations(self) -> None:
        k4, _ = download_models.recommend_bundle(download_models.detect_host_summary("cuda-4gb-12gb"))
        self.assertEqual(k4, "1")  # fast-5b lightweight starter for 4GB/12GB
        k20, _ = download_models.recommend_bundle(download_models.detect_host_summary("vulkan-6gb-20gb"))
        self.assertEqual(k20, "4")  # studio-compact Q3_K_M
        k32, _ = download_models.recommend_bundle(download_models.detect_host_summary("vulkan-6gb-32gb"))
        self.assertEqual(k32, "3")  # full Q4_K_M studio

    def test_python_fallback_downloader_local_http(self) -> None:
        from http.server import BaseHTTPRequestHandler
        payload = b"PicoVidModelData" * 1024

        class _RangeHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return
            def do_GET(self):
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    start = int(rng.split("=")[1].split("-")[0])
                    part = payload[start:]
                    self.send_response(206)
                    self.send_header("Content-Length", str(len(part)))
                    self.end_headers()
                    self.wfile.write(part)
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            with tempfile.TemporaryDirectory(prefix="picovid_dl_") as td:
                dest = pathlib.Path(td) / "model.bin"
                dest.write_bytes(payload[:512])  # test resume from 512 bytes
                ok = download_models._download_with_python_urllib(
                    f"http://127.0.0.1:{srv.server_address[1]}/model.bin", dest, len(payload)
                )
                self.assertTrue(ok)
                self.assertEqual(dest.read_bytes(), payload)
        finally:
            srv.shutdown()
            srv.server_close()


class TestProfileAndTokenMath(unittest.TestCase):
    def test_token_math_5b_vs_14b(self) -> None:
        # 640x384x81f: 14B stride=16 -> 40*24*21=20160; 5B stride=32 -> 20*12*21=5040 (4x fewer!)
        self.assertEqual(generate.compute_dit_tokens(640, 384, 81, engine="14b"), 20160)
        self.assertEqual(generate.compute_dit_tokens(640, 384, 81, engine="5b"), 5040)
        self.assertEqual(generate.compute_dit_tokens(832, 480, 81, engine="5b"), 8190)

    def test_5b_6gb_vulkan_profile(self) -> None:
        hw = generate.detect_hardware("vulkan-6gb")
        p81 = generate.select_profile(hw, frames=81, fast_preview=False, width=640, height=384, engine="5b")
        self.assertEqual(p81["dit_tokens"], 5040)
        self.assertEqual(p81["env_p1"]["SD_WAN_ATTN_HEAD_CHUNK"], "12")
        self.assertEqual(p81["env_p1"]["SD_WAN_MLP_CHUNK_MB"], "256")
        self.assertEqual(p81["flash_attn"], "off")

        p480 = generate.select_profile(hw, frames=81, fast_preview=False, width=832, height=480, engine="5b")
        self.assertEqual(p480["dit_tokens"], 8190)
        self.assertEqual(p480["env_p1"]["SD_WAN_ATTN_HEAD_CHUNK"], "4")
        self.assertEqual(p480["env_p1"]["SD_WAN_CROSS_Q_CHUNK"], "4096")

    def test_14b_fit_long_and_cuda_4gb(self) -> None:
        hw_vk6 = generate.detect_hardware("vulkan-6gb")
        p_long = generate.select_profile(hw_vk6, frames=33, fast_preview=False, width=832, height=480, engine="14b")
        self.assertEqual(p_long["profile_name"], "vulkan_6GB_fit_long")
        self.assertEqual(p_long["env_p1"]["SD_WAN_QKV_SPLIT"], "1")
        self.assertEqual(p_long["env_p1"]["SD_CUT_CACHE_HOST_MB"], "32")
        self.assertLessEqual(p_long["max_vram_gib"], 1.85)

        hw_cu4 = generate.detect_hardware("cuda-4gb")
        p_cu = generate.select_profile(hw_cu4, frames=17, fast_preview=False, width=640, height=384, engine="14b")
        self.assertEqual(p_cu["flash_attn"], "off,diffusion=on")
        self.assertEqual(p_cu["env_p1"]["SD_STAGING_POOL_MB"], "256")


class TestCliDryRuns(unittest.TestCase):
    def _run_dry(self, extra_args: list[str]) -> dict:
        cmd = [sys.executable, str(SCRIPTS_DIR / "generate.py"), *extra_args, "--simulate-profile", "vulkan-6gb", "--dry-run"]
        out = subprocess.check_output(cmd, cwd=str(REPO_ROOT), text=True)
        return json.loads(out)

    def test_dry_run_5b_segments(self) -> None:
        d = self._run_dry(["--engine", "5b", "--mode", "studio", "--segments", "3", "-f", "81", "--fps", "16", "--cinema-hd", "-p", "test", "--segment-prompts", "A | B | C"])
        self.assertEqual(d["segments"], 3)
        self.assertEqual(d["total_raw_frames"], 241)
        self.assertEqual(d["segment_prompts"], ["A", "B", "C"])
        self.assertEqual(d["target_output"], "1280x768 @ 32fps")

    def test_dry_run_extend_video(self) -> None:
        d = self._run_dry(["--engine", "5b", "--extend-video", "outputs/dummy.mp4", "-p", "continue shot", "-W", "832", "-H", "480", "-f", "81"])
        self.assertEqual(d["action"], "extend_video")
        self.assertEqual(d["resolution"], "832x480")


class TestPostprocessStitchAndLabAnchor(unittest.TestCase):
    def test_synthetic_stitch_and_color_anchor(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="picovid_test_") as td:
            d = pathlib.Path(td)
            seg1 = d / "s1.avi"
            seg2 = d / "s2.avi"
            out_mp4 = d / "stitched.mp4"

            # Segment 1: 5 warm frames; Segment 2: starts identical to seg1 tail, then drifts brighter
            w, h = 64, 64
            base = np.full((h, w, 3), (100, 120, 140), dtype=np.uint8)
            w1 = cv2.VideoWriter(str(seg1), cv2.VideoWriter_fourcc(*"MJPG"), 16.0, (w, h))
            for i in range(5):
                w1.write(np.clip(base.astype(np.int32) + i * 2, 0, 255).astype(np.uint8))
            w1.release()

            tail_png = postprocess.extract_last_frame(seg1, d / "tail.png")
            tail_im = cv2.imread(str(tail_png))
            self.assertIsNotNone(tail_im)

            w2 = cv2.VideoWriter(str(seg2), cv2.VideoWriter_fourcc(*"MJPG"), 16.0, (w, h))
            w2.write(tail_im)  # exact duplicate boundary frame
            for i in range(1, 5):
                w2.write(np.clip(tail_im.astype(np.int32) + i * 10, 0, 255).astype(np.uint8))
            w2.release()

            res = postprocess.stitch_video_segments([seg1, seg2], out_mp4, fps=16.0, drop_overlap_frames=1, color_anchor=True)
            self.assertEqual(res["total_frames"], 9)  # 5 + 5 - 1 duplicate boundary dropped
            self.assertTrue(out_mp4.exists())
            self.assertGreater(res["seam_psnr_db"][0], 30.0)


class TestWebStudioServer(unittest.TestCase):
    def test_http_endpoints(self) -> None:
        state = web_ui.StudioState()
        server = ThreadingHTTPServer(("127.0.0.1", 0), web_ui.make_handler(state))
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            base = f"http://127.0.0.1:{port}"
            with urllib.request.urlopen(f"{base}/", timeout=5) as r:
                self.assertEqual(r.status, 200)
                html = r.read().decode("utf-8")
                self.assertIn("PicoVid Studio", html)

            for ep in ("/api/hardware", "/api/status", "/api/gallery", "/api/models"):
                with urllib.request.urlopen(f"{base}{ep}", timeout=5) as r:
                    self.assertEqual(r.status, 200)
                    data = json.loads(r.read().decode("utf-8"))
                    self.assertIsInstance(data, dict)

            req = urllib.request.Request(f"{base}/api/cancel", data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=5) as r:
                self.assertEqual(r.status, 200)
                self.assertTrue(json.loads(r.read().decode("utf-8"))["ok"])
        finally:
            server.shutdown()
            server.server_close()
            state.shutdown()


if __name__ == "__main__":
    unittest.main()
