#!/usr/bin/env python3
"""
localVid Model Downloader (Apache 2.0 Default Set — Wan 2.2 14B Rapid-AIO v10).
Supports resume via curl (-C -) and exact byte-size verification.

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

MODELS = [
    {
        "id": "diffusion_rapid_v10",
        "required": True,
        "path": pathlib.Path("models/diffusion_models/wan2.2-t2v-rapid-aio-v10-Q4_K.gguf"),
        "expected_bytes": 9650100736,
        "url": "https://huggingface.co/befox/WAN2.2-14B-Rapid-AllInOne-GGUF/resolve/main/v10/wan2.2-t2v-rapid-aio-v10-Q4_K.gguf",
        "desc": "Wan 2.2 14B T2V Rapid-AIO v10 (pre-merged 4-step LightX2V + Lightning, Q4_K, 8.99 GB)",
    },
    {
        "id": "t5_encoder",
        "required": True,
        "path": pathlib.Path("models/text_encoders/umt5-xxl-encoder-Q4_K_M.gguf"),
        "expected_bytes": 3655145312,
        "url": "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q4_K_M.gguf",
        "desc": "UMT5-XXL Text Encoder (Q4_K_M, 3.40 GB, runs on CPU)",
    },
    {
        "id": "wan21_vae",
        "required": True,
        "path": pathlib.Path("models/vae/wan_2.1_vae.safetensors"),
        "expected_bytes": 253815318,
        "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
        "desc": "Wan 2.1 Video VAE (FP16/BF16, 242 MB)",
    },
    {
        "id": "taehv",
        "required": False,
        "path": pathlib.Path("models/vae/taew2_1.safetensors"),
        "expected_bytes": 22642902,
        "url": "https://github.com/madebyollin/taehv/raw/refs/heads/main/safetensors/taew2_1.safetensors",
        "desc": "TAEHV Tiny AutoEncoder for Wan 2.1/2.2 (--fast-preview, 21.6 MB)",
    },
]


def format_gb(nbytes: int) -> str:
    return f"{nbytes / (1024**3):.3f} GiB"


def check_file(item: dict) -> tuple[bool, str]:
    p: pathlib.Path = item["path"]
    if not p.exists():
        return False, "MISSING"
    sz = p.stat().st_size
    exp = item["expected_bytes"]
    if sz == exp:
        return True, f"OK ({format_gb(sz)})"
    return False, f"SIZE_MISMATCH (got {sz:,} B, expected {exp:,} B)"


def download_file(item: dict) -> bool:
    p: pathlib.Path = item["path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, msg = check_file(item)
    if ok:
        print(f"  [SKIP] {p} — {msg}")
        return True

    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        print("ERROR: curl is required for resumable downloads.", file=sys.stderr)
        return False

    print(f"  [GET ] {item['desc']}")
    print(f"         -> {p}")
    cmd = [curl, "-L", "-C", "-", "--retry", "3", "--retry-delay", "3", "-#", item["url"], "-o", str(p)]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"  [FAIL] curl exited with code {proc.returncode}", file=sys.stderr)
        return False

    ok, msg = check_file(item)
    print(f"  [{'OK  ' if ok else 'WARN'}] {p} — {msg}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Download localVid Apache-2.0 Wan 2.2 14B model set")
    parser.add_argument("--with-taehv", action="store_true", help="Also download TAEHV (21.6 MB) for --fast-preview")
    parser.add_argument("--check", action="store_true", help="Verify downloaded files without downloading")
    args = parser.parse_args()

    selected = [m for m in MODELS if m["required"] or args.with_taehv or args.check]
    total_bytes = sum(m["expected_bytes"] for m in selected if m["required"] or args.with_taehv)

    if args.check:
        print("=== localVid Model Integrity Check ===")
        all_req_ok = True
        for m in selected:
            ok, msg = check_file(m)
            tag = "REQ" if m["required"] else "OPT"
            print(f"  [{tag}] {m['path']}: {msg}")
            if m["required"] and not ok:
                all_req_ok = False
        return 0 if all_req_ok else 1

    print(f"Downloading {len(selected)} files (total {format_gb(total_bytes)})...")
    all_ok = True
    for m in selected:
        if not download_file(m):
            all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
