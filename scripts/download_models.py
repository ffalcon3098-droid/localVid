#!/usr/bin/env python3
"""
PicoVid Hardware-Aware Model Downloader driven by configs/models_registry.json.

Supports:
- Interactive Smart Download Wizard (when launched with no arguments or --interactive):
  detects GPU VRAM, total system RAM, and free disk space, and highlights the
  best-fit bundle with a '★ Recommended' badge.
- Direct profile downloads (--profile t2v|t2i|i2v|i2v-compact|studio|studio-compact|all)
- Simulated hardware dry-run (--interactive --dry-run --simulate-hw cuda-4gb-12gb)
- Resumable transfers via curl (-C - --speed-limit), free disk headroom check, and exact byte verification.

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "models_registry.json"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import probe_hardware  # noqa: E402

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def configure_utf8() -> None:
    for name in ("stdout", "stderr"):
        st = getattr(sys, name, None)
        if st is not None and hasattr(st, "reconfigure"):
            try:
                st.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def format_gb(nbytes: int) -> str:
    return f"{nbytes / (1024**3):.2f} GiB"


def load_registry(path: pathlib.Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(rel_or_abs: str) -> pathlib.Path:
    p = pathlib.Path(rel_or_abs)
    return p if p.is_absolute() else (REPO_ROOT / p)


def check_file(item: dict[str, Any]) -> tuple[bool, str, int]:
    p = resolve_path(item["path"])
    exp = int(item["expected_bytes"])
    if not p.exists():
        return False, "MISSING", exp
    sz = p.stat().st_size
    if sz == exp:
        return True, f"OK ({format_gb(sz)})", 0
    remaining = max(0, exp - sz)
    return False, f"PARTIAL/MISMATCH ({format_gb(sz)} / {format_gb(exp)})", remaining


def _download_with_python_urllib(url: str, dest: pathlib.Path, expected_bytes: int) -> bool:
    """Pure-Python resumable streaming downloader fallback (works even if curl is missing/broken)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 5):
        existing = dest.stat().st_size if dest.exists() else 0
        if expected_bytes > 0 and existing > expected_bytes:
            dest.unlink(missing_ok=True)
            existing = 0

        headers = {"User-Agent": "PicoVid/0.5 (+https://github.com/ffalcon3098-droid/localVid)"}
        mode = "wb"
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                status = getattr(resp, "status", 200)
                if existing > 0 and status == 200:
                    # Server ignored Range header; restart cleanly from 0
                    existing = 0
                    mode = "wb"
                cl = int(resp.headers.get("Content-Length", "0") or "0")
                total = (existing + cl) if (cl > 0 and status == 206) else (cl or expected_bytes)

                downloaded = existing
                t0 = time.monotonic()
                t_last = t0
                chunk_size = 4 * 1024 * 1024  # 4 MiB chunks
                with dest.open(mode) as fh:
                    while True:
                        buf = resp.read(chunk_size)
                        if not buf:
                            break
                        fh.write(buf)
                        downloaded += len(buf)
                        now = time.monotonic()
                        if now - t_last >= 0.4:
                            el = max(0.01, now - t0)
                            spd = (downloaded - existing) / el / (1024 * 1024)
                            pct = (downloaded / total * 100.0) if total > 0 else 0.0
                            print(
                                f"\r         [Python] {pct:5.1f}%  {format_gb(downloaded)} / {format_gb(total)}  ({spd:.1f} MB/s)   ",
                                end="",
                                flush=True,
                            )
                            t_last = now
                print()
                if expected_bytes <= 0 or dest.stat().st_size == expected_bytes:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                dest.unlink(missing_ok=True)
                continue
            print(f"\n  [WARN] Python download attempt {attempt}/4 HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        except Exception as exc:
            print(f"\n  [WARN] Python download attempt {attempt}/4 error: {exc}", file=sys.stderr)
        time.sleep(min(10, attempt * 2))
    return False


def download_file(item: dict[str, Any]) -> bool:
    p = resolve_path(item["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    exp = int(item.get("expected_bytes", 0))
    ok, msg, _ = check_file(item)
    if ok:
        print(f"  [SKIP] {item['path']} — {msg}")
        return True

    # Delete oversize/corrupt partial (e.g. HTML error page) before retrying
    if p.exists() and exp > 0 and p.stat().st_size > exp:
        p.unlink(missing_ok=True)

    print(f"  [GET ] {item['desc']}")
    print(f"         -> {item['path']}")

    curl = shutil.which("curl.exe") or shutil.which("curl")
    curl_ok = False
    if curl:
        has_partial = p.exists() and p.stat().st_size > 0
        for try_resume in ([True, False] if has_partial else [False]):
            cmd = [
                curl, "-L", "--fail",
                "--connect-timeout", "30",
                "--retry", "6", "--retry-delay", "4",
                "--speed-limit", "10000", "--speed-time", "45",
                "-#",
            ]
            if try_resume:
                cmd += ["-C", "-"]
            else:
                p.unlink(missing_ok=True)
            cmd += [item["url"], "-o", str(p)]
            proc = subprocess.run(cmd)
            if proc.returncode == 0:
                curl_ok = True
                break
            print(f"  [WARN] curl exit code {proc.returncode} (resume={try_resume})", file=sys.stderr)

    if not curl_ok:
        print("  [INFO] Falling back to built-in resumable Python downloader...")
        _download_with_python_urllib(item["url"], p, exp)

    ok, msg, _ = check_file(item)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {item['path']} — {msg}")
    return ok


def detect_host_summary(simulate_hw: str | None = None) -> dict[str, Any]:
    """Detect GPU VRAM, total system RAM, and target drive free disk space."""
    if simulate_hw:
        sim_presets: dict[str, dict[str, Any]] = {
            "cuda-4gb-12gb": {
                "gpu_name": "NVIDIA GeForce GTX 1650 (Simulated)",
                "backend": "cuda",
                "vram_total_gib": 4.0,
                "ram_total_gib": 12.0,
                "disk_free_gib": 40.0,
                "disk_drive": "C:",
            },
            "vulkan-4gb-16gb": {
                "gpu_name": "AMD Radeon RX 5500 XT 4GB (Simulated)",
                "backend": "vulkan",
                "vram_total_gib": 4.0,
                "ram_total_gib": 16.0,
                "disk_free_gib": 35.0,
                "disk_drive": "C:",
            },
            "vulkan-6gb-20gb": {
                "gpu_name": "AMD Radeon RX 5600 XT 6GB (Simulated)",
                "backend": "vulkan",
                "vram_total_gib": 6.0,
                "ram_total_gib": 20.0,
                "disk_free_gib": 27.0,
                "disk_drive": "C:",
            },
            "vulkan-6gb-32gb": {
                "gpu_name": "AMD Radeon RX 5600 XT 6GB (Simulated)",
                "backend": "vulkan",
                "vram_total_gib": 6.0,
                "ram_total_gib": 32.0,
                "disk_free_gib": 50.0,
                "disk_drive": "C:",
            },
            "cuda-8gb-32gb": {
                "gpu_name": "NVIDIA GeForce RTX 4060 8GB (Simulated)",
                "backend": "cuda",
                "vram_total_gib": 8.0,
                "ram_total_gib": 32.0,
                "disk_free_gib": 80.0,
                "disk_drive": "C:",
            },
        }
        return sim_presets[simulate_hw]

    sys_info = probe_hardware.probe_system(REPO_ROOT / "models")
    ram_gib = sys_info.get("ram_total_gib") or 16.0
    disk_free = sys_info.get("disk_free_gib") or 20.0
    disk_drive = sys_info.get("disk_drive") or ""

    cuda = probe_hardware.probe_nvidia_cuda()
    if cuda.get("available") and cuda.get("gpus"):
        g = cuda["gpus"][0]
        return {
            "gpu_name": g["name"],
            "backend": "cuda",
            "vram_total_gib": round(g["vram_total_mib"] / 1024.0, 2),
            "ram_total_gib": ram_gib,
            "disk_free_gib": disk_free,
            "disk_drive": disk_drive,
        }

    vk = probe_hardware.probe_vulkan()
    if vk.get("available") and vk.get("devices"):
        d = vk["devices"][0]
        return {
            "gpu_name": d.get("name") or "Vulkan GPU",
            "backend": "vulkan",
            "vram_total_gib": d.get("device_local_total_gib") or 6.0,
            "ram_total_gib": ram_gib,
            "disk_free_gib": disk_free,
            "disk_drive": disk_drive,
        }

    return {
        "gpu_name": "Unknown GPU",
        "backend": "vulkan",
        "vram_total_gib": 4.0,
        "ram_total_gib": ram_gib,
        "disk_free_gib": disk_free,
        "disk_drive": disk_drive,
    }


def recommend_bundle(host: dict[str, Any]) -> tuple[str, str]:
    """Choose best starter bundle key ('1'..'5') and short reason based on VRAM, RAM, and free disk."""
    vram = float(host["vram_total_gib"])
    ram = float(host["ram_total_gib"])
    disk = float(host["disk_free_gib"])

    if vram <= 4.5 or ram <= 16.5 or disk < 24.5:
        return "1", f"Lightest 4–6 GB VRAM + 16 GB RAM starter bundle (GPU {vram:.1f} GiB, RAM {ram:.1f} GiB)"
    if ram >= 24.0 and disk >= 29.5:
        return "3", f"Flagship LTX-2.3 22B Audio+Video for 6 GB+ VRAM ({vram:.1f} GiB) and >=24 GB RAM ({ram:.1f} GiB)"
    return "4", f"Compact Q3_K_M Studio bundle for lower disk/RAM (RAM {ram:.1f} GiB, Disk {disk:.1f} GiB)"


def download_file_ids(
    file_ids: list[str],
    reg: dict[str, Any],
    force_low_disk: bool = False,
) -> bool:
    """Check free space and download a specific list of registry file IDs."""
    files_map = reg["files"]
    unique_ids = list(dict.fromkeys(file_ids))
    needed_bytes = 0
    for fid in unique_ids:
        _, _, rem = check_file(files_map[fid])
        needed_bytes += rem

    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(models_dir).free
    headroom = 1_500_000_000  # 1.5 GB safety headroom

    if needed_bytes > 0 and (free_bytes - needed_bytes) < headroom and not force_low_disk:
        print(
            f"ERROR: Insufficient free disk space!\n"
            f"  Required : {format_gb(needed_bytes)}\n"
            f"  Free     : {format_gb(free_bytes)}\n"
            f"  Tip      : Free up disk space, choose 'i2v-compact' (Q3_K_M), or pass --force-low-disk.",
            file=sys.stderr,
        )
        return False

    print(
        f"[PicoVid] Checking {len(unique_ids)} file(s) "
        f"(to download: {format_gb(needed_bytes)}, free disk: {format_gb(free_bytes)})..."
    )
    all_ok = True
    for fid in unique_ids:
        if not download_file(files_map[fid]):
            all_ok = False
    return all_ok


def run_interactive_downloader(
    reg: dict[str, Any],
    simulate_hw: str | None = None,
    dry_run: bool = False,
    force_low_disk: bool = False,
) -> int:
    configure_utf8()
    host = detect_host_summary(simulate_hw)
    rec_key, rec_reason = recommend_bundle(host)
    files_map = reg["files"]
    profiles = reg["download_profiles"]

    menu_items = [
        ("1", "fast-5b", "Fast 5s Bundle: Wan 2.2 TI2V-5B Turbo (T2V + I2V)", "4–6 GB VRAM + 16 GB RAM (~3–6 min)"),
        ("2", "t2v", "14B T2V Starter: Wan 2.2 14B T2V Q4_K + TE + VAE", "14B single-model T2V"),
        ("3", "ltx23", "Flagship: LTX-2.3 22B Distilled Audio+Video + x2 Upscaler", "6 GB+ VRAM & >=24 GB RAM (48 kHz Stereo)"),
        ("4", "studio-compact", "14B Studio Compact: Z-Image T2I + Wan 2.2 I2V Q3_K_M", "Lower disk/RAM Studio alternative"),
        ("5", "ltx23-studio", "LTX-2.3 Studio: Z-Image-Turbo T2I + LTX-2.3 22B Audio+Video", "First-frame approval + Audio+Video"),
        ("6", "all", "Full Legacy Wan 2.2 Pack: 5B Turbo + 14B T2V + T2I + I2V", "All Wan 2.2 modes"),
    ]

    def calc_pack_sizes(prof_key: str) -> tuple[int, int]:
        ids = list(profiles[prof_key]["required"])
        if prof_key == "t2v":
            ids += list(profiles[prof_key].get("optional", []))
        tot = sum(int(files_map[fid]["expected_bytes"]) for fid in ids)
        rem = sum(check_file(files_map[fid])[2] for fid in ids)
        return tot, rem

    use_rich = HAS_RICH and (sys.stdout.isatty() or dry_run)
    if use_rich:
        console = Console(highlight=False)
        hdr = (
            f"[bold cyan]GPU:[/bold cyan] {host['gpu_name']} ({host['backend'].upper()}, {host['vram_total_gib']:.1f} GiB VRAM)  |  "
            f"[bold cyan]RAM:[/bold cyan] {host['ram_total_gib']:.1f} GiB  |  "
            f"[bold cyan]Free Disk ({host['disk_drive'] or 'target'}):[/bold cyan] {host['disk_free_gib']:.1f} GiB\n"
            f"[green]★ Recommendation:[/green] {rec_reason}"
        )
        console.print()
        console.print(Panel(hdr, title="[bold cyan]PicoVid v0.7.0 · Smart Model Download Wizard[/bold cyan]", border_style="cyan", box=box.ROUNDED))

        tbl = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
        tbl.add_column("Choice", style="bold cyan", width=6)
        tbl.add_column("Bundle", style="white", ratio=3)
        tbl.add_column("Total", justify="right", width=11)
        tbl.add_column("Remaining", justify="right", width=11)
        tbl.add_column("Notes", ratio=2)

        for key, prof_key, label, note in menu_items:
            tot_b, rem_b = calc_pack_sizes(prof_key)
            badge = "[bold green]★ Recommended[/bold green] · " + note if key == rec_key else f"[dim]{note}[/dim]"
            rem_txt = "[green]Ready[/green]" if rem_b == 0 else format_gb(rem_b)
            tbl.add_row(f"[{key}]", label, format_gb(tot_b), rem_txt, badge)
        console.print(tbl)
    else:
        print("=== PicoVid v0.7.0 · Smart Model Download Wizard ===")
        print(
            f"Hardware : {host['gpu_name']} ({host['backend'].upper()}, {host['vram_total_gib']:.1f} GiB VRAM) | "
            f"RAM: {host['ram_total_gib']:.1f} GiB | Free Disk: {host['disk_free_gib']:.1f} GiB"
        )
        print(f"Suggested: [{rec_key}] — {rec_reason}")
        for key, prof_key, label, note in menu_items:
            tot_b, rem_b = calc_pack_sizes(prof_key)
            mark = " ★ Recommended" if key == rec_key else ""
            print(f"  [{key}] {label} ({format_gb(tot_b)}, remaining {format_gb(rem_b)}){mark} — {note}")

    if dry_run:
        chosen_prof = dict((k, p) for k, p, _, _ in menu_items)[rec_key]
        print(f"\n[DRY-RUN] Recommended bundle: [{rec_key}] -> profile='{chosen_prof}' ({rec_reason})")
        return 0

    if use_rich:
        choice = Prompt.ask("Bundle to download", choices=["1", "2", "3", "4", "5", "6"], default=rec_key)
    else:
        choice = input(f"Bundle to download [{rec_key}]: ").strip() or rec_key
        if choice not in ("1", "2", "3", "4", "5", "6"):
            choice = rec_key

    chosen_prof = dict((k, p) for k, p, _, _ in menu_items)[choice]
    file_ids = list(profiles[chosen_prof]["required"])
    if chosen_prof in ("t2v", "all"):
        file_ids += list(profiles[chosen_prof].get("optional", []))

    # Optional Z-Image-Turbo add-on when user picks fast-5b or starter T2V
    if chosen_prof in ("fast-5b", "t2v"):
        z_ids = list(profiles["t2i"]["required"])
        z_rem = sum(check_file(files_map[fid])[2] for fid in z_ids)
        if z_rem > 0:
            if use_rich:
                add_t2i = Confirm.ask(
                    f"Optional: also include Z-Image-Turbo (+{format_gb(z_rem)}) for first-frame / photo generation?",
                    default=False,
                )
            else:
                ans = input(f"Optional: also include Z-Image-Turbo (+{format_gb(z_rem)})? [y/N]: ").strip().lower()
                add_t2i = ans in ("y", "yes", "e", "evet")
            if add_t2i:
                file_ids += z_ids

    ok = download_file_ids(file_ids, reg, force_low_disk=force_low_disk)
    return 0 if ok else 1


def main() -> int:
    configure_utf8()
    reg = load_registry()
    profiles = reg["download_profiles"]
    files_map = reg["files"]

    parser = argparse.ArgumentParser(description="PicoVid: Download supported models from configs/models_registry.json (see registry for upstream licenses)")
    parser.add_argument(
        "--profile",
        choices=list(profiles.keys()),
        default=None,
        help="Model profile to download or check (omit to open hardware-aware interactive wizard)",
    )
    parser.add_argument("--interactive", action="store_true", help="Open hardware-aware interactive download wizard")
    parser.add_argument("--simulate-hw", choices=["cuda-4gb-12gb", "vulkan-4gb-16gb", "vulkan-6gb-20gb", "vulkan-6gb-32gb", "cuda-8gb-32gb"], default=None, help="Simulate hardware specs for recommendation testing")
    parser.add_argument("--dry-run", action="store_true", help="Show hardware recommendation without downloading")
    parser.add_argument("--with-taehv", action="store_true", help="Also include optional TAEHV (21.6 MiB) for --fast-preview")
    parser.add_argument("--check", action="store_true", help="Verify downloaded files without downloading")
    parser.add_argument("--list-profiles", action="store_true", help="List available download profiles and sizes")
    parser.add_argument("--force-low-disk", action="store_true", help="Proceed even if free disk headroom < 1.5 GiB")
    args = parser.parse_args()

    if args.list_profiles:
        print("=== PicoVid Model Profiles ===")
        for pname, pdef in profiles.items():
            req_bytes = sum(files_map[fid]["expected_bytes"] for fid in pdef["required"])
            print(f"  {pname:15s} ({format_gb(req_bytes):>9s})  {pdef['desc']}")
        return 0

    if args.interactive or (args.profile is None and not args.check):
        return run_interactive_downloader(
            reg,
            simulate_hw=args.simulate_hw,
            dry_run=args.dry_run,
            force_low_disk=args.force_low_disk,
        )

    profile_name = args.profile or "t2v"
    pdef = profiles[profile_name]
    req_ids = list(pdef["required"])
    opt_ids = list(pdef.get("optional", []))
    if args.with_taehv and "taew2_1" not in req_ids and "taew2_1" not in opt_ids:
        opt_ids.append("taew2_1")

    selected: list[tuple[str, bool, dict[str, Any]]] = []
    for fid in req_ids:
        selected.append((fid, True, files_map[fid]))
    for fid in opt_ids:
        if args.with_taehv or args.check:
            selected.append((fid, False, files_map[fid]))

    if args.check:
        print(f"=== PicoVid Model Integrity Check (profile={profile_name}) ===")
        all_req_ok = True
        for fid, is_req, item in selected:
            ok, msg, _ = check_file(item)
            tag = "REQ" if is_req else "OPT"
            print(f"  [{tag}] {item['path']}: {msg}")
            if is_req and not ok:
                all_req_ok = False
        return 0 if all_req_ok else 1

    if args.dry_run:
        needed = sum(check_file(item)[2] for _, _, item in selected)
        print(f"[DRY-RUN] profile={profile_name} | files={len(selected)} | remaining={format_gb(needed)}")
        return 0

    ok = download_file_ids([fid for fid, _, _ in selected], reg, force_low_disk=args.force_low_disk)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
