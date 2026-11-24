#!/usr/bin/env python3
"""
PicoVid Hardware-Aware Model Downloader driven by configs/models_registry.json.

Supports:
- Interactive Smart Download Wizard (when launched with no arguments or --interactive):
  detects GPU VRAM, total system RAM, and free disk space, and highlights the
  best-fit bundle with a '★ Önerilen' badge.
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


def download_file(item: dict[str, Any]) -> bool:
    p = resolve_path(item["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, msg, _ = check_file(item)
    if ok:
        print(f"  [SKIP] {item['path']} — {msg}")
        return True

    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        print("HATA: Devam ettirilebilir indirme için curl gerekli.", file=sys.stderr)
        return False

    print(f"  [GET ] {item['desc']}")
    print(f"         -> {item['path']}")
    cmd = [
        curl, "-L", "-C", "-",
        "--retry", "5", "--retry-delay", "3",
        "--speed-limit", "50000", "--speed-time", "25",
        "-#", item["url"], "-o", str(p),
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"  [FAIL] curl çıkış kodu: {proc.returncode}", file=sys.stderr)
        return False

    ok, msg, _ = check_file(item)
    print(f"  [{'OK  ' if ok else 'WARN'}] {item['path']} — {msg}")
    return ok


def detect_host_summary(simulate_hw: str | None = None) -> dict[str, Any]:
    """Detect GPU VRAM, total system RAM, and target drive free disk space."""
    if simulate_hw:
        sim_presets: dict[str, dict[str, Any]] = {
            "cuda-4gb-12gb": {
                "gpu_name": "NVIDIA GeForce GTX 1650 (Simüle)",
                "backend": "cuda",
                "vram_total_gib": 4.0,
                "ram_total_gib": 12.0,
                "disk_free_gib": 40.0,
                "disk_drive": "C:",
            },
            "vulkan-4gb-16gb": {
                "gpu_name": "AMD Radeon RX 5500 XT 4GB (Simüle)",
                "backend": "vulkan",
                "vram_total_gib": 4.0,
                "ram_total_gib": 16.0,
                "disk_free_gib": 35.0,
                "disk_drive": "C:",
            },
            "vulkan-6gb-20gb": {
                "gpu_name": "AMD Radeon RX 5600 XT 6GB (Simüle)",
                "backend": "vulkan",
                "vram_total_gib": 6.0,
                "ram_total_gib": 20.0,
                "disk_free_gib": 27.0,
                "disk_drive": "C:",
            },
            "vulkan-6gb-32gb": {
                "gpu_name": "AMD Radeon RX 5600 XT 6GB (Simüle)",
                "backend": "vulkan",
                "vram_total_gib": 6.0,
                "ram_total_gib": 32.0,
                "disk_free_gib": 50.0,
                "disk_drive": "C:",
            },
            "cuda-8gb-32gb": {
                "gpu_name": "NVIDIA GeForce RTX 4060 8GB (Simüle)",
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
        "gpu_name": "Bilinmeyen GPU",
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
        return "1", f"4–6 GB VRAM / hızlı 5 sn video için en hafif paket (GPU {vram:.1f} GiB, RAM {ram:.1f} GiB)"
    if ram >= 24.0 and disk >= 29.5:
        return "3", f"6 GB+ VRAM ({vram:.1f} GiB) ve >=24 GB RAM ({ram:.1f} GiB) ile tam kalite 14B Stüdyo"
    return "4", f"Düşük disk/RAM için kompakt Q3_K_M Stüdyo (RAM {ram:.1f} GiB, Disk {disk:.1f} GiB)"


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
            f"HATA: Yetersiz boş disk alanı!\n"
            f"  Gereken : {format_gb(needed_bytes)}\n"
            f"  Boş     : {format_gb(free_bytes)}\n"
            f"  İpucu   : Yer açın, 'i2v-compact' (Q3_K_M) seçin veya --force-low-disk kullanın.",
            file=sys.stderr,
        )
        return False

    print(
        f"[PicoVid] {len(unique_ids)} dosya kontrol ediliyor "
        f"(indirilecek: {format_gb(needed_bytes)}, boş disk: {format_gb(free_bytes)})..."
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
        ("1", "fast-5b", "Hızlı 5sn Paket: Wan 2.2 TI2V-5B Turbo (T2V + I2V)", "5 saniyelik videolar (~3-6 dk), 4-6 GB VRAM"),
        ("2", "t2v", "14B T2V Başlangıç (Wan 2.2 14B T2V Q4_K + TE + VAE)", "14B tek model T2V"),
        ("3", "studio", "14B Stüdyo Paketi: T2I + I2V Q4_K_M", "6 GB+ VRAM ve >=24 GB RAM için"),
        ("4", "studio-compact", "14B Stüdyo Kompakt: T2I + I2V Q3_K_M", "Düşük disk/RAM alternatifi"),
        ("5", "all", "Tam Paket: 5B Turbo + 14B T2V + T2I + I2V", "Tüm modlar eksiksiz"),
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
            f"[bold cyan]Boş Disk ({host['disk_drive'] or 'hedef'}):[/bold cyan] {host['disk_free_gib']:.1f} GiB\n"
            f"[green]★ Öneri Nedeni:[/green] {rec_reason}"
        )
        console.print()
        console.print(Panel(hdr, title="[bold cyan]PicoVid v0.3.0 · Akıllı Model İndirme Sihirbazı[/bold cyan]", border_style="cyan", box=box.ROUNDED))

        tbl = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
        tbl.add_column("Seçim", style="bold cyan", width=6)
        tbl.add_column("Paket", style="white", ratio=3)
        tbl.add_column("Toplam", justify="right", width=11)
        tbl.add_column("Kalan", justify="right", width=11)
        tbl.add_column("Not / Öneri", ratio=2)

        for key, prof_key, label, note in menu_items:
            tot_b, rem_b = calc_pack_sizes(prof_key)
            badge = "[bold green]★ Önerilen[/bold green] · " + note if key == rec_key else f"[dim]{note}[/dim]"
            rem_txt = "[green]Hazır[/green]" if rem_b == 0 else format_gb(rem_b)
            tbl.add_row(f"[{key}]", label, format_gb(tot_b), rem_txt, badge)
        console.print(tbl)
    else:
        print("=== PicoVid v0.2.0 · Akıllı Model İndirme Sihirbazı ===")
        print(
            f"Donanım: {host['gpu_name']} ({host['backend'].upper()}, {host['vram_total_gib']:.1f} GiB VRAM) | "
            f"RAM: {host['ram_total_gib']:.1f} GiB | Boş Disk: {host['disk_free_gib']:.1f} GiB"
        )
        print(f"Öneri  : [{rec_key}] — {rec_reason}")
        for key, prof_key, label, note in menu_items:
            tot_b, rem_b = calc_pack_sizes(prof_key)
            mark = " ★ Önerilen" if key == rec_key else ""
            print(f"  [{key}] {label} ({format_gb(tot_b)}, kalan {format_gb(rem_b)}){mark} — {note}")

    if dry_run:
        chosen_prof = dict((k, p) for k, p, _, _ in menu_items)[rec_key]
        print(f"\n[DRY-RUN] Önerilen paket: [{rec_key}] -> profile='{chosen_prof}' ({rec_reason})")
        return 0

    if use_rich:
        choice = Prompt.ask("İndirmek istediğiniz paket", choices=["1", "2", "3", "4", "5"], default=rec_key)
    else:
        choice = input(f"İndirmek istediğiniz paket [{rec_key}]: ").strip() or rec_key
        if choice not in ("1", "2", "3", "4", "5"):
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
                    f"Opsiyonel: İlk kare / fotoğraf üretimi için Z-Image-Turbo (+{format_gb(z_rem)}) da eklensin mi?",
                    default=False,
                )
            else:
                ans = input(f"Opsiyonel: Z-Image-Turbo (+{format_gb(z_rem)}) da eklensin mi? [e/H]: ").strip().lower()
                add_t2i = ans in ("e", "evet", "y", "yes")
            if add_t2i:
                file_ids += z_ids

    ok = download_file_ids(file_ids, reg, force_low_disk=force_low_disk)
    return 0 if ok else 1


def main() -> int:
    configure_utf8()
    reg = load_registry()
    profiles = reg["download_profiles"]
    files_map = reg["files"]

    parser = argparse.ArgumentParser(description="PicoVid: Download Apache-2.0 models from configs/models_registry.json")
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
        print("=== PicoVid Model Profilleri ===")
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
        print(f"=== PicoVid Model Bütünlük Kontrolü (profile={profile_name}) ===")
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
