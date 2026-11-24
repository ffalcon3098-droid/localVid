#!/usr/bin/env python3
"""
localVid Rich Terminal UI, Live Dashboard, VRAM Sampler, and Interactive Wizard.

Provides:
- Interactive setup wizard (when launched without --prompt or via localvid.cmd)
- Single-screen animated Rich Live dashboard with 4 stages, smooth DiT block/step
  progress parsing, live GPU Dedicated/Shared VRAM bar, and completion summary card
- Clean plain-text fallback when Rich is unavailable, non-TTY, or --raw-logs/--verbose
- 5-6 second --ui-demo mode to verify terminal rendering on Windows PowerShell/CMD

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import os
import pathlib
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

try:
    from rich import box
    from rich.align import Align
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.prompt import Confirm, Prompt
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


DEFAULT_SAMPLE_PROMPT = (
    "A cinematic tracking shot of a vintage red sports car driving down a "
    "rain-slicked neon-lit city street at sunset, golden light reflecting off "
    "wet asphalt, realistic motion, 35mm film"
)

BANNER_ART = r"""
  _                 ___     ___     _
 | | ___   ___ __ _| \ \   / (_) __| |
 | |/ _ \ / __/ _` | |\ \ / /| |/ _` |
 | | (_) | (_| (_| | | \ V / | | (_| |
 |_|\___/ \___\__,_|_|  \_/  |_|\__,_|
"""


def configure_console_utf8() -> None:
    """Ensure stdout/stderr won't crash on Turkish/box characters on Windows."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def is_interactive_tty() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def should_use_rich(raw_logs: bool = False, verbose: bool = False, force_rich: bool = False) -> bool:
    if not HAS_RICH:
        return False
    if raw_logs or verbose:
        return False
    if force_rich:
        return True
    return is_interactive_tty()


def make_console(force_terminal: bool | None = None) -> Any:
    configure_console_utf8()
    if not HAS_RICH:
        return None
    return Console(force_terminal=force_terminal, highlight=False)


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(round(seconds))
    mins, secs = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def format_file_size(path: pathlib.Path) -> str:
    try:
        sz = path.stat().st_size
    except OSError:
        return "-"
    if sz >= 1024 * 1024:
        return f"{sz / (1024 * 1024):.2f} MB"
    if sz >= 1024:
        return f"{sz / 1024:.1f} KB"
    return f"{sz} B"


class LiveVramSampler:
    """Background GPU Dedicated & Shared VRAM poller (~1s interval).

    Uses nvidia-smi on NVIDIA GPUs, or a single persistent PowerShell
    Get-Counter loop on Windows AMD/Intel GPUs (avoids per-second process spawn).
    """

    def __init__(self, total_vram_gib: float = 6.0, interval_sec: float = 1.0) -> None:
        self.total_vram_mib = max(1024.0, total_vram_gib * 1024.0)
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._ps_proc: subprocess.Popen[str] | None = None
        self.has_nvidia = shutil.which("nvidia-smi") is not None

        self.current_ded_mib: float | None = None
        self.current_shr_mib: float | None = None
        self.peak_ded_mib: float = 0.0
        self.peak_shr_mib: float = 0.0

    def _record(self, ded_mib: float | None, shr_mib: float | None) -> None:
        with self._lock:
            if ded_mib is not None:
                self.current_ded_mib = ded_mib
                if ded_mib > self.peak_ded_mib:
                    self.peak_ded_mib = ded_mib
            if shr_mib is not None:
                self.current_shr_mib = shr_mib
                if shr_mib > self.peak_shr_mib:
                    self.peak_shr_mib = shr_mib

    def inject_sample(self, ded_mib: float, shr_mib: float) -> None:
        """Used by --ui-demo to drive realistic live VRAM curves."""
        self._record(ded_mib, shr_mib)

    def snapshot(self) -> dict[str, float | None]:
        with self._lock:
            return {
                "current_ded_mib": self.current_ded_mib,
                "current_shr_mib": self.current_shr_mib,
                "peak_ded_mib": self.peak_ded_mib if self.peak_ded_mib > 0 else None,
                "peak_shr_mib": self.peak_shr_mib if self.peak_shr_mib > 0 else None,
                "total_vram_mib": self.total_vram_mib,
            }

    def _poll_nvidia(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=3,
                    stderr=subprocess.DEVNULL,
                )
                first = out.strip().splitlines()[0]
                parts = [p.strip() for p in first.split(",")]
                used = float(parts[0])
                if len(parts) >= 2:
                    self.total_vram_mib = max(1024.0, float(parts[1]))
                self._record(used, 0.0)
            except Exception:
                pass
            self._stop.wait(self.interval_sec)

    def _poll_windows_perfcounter(self) -> None:
        interval_i = max(1, int(round(self.interval_sec)))
        ps_script = (
            "$ErrorActionPreference='SilentlyContinue'; "
            f"Get-Counter -Counter '\\GPU Adapter Memory(*)\\Dedicated Usage','\\GPU Adapter Memory(*)\\Shared Usage' "
            f"-SampleInterval {interval_i} -Continuous | ForEach-Object {{ "
            "$cs = $_.CounterSamples; "
            "$d = ($cs | Where-Object { $_.Path -like '*dedicated usage' } | "
            "Measure-Object -Property CookedValue -Maximum).Maximum; "
            "$s = ($cs | Where-Object { $_.Path -like '*shared usage' } | "
            "Measure-Object -Property CookedValue -Maximum).Maximum; "
            "if ($null -ne $d) { "
            "$di = [long][math]::Round([double]$d); "
            "$si = if ($null -ne $s) { [long][math]::Round([double]$s) } else { 0 }; "
            "[Console]::Out.WriteLine(\"$di,$si\"); [Console]::Out.Flush() "
            "} }"
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._ps_proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-Command", ps_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                creationflags=creationflags,
            )
            assert self._ps_proc.stdout is not None
            for line in self._ps_proc.stdout:
                if self._stop.is_set():
                    break
                line = line.strip()
                if not line or "," not in line:
                    continue
                try:
                    d_str, s_str = line.split(",", 1)
                    ded = float(d_str) / (1024.0 * 1024.0) if d_str else None
                    shr = float(s_str) / (1024.0 * 1024.0) if s_str else None
                    self._record(ded, shr)
                except ValueError:
                    continue
        except Exception:
            pass
        finally:
            self._terminate_ps()

    def _terminate_ps(self) -> None:
        proc = self._ps_proc
        self._ps_proc = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def start(self) -> None:
        self._stop.clear()
        target: Callable[[], None] | None = None
        if self.has_nvidia:
            target = self._poll_nvidia
        elif platform.system() == "Windows":
            target = self._poll_windows_perfcounter
        if target is not None:
            self._thread = threading.Thread(target=target, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._terminate_ps()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None


@dataclass
class StageState:
    key: str
    label: str
    status: str = "pending"  # pending | running | done | skipped | failed
    pct: float | None = None  # None = indeterminate spinner, 0..100 = determinate bar
    detail: str = ""
    eta_sec: float | None = None
    started_at: float | None = None
    elapsed_sec: float = 0.0


@dataclass
class DashboardConfig:
    gpu_name: str
    backend: str
    vram_class: str
    vram_total_gib: float
    vram_budget_gib: float
    profile_name: str
    width: int
    height: int
    frames: int
    fps: int
    steps: int
    vae_label: str
    log_path: pathlib.Path | None = None
    demo_mode: bool = False


class BaseDashboard:
    """Common interface for Rich Live Dashboard and Plain-Text fallback."""

    def __init__(self, config: DashboardConfig, vram_sampler: LiveVramSampler | None = None) -> None:
        self.config = config
        self.vram_sampler = vram_sampler
        self.t0 = time.monotonic()
        self.last_status = "Başlatılıyor..."
        self.error_lines: list[str] = []
        self._lock = threading.Lock()
        self.stages: dict[str, StageState] = {
            "te": StageState("te", "1. Metin Kodlama (UMT5-XXL CPU)"),
            "dit": StageState("dit", "2. DiT Örnekleme (Wan 2.2 14B)"),
            "vae": StageState("vae", f"3. VAE Decode ({config.vae_label})"),
            "mp4": StageState("mp4", "4. H.264 MP4 Paketleme"),
        }

    def start(self) -> None:
        self.t0 = time.monotonic()

    def stop(self) -> None:
        pass

    def start_stage(self, key: str, detail: str = "", determinate: bool = False) -> None:
        with self._lock:
            st = self.stages[key]
            if st.status != "running":
                st.status = "running"
                st.started_at = time.monotonic()
                st.pct = 0.0 if determinate else None
            if detail:
                st.detail = detail

    def update_stage(
        self,
        key: str,
        pct: float | None = None,
        detail: str | None = None,
        eta_sec: float | None = None,
    ) -> None:
        with self._lock:
            st = self.stages[key]
            if st.status == "pending":
                st.status = "running"
                st.started_at = time.monotonic()
            if pct is not None:
                st.pct = max(0.0, min(100.0, pct))
            if detail is not None:
                st.detail = detail
            if eta_sec is not None:
                st.eta_sec = eta_sec
            if st.started_at is not None and st.status == "running":
                st.elapsed_sec = time.monotonic() - st.started_at

    def finish_stage(self, key: str, detail: str = "", elapsed_override: float | None = None) -> None:
        with self._lock:
            st = self.stages[key]
            if elapsed_override is not None:
                st.elapsed_sec = elapsed_override
            elif st.started_at is not None:
                st.elapsed_sec = time.monotonic() - st.started_at
            st.status = "done"
            st.pct = 100.0
            st.eta_sec = 0.0
            if detail:
                st.detail = detail

    def fail_stage(self, key: str, detail: str = "") -> None:
        with self._lock:
            st = self.stages[key]
            if st.started_at is not None:
                st.elapsed_sec = time.monotonic() - st.started_at
            st.status = "failed"
            if detail:
                st.detail = detail

    def set_status(self, msg: str) -> None:
        with self._lock:
            self.last_status = msg[:140]

    def add_error(self, msg: str) -> None:
        with self._lock:
            self.error_lines.append(msg.strip())
            if len(self.error_lines) > 12:
                self.error_lines = self.error_lines[-12:]

    def render_summary(
        self,
        success: bool,
        output_path: pathlib.Path | None,
        encoder_label: str = "",
        error_msg: str = "",
    ) -> None:
        raise NotImplementedError


class PlainDashboard(BaseDashboard):
    """Safe plain-text fallback for non-TTY, CI, --raw-logs, or missing Rich."""

    def start(self) -> None:
        super().start()
        c = self.config
        print(
            f"[localVid] GPU: {c.gpu_name} ({c.backend.upper()}, {c.vram_class}, "
            f"bütçe {c.vram_budget_gib:.2f} GiB)"
        )
        print(
            f"[localVid] Profil: {c.profile_name} | {c.width}x{c.height} x {c.frames}f "
            f"({c.frames / max(1, c.fps):.1f}s) | {c.steps} adım | {c.vae_label}"
        )

    def start_stage(self, key: str, detail: str = "", determinate: bool = False) -> None:
        was_running = self.stages[key].status == "running"
        super().start_stage(key, detail, determinate)
        if not was_running:
            st = self.stages[key]
            suffix = f" - {detail}" if detail else ""
            print(f"[localVid] >> {st.label}{suffix}", flush=True)

    def finish_stage(self, key: str, detail: str = "", elapsed_override: float | None = None) -> None:
        super().finish_stage(key, detail, elapsed_override)
        st = self.stages[key]
        suffix = f" ({detail})" if detail else ""
        print(f"[localVid] [OK] {st.label} tamamlandı: {st.elapsed_sec:.1f}s{suffix}", flush=True)

    def fail_stage(self, key: str, detail: str = "") -> None:
        super().fail_stage(key, detail)
        st = self.stages[key]
        print(f"[localVid] [HATA] {st.label}: {detail}", file=sys.stderr, flush=True)

    def render_summary(
        self,
        success: bool,
        output_path: pathlib.Path | None,
        encoder_label: str = "",
        error_msg: str = "",
    ) -> None:
        total = time.monotonic() - self.t0
        vram = self.vram_sampler.snapshot() if self.vram_sampler else {}
        ded_peak = vram.get("peak_ded_mib")
        shr_peak = vram.get("peak_shr_mib")
        print("-" * 64)
        print(f"[localVid] Durum : {'TAMAMLANDI' if success else 'BAŞARISIZ'} ({total:.1f}s)")
        for st in self.stages.values():
            print(f"  - {st.label}: {st.status} ({st.elapsed_sec:.1f}s)")
        if ded_peak is not None:
            shr_txt = f" | Tepe Shared: {shr_peak:.0f} MiB" if shr_peak is not None else ""
            print(f"  - Tepe Dedicated VRAM: {ded_peak:.0f} MiB{shr_txt}")
        if output_path is not None and success:
            enc = f" ({encoder_label})" if encoder_label else ""
            print(f"  - Çıktı: {output_path}{enc} [{format_file_size(output_path)}]")
        if self.config.log_path:
            print(f"  - Log  : {self.config.log_path}")
        if error_msg and not success:
            print(f"  - Hata : {error_msg}", file=sys.stderr)
        print("-" * 64)


class RichDashboard(BaseDashboard):
    """Single-screen animated Rich Live dashboard."""

    SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(
        self,
        config: DashboardConfig,
        vram_sampler: LiveVramSampler | None = None,
        console: Any = None,
        use_alt_screen: bool = True,
    ) -> None:
        super().__init__(config, vram_sampler)
        self.console = console or make_console()
        self.use_alt_screen = use_alt_screen
        self._live: Live | None = None

    def _build_bar(self, pct: float, width: int = 24, filled_style: str = "cyan") -> Text:
        pct_clamped = max(0.0, min(100.0, pct))
        filled = int(round((pct_clamped / 100.0) * width))
        empty = max(0, width - filled)
        t = Text()
        t.append("█" * filled, style=filled_style)
        t.append("░" * empty, style="bright_black")
        return t

    def _build_renderable(self) -> Any:
        c = self.config
        now = time.monotonic()
        total_elapsed = now - self.t0
        spin_idx = int(now * 10) % len(self.SPINNER_FRAMES)
        spinner_char = self.SPINNER_FRAMES[spin_idx]
        term_w = getattr(self.console, "width", 100) or 100
        bar_w = 22 if term_w >= 104 else 14

        # 1. Single-line non-wrapping Header metadata strip (prevents line-wrap ghosting)
        header = Table.grid(expand=True, padding=(0, 1))
        header.add_column(style="bold cyan", ratio=3, no_wrap=True, overflow="ellipsis")
        header.add_column(style="white", ratio=2, justify="right", no_wrap=True, overflow="ellipsis")

        backend_tag = "Vulkan0" if c.backend == "vulkan" else "CUDA"
        demo_tag = " [DEMO]" if c.demo_mode else ""
        clip_sec = c.frames / max(1, c.fps)
        header.add_row(
            f"{c.gpu_name} · {backend_tag} · {c.profile_name}",
            f"{c.width}x{c.height} · {c.frames}f ({clip_sec:.1f}s) · {c.steps} adım · {format_duration(total_elapsed)}",
        )

        # 2. 4-Stage Progress Table (strictly no_wrap on every column)
        stage_tbl = Table(
            box=box.SIMPLE_HEAD,
            expand=True,
            show_edge=False,
            pad_edge=False,
            padding=(0, 1),
        )
        stage_tbl.add_column("Aşama", ratio=3, no_wrap=True, overflow="ellipsis")
        stage_tbl.add_column("İlerleme", width=bar_w + 2, no_wrap=True)
        stage_tbl.add_column("%", width=5, justify="right", no_wrap=True)
        stage_tbl.add_column("Süre / ETA", width=18, justify="right", no_wrap=True, overflow="ellipsis")
        stage_tbl.add_column("Detay", ratio=2, overflow="ellipsis", no_wrap=True)

        with self._lock:
            stages_copy = [StageState(**vars(st)) for st in self.stages.values()]
            status_line = self.last_status

        for st in stages_copy:
            elapsed = (
                (now - st.started_at)
                if (st.status == "running" and st.started_at is not None)
                else st.elapsed_sec
            )
            if st.status == "done":
                icon = Text("✓ ", style="bold green")
                label_txt = icon + Text(st.label, style="green")
                bar = self._build_bar(100.0, width=bar_w, filled_style="green")
                pct_txt = Text("100%", style="bold green")
                time_txt = Text(format_duration(elapsed), style="green")
                det_txt = Text(st.detail or "Tamamlandı", style="dim green")
            elif st.status == "failed":
                icon = Text("✗ ", style="bold red")
                label_txt = icon + Text(st.label, style="bold red")
                bar = self._build_bar(st.pct or 0.0, width=bar_w, filled_style="red")
                pct_txt = Text("HATA", style="bold red")
                time_txt = Text(format_duration(elapsed), style="red")
                det_txt = Text(st.detail or "Başarısız", style="red")
            elif st.status == "running":
                icon = Text(f"{spinner_char} ", style="bold cyan")
                label_txt = icon + Text(st.label, style="bold white")
                if st.pct is not None:
                    bar = self._build_bar(st.pct, width=bar_w, filled_style="cyan")
                    pct_txt = Text(f"{st.pct:4.0f}%", style="bold cyan")
                else:
                    # Animated sliding pulse for indeterminate stages (e.g. UMT5 CPU)
                    pulse_len = min(4, max(2, bar_w // 5))
                    span = max(1, bar_w - pulse_len)
                    pulse_pos = int(now * 12) % (span + 1)
                    bar = Text()
                    bar.append("░" * pulse_pos, style="bright_black")
                    bar.append("█" * pulse_len, style="cyan")
                    bar.append("░" * max(0, bar_w - pulse_len - pulse_pos), style="bright_black")
                    pct_txt = Text(" ...", style="cyan")
                if st.eta_sec is not None and st.eta_sec > 0.5:
                    time_txt = Text(
                        f"{format_duration(elapsed)} · ETA {format_duration(st.eta_sec)}",
                        style="cyan",
                    )
                else:
                    time_txt = Text(format_duration(elapsed), style="cyan")
                det_txt = Text(st.detail or "Çalışıyor...", style="white")
            else:
                icon = Text("○ ", style="dim")
                label_txt = icon + Text(st.label, style="dim")
                bar = self._build_bar(0.0, width=bar_w, filled_style="bright_black")
                pct_txt = Text("  0%", style="dim")
                time_txt = Text("--:--", style="dim")
                det_txt = Text("Bekliyor", style="dim")

            stage_tbl.add_row(label_txt, bar, pct_txt, time_txt, det_txt)

        # 3. Single-line Live VRAM Bars (no_wrap prevents height changes)
        vram_tbl = Table.grid(expand=True, padding=(0, 1))
        vram_tbl.add_column(width=15, style="bold", no_wrap=True)
        vram_tbl.add_column(width=bar_w + 2, no_wrap=True)
        vram_tbl.add_column(ratio=1, no_wrap=True, overflow="ellipsis")

        snap = self.vram_sampler.snapshot() if self.vram_sampler else {}
        cur_ded = snap.get("current_ded_mib")
        cur_shr = snap.get("current_shr_mib")
        peak_ded = snap.get("peak_ded_mib")
        peak_shr = snap.get("peak_shr_mib")
        tot_mib = snap.get("total_vram_mib") or (c.vram_total_gib * 1024.0)

        if cur_ded is not None:
            ded_pct = (cur_ded / max(1.0, tot_mib)) * 100.0
            v_color = "green" if ded_pct < 75 else ("yellow" if ded_pct < 90 else "red")
            ded_bar = self._build_bar(ded_pct, width=bar_w, filled_style=v_color)
            ded_info = (
                f"[{v_color}]{cur_ded / 1024.0:.2f}/{tot_mib / 1024.0:.2f} GiB "
                f"({ded_pct:.0f}%)[/{v_color}] · Tepe: {(peak_ded or cur_ded) / 1024.0:.2f} GiB"
            )
            vram_tbl.add_row("Dedicated VRAM", ded_bar, ded_info)

            if cur_shr is not None:
                shr_pct = min(100.0, (cur_shr / 4096.0) * 100.0)
                s_color = "green" if cur_shr < 768 else ("yellow" if cur_shr < 1536 else "red")
                shr_bar = self._build_bar(shr_pct, width=bar_w, filled_style=s_color)
                shr_info = (
                    f"[{s_color}]{cur_shr / 1024.0:.2f} GiB[/{s_color}] · "
                    f"Tepe: {(peak_shr or cur_shr) / 1024.0:.2f} GiB"
                )
                vram_tbl.add_row("Shared VRAM", shr_bar, shr_info)
        else:
            vram_tbl.add_row(
                "Canlı VRAM",
                self._build_bar(0.0, width=bar_w, filled_style="bright_black"),
                "[dim]GPU bellek sayacı okunuyor...[/dim]",
            )

        # 4. Single-line Footer / Last Event Line (no_wrap + ellipsis)
        log_hint = f" · Log: {c.log_path.name}" if c.log_path else ""
        footer = Text.assemble(
            ("Son Durum: ", "bold magenta"),
            (status_line, "white"),
            (log_hint, "dim"),
        )
        footer.no_wrap = True
        footer.overflow = "ellipsis"

        body = Group(
            header,
            Rule(style="bright_black"),
            stage_tbl,
            Rule(style="bright_black"),
            vram_tbl,
            Rule(style="bright_black"),
            footer,
        )

        return Panel(
            body,
            title=f"[bold cyan]localVid v0.1.0{demo_tag}[/bold cyan] · [white]Wan 2.2 14B Motor Paneli[/white]",
            subtitle="[dim]Ctrl+C ile iptal · Ham loglar .log dosyasına kaydedilir[/dim]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def start(self) -> None:
        super().start()
        self._live = Live(
            get_renderable=self._build_renderable,
            console=self.console,
            screen=self.use_alt_screen,
            auto_refresh=True,
            refresh_per_second=6,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            try:
                self._live.refresh()
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def render_summary(
        self,
        success: bool,
        output_path: pathlib.Path | None,
        encoder_label: str = "",
        error_msg: str = "",
    ) -> None:
        total = time.monotonic() - self.t0
        snap = self.vram_sampler.snapshot() if self.vram_sampler else {}
        peak_ded = snap.get("peak_ded_mib")
        peak_shr = snap.get("peak_shr_mib")

        tbl = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
        tbl.add_column("Anahtar", style="bold cyan", width=26)
        tbl.add_column("Değer", style="white")

        tbl.add_row(
            "Durum",
            "[bold green]✓ Başarıyla Tamamlandı[/bold green]"
            if success
            else f"[bold red]✗ Başarısız ({error_msg or 'hata'})[/bold red]",
        )
        tbl.add_row("Toplam Süre", f"[bold]{format_duration(total)}[/bold] ({total:.1f} sn)")

        breakdown = "  |  ".join(
            f"{st.key.upper()}: {st.elapsed_sec:.1f}s" for st in self.stages.values()
        )
        tbl.add_row("Aşama Kırılımı", breakdown)

        if peak_ded is not None:
            vram_txt = f"Dedicated: {peak_ded / 1024.0:.2f} GiB ({peak_ded:.0f} MiB)"
            if peak_shr is not None:
                vram_txt += f"  |  Shared: {peak_shr / 1024.0:.2f} GiB ({peak_shr:.0f} MiB)"
            tbl.add_row("Tepe GPU Belleği", vram_txt)

        if output_path is not None and success:
            enc_txt = f", {encoder_label}" if encoder_label else ""
            size_txt = format_file_size(output_path) if output_path.exists() else "demo"
            tbl.add_row("Üretilen Video", f"[bold green]{output_path}[/bold green] ({size_txt}{enc_txt})")

        if self.config.log_path is not None:
            tbl.add_row("Ham Motor Logu", f"[dim]{self.config.log_path}[/dim]")

        if not success and self.error_lines:
            tbl.add_row("Son Hata Satırları", "\n".join(self.error_lines[-4:]))

        panel = Panel(
            tbl,
            title="[bold green]Üretim Özeti[/bold green]" if success else "[bold red]Üretim Raporu (Hata)[/bold red]",
            border_style="green" if success else "red",
            box=box.ROUNDED,
            padding=(0, 1),
        )
        self.console.print()
        self.console.print(panel)


# Regexes for real-time sd-cli stdout/stderr parsing
RE_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
RE_TE_DONE_MS = re.compile(r"computing condition graph completed, taking ([\d.]+) ms")
RE_TE_DONE_S = re.compile(r"get_learned_condition completed, taking ([\d.]+)s")
RE_SAMPLE_START = re.compile(r"sampling started:\s*method=(\S+),\s*steps=(\d+)")
RE_TOTAL_SEGS = re.compile(r"Wan2\.x-T2V-14B using (\d+) segments")
RE_SEG_TIMING = re.compile(
    r"seg_timing:\s*(\S+)\s+seg\s+(\d+)/(\d+)\s+\(([^)]+)\):\s*([\d.]+)\s*ms"
)
RE_BLOCK_NAME = re.compile(r"wan\.blocks\.(\d+)\.([a-zA-Z0-9_]+)")
RE_PROGRESS_BAR = re.compile(
    r"\|[=#>\-\s]+\|\s*(\d+)/(\d+)\s*-\s*([\d.]+)(s/it|it/s)(?:\s*-\s*ETA\s*(\d+):(\d+))?"
)
RE_SAMPLE_DONE = re.compile(r"sampling completed, taking ([\d.]+)s")
RE_LATENT_SAVE = re.compile(r"SD_VIDEO_LATENT_SAVE:\s*saved\s+\d+\s+floats\s+\(([^)]+)\)")
RE_VAE_TILES_TOTAL = re.compile(r"processing (\d+) tiles")
RE_VAE_TILE_SIZE = re.compile(r"VAE decode tile size:\s*(\d+x\d+)")
RE_VAE_TEMP_TILE = re.compile(r"Wan VAE temporal tile (\d+)/(\d+)")
RE_VAE_DONE = re.compile(r"(?:computing vae decode graph|decode_first_stage) completed, taking ([\d.]+)s")


def iter_stream_lines(stream: Any):
    """Yield lines immediately on either \\r or \\n so progress bars never stall."""
    buf: list[str] = []
    while True:
        ch = stream.read(1)
        if not ch:
            if buf:
                yield "".join(buf)
            break
        if ch in ("\r", "\n"):
            if buf:
                yield "".join(buf)
                buf.clear()
        else:
            buf.append(ch)


def run_process_monitored(
    cmd: list[str],
    env: dict[str, str],
    log_file: Any,
    dashboard: BaseDashboard,
    phase: str,  # "process1" | "process2"
    total_steps: int,
    fast_preview: bool = False,
    echo_raw: bool = False,
) -> int:
    """Run sd-cli subprocess, log all raw lines, and feed real-time updates to dashboard."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=0,
    )

    dit_started = False
    dit_t0 = 0.0
    dit_steps_total = max(1, total_steps)
    dit_steps_done = 0
    dit_segs_total = 0
    dit_last_seg = 0
    in_taehv = False

    vae_tiles_total = 0
    vae_t0 = time.monotonic()

    if phase == "process1":
        dashboard.start_stage("te", detail="UMT5-XXL yükleniyor...", determinate=False)
        dashboard.set_status("Metin kodlayıcı (UMT5-XXL CPU) başlatılıyor...")
    else:
        dashboard.start_stage("vae", detail="Wan 2.1 VAE başlatılıyor...", determinate=False)
        dashboard.set_status("Ayrı süreç Wan 2.1 VAE yükleniyor...")

    assert proc.stdout is not None
    for raw_line in iter_stream_lines(proc.stdout):
        if log_file is not None:
            log_file.write(raw_line + "\n")
            log_file.flush()
        if echo_raw:
            print(raw_line, flush=True)

        line = RE_ANSI.sub("", raw_line).strip()
        if not line:
            continue

        # Capture errors
        if any(tok in line for tok in ("[E]", "ERROR", "ErrorDeviceLost", "VK_ERROR", "failed")):
            if "probe output check failed" not in line and "retrying with" not in line:
                dashboard.add_error(line)

        if phase == "process1":
            # Stage 1: Text Encoder
            if "umt5" in line.lower() and "loading" in line.lower():
                dashboard.update_stage("te", detail="UMT5-XXL ağırlıkları okunuyor...")
                dashboard.set_status("UMT5-XXL CPU tensörleri yükleniyor...")
                continue

            m_te_ms = RE_TE_DONE_MS.search(line)
            if m_te_ms:
                sec = float(m_te_ms.group(1)) / 1000.0
                dashboard.finish_stage("te", detail=f"{sec:.1f}s", elapsed_override=sec)
                dashboard.set_status(f"Metin kodlama tamamlandı ({sec:.1f}s) · DiT hazırlanıyor...")
                continue

            m_te_s = RE_TE_DONE_S.search(line)
            if m_te_s and dashboard.stages["te"].status != "done":
                sec = float(m_te_s.group(1))
                dashboard.finish_stage("te", detail=f"{sec:.1f}s", elapsed_override=sec)
                dashboard.set_status(f"Metin kodlama tamamlandı ({sec:.1f}s)")
                continue

            # Stage 2: DiT Sampling
            m_samp = RE_SAMPLE_START.search(line)
            if m_samp:
                if dashboard.stages["te"].status != "done":
                    dashboard.finish_stage("te", detail="Tamamlandı")
                dit_started = True
                dit_t0 = time.monotonic()
                dit_steps_total = max(1, int(m_samp.group(2)))
                dashboard.start_stage(
                    "dit",
                    detail=f"Adım 1/{dit_steps_total} · {m_samp.group(1)}",
                    determinate=True,
                )
                dashboard.set_status(f"DiT örnekleme başladı ({m_samp.group(1)}, {dit_steps_total} adım)")
                continue

            m_segs = RE_TOTAL_SEGS.search(line)
            if m_segs:
                dit_segs_total = max(1, int(m_segs.group(1)))
                if not dit_started:
                    dit_started = True
                    dit_t0 = time.monotonic()
                    dashboard.start_stage("dit", detail=f"Adım 1/{dit_steps_total}", determinate=True)
                dashboard.set_status(f"Wan 2.2 14B grafik planı hazır ({dit_segs_total} segment/adım)")
                continue

            m_st = RE_SEG_TIMING.search(line)
            if m_st and dit_started and not in_taehv:
                runner = m_st.group(1)
                if "Wan" in runner or "wan" in runner:
                    seg_idx = int(m_st.group(2))
                    seg_tot = max(1, int(m_st.group(3)))
                    seg_name = m_st.group(4)
                    seg_ms = float(m_st.group(5))
                    dit_segs_total = seg_tot

                    # Detect wrap from last segment into next step if progress line arrived later
                    if seg_idx < dit_last_seg and dit_steps_done < dit_steps_total - 1:
                        if dit_last_seg > seg_tot * 0.8:
                            dit_steps_done += 1
                    dit_last_seg = seg_idx

                    cur_step = min(dit_steps_total, dit_steps_done + 1)
                    frac = (dit_steps_done + (seg_idx / seg_tot)) / dit_steps_total
                    pct = min(99.5, frac * 100.0)
                    elapsed = time.monotonic() - dit_t0
                    eta = (elapsed * (1.0 - frac) / frac) if frac > 0.015 else None

                    m_blk = RE_BLOCK_NAME.search(seg_name)
                    if m_blk:
                        blk_num = int(m_blk.group(1)) + 1
                        sub = m_blk.group(2)
                        det = f"Adım {cur_step}/{dit_steps_total} · Blok {blk_num}/40 ({sub})"
                    else:
                        det = f"Adım {cur_step}/{dit_steps_total} · {seg_name}"

                    dashboard.update_stage("dit", pct=pct, detail=det, eta_sec=eta)
                    dashboard.set_status(f"{det} · {seg_ms:.0f} ms")
                    continue

            # Progress bar lines: either DiT step (1/4) or TAEHV tile (12/49)
            m_pb = RE_PROGRESS_BAR.search(line)
            if m_pb:
                cur_i = int(m_pb.group(1))
                tot_i = max(1, int(m_pb.group(2)))
                rate_val = float(m_pb.group(3))
                rate_unit = m_pb.group(4)
                eta_s = (
                    (int(m_pb.group(5)) * 60 + int(m_pb.group(6)))
                    if m_pb.group(5) is not None
                    else None
                )
                if in_taehv or (fast_preview and dashboard.stages["dit"].status == "done"):
                    in_taehv = True
                    pct = (cur_i / tot_i) * 100.0
                    det = f"TAEHV karo {cur_i}/{tot_i} ({rate_val:.1f} {rate_unit})"
                    dashboard.update_stage("vae", pct=pct, detail=det, eta_sec=eta_s)
                    dashboard.set_status(det)
                elif dit_started and tot_i == dit_steps_total:
                    dit_steps_done = max(dit_steps_done, cur_i)
                    dit_last_seg = 0
                    pct = (cur_i / tot_i) * 100.0
                    det = f"Adım {cur_i}/{tot_i} tamamlandı ({rate_val:.1f} {rate_unit})"
                    dashboard.update_stage("dit", pct=pct, detail=det, eta_sec=eta_s)
                    dashboard.set_status(det)
                continue

            m_sdone = RE_SAMPLE_DONE.search(line)
            if m_sdone:
                sec = float(m_sdone.group(1))
                dashboard.finish_stage("dit", detail=f"{dit_steps_total}/{dit_steps_total} adım ({sec:.1f}s)", elapsed_override=sec)
                if fast_preview:
                    in_taehv = True
                    dashboard.start_stage("vae", detail="TAEHV decode başlıyor...", determinate=True)
                    dashboard.set_status(f"DiT tamamlandı ({sec:.1f}s) · TAEHV hızlı decode başlıyor...")
                else:
                    dashboard.set_status(f"DiT tamamlandı ({sec:.1f}s) · Latent kaydediliyor...")
                continue

            m_lsave = RE_LATENT_SAVE.search(line)
            if m_lsave:
                dashboard.set_status(f"Float32 latent kaydedildi ({m_lsave.group(1)})")
                continue

            if fast_preview:
                if "taehv temporal tiling" in line and not in_taehv:
                    in_taehv = True
                    dashboard.start_stage("vae", detail="TAEHV çalışıyor...", determinate=True)
                m_vdone = RE_VAE_DONE.search(line)
                if m_vdone:
                    sec = float(m_vdone.group(1))
                    dashboard.finish_stage("vae", detail=f"TAEHV ({sec:.1f}s)", elapsed_override=sec)
                    dashboard.set_status(f"TAEHV decode tamamlandı ({sec:.1f}s)")

        else:
            # Process 2: Standalone Wan 2.1 VAE Decode
            if "conv2d direct auto:" in line or "conv3d direct auto:" in line:
                dashboard.set_status(line.split("---")[0].strip())
                continue

            m_tsize = RE_VAE_TILE_SIZE.search(line)
            if m_tsize:
                dashboard.set_status(f"Wan 2.1 VAE karo boyutu: {m_tsize.group(1)} px")
                continue

            m_ttot = RE_VAE_TILES_TOTAL.search(line)
            if m_ttot:
                vae_tiles_total = max(1, int(m_ttot.group(1)))
                vae_t0 = time.monotonic()
                dashboard.start_stage("vae", detail=f"0/{vae_tiles_total} karo", determinate=True)
                dashboard.set_status(f"Wan 2.1 VAE uzamsal+zamansal decode: {vae_tiles_total} karo")
                continue

            m_tmp = RE_VAE_TEMP_TILE.search(line)
            if m_tmp and vae_tiles_total == 0:
                dashboard.update_stage("vae", detail=f"Zamansal dilim {m_tmp.group(1)}/{m_tmp.group(2)}")
                continue

            m_pb = RE_PROGRESS_BAR.search(line)
            if m_pb and "=" in raw_line:
                cur_i = int(m_pb.group(1))
                tot_i = max(1, int(m_pb.group(2)))
                rate_val = float(m_pb.group(3))
                rate_unit = m_pb.group(4)
                eta_s = (
                    (int(m_pb.group(5)) * 60 + int(m_pb.group(6)))
                    if m_pb.group(5) is not None
                    else None
                )
                if eta_s is None and cur_i > 0:
                    elapsed = time.monotonic() - vae_t0
                    eta_s = elapsed * (tot_i - cur_i) / cur_i
                pct = (cur_i / tot_i) * 100.0
                det = f"Karo {cur_i}/{tot_i} ({rate_val:.2f} {rate_unit})"
                dashboard.update_stage("vae", pct=pct, detail=det, eta_sec=eta_s)
                dashboard.set_status(f"Wan 2.1 VAE: {det}")
                continue

            m_vdone = RE_VAE_DONE.search(line)
            if m_vdone:
                sec = float(m_vdone.group(1))
                dashboard.finish_stage("vae", detail=f"Wan 2.1 VAE ({sec:.1f}s)", elapsed_override=sec)
                dashboard.set_status(f"Wan 2.1 VAE decode tamamlandı ({sec:.1f}s)")
                continue

    rc = proc.wait()
    if rc == 0:
        if phase == "process1":
            if dashboard.stages["te"].status != "done":
                dashboard.finish_stage("te", detail="Tamamlandı")
            if dashboard.stages["dit"].status != "done":
                dashboard.finish_stage("dit", detail="Tamamlandı")
            if fast_preview and dashboard.stages["vae"].status != "done":
                dashboard.finish_stage("vae", detail="TAEHV tamamlandı")
        else:
            if dashboard.stages["vae"].status != "done":
                dashboard.finish_stage("vae", detail="Wan 2.1 VAE tamamlandı")
    return rc


def run_interactive_wizard(
    hw: dict[str, Any],
    select_profile_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Interactive TUI wizard when user runs without --prompt or via localvid.cmd."""
    configure_console_utf8()
    use_rich = HAS_RICH and is_interactive_tty()
    console = make_console() if use_rich else None

    frame_presets = [
        ("1", 17, "1.1 sn", "~4.5 dk"),
        ("2", 33, "2.1 sn", "~8.7 dk"),
        ("3", 49, "3.1 sn", "~15 dk"),
        ("4", 81, "5.1 sn", "~52 dk"),
    ]

    if use_rich and console is not None:
        live_free_txt = (
            f" · canlı boş: {hw['live_free_ded_gib']:.2f} GiB"
            if hw.get("live_free_ded_gib") is not None
            else ""
        )
        banner = Panel(
            Group(
                Align.center(Text(BANNER_ART.strip("\n"), style="bold cyan")),
                Align.center(
                    Text(
                        "v0.1.0  ·  Wan 2.2 14B on 4-6 GB VRAM GPUs (CUDA + Vulkan)",
                        style="bold white",
                    )
                ),
                Align.center(
                    Text(
                        f"Algılanan GPU: {hw['gpu_name']}  |  {hw['backend'].upper()}  |  "
                        f"{hw['vram_class']} ({hw['vram_budget_gib']:.2f} GiB bütçe{live_free_txt})",
                        style="green",
                    )
                ),
            ),
            border_style="cyan",
            box=box.DOUBLE,
            padding=(0, 2),
        )
        console.print()
        console.print(banner)
        console.print()

        console.print("[bold cyan]1) Video Prompt'u[/bold cyan] [dim](İngilizce detaylı sahne tanımı önerilir; boş bırakırsanız örnek prompt kullanılır)[/dim]")
        prompt_text = Prompt.ask("   Prompt", default="", show_default=False, console=console).strip()
        if not prompt_text:
            prompt_text = DEFAULT_SAMPLE_PROMPT
            console.print(f"   [dim]Varsayılan örnek seçildi: {prompt_text[:78]}...[/dim]")

        console.print()
        console.print("[bold cyan]2) Süre / Kare Sayısı[/bold cyan]")
        p_tbl = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        p_tbl.add_column("Seçim", style="bold cyan", width=6)
        p_tbl.add_column("Kare / Süre", style="white", width=16)
        p_tbl.add_column("Profil", style="green", width=22)
        p_tbl.add_column("Tahmini Süre (RX 5600 XT, 4 adım)", style="yellow")
        for key, fcnt, dur, eta_est in frame_presets:
            prof_name = select_profile_fn(hw, fcnt, False)["profile_name"]
            default_mark = " (Varsayılan)" if key == "2" else ""
            p_tbl.add_row(f"[{key}]", f"{fcnt}f / {dur}{default_mark}", prof_name, eta_est)
        console.print(p_tbl)
        frame_choice = Prompt.ask("   Seçiminiz", choices=["1", "2", "3", "4"], default="2", console=console)
        frames = dict((k, f) for k, f, _, _ in frame_presets)[frame_choice]

        console.print()
        console.print("[bold cyan]3) Çözünürlük[/bold cyan]")
        console.print("   [bold cyan][1][/bold cyan] 640x384 [dim](Varsayılan · Hızlı ve dengeli)[/dim]")
        console.print("   [bold cyan][2][/bold cyan] 832x480 [green](480p Keskin · Wan 2.2 doğal çözünürlüğü)[/green]")
        console.print("   [bold cyan][3][/bold cyan] 480x320 [dim](Ultra hızlı · ~1.5x daha kısa süre)[/dim]")
        res_choice = Prompt.ask("   Seçiminiz", choices=["1", "2", "3"], default="1", console=console)
        width, height = {"1": (640, 384), "2": (832, 480), "3": (480, 320)}[res_choice]

        console.print()
        console.print("[bold cyan]4) Kalite / VAE Modu[/bold cyan]")
        console.print("   [bold cyan][1][/bold cyan] Wan 2.1 VAE [green](Tam Kalite · Ayrı Süreç)[/green]")
        console.print("   [bold cyan][2][/bold cyan] TAEHV [yellow](Hızlı Önizleme · ~4 sn decode)[/yellow]")
        vae_choice = Prompt.ask("   Seçiminiz", choices=["1", "2"], default="1", console=console)
        fast_preview = vae_choice == "2"

        console.print()
        seed_raw = Prompt.ask(
            "[bold cyan]5) Seed[/bold cyan] [dim](boş bırakılırsa rastgele)[/dim]",
            default="",
            show_default=False,
            console=console,
        ).strip()
        seed = int(seed_raw) if seed_raw.isdigit() else random.randint(1, 2_147_483_647)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_out = f"outputs/localvid_{ts}_{width}x{height}_{frames}f.mp4"
        out_raw = Prompt.ask(
            "[bold cyan]6) Çıktı MP4 Yolu[/bold cyan]",
            default=default_out,
            console=console,
        ).strip()
        output_path = pathlib.Path(out_raw or default_out)

        selected_prof = select_profile_fn(hw, frames, fast_preview, width, height)
        confirm_tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        confirm_tbl.add_column(style="bold cyan", width=18)
        confirm_tbl.add_column(style="white")
        confirm_tbl.add_row("Prompt", prompt_text if len(prompt_text) <= 90 else prompt_text[:87] + "...")
        confirm_tbl.add_row("Hedef", f"{width}x{height} · {frames} kare ({frames / 16.0:.1f}s) · 4 adım")
        confirm_tbl.add_row("Profil / VRAM", f"{selected_prof['profile_name']} (--max-vram {selected_prof['max_vram_gib']} GiB)")
        confirm_tbl.add_row("VAE Modu", "TAEHV (Hızlı Önizleme)" if fast_preview else "Wan 2.1 VAE (Tam Kalite)")
        confirm_tbl.add_row("Seed / Çıktı", f"{seed}  ->  {output_path}")
        if selected_prof.get("vram_warning"):
            confirm_tbl.add_row("VRAM Uyarısı", f"[bold yellow]{selected_prof['vram_warning']}[/bold yellow]")
        console.print()
        console.print(Panel(confirm_tbl, title="[bold]Üretim Onayı[/bold]", border_style="green", box=box.ROUNDED))

        if not Confirm.ask("Üretimi başlat?", default=True, console=console):
            raise KeyboardInterrupt("Kullanıcı tarafından iptal edildi.")
    else:
        print("=== localVid v0.1.0 İnteraktif Sihirbaz ===")
        print(f"GPU: {hw['gpu_name']} ({hw['backend'].upper()}, {hw['vram_class']})")
        prompt_text = input("1) Prompt (İngilizce önerilir, boş = örnek): ").strip() or DEFAULT_SAMPLE_PROMPT
        print("2) Süre/Kare: [1] 17f (1.1s)  [2] 33f (2.1s, varsayılan)  [3] 49f (3.1s)  [4] 81f (5.1s)")
        fc = input("   Seçim [2]: ").strip() or "2"
        frames = {"1": 17, "2": 33, "3": 49, "4": 81}.get(fc, 33)
        rc = input("3) Çözünürlük: [1] 640x384 (varsayılan)  [2] 832x480 (480p keskin)  [3] 480x320 (hızlı) [1]: ").strip() or "1"
        width, height = {"1": (640, 384), "2": (832, 480), "3": (480, 320)}.get(rc, (640, 384))
        vc = input("4) Kalite: [1] Wan 2.1 VAE (Tam Kalite)  [2] TAEHV (Hızlı) [1]: ").strip() or "1"
        fast_preview = vc == "2"
        sr = input("5) Seed (boş = rastgele): ").strip()
        seed = int(sr) if sr.isdigit() else random.randint(1, 2_147_483_647)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_out = f"outputs/localvid_{ts}_{width}x{height}_{frames}f.mp4"
        output_path = pathlib.Path(input(f"6) Çıktı MP4 [{default_out}]: ").strip() or default_out)
        selected_prof = select_profile_fn(hw, frames, fast_preview, width, height)
        if selected_prof.get("vram_warning"):
            print(f"[localVid] {selected_prof['vram_warning']}")

    return {
        "prompt": prompt_text,
        "frames": frames,
        "width": width,
        "height": height,
        "fast_preview": fast_preview,
        "seed": seed,
        "output": output_path,
    }


def ask_open_video(output_path: pathlib.Path, use_rich: bool) -> None:
    """Offer to launch the finished MP4 on Windows via os.startfile."""
    if not output_path.exists() or platform.system() != "Windows":
        return
    try:
        if use_rich and HAS_RICH:
            console = make_console()
            ans = Prompt.ask("Videoyu şimdi aç? [E/h]", default="E", console=console).strip().lower()
            should_open = ans in ("", "e", "evet", "y", "yes")
        else:
            ans = input("Videoyu şimdi aç? [E/h]: ").strip().lower()
            should_open = ans in ("", "e", "evet", "y", "yes")
        if should_open:
            os.startfile(str(output_path))  # type: ignore[attr-defined]
    except Exception:
        pass


def run_ui_demo(
    hw: dict[str, Any],
    prof: dict[str, Any],
    raw_logs: bool = False,
    force_rich: bool = False,
) -> int:
    """Run a 5.5-second animated dashboard self-test without executing sd-cli."""
    use_rich = should_use_rich(raw_logs=raw_logs, verbose=False, force_rich=force_rich)
    sampler = LiveVramSampler(total_vram_gib=hw["vram_total_gib"], interval_sec=1.0)
    base_ded = min(2950.0, hw["vram_total_gib"] * 1024.0 * 0.48)
    sampler.inject_sample(base_ded, 460.0)

    cfg = DashboardConfig(
        gpu_name=hw["gpu_name"],
        backend=hw["backend"],
        vram_class=hw["vram_class"],
        vram_total_gib=hw["vram_total_gib"],
        vram_budget_gib=hw["vram_budget_gib"],
        profile_name=prof["profile_name"],
        width=640,
        height=384,
        frames=33,
        fps=16,
        steps=4,
        vae_label="Wan 2.1 VAE",
        log_path=pathlib.Path("demo_output.log"),
        demo_mode=True,
    )
    dash: BaseDashboard = (
        RichDashboard(cfg, sampler, console=make_console(force_terminal=force_rich or None))
        if use_rich
        else PlainDashboard(cfg, sampler)
    )

    dash.start()
    try:
        # Stage 1: UMT5-XXL CPU (~0.9s)
        dash.start_stage("te", detail="UMT5-XXL CPU çalışıyor...", determinate=False)
        dash.set_status("UMT5-XXL 242 tensör mmap edildi · prompt kodlanıyor...")
        for i in range(6):
            sampler.inject_sample(base_ded + i * 12.0, 465.0)
            time.sleep(0.14)
        dash.finish_stage("te", detail="13.4s (simüle)")

        # Stage 2: Wan 2.2 14B DiT (~2.8s across 4 steps x 40 blocks)
        dash.start_stage("dit", detail="Adım 1/4 · Blok 1/40", determinate=True)
        total_ticks = 32
        for tick in range(1, total_ticks + 1):
            step = ((tick - 1) // 8) + 1
            blk = min(40, (((tick - 1) % 8) + 1) * 5)
            sub = ("sa_qkv", "sa_h12", "cross_attn", "ffn")[tick % 4]
            pct = (tick / total_ticks) * 100.0
            eta = max(0.0, (total_ticks - tick) * 0.085)
            det = f"Adım {step}/4 · Blok {blk}/40 ({sub})"
            dash.update_stage("dit", pct=pct, detail=det, eta_sec=eta)
            dash.set_status(f"{det} · 17.2 ms · SD_EVICT_PAST=1")
            sampler.inject_sample(base_ded + 1150.0 + (tick % 6) * 48.0, 480.0 + (tick % 4) * 10.0)
            time.sleep(0.085)
        dash.finish_stage("dit", detail="4/4 adım (simüle)")

        # Stage 3: Separate-Process Wan 2.1 VAE (~1.3s across 49 tiles)
        dash.start_stage("vae", detail="Karo 0/49", determinate=True)
        for tile in range(1, 50, 3):
            pct = (tile / 49.0) * 100.0
            eta = max(0.0, ((49 - tile) / 3.0) * 0.075)
            det = f"Karo {tile}/49 (1.32 s/it)"
            dash.update_stage("vae", pct=pct, detail=det, eta_sec=eta)
            dash.set_status(f"Wan 2.1 VAE (conv_direct=on): {det}")
            sampler.inject_sample(base_ded + 1480.0 + (tile % 5) * 35.0, 510.0)
            time.sleep(0.075)
        dash.finish_stage("vae", detail="49/49 karo (simüle)")

        # Stage 4: MP4 Packaging (~0.45s)
        dash.start_stage("mp4", detail="OpenCV/FFmpeg MP4 paketleniyor...", determinate=True)
        for p in (35.0, 75.0, 100.0):
            dash.update_stage("mp4", pct=p, detail="H.264 / MP4V yazılıyor...")
            sampler.inject_sample(base_ded + 120.0, 460.0)
            time.sleep(0.14)
        dash.finish_stage("mp4", detail="demo_output.mp4")
        dash.set_status("Demo başarıyla tamamlandı!")
        time.sleep(0.2)
    finally:
        dash.stop()

    dash.render_summary(
        success=True,
        output_path=pathlib.Path("demo_output.mp4"),
        encoder_label="ui-demo",
    )
    return 0
