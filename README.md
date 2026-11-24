# localVid

**[English](README.md) · [Türkçe](README.tr.md)**

> **Run flagship 14B video diffusion models (Wan 2.2) on 4–6 GB VRAM consumer GPUs (AMD Vulkan & NVIDIA CUDA) — crash-free, up to 9.28× faster than stock, and up to 81 frames (5.1s @ 640×384) — Apache 2.0 licensed.**

![Wan 2.2 14B 640x384 4-step output on AMD RX 5600 XT 6GB Vulkan](docs/media/wan22_640x384_33f_frame12_rx5600xt.png)
*Generated locally on an **AMD Radeon RX 5600 XT (6 GB VRAM, Vulkan, Windows 11)** — `640×384`, 33 frames (2.1s), 4 steps (`wan2.2-t2v-rapid-aio-v10-Q4_K`), real Wan 2.1 VAE, ~8.5 min end-to-end.*

| Frame 12 (`17f`, 4.5 min total) | Frame 32 (`33f`, `Q_CHUNK=2048`) |
|---|---|
| ![17f sample](docs/media/wan22_640x384_17f_rx5600xt.png) | ![33f end frame](docs/media/wan22_640x384_33f_frame32_rx5600xt.png) |

---

## Why localVid Exists

Stock [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) is an incredible C++ inference engine, but running a heavy **14B video model (Wan 2.2)** on a **4–6 GB consumer GPU** hits 5 immediate walls:

1. **WDDM Shared Memory Thrashing (2.75–4.03 GiB PCIe Spill):** Un-evicted transformer blocks (`40 × 219 MB = 8.75 GB`) silently spill over PCIe into System RAM, slowing compute down by ~3–10×.
2. **Windows 2-Second TDR Crashes (`vk::Queue::submit: ErrorDeviceLost`):** At `640×384`, a single unsplit Wan block takes >6 seconds on RDNA 1 Vulkan Flash Attention, triggering an instant GPU driver reset on `blocks.0`.
3. **Vulkan `maxMemoryAllocationSize = 2.0 GiB` Wall:** Long clips (`49f–81f`) and Wan 2.1 VAE 3D `im2col` buffers exceed 2 GiB single-allocation limits.
4. **Over-Strict `conv2d/conv3d direct` Auto-Probe:** On RDNA 1 Vulkan, harmless `0.0037` FP16 accumulation rounding (`non_finite = 0`) fails the `0.001` probe tolerance, disabling direct 3D convolution and inflating Wan 2.1 VAE memory to 6.97–8.53 GB.
5. **In-Process VAE OOM & Separate-Process Normalization Trap:** Decoding inside the DiT process fragments VRAM, while saving raw DiT latents via `--taesd` without `dit_to_vae_latents` causes cross-hatch artifacts in Wan 2.1 VAE.

**localVid** solves all five via **9 modular, default-OFF, 100% backend-agnostic (`ggml_backend_*`) patches** that work across both **Vulkan** and **CUDA**.

---

## Measured Benchmarks (AMD Radeon RX 5600 XT 6 GB Vulkan, Windows 11, 32 GB RAM)

*Note: Idle desktop VRAM baseline on the test machine was `2,914 MiB` Dedicated / `473 MiB` Shared — net `sd-cli` deltas (`Δ`) are shown alongside total GPU usage.*

### 1. Speedup Breakdown (1 Step, Wan 2.2 14B Q4_K_M)

| Resolution × Frames | Configuration | Self-Attn / Block (40 heads) | DiT 1-Step Time | Speedup | DiT Compute Buffer | Net Dedicated VRAM (`Δ`) | Status |
|---|---|---:|---:|---:|---:|---:|---|
| `480×320 × 17f` | Stock `sd.cpp` (`--max-vram 3.8`, FA on) | — | `356.10 s` | `1.00×` | `345.2 MB` | `+1,563 MiB` (`+2,281` Shared!) | VAE OOM fallback |
| `480×320 × 17f` | P1 (`mv=1.5`, `evict_on_end`, FA on) | — | `137.73 s` | `2.59×` | `345.2 MB` | **`+1,333 MiB`** | OK |
| **`480×320 × 17f`** | **localVid P1+P2 (`HEAD_CHUNK=4`, FA off)** | **`78.0 ms`** | **`38.36 s`** | **`9.28×`** | **`381.6 MB`** | **`+1,518 MiB`** | **OK** |
| `640×384 × 17f` | Stock `sd.cpp` / P1 (FA on) | *(TDR reset)* | **CRASH (`23 s`)** | — | `547.3 MB` | — | `ErrorDeviceLost` |
| `640×384 × 17f` | P2 (`HEAD_CHUNK=8`, FA on) | `6,265.0 ms` | `296.50 s` | `1.00×` | `375.0 MB` | `+1,300 MiB` | TDR Fixed |
| **`640×384 × 17f`** | **localVid P1+P2 (`HEAD_CHUNK=4`, FA off)** | **`168.0 ms` (`37.3×`)** | **`56.60–58.70 s`** | **`5.05×`** | **`592.5 MB`** | **`+1,187–1,507 MiB`** | **OK** |

### 2. `640×384` Frame Ceiling Ladder (up to 5.1s / 81 Frames, 1 Step + Separate-Process Wan 2.1 VAE)

| Frames (`@16fps`) | Tokens (`N`) | Profile | DiT (1 Step) | DiT Compute Buffer | Separate-Process Wan 2.1 VAE (`conv_direct=on`) | Net DiT Dedicated (`Δ`) | Status |
|---|---:|---|---:|---:|---:|---:|---|
| **`17f`** (`1.1s`) | `4,800` | Fast (`q=off, mv=1.5`) | **`56.60 s`** | `592.5 MB` | **`34.63 s`** (`1,441 MB`) | `+1,187 MiB` | **OK** |
| **`33f`** (`2.1s`) | `8,640` | Fast (`q=2048, mv=2.0`) | **`114.51 s`** | `675.0 MB` | **`64.78 s`** (`1,617 MB`) | `+1,656 MiB` | **OK** |
| **`49f`** (`3.1s`) | `12,480` | Fast (`q=1024, mv=2.5`) | **`195.97 s`** | `1,500.5 MB` | **`131.94 s`** (`1,617–1,922 MB`) | `+1,834 MiB` | **OK** |
| **`81f`** (`5.1s`) | `20,160` | Fit (`QKV_SPLIT=1`, `CUT_HOST=32`, `q=1024`, `mv=2.5`) | **`728.03 s`** | **`1,890.0 MB`** (`< 2 GiB`) | **`230.16 s`** (`1,617–1,922 MB`) | `+2,110 MiB` | **OK** |

### 3. Real 4-Step End-to-End Video Generation (`wan2.2-t2v-rapid-aio-v10-Q4_K`, `CFG=1.0`, `euler_a + beta`, `flow-shift 5.0`)

| Output (`640×384`) | TE (CPU) | DiT (4 Steps) | Separate-Process Wan 2.1 VAE | Total End-to-End |
|---|---:|---:|---:|---:|
| **17 frames (1.1s)** | `22.2 s` | `209.4 s` | `34.6 s` | **~4.5 min** |
| **33 frames (2.1s)** | `13.4 s` | `444.1 s` | `64.8 s` | **~8.7 min** |

---

## Architecture & Patch Series (`patches/SERIES.txt`)

Base: `leejet/stable-diffusion.cpp @ 3b12781` · `leejet/ggml @ eeda9ec`  
All patches are **default OFF** (enabled strictly via environment variables) and use backend-agnostic `ggml_backend_*` APIs:

| Patch | Env Flags | What It Solves |
|---|---|---|
| `00-diag-conv-probe.patch` | `SD_PROBE_DIAG=1` | Logs `max_abs_diff` & `non_finite` on probe mismatch |
| `01-p1-model-loader-sleep.patch` | `SD_LOADER_FAST_POLL=1` | Splits 200 ms loader polling sleep into 5 ms (`-8s` across 40 blocks) |
| `02-p1-evict-and-drop-queued.patch` | `SD_EVICT_ON_END`, `SD_EVICT_PAST`, `SD_DROP_QUEUED` | Evicts finished block weights immediately to stop WDDM Shared spill |
| `03-p1-staging-pool-and-capacity-log.patch` | `SD_STAGING_POOL_MB`, `SD_VRAM_MARGIN_MB`, `SD_CAPACITY_LOG` | Reuses device staging buffers across blocks (`0.09–0.12s` per 219 MB block) |
| `04-p2-seg-timing.patch` | `SD_SEG_TIMING=1\|2` | Per-segment ms timing breakdown |
| `05-p2-wan-split-and-chunk.patch` | `SD_WAN_SPLIT_BLOCK`, `SD_WAN_QKV_SPLIT`, `SD_WAN_ATTN_HEAD_CHUNK/CUT`, `SD_WAN_ATTN_Q_CHUNK`, `SD_WAN_MLP_CHUNK_MB` | Disjoint sub-block graph cuts + head/Q/cross-attn/MLP chunking (eliminates 2s TDR, enables 37× faster normal attention & 81f under 2 GiB) |
| `06-p2-cut-cache-host.patch` | `SD_CUT_CACHE_HOST_MB` | Disables `bind_outputs` for large cuts and parks them in Host RAM |
| `07-p3-conv-direct-tol.patch` | `SD_CONV_DIRECT_TOL` | Relaxes FP16 rounding tolerance (`0.005`) so Vulkan `conv2d/conv3d direct` turns ON (`1.8×` faster VAE) |
| `08-p3-separate-process-vae.patch` | `SD_VIDEO_LATENT_SAVE[_ONLY]`, `SD_VIDEO_LATENT_LOAD`, `SD_VIDEO_LATENT_DIT_TO_VAE` | Bit-exact float32 latent handoff + optional `dit_to_vae_latents` normalization for clean two-process VAE decode |

---

## Quick Start (Windows)

### 1. Clone `stable-diffusion.cpp`, Apply Patches & Build

```powershell
# Clone pinned base
git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/leejet/stable-diffusion.cpp.git third_party/stable-diffusion.cpp
git -C third_party/stable-diffusion.cpp checkout 3b12781

# Apply all 9 localVid patches
powershell -ExecutionPolicy Bypass -File scripts/apply_patches.ps1

# Build (Vulkan for AMD/Intel/NVIDIA, or -Backend cuda for NVIDIA CUDA)
powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Backend vulkan
# powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Backend cuda
```

*Prerequisites: Visual Studio 2022 (MSVC + CMake + Ninja), Python 3.10+ (`py -m pip install -r requirements.txt`), and [Vulkan SDK](https://vulkan.lunarg.com/) (or CUDA Toolkit for `-Backend cuda`).*

### 2. Download Models (Apache 2.0 Default Set — ~12.6 GiB)

```powershell
py scripts/download_models.py --with-taehv
```

### 3. Generate Video (Interactive TUI or One-Command CLI)

```powershell
# Interactive wizard + live Rich dashboard (or double-click localvid.cmd)
.\localvid.cmd
# py scripts/generate.py

# Full quality (2-process DiT + Wan 2.1 VAE -> .mp4, raw logs saved to .log)
py scripts/generate.py -p "A cinematic tracking shot of a vintage red sports car driving down a rain-slicked neon-lit city street at sunset" -W 640 -H 384 -f 33 -o output_33f.mp4

# Ultra-fast preview (~2-6s TAEHV decode)
py scripts/generate.py -p "Ocean waves crashing on dark volcanic rocks at golden hour" -f 17 --fast-preview -o preview.mp4

# Quick 5s live dashboard self-test / hardware dry-run
py scripts/generate.py --ui-demo
py scripts/generate.py -p "test" -f 81 --dry-run
```

---

## Hardware Support Matrix (Honest Verification Status)

| Hardware / Platform | Backend | Status | Notes |
|---|---|---|---|
| **AMD Radeon RX 5600 XT (6 GB, RDNA 1), Windows 11, 32 GB RAM** | Vulkan | **Physically Verified** | All benchmarks (`17f–81f`, 4-step MP4s, `9.28×` speedup, `1.30–2.11 GiB` net VRAM) measured on this GPU. |
| **AMD / Intel 4–8 GB GPUs (Windows / Linux)** | Vulkan | **Architecture-Ready (Community Test Welcome)** | Uses pure `ggml-vulkan` + `probe_hardware.py` heap budget detection. |
| **NVIDIA 4–8 GB GPUs (GTX 1050 Ti / RTX 2060 / 3050 / 3060 / 4060)** | CUDA | **Architecture-Ready (Community Test Welcome)** | All 9 patches use backend-agnostic `ggml_backend_*` APIs; `generate.py` auto-selects CUDA profiles (`--simulate-profile cuda-4gb` verified). |

*Note: System RAM requirement is **16–32 GB** (mmap-backed 9 GB DiT + 3.4 GB UMT5-XXL on CPU; 32 GB recommended for `49f–81f` with `SD_CUT_CACHE_HOST_MB`).*

---

## License & Acknowledgments

- **localVid** is released under the **[Apache License 2.0](LICENSE)**.
- Built on [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) and [`ggml`](https://github.com/ggml-org/ggml).
- Memory optimization research inspired in part by [`alexturkey/VideoTurko`](https://github.com/alexturkey/VideoTurko)'s low-VRAM patch methodology.
