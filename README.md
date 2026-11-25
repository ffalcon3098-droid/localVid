# PicoVid

**[English](README.md) · [Türkçe](README.tr.md)**

> **Run flagship 14B video diffusion models (Wan 2.2 T2V + I2V + Z-Image-Turbo Studio) on 4–6 GB VRAM consumer GPUs (AMD Vulkan & NVIDIA CUDA) — crash-free, up to 9.28× faster than stock, and up to 81 frames (5.1s @ 640×384) — Apache 2.0 licensed.**

![Wan 2.2 14B 640x384 4-step output on AMD RX 5600 XT 6GB Vulkan](docs/media/wan22_640x384_33f_frame12_rx5600xt.png)
*Generated locally on an **AMD Radeon RX 5600 XT (6 GB VRAM, Vulkan, Windows 11)** — `640×384`, 33 frames (2.1s), 4 steps (`wan2.2-t2v-rapid-aio-v10-Q4_K`), real Wan 2.1 VAE, ~8.5 min end-to-end.*

| Frame 12 (`17f`, 4.5 min total) | Frame 32 (`33f`, `Q_CHUNK=2048`) |
|---|---|
| ![17f sample](docs/media/wan22_640x384_17f_rx5600xt.png) | ![33f end frame](docs/media/wan22_640x384_33f_frame32_rx5600xt.png) |

---

## Why PicoVid Exists

Stock [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) is an incredible C++ inference engine, but running a heavy **14B video model (Wan 2.2)** on a **4–6 GB consumer GPU** hits 5 immediate walls:

1. **WDDM Shared Memory Thrashing (2.75–4.03 GiB PCIe Spill):** Un-evicted transformer blocks (`40 × 219 MB = 8.75 GB`) silently spill over PCIe into System RAM, slowing compute down by ~3–10×.
2. **Windows 2-Second TDR Crashes (`vk::Queue::submit: ErrorDeviceLost`):** At `640×384`, a single unsplit Wan block takes >6 seconds on RDNA 1 Vulkan Flash Attention, triggering an instant GPU driver reset on `blocks.0`.
3. **Vulkan `maxMemoryAllocationSize = 2.0 GiB` Wall:** Long clips (`49f–81f`) and Wan 2.1 VAE 3D `im2col` buffers exceed 2 GiB single-allocation limits.
4. **Over-Strict `conv2d/conv3d direct` Auto-Probe:** On RDNA 1 Vulkan, harmless `0.0037` FP16 accumulation rounding (`non_finite = 0`) fails the `0.001` probe tolerance, disabling direct 3D convolution and inflating Wan 2.1 VAE memory to 6.97–8.53 GB.
5. **In-Process VAE OOM & Separate-Process Normalization Trap:** Decoding inside the DiT process fragments VRAM, while saving raw DiT latents via `--taesd` without `dit_to_vae_latents` causes cross-hatch artifacts in Wan 2.1 VAE.

**PicoVid** solves all five via **9 modular, default-OFF, 100% backend-agnostic (`ggml_backend_*`) patches** that work across both **Vulkan** and **CUDA**.

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

### 3. Real 4-Step End-to-End Video Generation (`640×384`, Separate-Process Wan 2.1 VAE, H.264 CRF 16)

| Pipeline (`640×384`) | First Frame (T2I) | I2V `vae_encode` | TE (CPU) | DiT (4 Steps) | Wan 2.1 VAE (`conv_direct=on`) | Total End-to-End |
|---|---:|---:|---:|---:|---:|---:|
| **T2V — 17 frames (1.1s)** (`Rapid-AIO v10 Q4_K`) | — | — | `22.2 s` | `209.4 s` | `34.6 s` | **~4.5 min** |
| **T2V — 33 frames (2.1s)** (`Rapid-AIO v10 Q4_K`) | — | — | `13.4 s` | `444.1 s` | `64.8 s` | **~8.7 min** |
| **v0.2 Studio Mode (T2I → I2V 17f)** (`Z-Image-Turbo` + `Wan 2.2 I2V LightX2V MoE Q4_K_M`, GPU-tiled `vae_encode`) | **`21.9 s`** *(832×480: `32.1 s`)* | **`19.3 s`** *(11.2× faster than 216s CPU)* | `18.2 s` | **`234.5 s`** *(2 High + 2 Low)* | **`35.7 s`** | **~5.6 min** |
| **v0.3 Cinema HD (`--cinema-hd`, `640×384×17f` → `1280×768 HD @ 32fps`)** | **`21.9 s`** | **`19.3 s`** | `18.2 s` | **`~195–210 s`** *(Phase A: `HEAD_CHUNK=8`, `clear()` prefetch, NVMe pre-warm)* | **`35.7 s`** + **`49.7 s`** (`x4plus` 2× HD + 32fps AOBMC) *(or `~8 s` w/ `animevideov3`)* | **~5.5 min (`1280×768 @ 32fps`)** |
| **v0.3 Fast 5-Second (`--engine 5b`, `Wan 2.2 TI2V-5B Turbo`, `640×384 × 81f @ 16→32fps` = 5.0s!)** | *(opt. `21.9 s`)* | **`1.3 s`** *(single-frame frozen I2V!)* | `13.1 s` | **`74.2 s`** *(`18.5 s/step`, 80.4% GPU eff.)* | **`167.8 s`** (`Wan 2.2 VAE 224×224`, 6 tiles) + **`30.1 s`** (`1280×768 @ 32fps`) | **~4.8 min (`1280×768 @ 32fps`!)** |
| **v0.3 Photoreal 5-Second (`--engine 5b`, native `832×480 × 81f @ 16→32fps`, `beta` scheduler)** | *(opt. `32.1 s`)* | **`2.1 s`** | `13.4 s` | **`140.0 s`** *(`35.0 s/step`)* | **`334.9 s`** (`12` tiles → `8` tiles w/ `224×256`!) + **`11.3 s`** (`32fps` AOBMC) | **~8.4 min (`832×480 @ 32fps`, DSLR-sharp)** |
| **v0.3 Fast 5-Second (`--engine 5b`, `640×384 × 121f @ 24fps` native = 5.04s)** | *(opt. `21.9 s`)* | **`1.3 s`** | `15.6 s` | **`124.1 s`** *(`30.2 s/step`, PeakDed `4,346 MiB`)* | **`352.5 s`** (`Wan 2.2 VAE 192×192`) / **`1.9 s`** (`--fast-preview` `taew2_2`) | **~2.4 min (TAEW) · ~7.9 min (Full VAE)** |
| **`832×480 × 33f` Native 14B I2V (Phase A Shared=0)** | — | — | — | **`232.7 s/step`** *(was `613.2 s/step` w/ `2.18 GiB` Shared spill → **`2.64×` faster**)* | — | **~15.5 min** *(down from ~45 min)* |

| Studio First Frame (`Z-Image-Turbo 6B`, `832×480`, `32.1 s`) | Animated I2V End Frame (`Wan 2.2 I2V MoE`, GPU-tiled `vae_encode`, `PSNR 39.35 dB`) | v0.3 Cinema HD Output (`1280×768 HD @ 32fps`, `realesrgan-x4plus` 2× + AOBMC) |
|---|---|---|
| ![Z-Image-Turbo 832x480 Bosphorus](docs/media/studio_bosphorus_832x480_zimage_rx5600xt.png) | ![Wan 2.2 I2V 17f Bosphorus](docs/media/studio_bosphorus_17f_i2v_rx5600xt.png) | ![Cinema HD 1280x768 Bosphorus](docs/media/cinema_hd_1280x768_bosphorus_rx5600xt.png) |

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
| `05-p2-wan-split-and-chunk.patch` | `SD_WAN_SPLIT_BLOCK`, `SD_WAN_QKV_SPLIT`, `SD_WAN_ATTN_HEAD_CHUNK/CUT`, `SD_WAN_ATTN_Q_CHUNK`, `SD_WAN_CROSS_Q_CHUNK`, `SD_WAN_MLP_CHUNK_MB` | Disjoint sub-block graph cuts + head/Q/cross-attn/MLP chunking (eliminates 2s TDR, enables 37× faster normal attention & 81f under 2 GiB) |
| `06-p2-cut-cache-host.patch` | `SD_CUT_CACHE_HOST_MB` | Disables `bind_outputs` for large cuts and parks them in Host RAM |
| `07-p3-conv-direct-tol.patch` | `SD_CONV_DIRECT_TOL` | Relaxes FP16 rounding tolerance (`0.005`) so Vulkan `conv2d/conv3d direct` turns ON (`1.8×` faster VAE) |
| `08-p3-separate-process-vae.patch` | `SD_VIDEO_LATENT_SAVE[_ONLY]`, `SD_VIDEO_LATENT_LOAD`, `SD_VIDEO_LATENT_DIT_TO_VAE` | Bit-exact float32 latent handoff + optional `dit_to_vae_latents` normalization for clean two-process VAE decode |

---

## Quick Start (Windows)

### 0. One-Click Zero-Prerequisite Setup (Recommended on a Fresh PC)

```powershell
# Automatically installs missing Git / Python / VS 2022 C++ Tools / CUDA or Vulkan SDK,
# clones & patches stable-diffusion.cpp, compiles sd-cli.exe, and downloads recommended models!
.\setup.cmd
```

### 1. Manual Clone, Apply Patches & Build

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

### 2. Smart Hardware-Aware Model Download (`configs/models_registry.json`, Apache 2.0)

```powershell
# Interactive hardware-aware wizard (reads GPU VRAM, System RAM & free disk and marks ★ Recommended)
py scripts/download_models.py

# Or direct profile download:
py scripts/download_models.py --profile t2v --with-taehv       # Starter T2V (~12.6 GiB)
py scripts/download_models.py --profile studio                 # Studio T2I + I2V Q4_K_M (~27.9 GiB)
py scripts/download_models.py --profile studio-compact         # Studio Compact Q3_K_M (~23.3 GiB)
```

### 3. Generate Video (Local Web Studio, Interactive TUI, or One-Command CLI)

```powershell
# Local Web Studio UI (zero extra deps — opens http://127.0.0.1:7860 with live VRAM bars, Studio T2I->I2V loop, gallery & +5s Extend)
.\web-studio.cmd
# or: .\picovid.cmd --web

# Interactive terminal wizard (5s/10s/15s presets, T2V / I2V / Studio Mode)
.\picovid.cmd

# Fast 5-Second Video (Wan 2.2 TI2V-5B Turbo: 81f @ 16->32fps = 5.0s, ~4.8 min!)
py scripts/generate.py --engine 5b --mode studio --cinema-hd -p "A cinematic wide shot of the Bosphorus at golden hour, historic ferry gliding across sparkling water, 35mm film" -W 640 -H 384 -f 81 --fps 16 -o bosphorus_5sec_hd32.mp4

# Continuous 10-Second Multi-Segment Video (--segments 2 + per-segment director prompts + Lab color anchoring)
py scripts/generate.py --engine 5b --mode studio --segments 2 -f 81 --fps 16 --cinema-hd --segment-prompts "Historic ferry gliding across the golden Bosphorus | Sweeping aerial pull-back over Istanbul skyline, seagulls soaring" -W 640 -H 384 -o bosphorus_10sec_32fps.mp4

# Extend ANY existing MP4 seamlessly from its last frame (+5s continuation)
py scripts/generate.py --engine 5b --extend-video bosphorus_5sec_hd32.mp4 -p "Camera rises smoothly into an aerial view at sunset" -W 640 -H 384 -f 81 --cinema-hd -o bosphorus_extended_10sec.mp4

# Studio Mode + Cinema HD (14B quality, 640x384 -> 1280x768 HD @ 32fps H.264 CRF 15 in ~5.5 min)
py scripts/generate.py --mode studio --cinema-hd -p "A cinematic wide shot of the Bosphorus at golden hour, historic ferry gliding across sparkling water, 35mm film" -W 640 -H 384 -f 17 -o bosphorus_hd32.mp4

# Standalone post-process (2x Vulkan Real-ESRGAN + 2x 32fps optical flow on any existing clip)
py scripts/postprocess.py -i bosphorus_studio.mp4 -o bosphorus_hd32.mp4 --upscale 2 --rife 2

# Direct Image-to-Video (Wan 2.2 I2V LightX2V MoE)
py scripts/generate.py --mode i2v -i my_image.png -p "Slow cinematic camera pan, gentle natural motion" -W 832 -H 480 -f 17 -o animated.mp4

# Text-to-Video (Wan 2.2 T2V Rapid-AIO v10)
py scripts/generate.py -p "A cinematic tracking shot of a vintage red sports car driving down a rain-slicked neon-lit city street at sunset" -W 640 -H 384 -f 33 -o output_33f.mp4

# Quick 5s live dashboard self-test / hardware dry-run
py scripts/generate.py --ui-demo
py scripts/generate.py --mode studio -p "test" -W 832 -H 480 -f 33 --dry-run
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

- **PicoVid** is released under the **[Apache License 2.0](LICENSE)**.
- Built on [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) and [`ggml`](https://github.com/ggml-org/ggml).
- Memory optimization research inspired in part by [`alexturkey/VideoTurko`](https://github.com/alexturkey/VideoTurko)'s low-VRAM patch methodology.
