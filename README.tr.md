# localVid

**[English](README.md) · [Türkçe](README.tr.md)**

> **Normalde 16–24 GB VRAM gerektiren amiral gemisi 14B video üretim modellerini (Wan 2.2), 4–6 GB VRAM'li düşük ekran kartlarında (AMD Vulkan & NVIDIA CUDA) çökmeden, stok sürüme göre 9.28×'e kadar daha hızlı ve 81 kareye (5.1 sn @ 640×384) kadar çalıştıran Apache 2.0 bellek & performans optimizasyon projesi.**

![AMD RX 5600 XT 6GB Vulkan üzerinde Wan 2.2 14B 640x384 4-adım çıktısı](docs/media/wan22_640x384_33f_frame12_rx5600xt.png)
***AMD Radeon RX 5600 XT (6 GB VRAM, Vulkan, Windows 11)** üzerinde yerel olarak üretilmiştir — `640×384`, 33 kare (2.1 sn), 4 adım (`wan2.2-t2v-rapid-aio-v10-Q4_K`), gerçek Wan 2.1 VAE, uçtan uca ~8.7 dk.*

| Kare 12 (`17f`, toplam 4.5 dk) | Kare 32 (`33f`, `Q_CHUNK=2048`) |
|---|---|
| ![17f örnek](docs/media/wan22_640x384_17f_rx5600xt.png) | ![33f son kare](docs/media/wan22_640x384_33f_frame32_rx5600xt.png) |

---

## Neden localVid?

Stok [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) üzerinde ağır bir **14B video modeli (Wan 2.2)** 4–6 GB VRAM'li kartlarda çalıştırıldığında 5 kritik duvara çarpar:

1. **WDDM Shared Memory Taşması (2.75–4.03 GiB PCIe Thrashing):** Tahliye edilmeyen blok ağırlıkları (`40 × 219 MB = 8.75 GB`) sessizce PCIe üzerinden sistem RAM'ine taşar ve hızı ~3–10× düşürür.
2. **Windows 2 Saniyelik TDR Çöküşü (`vk::Queue::submit: ErrorDeviceLost`):** `640×384`'te bölünmemiş tek bir Wan bloğu RDNA 1 Vulkan Flash Attention ile >6 sn sürdüğü için `blocks.0`'da sürücü sıfırlanır.
3. **Vulkan `maxMemoryAllocationSize = 2.0 GiB` Sınırı:** Uzun klipler (`49f–81f`) ve Wan 2.1 VAE 3D `im2col` tamponları 2 GiB tek buffer sınırını aşar.
4. **Aşırı Dar `conv2d/conv3d direct` Probe Eşiği:** RDNA 1 Vulkan'da zararsız `0.0037` FP16 yuvarlama farkı (`non_finite = 0`) `0.001` eşiğini geçtiği için `conv3d direct` kapatılır ve Wan 2.1 VAE 6.97–8.53 GB VRAM ister.
5. **Süreç-İçi VAE OOM & Normalizasyon Tuzağı:** DiT sürecinde VAE çalıştırmak VRAM'i parçalar; `--taesd` ile kaydedilen ham DiT latent'i `dit_to_vae_latents` uygulanmadan Wan 2.1 VAE'ye vermek diyagonal kafes artefaktı oluşturur.

**localVid**, bu 5 sorunun tamamını **9 adet modüler, varsayılanı OFF, %100 backend-bağımsız (`ggml_backend_*`) yama** ile hem **Vulkan** hem **CUDA** için çözer.

---

## Ölçülmüş Benchmark Sonuçları (AMD Radeon RX 5600 XT 6 GB Vulkan, Windows 11, 32 GB RAM)

*Not: Test sisteminde boşta masaüstü tabanı `2,914 MiB` Dedicated / `473 MiB` Shared VRAM'dir — tabloda `sd-cli` net VRAM artışı (`Δ`) ayrıca gösterilmiştir.*

### 1. Hızlanma Kırılımı (1 Adım, Wan 2.2 14B Q4_K_M)

| Çözünürlük × Kare | Yapılandırma | Self-Attn / Blok (40 head) | DiT 1 Adım | Hızlanma | DiT Compute Buffer | Net Dedicated VRAM (`Δ`) | Durum |
|---|---|---:|---:|---:|---:|---:|---|
| `480×320 × 17f` | Stok `sd.cpp` (`--max-vram 3.8`, FA on) | — | `356.10 s` | `1.00×` | `345.2 MB` | `+1,563 MiB` (`+2,281` Shared!) | VAE OOM fallback |
| `480×320 × 17f` | P1 (`mv=1.5`, `evict_on_end`, FA on) | — | `137.73 s` | `2.59×` | `345.2 MB` | **`+1,333 MiB`** | OK |
| **`480×320 × 17f`** | **localVid P1+P2 (`HEAD_CHUNK=4`, FA off)** | **`78.0 ms`** | **`38.36 s`** | **`9.28×`** | **`381.6 MB`** | **`+1,518 MiB`** | **OK** |
| `640×384 × 17f` | Stok `sd.cpp` / P1 (FA on) | *(TDR çöküş)* | **ÇÖKTÜ (`23 s`)** | — | `547.3 MB` | — | `ErrorDeviceLost` |
| `640×384 × 17f` | P2 (`HEAD_CHUNK=8`, FA on) | `6,265.0 ms` | `296.50 s` | `1.00×` | `375.0 MB` | `+1,300 MiB` | TDR Çözüldü |
| **`640×384 × 17f`** | **localVid P1+P2 (`HEAD_CHUNK=4`, FA off)** | **`168.0 ms` (`37.3×`)** | **`56.60–58.70 s`** | **`5.05×`** | **`592.5 MB`** | **`+1,187–1,507 MiB`** | **OK** |

### 2. `640×384` Kare Tavanı Merdiveni (81 Kare / 5.1 sn, 1 Adım + Ayrı Süreç Wan 2.1 VAE)

| Kare (`@16fps`) | Token (`N`) | Profil | DiT (1 Adım) | DiT Compute Buffer | Ayrı Süreç Wan 2.1 VAE (`conv_direct=on`) | Net DiT Dedicated (`Δ`) | Durum |
|---|---:|---|---:|---:|---:|---:|---|
| **`17f`** (`1.1 sn`) | `4,800` | Hızlı (`q=off, mv=1.5`) | **`56.60 s`** | `592.5 MB` | **`34.63 s`** (`1,441 MB`) | `+1,187 MiB` | **OK** |
| **`33f`** (`2.1 sn`) | `8,640` | Hızlı (`q=2048, mv=2.0`) | **`114.51 s`** | `675.0 MB` | **`64.78 s`** (`1,617 MB`) | `+1,656 MiB` | **OK** |
| **`49f`** (`3.1 sn`) | `12,480` | Hızlı (`q=1024, mv=2.5`) | **`195.97 s`** | `1,500.5 MB` | **`131.94 s`** (`1,617–1,922 MB`) | `+1,834 MiB` | **OK** |
| **`81f`** (`5.1 sn`) | `20,160` | Sığdırma (`QKV_SPLIT=1`, `CUT_HOST=32`, `q=1024`, `mv=2.5`) | **`728.03 s`** | **`1,890.0 MB`** (`< 2 GiB`) | **`230.16 s`** (`1,617–1,922 MB`) | `+2,110 MiB` | **OK** |

### 3. Gerçek 4-Adımlı Uçtan Uca Video Üretimi (`wan2.2-t2v-rapid-aio-v10-Q4_K`, `CFG=1.0`, `euler_a + beta`, `flow-shift 5.0`)

| Çıktı (`640×384`) | TE (CPU) | DiT (4 Adım) | Ayrı Süreç Wan 2.1 VAE | Toplam Uçtan Uca |
|---|---:|---:|---:|---:|
| **17 kare (1.1 sn)** | `22.2 s` | `209.4 s` | `34.6 s` | **~4.5 dk** |
| **33 kare (2.1 sn)** | `13.4 s` | `444.1 s` | `64.8 s` | **~8.7 dk** |

---

## Hızlı Kurulum ve Çalıştırma (Windows)

### 1. Kaynak Çekme, Yama Uygulama ve Derleme

```powershell
git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/leejet/stable-diffusion.cpp.git third_party/stable-diffusion.cpp
git -C third_party/stable-diffusion.cpp checkout 3b12781

# 9 modüler yamayı uygula
powershell -ExecutionPolicy Bypass -File scripts/apply_patches.ps1

# Derle (AMD/Intel/NVIDIA için -Backend vulkan, NVIDIA CUDA için -Backend cuda)
powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Backend vulkan
```

### 2. Python Bağımlılıkları ve Modelleri İndir (Apache 2.0 Varsayılan Set — ~12.6 GiB)

```powershell
py -m pip install -r requirements.txt
py scripts/download_models.py --with-taehv
```

### 3. Video Üret (İnteraktif TUI Sihirbazı veya Tek Komut CLI)

```powershell
# İnteraktif sihirbaz + canlı Rich kontrol paneli (veya localvid.cmd'ye çift tıklayın)
.\localvid.cmd
# py scripts/generate.py

# Tam kalite (Ayrı süreç DiT + Wan 2.1 VAE -> .mp4, ham loglar .log dosyasına yazılır)
py scripts/generate.py -p "A cinematic tracking shot of a vintage red sports car driving down a rain-slicked neon-lit city street at sunset" -W 640 -H 384 -f 33 -o cikti_33f.mp4

# Hızlı TAEHV önizleme (~2–6 sn decode)
py scripts/generate.py -p "Ocean waves crashing on dark volcanic rocks at golden hour" -f 17 --fast-preview -o onizleme.mp4

# 5 saniyelik canlı arayüz demosu / kuru donanım doğrulaması
py scripts/generate.py --ui-demo
py scripts/generate.py -p "test" -f 81 --dry-run
```

---

## Donanım Destek Matrisi (Dürüst Doğrulama Durumu)

| Donanım / Platform | Backend | Durum | Notlar |
|---|---|---|---|
| **AMD Radeon RX 5600 XT (6 GB, RDNA 1), Windows 11, 32 GB RAM** | Vulkan | **Fiziksel Olarak Doğrulandı** | Tüm benchmark'lar (`17f–81f`, 4-adımlı temiz MP4'ler, `9.28×` hızlanma, `1.30–2.11 GiB` net VRAM) bu kartta ölçüldü. |
| **AMD / Intel 4–8 GB Kartlar (Windows / Linux)** | Vulkan | **Mimari Hazır (Topluluk Testi Bekleniyor)** | Saf `ggml-vulkan` + `probe_hardware.py` heap bütçesi tespiti kullanır. |
| **NVIDIA 4–8 GB Kartlar (GTX 1050 Ti / RTX 2060 / 3050 / 3060 / 4060)** | CUDA | **Mimari Hazır (Topluluk Testi Bekleniyor)** | 9 yamanın tamamı backend-bağımsız `ggml_backend_*` API kullanır; `generate.py` CUDA profillerini otomatik seçer (`--simulate-profile cuda-4gb` doğrulandı). |

*Not: Sistem RAM gereksinimi **16–32 GB**'dır (`49f–81f` uzun kliplerde `SD_CUT_CACHE_HOST_MB` için 32 GB önerilir).*

---

## Lisans ve Teşekkür

- **localVid**, **[Apache License 2.0](LICENSE)** ile lisanslanmıştır.
- [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) ve [`ggml`](https://github.com/ggml-org/ggml) üzerine inşa edilmiştir.
- Düşük VRAM yama metodolojisi kısmen [`alexturkey/VideoTurko`](https://github.com/alexturkey/VideoTurko) projesinden ilham almıştır.
