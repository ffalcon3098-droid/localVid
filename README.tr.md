# PicoVid

**[English](README.md) · [Türkçe](README.tr.md)**

> **Normalde 16–24 GB VRAM gerektiren amiral gemisi 14B video üretim modellerini (Wan 2.2 T2V + I2V + Z-Image-Turbo Stüdyo), 4–6 GB VRAM'li düşük ekran kartlarında (AMD Vulkan & NVIDIA CUDA) çökmeden, stok sürüme göre 9.28×'e kadar daha hızlı ve 81 kareye (5.1 sn @ 640×384) kadar çalıştıran Apache 2.0 bellek & performans optimizasyon projesi.**

![AMD RX 5600 XT 6GB Vulkan üzerinde Wan 2.2 14B 640x384 4-adım çıktısı](docs/media/wan22_640x384_33f_frame12_rx5600xt.png)
***AMD Radeon RX 5600 XT (6 GB VRAM, Vulkan, Windows 11)** üzerinde yerel olarak üretilmiştir — `640×384`, 33 kare (2.1 sn), 4 adım (`wan2.2-t2v-rapid-aio-v10-Q4_K`), gerçek Wan 2.1 VAE, uçtan uca ~8.7 dk.*

| Kare 12 (`17f`, toplam 4.5 dk) | Kare 32 (`33f`, `Q_CHUNK=2048`) |
|---|---|
| ![17f örnek](docs/media/wan22_640x384_17f_rx5600xt.png) | ![33f son kare](docs/media/wan22_640x384_33f_frame32_rx5600xt.png) |

---

## Neden PicoVid?

Stok [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) üzerinde ağır bir **14B video modeli (Wan 2.2)** 4–6 GB VRAM'li kartlarda çalıştırıldığında 5 kritik duvara çarpar:

1. **WDDM Shared Memory Taşması (2.75–4.03 GiB PCIe Thrashing):** Tahliye edilmeyen blok ağırlıkları (`40 × 219 MB = 8.75 GB`) sessizce PCIe üzerinden sistem RAM'ine taşar ve hızı ~3–10× düşürür.
2. **Windows 2 Saniyelik TDR Çöküşü (`vk::Queue::submit: ErrorDeviceLost`):** `640×384`'te bölünmemiş tek bir Wan bloğu RDNA 1 Vulkan Flash Attention ile >6 sn sürdüğü için `blocks.0`'da sürücü sıfırlanır.
3. **Vulkan `maxMemoryAllocationSize = 2.0 GiB` Sınırı:** Uzun klipler (`49f–81f`) ve Wan 2.1 VAE 3D `im2col` tamponları 2 GiB tek buffer sınırını aşar.
4. **Aşırı Dar `conv2d/conv3d direct` Probe Eşiği:** RDNA 1 Vulkan'da zararsız `0.0037` FP16 yuvarlama farkı (`non_finite = 0`) `0.001` eşiğini geçtiği için `conv3d direct` kapatılır ve Wan 2.1 VAE 6.97–8.53 GB VRAM ister.
5. **Süreç-İçi VAE OOM & Normalizasyon Tuzağı:** DiT sürecinde VAE çalıştırmak VRAM'i parçalar; `--taesd` ile kaydedilen ham DiT latent'i `dit_to_vae_latents` uygulanmadan Wan 2.1 VAE'ye vermek diyagonal kafes artefaktı oluşturur.

**PicoVid**, bu 5 sorunun tamamını **9 adet modüler, varsayılanı OFF, %100 backend-bağımsız (`ggml_backend_*`) yama** ile hem **Vulkan** hem **CUDA** için çözer.

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

### 3. Gerçek 4-Adımlı Uçtan Uca Video & v0.2 Stüdyo Modu (`640×384`, Ayrı Süreç Wan 2.1 VAE, H.264 CRF 16)

| Hat (`640×384`) | İlk Kare (T2I) | I2V `vae_encode` | TE (CPU) | DiT (4 Adım) | Wan 2.1 VAE (`conv_direct=on`) | Toplam Uçtan Uca |
|---|---:|---:|---:|---:|---:|---:|
| **T2V — 17 kare (1.1 sn)** (`Rapid-AIO v10 Q4_K`) | — | — | `22.2 s` | `209.4 s` | `34.6 s` | **~4.5 dk** |
| **T2V — 33 kare (2.1 sn)** (`Rapid-AIO v10 Q4_K`) | — | — | `13.4 s` | `444.1 s` | `64.8 s` | **~8.7 dk** |
| **v0.2 Stüdyo Modu (T2I → I2V 17f)** (`Z-Image-Turbo` + `Wan 2.2 I2V LightX2V MoE Q4_K_M`, GPU-tiled `vae_encode`) | **`21.9 s`** *(832×480: `32.1 s`)* | **`19.3 s`** *(216s CPU'dan 11.2× hızlı)* | `18.2 s` | **`234.5 s`** *(2 High + 2 Low)* | **`35.7 s`** | **~5.6 dk** |
| **v0.3 Sinema HD (`--cinema-hd`, `640×384×17f` → `1280×768 HD @ 32fps`)** | **`21.9 s`** | **`19.3 s`** | `18.2 s` | **`~195–210 s`** *(Faz A: `HEAD_CHUNK=8`, `clear()` prefetch, NVMe ön-ısıtma)* | **`35.7 s`** + **`49.7 s`** (`x4plus` 2× HD + 32fps optik akış) *(veya `~8 s` `animevideov3`)* | **~5.5 dk (`1280×768 @ 32fps`)** |
| **v0.3 Hızlı 5 Saniye (`--engine 5b`, `Wan 2.2 TI2V-5B Turbo`, `640×384 × 81f @ 16→32fps` = 5.0s!)** | *(ops. `21.9 s`)* | **`1.3 s`** *(tek-kare frozen I2V!)* | `13.1 s` | **`74.2 s`** *(`18.5 s/adım`, `%80.4` GPU verimi)* | **`167.8 s`** (`Wan 2.2 VAE 224×224`, 6 karo) + **`30.1 s`** (`1280×768 @ 32fps`) | **~4.8 dk (`1280×768 @ 32fps`!)** |
| **v0.3 Fotogerçekçi 5 Saniye (`--engine 5b`, doğal `832×480 × 81f @ 16→32fps`, `beta` scheduler)** | *(ops. `32.1 s`)* | **`2.1 s`** | `13.4 s` | **`140.0 s`** *(`35.0 s/adım`)* | **`334.9 s`** (`224×256` ile 8 karoya indi!) + **`11.3 s`** (`32fps` AOBMC) | **~8.4 dk (`832×480 @ 32fps`, keskin doku)** |
| **v0.3 Hızlı 5 Saniye (`--engine 5b`, `640×384 × 121f @ 24fps` native = 5.04s)** | *(ops. `21.9 s`)* | **`1.3 s`** | `15.6 s` | **`124.1 s`** *(`30.2 s/adım`, PeakDed `4,346 MiB`)* | **`352.5 s`** (`Wan 2.2 VAE 192×192`) / **`1.9 s`** (`--fast-preview` `taew2_2`) | **~2.4 dk (TAEW) · ~7.9 dk (Tam VAE)** |
| **`832×480 × 33f` Native 14B I2V (Faz A Shared=0)** | — | — | — | **`232.7 s/adım`** *(öncesi `613.2 s/adım` → **`2.64×` hızlı**)* | — | **~15.5 dk** *(~45 dk'dan düştü)* |

| Stüdyo İlk Kare (`Z-Image-Turbo 6B`, `832×480`, `32.1 s`) | Canlandırılmış I2V Son Kare (`Wan 2.2 I2V MoE`, GPU-tiled `vae_encode`, `PSNR 39.35 dB`) | v0.3 Sinema HD Çıktısı (`1280×768 HD @ 32fps`, `realesrgan-x4plus` 2× + AOBMC) |
|---|---|---|
| ![Z-Image-Turbo 832x480 Boğaziçi](docs/media/studio_bosphorus_832x480_zimage_rx5600xt.png) | ![Wan 2.2 I2V 17f Boğaziçi](docs/media/studio_bosphorus_17f_i2v_rx5600xt.png) | ![Sinema HD 1280x768 Boğaziçi](docs/media/cinema_hd_1280x768_bosphorus_rx5600xt.png) |

---

## Hızlı Kurulum ve Çalıştırma (Windows)

### 0. Sıfırdan Tek Tıkla Otomatik Kurulum (Temiz Bilgisayarda Önerilen)

```powershell
# Eksik Git / Python / VS 2022 C++ Build Tools / CUDA veya Vulkan SDK'yi otomatik kurar,
# 9 yamayı uygular, sd-cli.exe'yi derler, Cinema HD aracını ve donanıma uygun modelleri indirir!
.\setup.cmd
```

### 1. Manuel Kaynak Çekme, Yama Uygulama ve Derleme

```powershell
git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/leejet/stable-diffusion.cpp.git third_party/stable-diffusion.cpp
git -C third_party/stable-diffusion.cpp checkout 3b12781

# 9 modüler yamayı uygula
powershell -ExecutionPolicy Bypass -File scripts/apply_patches.ps1

# Derle (AMD/Intel/NVIDIA için -Backend vulkan, NVIDIA CUDA için -Backend cuda)
powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Backend vulkan
```

### 2. Akıllı Donanım Duyarlı Model İndirme (`configs/models_registry.json`, Apache 2.0)

```powershell
py -m pip install -r requirements.txt

# Donanım duyarlı sihirbaz (GPU VRAM, Sistem RAM ve boş diski okuyup ★ Önerilen paketi seçer)
py scripts/download_models.py

# Veya doğrudan profil indir:
py scripts/download_models.py --profile t2v --with-taehv       # T2V Başlangıç (~12.6 GiB)
py scripts/download_models.py --profile studio                 # Stüdyo T2I + I2V Q4_K_M (~27.9 GiB)
py scripts/download_models.py --profile studio-compact         # Stüdyo Kompakt Q3_K_M (~23.3 GiB)
```

### 3. Video Üret (Yerel Web Stüdyosu, İnteraktif TUI veya Tek Komut CLI)

```powershell
# Yerel Web Stüdyosu (ek kurulum gerektirmez — http://127.0.0.1:7860, canlı VRAM, önizleme/onay, model durumu & +5sn Uzat)
.\web-studio.cmd
# veya: .\picovid.cmd --web

# İnteraktif terminal sihirbazı (5sn/10sn/15sn seçenekleri, T2V / I2V / Stüdyo Modu)
.\picovid.cmd

# Hızlı 5 Saniyelik Video (Wan 2.2 TI2V-5B Turbo: 81f @ 16->32fps = 5.0 sn, ~4.8 dk!)
py scripts/generate.py --engine 5b --mode studio --cinema-hd -p "A cinematic wide shot of the Bosphorus at golden hour, historic ferry gliding across sparkling water, 35mm film" -W 640 -H 384 -f 81 --fps 16 -o bogaz_5sn_hd32.mp4

# Kesintisiz 10 Saniyelik Çoklu-Segment Video (--segments 2 + bölüm bazlı yönetmen prompt'u + Lab renk çıpası)
py scripts/generate.py --engine 5b --mode studio --segments 2 -f 81 --fps 16 --cinema-hd --segment-prompts "Historic ferry gliding across the golden Bosphorus | Sweeping aerial pull-back over Istanbul skyline, seagulls soaring" -W 640 -H 384 -o bogaz_10sn_32fps.mp4

# Mevcut herhangi bir MP4 videoyu son karesinden kesintisiz uzat (+5 sn devam)
py scripts/generate.py --engine 5b --extend-video bogaz_5sn_hd32.mp4 -p "Camera rises smoothly into an aerial view at sunset" -W 640 -H 384 -f 81 --cinema-hd -o bogaz_uzatilmis_10sn.mp4

# Stüdyo Modu + Sinema HD (14B kalite, 640x384 -> 1280x768 HD @ 32fps H.264 CRF 15, ~5.5 dk)
py scripts/generate.py --mode studio --cinema-hd -p "A cinematic wide shot of the Bosphorus at golden hour, historic ferry gliding across sparkling water, 35mm film" -W 640 -H 384 -f 17 -o bogaz_hd32.mp4

# Bağımsız son-işleme (herhangi bir videoya 2x Vulkan Real-ESRGAN + 2x 32fps optik akış)
py scripts/postprocess.py -i bogaz_studio.mp4 -o bogaz_hd32.mp4 --upscale 2 --rife 2

# Doğrudan Görselden Videoya (Wan 2.2 I2V LightX2V MoE)
py scripts/generate.py --mode i2v -i gorselim.png -p "Slow cinematic camera pan, gentle natural motion" -W 832 -H 480 -f 17 -o canlandirma.mp4

# Metinden Videoya (Wan 2.2 T2V Rapid-AIO v10)
py scripts/generate.py -p "A cinematic tracking shot of a vintage red sports car driving down a rain-slicked neon-lit city street at sunset" -W 640 -H 384 -f 33 -o cikti_33f.mp4

# 5 saniyelik canlı arayüz demosu / kuru donanım doğrulaması
py scripts/generate.py --ui-demo
py scripts/generate.py --mode studio -p "test" -W 832 -H 480 -f 33 --dry-run
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

- **PicoVid**, **[Apache License 2.0](LICENSE)** ile lisanslanmıştır.
- [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) ve [`ggml`](https://github.com/ggml-org/ggml) üzerine inşa edilmiştir.
- Düşük VRAM yama metodolojisi kısmen [`alexturkey/VideoTurko`](https://github.com/alexturkey/VideoTurko) projesinden ilham almıştır.
