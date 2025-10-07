# LED BodyGlow

Fast, audio‑reactive **body glow + halo** visualizer with live camera **or** video input. Includes heatmap palettes, gradient/rainbow modes, white‑noise “sparks” background, scanline overlays, and a safe fullscreen toggle. Built with **OpenCV**, **pygame**, and an **ONNX** human segmentation model (e.g., U²‑Net).

> Tested on macOS with Python **3.13.5**, OpenCV‑Python **4.12**, pygame **2.6**. Works without mediapipe.

---

## Features

- **Threaded person segmentation** (OpenCV DNN + ONNX) for smooth UI.
- **Inner glow** (contour bloom) + **outer halo** (distance transform) that extends beyond the silhouette.
- **Color systems**: single hue, two‑color gradients (presets), fast **rainbow LUT**, heatmap palettes.
- **Background options**: original camera, **black**, **heatmap blobs**, **white‑noise “sparks”** (popcorn look), optional scanlines.
- **Audio reactive**: low/mid/high bands (via `sounddevice`), **bass‑only** mode, color shifts from bass/treble, tweakable responsiveness.
- **Performance helpers**: ROI tracking, segmentation FPS decoupled from UI FPS, **downscaled glow/halo**, CoreML/OpenCL targets.
- **Video file input** with `--loop-video` (mp4/mov) in addition to camera.
- **Safe fullscreen** toggle (recreates the window; avoids SDL macOS glitches).
- Built‑in **preset**: `--preset popcorn` (black stage + white sparks + bassy glow).

---

## Install

> Python **3.11+** recommended (tested on **3.13.5**).

```bash
python3 -m pip install --upgrade pygame opencv-python numpy
# Optional for audio-reactive:
python3 -m pip install --upgrade sounddevice
# Optional for quick ffmpeg without Homebrew (video transcode helper):
python3 -m pip install --upgrade imageio-ffmpeg
```

**Segmentation model**: place a U²‑Net ONNX file (e.g., `u2net.onnx`) somewhere on disk and pass its path via `--seg-model`. The model is not bundled.

---

## Run (examples)

### 1) Live camera (fast, two‑color gradient, black background)
```bash
python3 led_bodyglow.py \
  --seg-model "/Users/samkulka/Desktop/Coding/Python/Plugins/u2net.onnx" \
  --dnn-backend coreml \
  --cam-width 960 --cam-height 540 --fps 60 \
  --seg-size 192 --seg-fps 24 --roi --roi-pad 80 \
  --glow-downscale 2 --halo-downscale 2 \
  --gradient --gradient-mode two --no-hud --black-bg \
  --halo-size 48 --halo-strength 1.2 --glow-size 21 --strength 1.2 \
  --audio-react --audio-bass-only --audio-mode glow \
  --audio-color --audio-hue-gain 240 --audio-val-gain 0.0 \
  --audio-device "BlackHole 2ch"
```

### 2) Video file input (loop)
```bash
python3 led_bodyglow.py \
  --input "/Users/samkulka/Movies/zacharyscott.mp4" --loop-video \
  --seg-model "/Users/samkulka/Desktop/Coding/Python/Plugins/u2net.onnx" \
  --dnn-backend coreml --fps 60 --no-hud \
  --seg-size 160 --seg-fps 18 --roi --roi-pad 48 \
  --glow-downscale 3 --halo-downscale 3 \
  --gradient --gradient-mode two --outline-thickness 0 --silhouette
```

### 3) Black “popcorn” preset (white sparks + bassy glow)
```bash
python3 led_bodyglow.py \
  --preset popcorn \
  --seg-model "/Users/samkulka/Desktop/Coding/Python/Plugins/u2net.onnx" \
  --dnn-backend coreml \
  --cam-width 960 --cam-height 540 --fps 60
```

> Add `--audio-device "BlackHole 2ch"` to react to internal audio (see below).

---

## Audio (macOS loopback)

To drive visuals from **Spotify/system audio**:

1. Create a **Multi‑Output Device** in *Audio MIDI Setup* (Built‑in Output + **BlackHole 2ch**).  
2. Set **System Output** to that Multi‑Output.  
3. Run app with `--audio-react --audio-bass-only --audio-mode glow --audio-device "BlackHole 2ch"`.

**Live tuning**:  
- **F2/F1** gain up/down, **Shift+F3/F3** less/more smoothing.  
- **t** cycles what audio drives (halo/glow/alpha).  
- **Shift+B** toggles bass‑only.  
- Extra response knobs: `--audio-react-boost`, `--audio-react-gamma`, `--audio-floor`.

---

## Performance tips

- Lower `--seg-size` (e.g., 160 or 144).  
- Lower `--seg-fps` (e.g., 18–24) — UI stays 60 fps.  
- Use `--roi --roi-pad 48` so only the person region is segmented after lock.  
- Use `--glow-downscale` / `--halo-downscale` (2–3).  
- Prefer **two‑color** gradients over rainbow; or keep rainbow but avoid huge windows/fullscreen.  
- For video files, **transcode** once to ~**960×540 @ 24/30fps**. If you lack `ffmpeg`, install `imageio-ffmpeg` and call its bundled binary.

**Example transcode (no Homebrew):**
```bash
python3 -m pip install --upgrade imageio-ffmpeg
FFMPEG="$(python3 - <<'PY'
import imageio_ffmpeg as i; print(i.get_ffmpeg_exe())
PY
)"
"$FFMPEG" -hide_banner -loglevel error \
  -i "/path/to/input.mp4" \
  -vf "scale=-2:540:flags=lanczos,fps=24" \
  -c:v libx264 -preset veryfast -crf 22 -c:a aac -movflags +faststart \
  "/path/to/output_540p24.mp4"
```

---

## Hotkeys

**Global**
- `q` / `Esc` — quit  
- `Space` — pause/resume  
- `Ctrl+F` or `Shift+Enter` — toggle fullscreen (safe)  
- `\` — toggle global scanlines overlay  
- `F9/F10` — overlay scanline gap −/+  
- `F11/F12` — overlay scanline strength −/+  
- `Shift+S` — toggle silhouette depth overlay (dark border)

**Body Glow**
- `g` — inner glow on/off  
- `p` — show person fill on/off  
- `b` — black background on/off  
- `Shift+B` — **bass‑only** audio level on/off  
- `o` — outline on/off  
- `s` — silhouette‑only outlines  
- `[` / `]` — inner glow size −/+ (odd)  
- `-` / `=` — inner glow strength −/+  
- `d` / `f` — halo size −/+  
- `,` / `.` — segmentation threshold −/+  
- `m` — median blur on mask on/off  
- `v` — bilateral smoothing on/off  
- `;` / `'` — segmentation FPS −/+  
- `3` / `4` — outline thickness −/+ (0 hides)  
- `c` — hue cycling on/off  
- `h` — gradient on/off  
- `j` / `l` — gradient mode (two ↔ rainbow)  
- `1` / `2` — two‑color preset prev/next  
- `u` / `i` — hue speed −/+  
- `x` — swap two‑color gradient ends  
- `k` — person heatmap on/off  
- `r` — heatmap edge ↔ interior  
- `e` / `9` / `0` — cycle heatmap & bg palettes  
- `z` — background heatmap/noise toggle  
- `/` — scanlines on background heatmap toggle  
- `n` — white‑noise **sparks** on/off  
- `Shift+N` — one‑shot spark burst  
- `5` / `6` — background blob scale −/+  
- `7` / `8` — background drift speed −/+  
- `y` — audio reactive on/off  
- `Shift+Y` — audio→color on/off  
- `t` — audio target (halo / glow / alpha)  
- `F1/F2` — audio input gain −/+  
- `F3 (Shift+F3)` — audio smoothing + (−)  
- `F5/F6` — audio hue‑gain −/+  
- `F7/F8` — audio brightness‑gain −/+

---

## CLI reference (most‑used)

- `--seg-model PATH` (required): ONNX model path.  
- `--input PATH` / `--camera IDX`: choose video file or camera.  
- `--loop-video`: loop file playback.  
- `--dnn-backend {default,opencl,coreml}` / `--dnn-target {cpu,opencl,opencl_fp16}`  
- `--seg-size N` (e.g., 144–256), `--seg-fps N` (e.g., 18–24), `--roi`, `--roi-pad PX`  
- `--glow-downscale N`, `--halo-downscale N` (2–3 is fast & soft)  
- `--gradient`, `--gradient-mode {two,rainbow}`, `--color-cycle`, `--hue-speed DEG_PER_SEC`  
- `--heatmap`, `--heatmap-mode {edge,interior}`, `--heatmap-palette`, `--heatmap-alpha`  
- `--white-noise`, `--spark-rate`, `--spark-decay`, `--spark-size`, `--spark-brightness`, `--spark-strobe-chance`, `--spark-strobe-mult`  
- `--audio-react`, `--audio-mode {halo,glow,alpha}`, `--audio-device NAME`, `--audio-bass-only`  
- `--audio-react-boost`, `--audio-react-gamma`, `--audio-floor`, `--audio-hue-gain`, `--audio-val-gain`

---

## Troubleshooting

- **`dquote>` prompt in zsh**: your paste has an unmatched `"`; press **Ctrl‑C** and re‑paste with straight quotes and proper `\` line continuations (no trailing spaces).
- **No `ffmpeg` command**: install the bundled binary via `imageio-ffmpeg` (see Performance tips).
- **No audio reactivity**: ensure System Output is the **Multi‑Output Device** and `--audio-device "BlackHole 2ch"` is set; raise gain (**F2**), reduce smoothing (**Shift+F3**) and enable **bass‑only** (**Shift+B**).
- **Camera not opening**: try `--camera 1` or a video file via `--input`.
- **Slow video**: transcode to 540p/24–30fps, use downscales (glow/halo 2–3), lower `--seg-size`, enable ROI.

---

## Notes

- This repo does **not** include model weights. Provide your own `u2net.onnx` (or a compatible human‑segmentation ONNX).  
- `coreml` backend may improve DNN speed on Apple Silicon; otherwise use `default`/`opencl` as available.

---

## License

Add your license of choice (e.g., MIT) if you plan to distribute.
