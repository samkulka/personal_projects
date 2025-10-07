# led_bodyglow.py — fast body glow + halo, audio-reactive (+bass-only), white-noise “sparks” bg,
#                    perf downscales, MP4 input, safe fullscreen, rainbow LUT, preset "popcorn"
#
# ---------------- Hotkeys ----------------
# Global:
#   q / ESC               → quit
#   Space                 → pause / resume
#   Ctrl+F / Shift+Enter  → toggle fullscreen (safe)
#   \                     → toggle GLOBAL scanlines overlay (dark, crisp)
#   F9 / F10              → global scanline gap − / +
#   F11 / F12             → global scanline strength − / +
#   Shift+S               → toggle silhouette depth overlay (dark border)
#
# Body Glow mode:
#   g         → toggle inner glow on/off
#   p         → toggle showing person fill
#   b         → toggle black background
#   Shift+B   → toggle “bass-only” audio level (use low band only)
#   o         → toggle outline on/off
#   s         → toggle silhouette-only outlines
#   [ / ]     → decrease / increase inner glow size
#   - / =     → decrease / increase inner glow strength
#   d / f     → decrease / increase OUTER HALO size
#   , / .     → lower / raise segmentation threshold
#   m         → toggle median blur on mask
#   v         → toggle bilateral smoothing
#   ; / '     → decrease / increase segmentation FPS (worker thread)
#   3 / 4     → decrease / increase outline thickness (0 hides outline)
#   c         → toggle color cycling (rainbow over time)  [starts OFF]
#   h         → toggle gradient mode on/off
#   j / l     → switch gradient mode (two-color ↔ rainbow)
#   1 / 2     → previous / next two-color gradient preset
#   u / i     → slow down / speed up hue cycling (±60 dps per press)
#   x         → swap the two gradient colors
#   k         → toggle heatmap on person on/off
#   r         → switch heatmap mode (edge ↔ interior)
#   e / 9 / 0 → cycle heatmap & background palettes (prev/next)
#   z         → toggle background heatmap/noise
#   /         → toggle scanlines on BACKGROUND heatmap
#   n         → toggle WHITE-NOISE "sparks" background
#   Shift+N   → one-shot spark burst
#   5 / 6     → decrease / increase background blob scale
#   7 / 8     → decrease / increase background drift speed
#   y         → toggle audio reactive on/off
#   Shift+Y   → toggle audio→color (bass→hue, treble→brightness)
#   t         → cycle audio mode (halo / glow / alpha)
#   F1 / F2   → audio input gain − / +
#   F3        → audio smoothing (hold Shift to decrease)
#   F5 / F6   → audio hue-gain − / +
#   F7 / F8   → audio brightness-gain − / +
#
# ---------------- Features & Notes ----------------
# • Perf flags: --glow-downscale N, --halo-downscale N (compute heavy FX at lower res, upscale)
# • MP4 input: --input path/to/video.mp4 (instead of --camera). Add --loop-video to loop.
# • Fast rainbow gradient via LUT (no Python loops).
# • Global scanlines overlay (dark, crisp) + silhouette depth overlay for “body depth”.
# • Segmentation runs in a worker thread; UI draws at target FPS.
# • Outer HALO uses a distance transform to bloom beyond the silhouette (less blocky).
# • Preset: --preset popcorn  → black bg + punchy white “sparks”, bassy glow.
#
# Example (live camera, fast):
#   python3 led_bodyglow.py \
#     --seg-model /path/to/u2net.onnx \
#     --dnn-backend coreml \
#     --cam-width 960 --cam-height 540 --fps 60 \
#     --seg-size 192 --seg-fps 24 --roi --roi-pad 80 \
#     --glow-downscale 2 --halo-downscale 2 \
#     --gradient --gradient-mode two --no-hud --black-bg \
#     --halo-size 48 --halo-strength 1.2 --glow-size 21 --strength 1.2 \
#     --audio-react --audio-bass-only --audio-mode glow \
#     --audio-color --audio-hue-gain 240 --audio-val-gain 0.0 \
#     --audio-device "BlackHole 2ch"
#
# Example (MP4 loop):
#   python3 led_bodyglow.py \
#     --input /path/to/clip.mp4 --loop-video \
#     --seg-model /path/to/u2net.onnx \
#     --seg-size 192 --seg-fps 24 --roi --roi-pad 80 \
#     --glow-downscale 2 --halo-downscale 2 \
#     --gradient --gradient-mode rainbow --black-bg --no-hud
#
import os, time, argparse, threading, collections
from dataclasses import dataclass
import numpy as np
import pygame
import cv2 as cv

# Optional audio input for reactive effects
try:
    import sounddevice as sd
except Exception:
    sd = None

# ---------------- Utility helpers ----------------
def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

def odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1

def parse_rgb(s):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3 or any(p < 0 or p > 255 for p in parts):
        raise ValueError("Use R,G,B in 0..255")
    return tuple(parts)

def bgr_from_rgb(rgb):
    r, g, b = rgb
    return (b, g, r)

def hsv_to_bgr(h, s=1.0, v=1.0):
    h = float(h) % 360.0
    s = clamp(float(s), 0.0, 1.0)
    v = clamp(float(v), 0.0, 1.0)
    c = v * s
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    m = v - c
    if   0 <= h < 60:   r,g,b = (c, x, 0)
    elif 60 <= h <120:  r,g,b = (x, c, 0)
    elif 120<= h <180:  r,g,b = (0, c, x)
    elif 180<= h <240:  r,g,b = (0, x, c)
    elif 240<= h <300:  r,g,b = (x, 0, c)
    else:               r,g,b = (c, 0, x)
    r,g,b = (r+m, g+m, b+m)
    return (int(b*255), int(g*255), int(r*255))

def lerp_color_bgr(c0, c1, t):
    t = clamp(float(t), 0.0, 1.0)
    b0,g0,r0 = c0; b1,g1,r1 = c1
    return (int(b0 + (b1-b0)*t),
            int(g0 + (g1-g0)*t),
            int(r0 + (r1-r0)*t))

# --- FAST RAINBOW LUT (for gradient_mode='rainbow') ---
def make_rainbow_lut():
    """256×3 BGR LUT spanning ~0..240° hue (S=1,V=1) for fast rainbow colorize."""
    hues = np.linspace(0, 240, 256, dtype=np.float32)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i, h in enumerate(hues):
        lut[i] = hsv_to_bgr(float(h), 1.0, 1.0)
    return lut

# ---------------- Safe fullscreen toggler (recreate window) ----------------
def _toggle_fullscreen_safe(state):
    flags = pygame.SCALED | pygame.RESIZABLE
    if not state.get("is_fullscreen", False):
        info = pygame.display.Info()
        screen = pygame.display.set_mode((info.current_w, info.current_h), flags | pygame.FULLSCREEN)
        state["is_fullscreen"] = True
        state["screen"] = screen
    else:
        w, h = state.get("windowed_size", (1280, 720))
        screen = pygame.display.set_mode((w, h), flags)
        state["is_fullscreen"] = False
        state["screen"] = screen
    return state["screen"]

# ---------------- Palettes & gradients ----------------
def _palette_code(name: str):
    name = (name or "").lower()
    table = {
        "inferno": getattr(cv, "COLORMAP_INFERNO", cv.COLORMAP_JET),
        "turbo": getattr(cv, "COLORMAP_TURBO", getattr(cv, "COLORMAP_JET", 2)),
        "plasma": getattr(cv, "COLORMAP_PLASMA", cv.COLORMAP_JET),
        "magma": getattr(cv, "COLORMAP_MAGMA", cv.COLORMAP_JET),
        "jet": getattr(cv, "COLORMAP_JET", 2),
        "hot": getattr(cv, "COLORMAP_HOT", cv.COLORMAP_JET),
    }
    return table.get(name, table["inferno"])

_PALETTE_ORDER = ["inferno","turbo","plasma","magma","jet","hot"]

# Pairs for two-color gradient mode (BGR). Name, (c1), (c2)
GRADIENT_PRESETS = [
    ("magenta-yellow", (255, 0, 255), (0, 255, 255)),
    ("cyan-magenta",   (255, 255, 0), (255, 0, 255)),
    ("blue-cyan",      (255, 0, 0),   (255, 255, 0)),
    ("red-yellow",     (0, 0, 255),   (0, 255, 255)),
    ("green-cyan",     (0, 255, 0),   (255, 255, 0)),
    ("violet-blue",    (255, 0, 127), (255, 0, 0)),
    ("white-cyan",     (255, 255, 255), (255, 255, 0)),
    ("orange-pink",    (0, 165, 255), (255, 0, 180)),
]

# ---------------- Audio Meter (optional, uses sounddevice) ----------------
class AudioMeter:
    def __init__(self, samplerate=44100, blocksize=1024, channels=1, gain=1.0, decay=0.9):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.gain = float(gain)
        self.decay = float(decay)
        self.level_smoothed = 0.0
        self._stream = None
        self._ok = False
        self._last_err = None
        self._buf = collections.deque(maxlen=20)
        self._bands = collections.deque(maxlen=8)

    def start(self, device=None):
        if sd is None:
            self._last_err = "sounddevice not available"
            return False
        try:
            dev = None
            if device is not None:
                try: dev = int(device)
                except Exception: dev = device

            def cb(indata, frames, time_info, status):
                if status: pass
                x = indata if indata.ndim == 1 else indata[:, 0]
                rms = float(np.sqrt(np.mean(np.square(x)))) * self.gain
                rms = max(0.0, min(1.0, rms * 10.0))
                self._buf.append(rms)
                try:
                    X = np.fft.rfft(x.astype(np.float32) * np.hanning(len(x)))
                    mag = np.abs(X); freqs = np.fft.rfftfreq(len(x), d=1.0/self.samplerate)
                    b = mag[(freqs >= 20) & (freqs < 200)].mean() if np.any((freqs >= 20) & (freqs < 200)) else 0.0
                    m = mag[(freqs >= 200) & (freqs < 2000)].mean() if np.any((freqs >= 200) & (freqs < 2000)) else 0.0
                    t = mag[(freqs >= 2000) & (freqs < 8000)].mean() if np.any((freqs >= 2000) & (freqs < 8000)) else 0.0
                    s = b + m + t + 1e-9; b, m, t = (b/s, m/s, t/s)
                    self._bands.append((b, m, t))
                except Exception:
                    self._bands.append((0.0, 0.0, 0.0))

            self._stream = sd.InputStream(samplerate=self.samplerate, blocksize=self.blocksize, channels=self.channels, callback=cb, device=dev)
            self._stream.start(); self._ok = True; return True
        except Exception as e:
            self._last_err = str(e); self._ok = False; return False

    def stop(self):
        try:
            if self._stream: self._stream.close()
        except Exception: pass
        self._stream = None; self._ok = False

    def level(self):
        if not self._ok or not self._buf:
            self.level_smoothed *= self.decay
            return self.level_smoothed
        avg = float(np.mean(self._buf))
        self.level_smoothed = self.decay * self.level_smoothed + (1.0 - self.decay) * avg
        return max(0.0, min(1.0, self.level_smoothed))

    def bands(self):
        if not self._ok or not self._bands:
            return (0.0, 0.0, 0.0)
        b = np.mean([x[0] for x in self._bands]); m = np.mean([x[1] for x in self._bands]); t = np.mean([x[2] for x in self._bands])
        return (float(b), float(m), float(t))

# ---------------- Segmentation (OpenCV DNN, ONNX) ----------------
@dataclass
class SegConfig:
    model_path: str
    seg_type: str = "u2net"
    input_size: int = 256
    mean: tuple = (0,0,0)
    scale: float = 1/255.0
    swapRB: bool = True
    backend: str = "default"
    target: str = "cpu"

class SegNet:
    def __init__(self, cfg: SegConfig):
        if not os.path.exists(cfg.model_path):
            raise FileNotFoundError(f"Segmentation model not found: {cfg.model_path}")
        self.cfg = cfg
        self.net = cv.dnn.readNetFromONNX(cfg.model_path)
        try:
            if cfg.backend == "coreml" and hasattr(cv.dnn, "DNN_BACKEND_COREML"):
                self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_COREML)
            else:
                self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        except Exception: pass
        try:
            if cfg.target == "opencl" and hasattr(cv.dnn, "DNN_TARGET_OPENCL"):
                self.net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
            elif cfg.target == "opencl_fp16" and hasattr(cv.dnn, "DNN_TARGET_OPENCL_FP16"):
                self.net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL_FP16)
            else:
                self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        except Exception: pass

    def _blob(self, img_bgr):
        s = int(self.cfg.input_size)
        return cv.dnn.blobFromImage(img_bgr, scalefactor=self.cfg.scale, size=(s, s), mean=self.cfg.mean, swapRB=self.cfg.swapRB, crop=False)

    def _forward_mask_resized(self, img_bgr, out_w, out_h):
        inp = self._blob(img_bgr); self.net.setInput(inp); out = self.net.forward()
        mask_small = out.squeeze()
        mmin, mmax = float(mask_small.min()), float(mask_small.max())
        mask_small = (mask_small - mmin) / (mmax - mmin + 1e-6)
        return cv.resize(mask_small, (out_w, out_h), interpolation=cv.INTER_LINEAR)

    def infer_mask(self, frame_bgr, roi=None):
        H, W = frame_bgr.shape[:2]
        if roi is None:
            return self._forward_mask_resized(frame_bgr, W, H)
        x,y,w,h = roi
        x0,y0 = max(0,x), max(0,y); x1,y1 = min(W,x+w), min(H,y+h)
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0: return np.zeros((H,W), np.float32)
        m = self._forward_mask_resized(crop, w, h)
        full = np.zeros((H,W), np.float32); full[y0:y1, x0:x1] = m[:(y1-y0), :(x1-x0)]
        return full

# ---------------- Glow state & helpers ----------------
@dataclass
class GlowState:
    show_person: bool = True
    show_outline: bool = True
    silhouette_only: bool = False   # outline only the outer silhouette
    black_bg: bool = False
    glow_size: int = 31
    glow_strength: float = 1.3
    seg_thresh: float = 0.5
    mask_median: bool = True
    bilateral: bool = False
    glow_enabled: bool = True
    color_bgr: tuple = (255, 255, 255)

    # Outline
    outline_thickness: int = 1      # 0 = hidden

    # Color/gradient
    color_cycle: bool = False
    hue_deg: float = 180.0
    hue_speed_dps: float = 240.0
    gradient_enabled: bool = False
    gradient_mode: str = "two"      # "two" or "rainbow"
    grad_color1_bgr: tuple = (255, 0, 255)
    grad_color2_bgr: tuple = (0, 255, 255)
    grad_preset_idx: int = 0
    grad_preset_name: str = "magenta-yellow"

    # Heatmap fill (person)
    heatmap_enabled: bool = False
    heatmap_mode: str = "edge"
    heatmap_palette: str = "inferno"
    heatmap_alpha: float = 0.7

    # Background heatmap/noise
    heat_bg_enabled: bool = False
    bg_palette: str = "inferno"
    bg_scale: int = 41
    bg_speed: int = 2
    scanlines: bool = False         # background heatmap scanlines (bright additive)

    # OUTER HALO
    halo_enabled: bool = True
    halo_size: int = 40
    halo_strength: float = 1.0
    halo_use_gradient: bool = True

    # Audio reaction
    audio_enabled: bool = False
    audio_mode: str = 'halo'        # 'halo','glow','alpha'
    audio_gain: float = 1.0
    audio_decay: float = 0.9
    audio_color: bool = False
    audio_hue_gain: float = 180.0
    audio_val_gain: float = 0.7
    audio_bass_only: bool = False   # use only low band as audio level

    # Audio reactivity curve (for stronger glow/halo)
    audio_react_boost: float = 2.0  # overall boost for visual response
    audio_react_gamma: float = 0.75 # <1 more sensitive to small hits
    audio_floor: float = 0.05       # ignore levels below this

    # Silhouette depth overlay
    silhouette_overlay: bool = True
    silhouette_edge_size: int = 1      # 1..6
    silhouette_strength: float = 0.35  # 0..1

    # GLOBAL scanlines overlay (dark)
    scanlines_overlay: bool = True
    scan_gap_ov: int = 2               # px between line starts (1..8)
    scan_thickness_ov: int = 1         # line thickness (px)
    scan_strength_ov: float = 0.35     # 0..1 darkness

    # Perf downscales
    glow_downscale: int = 1            # 1=no downscale, 2/3 faster, softer
    halo_downscale: int = 1            # 1=no downscale, 2/3 faster, softer

    # White-noise spark background
    white_noise: bool = False
    spark_rate: float = 160.0      # seeds per frame per 100k px
    spark_decay: float = 0.82      # 0.5..0.99 (lower = faster fade)
    spark_size: int = 7            # Gaussian blur kernel for dot size (odd)
    spark_brightness: float = 1.6  # 1..3
    spark_strobe_chance: float = 0.06  # chance per frame
    spark_strobe_mult: float = 3.0     # burst intensity multiplier

def make_glow(edge_img, glow_size=31, strength=1.3):
    glow_size = odd(glow_size)
    thick = cv.dilate(edge_img, np.ones((3, 3), np.uint8), iterations=1)
    glow = cv.GaussianBlur(thick, (glow_size, glow_size), 0)
    gmin, gmax = float(glow.min()), float(glow.max())
    glow = (glow - gmin) * (255.0 / (gmax - gmin + 1e-6))
    glow = np.clip(glow.astype(np.float32) * float(strength), 0, 255).astype(np.uint8)
    return glow

# ---------------- Threaded BodyGlow ----------------
class BodyGlowThreaded:
    def __init__(self, camera=0, cam_w=1280, cam_h=720, fps=60.0,
                 seg_model=None, seg_type="u2net", state: GlowState=None,
                 seg_size=256, seg_fps=20, roi=False, roi_pad=64,
                 dnn_backend="default", dnn_target="cpu", loop_video=False):
        if seg_model is None:
            raise RuntimeError("--seg-model is required")

        # Support int camera index OR a file path
        self.loop_video = bool(loop_video)
        if isinstance(camera, str):
            self.cap = cv.VideoCapture(camera)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video file: {camera}")
            self.is_video = True
            self.file_fps = self.cap.get(cv.CAP_PROP_FPS) or fps
        else:
            self.cap = cv.VideoCapture(int(camera))
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera. Try --camera 1")
            self.is_video = False
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, cam_w)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, cam_h)

        self.ui_interval = 1.0 / max(fps, 1e-6); self.last_ui = time.time()

        self.state = state or GlowState()
        self.seg = SegNet(SegConfig(model_path=seg_model, seg_type=seg_type, input_size=seg_size, backend=dnn_backend, target=dnn_target))
        self.seg_interval = 1.0 / max(seg_fps, 1e-6)
        self.roi_enabled = bool(roi); self.roi_pad = int(roi_pad)

        # Shared buffers
        self._lock = threading.Lock()
        self._latest_frame = None; self._mask_f = None; self._last_roi = None
        self._stop = False

        # Worker thread
        self._worker = threading.Thread(target=self._seg_loop, daemon=True); self._worker.start()

        # Background states
        self._bg_noise = None
        self._spark = None   # float32 HxW, decays each frame

        # Audio meter reference
        self.audio_meter = None

        # Preallocs for overlays & LUT
        self._scanmask_ov = None
        self._scanmask_sig_ov = None   # (H,W,gap,thick)
        self._rainbow_lut = make_rainbow_lut()  # 256x3 uint8

    def set_seg_fps(self, new_fps: float):
        new_fps = max(1.0, float(new_fps)); self.seg_interval = 1.0 / new_fps

    def _bbox_from_mask(self, mask_bin):
        ys, xs = np.where(mask_bin > 0)
        if len(xs) == 0 or len(ys) == 0: return None
        x0, x1 = int(xs.min()), int(xs.max()); y0, y1 = int(ys.min()), int(ys.max())
        return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    def _pad_roi(self, roi, W, H, pad):
        x, y, w, h = roi
        x2 = max(0, x - pad); y2 = max(0, y - pad)
        w2 = min(W - x2, max(1, w + 2 * pad)); h2 = min(H - y2, max(1, h + 2 * pad))
        return (x2, y2, w2, h2)

    def _seg_loop(self):
        next_time = time.time()
        while not self._stop:
            now = time.time()
            if now < next_time: time.sleep(max(0.0, next_time - now))
            next_time += self.seg_interval
            with self._lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
                last_roi = self._last_roi
            if frame is None: continue
            H, W = frame.shape[:2]
            roi = None
            if self.roi_enabled and last_roi is not None:
                roi = self._pad_roi(last_roi, W, H, self.roi_pad)
            try:
                mask_f = self.seg.infer_mask(frame, roi=roi)
            except Exception:
                continue
            t = clamp(self.state.seg_thresh, 0.05, 0.95)
            mb = (mask_f > t).astype(np.uint8) * 255
            bbox = self._bbox_from_mask(mb)
            with self._lock:
                self._mask_f = mask_f; self._last_roi = bbox

    # ---------- overlays ----------
    def _ensure_scanmask_overlay(self, H, W):
        sig = (H, W, self.state.scan_gap_ov, self.state.scan_thickness_ov)
        if self._scanmask_ov is not None and self._scanmask_sig_ov == sig:
            return
        gap = max(1, int(self.state.scan_gap_ov))
        thick = max(1, int(self.state.scan_thickness_ov))
        period = gap + thick
        rows = np.arange(H, dtype=np.int32)
        line_rows = (rows % period) < thick
        mask = line_rows.astype(np.float32).reshape(H, 1, 1)  # 1 where line exists
        self._scanmask_ov = np.repeat(mask, W, axis=1)        # H x W x 1
        self._scanmask_sig_ov = sig

    # ---------- colorizer ----------
    def _colorize_map(self, norm):
        st = self.state
        if st.gradient_enabled:
            if st.gradient_mode == "two":
                timg = norm[..., None]
                c1 = np.array(st.grad_color1_bgr, dtype=np.float32).reshape(1,1,3)
                c2 = np.array(st.grad_color2_bgr, dtype=np.float32).reshape(1,1,3)
                return (c1 * (1.0 - timg) + c2 * timg).astype(np.uint8)
            else:
                # FAST RAINBOW: LUT + circular shift + vectorized indexing
                base = float(st.hue_deg) % 360.0
                k = int((base / 360.0) * 256) % 256
                lut = np.roll(self._rainbow_lut, k, axis=0)  # 256x3 BGR
                idx = np.clip((norm * 255.0).astype(np.uint8), 0, 255)
                out = lut[idx]
                return np.clip(out.astype(np.float32) * norm[..., None], 0, 255).astype(np.uint8)
        else:
            tint_bgr = hsv_to_bgr(st.hue_deg, 1.0, 1.0) if st.color_cycle else st.color_bgr
            t3 = np.dstack([norm]*3).astype(np.float32)
            return (t3 * (np.array(tint_bgr, np.float32))).astype(np.uint8)

    # ---------- main step ----------
    def step(self):
        now = time.time()
        dt = now - self.last_ui
        if dt < self.ui_interval:
            time.sleep(self.ui_interval - dt); dt = self.ui_interval
        self.last_ui = time.time()

        ok, frame = self.cap.read()
        if not ok:
            if getattr(self, "is_video", False) and self.loop_video:
                try:
                    self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.cap.read()
                except Exception:
                    ok = False
            if not ok:
                return None

        src = frame
        if self.state.bilateral:
            src = cv.bilateralFilter(src, d=7, sigmaColor=50, sigmaSpace=7)

        with self._lock:
            self._latest_frame = src
            mask_f = None if self._mask_f is None else self._mask_f.copy()

        if mask_f is None:
            return np.zeros_like(src)

        t = clamp(self.state.seg_thresh, 0.05, 0.95)
        mask_bin = (mask_f > t).astype(np.uint8) * 255
        if self.state.mask_median: mask_bin = cv.medianBlur(mask_bin, 5)
        mask = (mask_bin // 255).astype(np.uint8)
        mask3 = np.dstack([mask] * 3)

        person = src * mask3
        gray = cv.cvtColor(person, cv.COLOR_BGR2GRAY)

        if self.state.silhouette_only:
            outline = np.zeros(gray.shape, dtype=np.uint8)
            contours, _ = cv.findContours(mask_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                cv.drawContours(outline, contours, -1, 255, max(1, int(self.state.outline_thickness)))
        else:
            edges = cv.Canny(gray, 80, 160)
            outline = np.zeros(gray.shape, dtype=np.uint8)
            contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            if len(contours) > 0 and self.state.show_outline:
                cv.drawContours(outline, contours, -1, 255, max(1, int(self.state.outline_thickness)))

        # advance hue if cycling
        if self.state.color_cycle:
            self.state.hue_deg = (self.state.hue_deg + self.state.hue_speed_dps * dt) % 360.0

        # ---------------- Audio-reactive modulation (BASS-ONLY option) ----------------
        audio_level = 0.0
        bass = mid = treb = 0.0
        if self.audio_meter is not None and self.state.audio_enabled:
            try:
                bass, mid, treb = self.audio_meter.bands()
            except Exception:
                bass = mid = treb = 0.0
            audio_level = float(bass) if self.state.audio_bass_only else float(self.audio_meter.level())

        # Map audio_level → “react” (stronger, gated, curved)
        def react_map(level):
            st = self.state
            floor = max(0.0, min(0.95, st.audio_floor))
            rng = 1.0 - floor + 1e-6
            lv = max(0.0, level - floor) / rng
            lv = pow(lv, max(0.1, st.audio_react_gamma))
            return st.audio_react_boost * lv  # 0..~boost

        # Temporary hue offset from bass (if audio→color enabled)
        hue_saved = float(self.state.hue_deg)
        if self.state.audio_enabled and self.state.audio_color:
            self.state.hue_deg = (self.state.hue_deg + bass * self.state.audio_hue_gain) % 360.0

        # ---------------- Inner glow (downscale optional) ----------------
        if self.state.glow_enabled:
            if self.state.audio_enabled and self.state.audio_mode == 'glow':
                glow_strength = self.state.glow_strength * (1.0 + react_map(audio_level))
            else:
                glow_strength = self.state.glow_strength

            H, W = src.shape[:2]
            ds = max(1, int(self.state.glow_downscale))
            if ds > 1:
                small_w, small_h = max(1, W//ds), max(1, H//ds)
                outline_small = cv.resize(outline, (small_w, small_h), interpolation=cv.INTER_AREA)
                glow_small = make_glow(outline_small, max(3, int(self.state.glow_size//ds)), glow_strength)
                glow_map = cv.resize(glow_small, (W, H), interpolation=cv.INTER_CUBIC)
            else:
                glow_map = make_glow(outline, self.state.glow_size, glow_strength)

            gmin, gmax = float(glow_map.min()), float(glow_map.max())
            norm_in = (glow_map.astype(np.float32) - gmin) / (gmax - gmin + 1e-6)
            glow_bgr = self._colorize_map(norm_in)
            glow_bgr = (glow_bgr * mask3).astype(np.uint8)
        else:
            glow_bgr = np.zeros_like(src)

        # ---------------- Outer halo (downscale optional) ----------------
        halo_bgr = np.zeros_like(src)
        if self.state.halo_enabled and self.state.halo_size > 0:
            if self.state.audio_enabled and self.state.audio_mode in ('halo','alpha'):
                halo_size = int(max(1, self.state.halo_size * (1.0 + react_map(audio_level))))
                halo_strength = float(self.state.halo_strength * (1.0 + 0.6 * react_map(audio_level)))
            else:
                halo_size = int(self.state.halo_size)
                halo_strength = float(self.state.halo_strength)

            H, W = src.shape[:2]
            ds = max(1, int(self.state.halo_downscale))
            if ds > 1:
                small_w, small_h = max(1, W//ds), max(1, H//ds)
                mask_u8 = (mask*255).astype(np.uint8)
                mask_small = cv.resize(mask_u8, (small_w, small_h), interpolation=cv.INTER_NEAREST)
                inv_small = (1 - (mask_small // 255)).astype(np.uint8)
                dist_small = cv.distanceTransform((inv_small*255).astype(np.uint8), cv.DIST_L2, 3)
                s = max(1.0, float(halo_size)/ds)
                outer_small = np.exp(- (dist_small / s) ** 2).astype(np.float32)
                outer_small = (outer_small * 255.0 * halo_strength).clip(0,255).astype(np.uint8)
                k = odd(max(3, int(s//2)*2 + 1))
                outer_small = cv.GaussianBlur(outer_small, (k, k), 0)
                outer = cv.resize(outer_small, (W, H), interpolation=cv.INTER_CUBIC)
            else:
                inv = (1 - mask).astype(np.uint8)
                dist = cv.distanceTransform((inv*255).astype(np.uint8), cv.DIST_L2, 3)
                s = max(1.0, float(halo_size))
                outer = np.exp(- (dist / s) ** 2).astype(np.float32)
                outer = (outer * 255.0 * halo_strength).clip(0,255).astype(np.uint8)
                k = odd(max(3, int(s//2)*2 + 1))
                outer = cv.GaussianBlur(outer, (k, k), 0)

            omin, omax = float(outer.min()), float(outer.max())
            norm_out = (outer.astype(np.float32) - omin) / (omax - omin + 1e-6)
            halo_bgr = self._colorize_map(norm_out)
            halo_bgr = (halo_bgr * (1 - mask3)).astype(np.uint8)

        # Brightness boost from treble — disabled if bass-only is on
        if self.state.audio_enabled and self.state.audio_color:
            if self.state.audio_bass_only:
                val_mult = 1.0
            else:
                val_mult = float(1.0 + max(0.0, min(1.0, treb)) * self.state.audio_val_gain)
            halo_bgr = np.clip(halo_bgr.astype(np.float32) * val_mult, 0, 255).astype(np.uint8)
            glow_bgr = np.clip(glow_bgr.astype(np.float32) * val_mult, 0, 255).astype(np.uint8)

        # ---------------- Outline overlay ----------------
        outline_bgr = np.zeros_like(src)
        if self.state.show_outline and self.state.outline_thickness > 0:
            if self.state.silhouette_only:
                cont2, _ = cv.findContours(mask_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                if len(cont2) > 0:
                    cv.drawContours(outline_bgr, cont2, -1, self.state.color_bgr, self.state.outline_thickness)
            else:
                if 'contours' in locals() and len(contours) > 0:
                    cv.drawContours(outline_bgr, contours, -1, self.state.color_bgr, self.state.outline_thickness)
            outline_bgr = (outline_bgr * mask3).astype(np.uint8)

        # ---------------- Background ----------------
        if self.state.heat_bg_enabled:
            Ht, Wt = src.shape[:2]
            if self._bg_noise is None or self._bg_noise.shape[:2] != (Ht, Wt):
                self._bg_noise = np.random.rand(Ht, Wt).astype(np.float32)
            if self.state.bg_speed != 0:
                self._bg_noise = np.roll(self._bg_noise, int(self.state.bg_speed), axis=1)
            self._bg_noise = 0.98 * self._bg_noise + 0.02 * np.random.rand(Ht, Wt).astype(np.float32)
            ksz = odd(max(3, int(self.state.bg_scale)))
            blob = cv.GaussianBlur((self._bg_noise*255).astype(np.uint8), (ksz, ksz), 0)
            bg = cv.applyColorMap(blob, _palette_code(self.state.bg_palette))
            if self.state.scanlines:
                lines = (np.arange(Ht, dtype=np.uint8) % 2) * 20
                lines_img = np.tile(lines[:, None], (1, Wt))
                lines_bgr = cv.merge([lines_img, lines_img, lines_img])
                bg = cv.add(bg, lines_bgr)
            background = (bg * (1 - mask3)).astype(np.uint8)
        else:
            background = np.zeros_like(src) if self.state.black_bg else src * (1 - mask3)

        # ---- White-noise sparks (outside the person) ----
        if self.state.white_noise:
            Ht, Wt = src.shape[:2]
            if self._spark is None or self._spark.shape[:2] != (Ht, Wt):
                self._spark = np.zeros((Ht, Wt), dtype=np.float32)

            # Decay
            decay = float(clamp(self.state.spark_decay, 0.5, 0.995))
            self._spark *= decay

            # Seed new sparks
            rate = max(0.0, float(self.state.spark_rate))
            p = rate / 100000.0
            if p > 0.0:
                seeds = (np.random.rand(Ht, Wt) < p)
                if seeds.any():
                    self._spark[seeds] = 1.0

            # Occasional burst
            if np.random.rand() < clamp(self.state.spark_strobe_chance, 0.0, 1.0):
                kcnt = max(1, int(0.0005 * Ht * Wt))
                ys = np.random.randint(0, Ht, size=kcnt)
                xs = np.random.randint(0, Wt, size=kcnt)
                self._spark[ys, xs] = np.maximum(self._spark[ys, xs], float(self.state.spark_strobe_mult))

            # Soften dots & normalize
            ksz = odd(max(1, int(self.state.spark_size)))
            spark = cv.GaussianBlur(self._spark, (ksz, ksz), 0) if ksz > 1 else self._spark
            smax = float(spark.max())
            if smax > 1e-6:
                sn = np.clip((spark / smax) * float(self.state.spark_brightness), 0.0, 1.0)
            else:
                sn = spark

            spark_u8 = (sn * 255.0).astype(np.uint8)
            spark_bgr = cv.merge([spark_u8, spark_u8, spark_u8])  # WHITE sparks
            spark_bgr = (spark_bgr * (1 - mask3)).astype(np.uint8)
            background = cv.add(background, spark_bgr)

        # ---------------- Composite core ----------------
        result = cv.add(background, halo_bgr)
        result = cv.add(result, glow_bgr)
        result = cv.add(result, outline_bgr)

        # Restore hue after audio-color temp shift
        if self.state.audio_enabled and self.state.audio_color:
            self.state.hue_deg = hue_saved

        # Person fill (with optional heatmap)
        if self.state.show_person:
            if self.state.heatmap_enabled:
                if self.state.heatmap_mode == 'interior':
                    dist_in = cv.distanceTransform((mask*255).astype(np.uint8), cv.DIST_L2, 3)
                    if dist_in.max() > 1e-6:
                        dist_in = (dist_in / (dist_in.max()+1e-6) * 255).astype(np.uint8)
                    else:
                        dist_in = (mask*255).astype(np.uint8)
                    heat = cv.applyColorMap(dist_in, _palette_code(self.state.heatmap_palette))
                else:
                    base = cv.GaussianBlur(mask_bin, (odd(31), odd(31)), 0)
                    heat = cv.applyColorMap(base, _palette_code(self.state.heatmap_palette))
                heat = (heat * mask3).astype(np.uint8)
                a = float(min(1.0, self.state.heatmap_alpha * (0.6 + 0.8*react_map(audio_level)) if (self.state.audio_enabled and self.state.audio_mode=='alpha') else self.state.heatmap_alpha))
                person_colored = cv.addWeighted(person.astype(np.uint8), 1.0 - a, heat.astype(np.uint8), a, 0)
                result = cv.add(result, person_colored)
            else:
                result = cv.addWeighted(result, 1.0, person, 1.0, 0)

        # ---------------- Silhouette depth overlay (darken border) ----------------
        if self.state.silhouette_overlay:
            edge = cv.morphologyEx(mask_bin, cv.MORPH_GRADIENT, np.ones((3,3), np.uint8))
            if self.state.silhouette_edge_size > 1:
                edge = cv.dilate(edge, np.ones((self.state.silhouette_edge_size, self.state.silhouette_edge_size), np.uint8), 1)
            edge = cv.GaussianBlur(edge, (3,3), 0)
            edge_f = (edge.astype(np.float32) / 255.0) * float(clamp(self.state.silhouette_strength, 0.0, 1.0))
            result = np.clip(result.astype(np.float32) * (1.0 - edge_f[...,None]), 0, 255).astype(np.uint8)

        # ---------------- GLOBAL scanlines overlay (dark) ----------------
        if self.state.scanlines_overlay:
            H, W = result.shape[:2]
            self._ensure_scanmask_overlay(H, W)
            strength = clamp(self.state.scan_strength_ov, 0.0, 1.0)
            result = np.clip(result.astype(np.float32) * (1.0 - strength * self._scanmask_ov), 0, 255).astype(np.uint8)

        return cv.cvtColor(result, cv.COLOR_BGR2RGB)

    def release(self):
        self._stop = True
        try: self._worker.join(1.0)
        except Exception: pass
        try: self.cap.release()
        except Exception: pass

# ---------------- HUD helper ----------------
def draw_hud(screen, lines):
    try:
        font = pygame.font.SysFont(None, 20)
        y = 6
        for line in lines:
            s = font.render(line, True, (255,255,255))
            screen.blit(s, (8, y)); y += 20
    except Exception:
        pass

# ---------------- Presets ----------------
def apply_preset(args):
    if args.preset == "popcorn":
        # Force a black stage with punchy white “sparks” and bassy reactive glow
        args.black_bg = True
        args.white_noise = True
        args.spark_rate = 220.0
        args.spark_decay = 0.80
        args.spark_size = 7
        args.spark_brightness = 2.0
        args.spark_strobe_chance = 0.08
        args.spark_strobe_mult = 3.2

        args.gradient = True
        args.gradient_mode = "two"     # two-color is cheaper & punchier
        args.color_cycle = False

        args.audio_react = True
        args.audio_bass_only = True
        args.audio_mode = "glow"
        args.audio_color = True
        args.audio_hue_gain = 240.0
        args.audio_val_gain = 0.0
        args.audio_react_boost = 3.0
        args.audio_react_gamma = 0.7
        args.audio_floor = 0.06

        # Perf tilt
        args.glow_downscale = max(1, args.glow_downscale or 2)
        args.halo_downscale = max(1, args.halo_downscale or 2)
        args.no_hud = True

# ---------------- Main app ----------------
def main():
    ap = argparse.ArgumentParser(description="LED BodyGlow (fast + halo + audio-react)")
    ap.add_argument("--mode", choices=["body_glow"], default="body_glow")
    ap.add_argument("--preset", choices=["popcorn"], default=None, help="Apply a named preset (e.g., 'popcorn').")

    # Window
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--no-hud", action="store_true", help="Disable HUD overlay for a tiny perf win")

    # Input source
    ap.add_argument("--input", type=str, default=None, help="Path to a video file (mp4/mov). If set, overrides --camera.")
    ap.add_argument("--loop-video", action="store_true", help="Loop the video when it reaches the end.")

    # Camera/UI
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=60.0)  # UI FPS

    # Segmentation
    ap.add_argument("--seg-model", type=str, required=True)
    ap.add_argument("--seg-type", choices=["u2net","modnet"], default="u2net")
    ap.add_argument("--seg-size", type=int, default=256)
    ap.add_argument("--seg-fps", type=float, default=20.0)

    # Visual/Effect toggles
    ap.add_argument("--black-bg", action="store_true")
    ap.add_argument("--no-person", action="store_true")
    ap.add_argument("--no-outline", action="store_true")
    ap.add_argument("--outline-thickness", type=int, default=1, help="Outline thickness in px (0-10, 0=hidden)")
    ap.add_argument("--glow-size", type=int, default=31)
    ap.add_argument("--strength", type=float, default=1.3)
    ap.add_argument("--color", type=str, default="255,255,255")  # R,G,B
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--median-mask", action="store_true")
    ap.add_argument("--bilateral", action="store_true")
    ap.add_argument("--silhouette", action="store_true")

    # Color & gradients
    ap.add_argument("--color-cycle", action="store_true", help="Start with hue cycling on")
    ap.add_argument("--hue-speed", type=float, default=240.0, help="Hue speed in degrees/sec when cycling")
    ap.add_argument("--gradient", action="store_true", help="Start with gradient mode on")
    ap.add_argument("--gradient-mode", choices=["two","rainbow"], default="two")

    # Heatmap fill (person)
    ap.add_argument("--heatmap", action="store_true")
    ap.add_argument("--heatmap-mode", choices=["edge","interior"], default="edge")
    ap.add_argument("--heatmap-palette", choices=_PALETTE_ORDER, default="inferno")
    ap.add_argument("--heatmap-alpha", type=float, default=0.7)

    # Background heatmap/noise
    ap.add_argument("--heat-bg", action="store_true")
    ap.add_argument("--bg-scale", type=int, default=41)
    ap.add_argument("--bg-speed", type=int, default=2)
    ap.add_argument("--bg-palette", choices=_PALETTE_ORDER, default="inferno")
    ap.add_argument("--scanlines", action="store_true")  # background scanlines

    # OUTER HALO controls
    ap.add_argument("--no-halo", action="store_true", help="Disable outer halo extension")
    ap.add_argument("--halo-size", type=int, default=40, help="Outer halo falloff size (px)")
    ap.add_argument("--halo-strength", type=float, default=1.0, help="Outer halo intensity multiplier")

    # Audio reactive options
    ap.add_argument("--audio-react", action="store_true", help="Enable audio reactive visuals (requires sounddevice)")
    ap.add_argument("--audio-mode", choices=["halo","glow","alpha"], default="halo", help="What to modulate with audio")
    ap.add_argument("--audio-gain", type=float, default=1.0, help="Input gain multiplier")
    ap.add_argument("--audio-decay", type=float, default=0.9, help="Smoothing factor 0..1 (higher = smoother)")
    ap.add_argument("--audio-color", action="store_true", help="Drive colors with 3-band audio (bass→hue, treble→brightness)")
    ap.add_argument("--audio-hue-gain", type=float, default=180.0, help="Degrees of hue shift at max bass")
    ap.add_argument("--audio-val-gain", type=float, default=0.7, help="Extra brightness at max highs")
    ap.add_argument("--audio-device", type=str, default=None, help="Audio input device (index or name, e.g. 'BlackHole 2ch')")
    ap.add_argument("--audio-bass-only", action="store_true", help="Use only low-band (bass) as the audio level driver.")
    ap.add_argument("--audio-react-boost", type=float, default=2.0, help="Overall visual boost from audio level.")
    ap.add_argument("--audio-react-gamma", type=float, default=0.75, help="Response curve exponent (<1 more sensitive).")
    ap.add_argument("--audio-floor", type=float, default=0.05, help="Noise floor (0..1) subtracted before mapping.")

    # DNN
    ap.add_argument("--roi", action="store_true", help="Infer on a padded ROI around last mask")
    ap.add_argument("--roi-pad", type=int, default=64, help="Padding in px for --roi")
    ap.add_argument("--dnn-backend", choices=["default","opencl","coreml"], default="default")
    ap.add_argument("--dnn-target", choices=["cpu","opencl","opencl_fp16"], default="cpu")

    # Perf downscales
    ap.add_argument("--glow-downscale", type=int, default=1, help="Compute inner glow at 1/N resolution (1=no downscale).")
    ap.add_argument("--halo-downscale", type=int, default=1, help="Compute outer halo at 1/N resolution (1=no downscale).")

    # White-noise spark bg
    ap.add_argument("--white-noise", action="store_true", help="Enable popping white-noise sparks on background.")
    ap.add_argument("--spark-rate", type=float, default=160.0, help="Seeds per frame per 100k pixels.")
    ap.add_argument("--spark-decay", type=float, default=0.82, help="Per-frame decay (0.5..0.99).")
    ap.add_argument("--spark-size", type=int, default=7, help="Blur kernel (odd) for spark size.")
    ap.add_argument("--spark-brightness", type=float, default=1.6, help="Brightness multiplier (1..3).")
    ap.add_argument("--spark-strobe-chance", type=float, default=0.06, help="Chance of burst per frame (0..1).")
    ap.add_argument("--spark-strobe-mult", type=float, default=3.0, help="Burst intensity multiplier.")

    args = ap.parse_args()

    # Apply preset (strong override)
    if args.preset:
        apply_preset(args)

    pygame.init()
    flags = pygame.SCALED | pygame.RESIZABLE
    if args.fullscreen: flags |= pygame.FULLSCREEN
    screen = pygame.display.set_mode((args.width, args.height), flags)
    pygame.display.set_caption("LED BodyGlow (fast + halo + audio-react)")

    app_state = {"is_fullscreen": bool(args.fullscreen),
                 "windowed_size": (args.width, args.height),
                 "screen": screen}
    if args.fullscreen:
        screen = _toggle_fullscreen_safe(app_state)

    clock = pygame.time.Clock()
    paused = False

    # Audio meter (optional)
    audio_meter = None
    if args.audio_react:
        audio_meter = AudioMeter(gain=args.audio_gain, decay=args.audio_decay)
        ok = audio_meter.start(device=args.audio_device)
        if not ok:
            print(f"[audio] disabled: {audio_meter._last_err}")
            audio_meter = None
        else:
            # Print resolved device name
            try:
                dev = args.audio_device
                if dev is None:
                    idx = sd.default.device[0]  # default input index
                    info = sd.query_devices(idx)
                    print(f"[audio] using input: {info['name']} (default index {idx})")
                else:
                    info = sd.query_devices(dev)  # works with index or name
                    label = dev if isinstance(dev, str) else f"index {dev}"
                    print(f"[audio] using input: {info['name']} ({label})")
            except Exception as e:
                print(f"[audio] device info unavailable: {e}")

    glow_state = GlowState(
        show_person=not args.no_person,
        show_outline=not args.no_outline,
        silhouette_only=bool(args.silhouette),
        black_bg=args.black_bg,
        glow_size=args.glow_size,
        glow_strength=args.strength,
        seg_thresh=args.threshold,
        mask_median=args.median_mask,
        bilateral=args.bilateral,
        color_bgr=bgr_from_rgb(parse_rgb(args.color)),
        outline_thickness=max(0, min(10, int(args.outline_thickness))),

        color_cycle=bool(args.color_cycle),
        hue_speed_dps=float(args.hue_speed),
        gradient_enabled=bool(args.gradient),
        gradient_mode=args.gradient_mode,

        heatmap_enabled=bool(args.heatmap),
        heatmap_mode=args.heatmap_mode,
        heatmap_palette=args.heatmap_palette,
        heatmap_alpha=float(max(0.0, min(1.0, args.heatmap_alpha))),

        heat_bg_enabled=bool(args.heat_bg),
        bg_scale=int(max(3, args.bg_scale)),
        bg_speed=int(args.bg_speed),
        bg_palette=args.bg_palette,
        scanlines=bool(args.scanlines),

        halo_enabled=not bool(args.no_halo),
        halo_size=max(1, int(args.halo_size)),
        halo_strength=float(max(0.0, args.halo_strength)),

        audio_enabled=bool(args.audio_react),
        audio_mode=args.audio_mode,
        audio_gain=float(args.audio_gain),
        audio_decay=float(args.audio_decay),
        audio_color=bool(args.audio_color),
        audio_hue_gain=float(args.audio_hue_gain),
        audio_val_gain=float(args.audio_val_gain),
        audio_bass_only=bool(args.audio_bass_only),
        audio_react_boost=float(args.audio_react_boost),
        audio_react_gamma=float(args.audio_react_gamma),
        audio_floor=float(args.audio_floor),

        silhouette_overlay=True,
        silhouette_edge_size=1,
        silhouette_strength=0.35,

        scanlines_overlay=True,
        scan_gap_ov=2,
        scan_thickness_ov=1,
        scan_strength_ov=0.35,

        glow_downscale=max(1, int(args.glow_downscale)),
        halo_downscale=max(1, int(args.halo_downscale)),

        white_noise=bool(args.white_noise),
        spark_rate=float(args.spark_rate),
        spark_decay=float(args.spark_decay),
        spark_size=int(args.spark_size),
        spark_brightness=float(args.spark_brightness),
        spark_strobe_chance=float(args.spark_strobe_chance),
        spark_strobe_mult=float(args.spark_strobe_mult),
    )

    # Initialize two-color gradient from preset if enabled
    if glow_state.gradient_enabled and 0 <= glow_state.grad_preset_idx < len(GRADIENT_PRESETS):
        name, c1, c2 = GRADIENT_PRESETS[glow_state.grad_preset_idx]
        glow_state.grad_preset_name = name
        glow_state.grad_color1_bgr = c1
        glow_state.grad_color2_bgr = c2

    # Choose capture: file or camera
    cam_param = args.input if args.input else args.camera

    try:
        glow = BodyGlowThreaded(
            camera=cam_param, cam_w=args.cam_width, cam_h=args.cam_height, fps=args.fps,
            seg_model=args.seg_model, seg_type=args.seg_type, state=glow_state,
            seg_size=args.seg_size, seg_fps=args.seg_fps,
            roi=args.roi, roi_pad=args.roi_pad,
            dnn_backend=args.dnn_backend, dnn_target=args.dnn_target,
            loop_video=bool(args.loop_video)
        )
        glow.audio_meter = audio_meter
    except Exception as e:
        print(f"body_glow init failed: {e}")
        return

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                mods = pygame.key.get_mods()
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif (k == pygame.K_f and (mods & (pygame.KMOD_CTRL | pygame.KMOD_META))) or (k == pygame.K_RETURN and (mods & pygame.KMOD_SHIFT)):
                    screen = _toggle_fullscreen_safe(app_state)

                # Global overlays
                elif k == pygame.K_BACKSLASH:
                    glow_state.scanlines_overlay = not glow_state.scanlines_overlay
                elif k == pygame.K_F9:  glow_state.scan_gap_ov = max(1, glow_state.scan_gap_ov - 1); glow._scanmask_ov = None
                elif k == pygame.K_F10: glow_state.scan_gap_ov = min(8, glow_state.scan_gap_ov + 1); glow._scanmask_ov = None
                elif k == pygame.K_F11: glow_state.scan_strength_ov = round(max(0.0, glow_state.scan_strength_ov - 0.05), 2)
                elif k == pygame.K_F12: glow_state.scan_strength_ov = round(min(1.0, glow_state.scan_strength_ov + 0.05), 2)
                elif k == pygame.K_s and (mods & pygame.KMOD_SHIFT):
                    glow_state.silhouette_overlay = not glow_state.silhouette_overlay

                # Body-glow hotkeys
                elif k == pygame.K_g: glow_state.glow_enabled = not glow_state.glow_enabled
                elif k == pygame.K_p: glow_state.show_person = not glow_state.show_person
                elif k == pygame.K_b and (mods & pygame.KMOD_SHIFT):
                    glow_state.audio_bass_only = not glow_state.audio_bass_only
                elif k == pygame.K_b:
                    glow_state.black_bg = not glow_state.black_bg
                elif k == pygame.K_o: glow_state.show_outline = not glow_state.show_outline
                elif k == pygame.K_s: glow_state.silhouette_only = not glow_state.silhouette_only
                elif k == pygame.K_LEFTBRACKET:
                    glow_state.glow_size = max(3, glow_state.glow_size - 2)
                    if glow_state.glow_size % 2 == 0: glow_state.glow_size -= 1
                elif k == pygame.K_RIGHTBRACKET:
                    glow_state.glow_size += 2
                    if glow_state.glow_size % 2 == 0: glow_state.glow_size += 1
                elif k == pygame.K_MINUS: glow_state.glow_strength = round(max(0.1, glow_state.glow_strength - 0.1), 2)
                elif k == pygame.K_EQUALS: glow_state.glow_strength = round(min(5.0, glow_state.glow_strength + 0.1), 2)
                elif k == pygame.K_d: glow_state.halo_size = max(1, glow_state.halo_size - 2)
                elif k == pygame.K_f and not (mods & (pygame.KMOD_CTRL | pygame.KMOD_META)): glow_state.halo_size = glow_state.halo_size + 2
                elif k == pygame.K_COMMA: glow_state.seg_thresh = round(max(0.05, glow_state.seg_thresh - 0.02), 2)
                elif k == pygame.K_PERIOD: glow_state.seg_thresh = round(min(0.95, glow_state.seg_thresh + 0.02), 2)
                elif k == pygame.K_m: glow_state.mask_median = not glow_state.mask_median
                elif k == pygame.K_v: glow_state.bilateral = not glow_state.bilateral
                elif k == pygame.K_QUOTE:
                    current = 1.0 / max(1e-6, glow.seg_interval); glow.set_seg_fps(current + 2.0)
                elif k == pygame.K_SEMICOLON:
                    current = 1.0 / max(1e-6, glow.seg_interval); glow.set_seg_fps(max(1.0, current - 2.0))
                elif k == pygame.K_3: glow_state.outline_thickness = max(0, glow_state.outline_thickness - 1)
                elif k == pygame.K_4: glow_state.outline_thickness = min(10, glow_state.outline_thickness + 1)
                elif k == pygame.K_c: glow_state.color_cycle = not glow_state.color_cycle
                elif k == pygame.K_h: glow_state.gradient_enabled = not glow_state.gradient_enabled
                elif k == pygame.K_j: glow_state.gradient_mode = "two" if glow_state.gradient_mode == "rainbow" else "rainbow"
                elif k == pygame.K_l: glow_state.gradient_mode = "rainbow" if glow_state.gradient_mode == "two" else "two"
                elif k == pygame.K_1:
                    glow_state.grad_preset_idx = (glow_state.grad_preset_idx - 1) % len(GRADIENT_PRESETS)
                    name, c1, c2 = GRADIENT_PRESETS[glow_state.grad_preset_idx]
                    glow_state.grad_preset_name = name
                    glow_state.grad_color1_bgr, glow_state.grad_color2_bgr = c1, c2
                elif k == pygame.K_2:
                    glow_state.grad_preset_idx = (glow_state.grad_preset_idx + 1) % len(GRADIENT_PRESETS)
                    name, c1, c2 = GRADIENT_PRESETS[glow_state.grad_preset_idx]
                    glow_state.grad_preset_name = name
                    glow_state.grad_color1_bgr, glow_state.grad_color2_bgr = c1, c2
                elif k == pygame.K_u: glow_state.hue_speed_dps = max(0.0, glow_state.hue_speed_dps - 60.0)
                elif k == pygame.K_i: glow_state.hue_speed_dps = min(1440.0, glow_state.hue_speed_dps + 60.0)
                elif k == pygame.K_x: glow_state.grad_color1_bgr, glow_state.grad_color2_bgr = glow_state.grad_color2_bgr, glow_state.grad_color1_bgr
                elif k == pygame.K_k: glow_state.heatmap_enabled = not glow_state.heatmap_enabled
                elif k == pygame.K_r: glow_state.heatmap_mode = 'interior' if glow_state.heatmap_mode == 'edge' else 'edge'
                elif k == pygame.K_e:
                    try:
                        idx = _PALETTE_ORDER.index(glow_state.heatmap_palette)
                        glow_state.heatmap_palette = _PALETTE_ORDER[(idx+1)%len(_PALETTE_ORDER)]; glow_state.bg_palette = glow_state.heatmap_palette
                    except Exception: glow_state.heatmap_palette = 'inferno'; glow_state.bg_palette = 'inferno'
                elif k == pygame.K_9:
                    try:
                        idx = _PALETTE_ORDER.index(glow_state.heatmap_palette)
                        glow_state.heatmap_palette = _PALETTE_ORDER[(idx-1)%len(_PALETTE_ORDER)]; glow_state.bg_palette = glow_state.heatmap_palette
                    except Exception: glow_state.heatmap_palette = 'inferno'; glow_state.bg_palette = 'inferno'
                elif k == pygame.K_0:
                    try:
                        idx = _PALETTE_ORDER.index(glow_state.heatmap_palette)
                        glow_state.heatmap_palette = _PALETTE_ORDER[(idx+1)%len(_PALETTE_ORDER)]; glow_state.bg_palette = glow_state.heatmap_palette
                    except Exception: glow_state.heatmap_palette = 'inferno'; glow_state.bg_palette = 'inferno'
                elif k == pygame.K_z: glow_state.heat_bg_enabled = not glow_state.heat_bg_enabled
                elif k == pygame.K_SLASH: glow_state.scanlines = not glow_state.scanlines
                elif k == pygame.K_n and (mods & pygame.KMOD_SHIFT):
                    if glow._spark is not None:
                        H, W = glow._spark.shape[:2]
                        kcnt = max(1, int(0.0007 * H * W))
                        ys = np.random.randint(0, H, size=kcnt)
                        xs = np.random.randint(0, W, size=kcnt)
                        glow._spark[ys, xs] = np.maximum(glow._spark[ys, xs], glow_state.spark_strobe_mult)
                elif k == pygame.K_n:
                    glow_state.white_noise = not glow_state.white_noise
                elif k == pygame.K_5: glow_state.bg_scale = max(3, glow_state.bg_scale - 2)
                elif k == pygame.K_6: glow_state.bg_scale = glow_state.bg_scale + 2
                elif k == pygame.K_7: glow_state.bg_speed = max(0, glow_state.bg_speed - 1)
                elif k == pygame.K_8: glow_state.bg_speed = glow_state.bg_speed + 1
                elif k == pygame.K_y and not (mods & pygame.KMOD_SHIFT): glow_state.audio_enabled = not glow_state.audio_enabled
                elif k == pygame.K_y and (mods & pygame.KMOD_SHIFT): glow_state.audio_color = not glow_state.audio_color
                elif k == pygame.K_t:
                    modes = ['halo','glow','alpha']
                    try:
                        i = (modes.index(glow_state.audio_mode) + 1) % len(modes)
                        glow_state.audio_mode = modes[i]
                    except Exception:
                        glow_state.audio_mode = 'halo'
                elif k == pygame.K_F1: glow_state.audio_gain = max(0.1, glow_state.audio_gain - 0.1)
                elif k == pygame.K_F2: glow_state.audio_gain = min(10.0, glow_state.audio_gain + 0.1)
                elif k == pygame.K_F3:
                    if (mods & pygame.KMOD_SHIFT): glow_state.audio_decay = max(0.50, glow_state.audio_decay - 0.05)
                    else:                           glow_state.audio_decay = min(0.99, glow_state.audio_decay + 0.05)
                elif k == pygame.K_F5: glow_state.audio_hue_gain = max(0.0, glow_state.audio_hue_gain - 30.0)
                elif k == pygame.K_F6: glow_state.audio_hue_gain = min(720.0, glow_state.audio_hue_gain + 30.0)
                elif k == pygame.K_F7: glow_state.audio_val_gain = max(0.0, glow_state.audio_val_gain - 0.05)
                elif k == pygame.K_F8: glow_state.audio_val_gain = min(3.0, glow_state.audio_val_gain + 0.05)

            elif ev.type == pygame.VIDEORESIZE:
                app_state['windowed_size'] = (ev.w, ev.h)
                screen = pygame.display.set_mode((ev.w, ev.h), pygame.SCALED | pygame.RESIZABLE)

        if paused:
            pygame.display.flip(); clock.tick(60); continue

        if audio_meter is not None:
            audio_meter.gain = glow_state.audio_gain
            audio_meter.decay = glow_state.audio_decay

        rgb = glow.step()
        if rgb is None:
            screen.fill((0, 0, 0))
        else:
            win_w, win_h = screen.get_size()
            h, w = rgb.shape[0], rgb.shape[1]
            scale = min(win_w / w, win_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
            surf = pygame.transform.smoothscale(surf, (new_w, new_h))
            screen.fill((0, 0, 0))
            screen.blit(surf, ((win_w - new_w) // 2, (win_h - new_h) // 2))

        if not args.no_hud:
            try:
                draw_hud(screen, [
                    f"SegFPS:{int(1.0/max(1e-6, glow.seg_interval))} | InnerGlow sz:{glow_state.glow_size} str:{glow_state.glow_strength:.2f} | Halo sz:{glow_state.halo_size}",
                    f"BG:{'black' if glow_state.black_bg else 'orig' if not glow_state.heat_bg_enabled else 'heatmap'} Outline:{glow_state.show_outline} Thick:{glow_state.outline_thickness} SilhouetteOnly:{glow_state.silhouette_only}",
                    f"Cycle:{glow_state.color_cycle} Grad:{glow_state.gradient_enabled}({glow_state.gradient_mode}{' '+glow_state.grad_preset_name if glow_state.gradient_enabled and glow_state.gradient_mode=='two' else ''}) HueSpd:{int(glow_state.hue_speed_dps)}",
                    f"Heatmap:{glow_state.heatmap_enabled}({glow_state.heatmap_mode},{glow_state.heatmap_palette},{glow_state.heatmap_alpha:.1f})",
                    f"Audio:{glow_state.audio_enabled}({glow_state.audio_mode},G{glow_state.audio_gain:.1f},D{glow_state.audio_decay:.2f},Color:{glow_state.audio_color},BassOnly:{glow_state.audio_bass_only})",
                    f"Overlay scanlines:{glow_state.scanlines_overlay} gap:{glow_state.scan_gap_ov} str:{glow_state.scan_strength_ov:.2f} | DS glow:{glow_state.glow_downscale} halo:{glow_state.halo_downscale}",
                    f"Sparks:{glow_state.white_noise} rate:{int(glow_state.spark_rate)} size:{glow_state.spark_size}"
                ])
            except Exception:
                pass

        pygame.display.flip()
        clock.tick(60)

    glow.release()
    pygame.quit()

if __name__ == "__main__":
    main()
