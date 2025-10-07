# led_board_emulator_now_playing.py
# LED dot-matrix emulator with Video mode + robust "Now Playing" file watcher
import math, time, sys, os
from pathlib import Path
from dataclasses import dataclass
import pygame as pg
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

# ------------- CONFIG -------------
WINDOW_START = (1280, 720)
FPS = 60

MATRIX_W = 96
MATRIX_H = 54

LED_SIZE = 10
LED_GAP  = 2

ROUND_LEDS = True
GLOW = True
SCANLINE = True
SHOW_HUD = True
BRIGHTNESS = 0.85
# --- White pixel camera mode ---
WHITE_MASK = False    # toggle with 'W'
WHITE_THRESH = 0.35   # 0..1 luminance threshold


# >>> EDIT THIS to your real file (absolute path recommended) <<<
NOW_PLAYING_FILE = "/Users/samkulka/Desktop/Club Fast/Visuals/now_playing.txt"
# or "./now_playing.txt" if kept in same folder as this script

VIDEO_FILE = "/Users/samkulka/Desktop/Club Fast/Videos/RECAPVIDEO.mp4"  # optional: set to a video path and press 'v' to toggle

THEMES = [
    ((6,8,12), (255,200,40), (255,80,30),  (235,235,240)),
    ((4,6,10), (40,220,255), (255,80,180), (230,240,255)),
    ((8,6,6),  (255,160,20), (220,40,40),  (245,234,210)),
    ((5,5,7),  (160,220,120), (120,160,255),(230,238,255)),
]
THEME_IDX = 0
# ----------------------------------

def lerp(a,b,t): return a + (b-a)*t
def lerp_color(c1,c2,t): return (int(lerp(c1[0],c2[0],t)), int(lerp(c1[1],c2[1],t)), int(lerp(c1[2],c2[2],t)))

def make_text_surf(text, size, color=(255,255,255), font_name="PressStart2P, PixelMplus10, Verdana, Arial, sans"):
    try: font = pg.font.SysFont(font_name, size, bold=True)
    except Exception: font = pg.font.SysFont(None, size, bold=True)
    return font.render(text, True, color).convert_alpha()

@dataclass
class LedGeom: w:int; h:int; led_px:int; gap_px:int

class LedMatrix:
    def __init__(self, mw, mh):
        self.mw, self.mh = mw, mh
        self.canvas = pg.Surface((mw, mh)).convert()
    def compute_geom(self, window_size):
        W,H = window_size
        cols, rows = self.mw, self.mh
        cell = min(W/cols, H/rows)
        scale = cell / (LED_SIZE + LED_GAP)
        return LedGeom(W,H, max(2,int(LED_SIZE*scale)), max(1,int(LED_GAP*scale)))
    def blit_to_leds(self, screen, theme, geom, brightness=1.0, round_leds=True,
                 glow=True, scanline=False, t=0.0):
        bg, primary, accent, *rest = theme if len(theme) >= 3 else (theme + [(235,235,235)])
        # Read canvas -> numpy
        arr = pg.surfarray.pixels3d(self.canvas)             # (w,h,3)
        img = np.transpose(arr, (1,0,2)).astype(np.float32)  # (h,w,3)
        del arr

        # Luminance 0..1
        luma = (0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]) / 255.0
        luma = np.clip(luma, 0, 1)

        global WHITE_MASK, WHITE_THRESH

        if WHITE_MASK:
            # Hard/soft “white pixels only”:
            # Use soft step so bright edges still show: ramp from (th-0.1) to (th+0.1)
            th = float(WHITE_THRESH)
            soft = np.clip((luma - (th - 0.10)) / 0.20, 0.0, 1.0)  # smooth ramp around threshold
            # Pure white where present, fully black elsewhere
            led_rgb = soft[..., None] * brightness  # (h,w,1)
            # expand to RGB (white)
            led_rgb = np.repeat(led_rgb, 3, axis=2)
        else:
            # Original colorized pipeline (primary/accent tint)
            luma = np.power(luma, 1.4)
            hue_mix = np.clip((img[...,1] - img[...,2] + 128)/255.0, 0, 1)
            col_primary = np.array(primary, dtype=np.float32)/255.0
            col_accent  = np.array(accent,  dtype=np.float32)/255.0
            base_color = col_primary*(1.0-hue_mix)[...,None] + col_accent*hue_mix[...,None]
            led_rgb = base_color * luma[...,None] * brightness

        # Optional scanline dim
        if scanline:
            active_row = int((t*120.0) % self.mh)
            mask = np.ones((self.mh,), dtype=np.float32)*(0.35 if not WHITE_MASK else 0.25)
            mask[active_row] = 1.0
            led_rgb *= mask[:,None,None]

        # --- draw LEDs (unchanged) ---
        screen.fill(bg)
        cols, rows = self.mw, self.mh
        cell_w = geom.led_px + geom.gap_px
        cell_h = geom.led_px + geom.gap_px
        start_x = (geom.w - cols*cell_w + geom.gap_px)//2
        start_y = (geom.h - rows*cell_h + geom.gap_px)//2
        radius = geom.led_px//2

        glow_surf = None
        if glow:
            glow_surf = pg.Surface((geom.led_px*2, geom.led_px*2), pg.SRCALPHA)
            pg.draw.circle(glow_surf, (255,255,255,40), (geom.led_px, geom.led_px), geom.led_px)

        for j in range(rows):
            y = start_y + j*cell_h
            row = led_rgb[j]
            for i in range(cols):
                r,g,b = row[i]
                if r<=0.004 and g<=0.004 and b<=0.004: continue
                x = start_x + i*cell_w
                color = (int(255*r), int(255*g), int(255*b))
                if round_leds:
                    if glow_surf: screen.blit(glow_surf, (x-radius, y-radius), special_flags=pg.BLEND_ADD)
                    pg.draw.circle(screen, color, (x, y), radius)
                else:
                    pg.draw.rect(screen, color, (x-radius, y-radius, geom.led_px, geom.led_px))


# ---------- DEMOS ----------
class ScoreboardDemo:
    def __init__(self, m, theme): self.m, self.theme = m, theme
    def draw(self, dt):
        canvas = self.m.canvas; mw, mh = canvas.get_size()
        canvas.fill((0,0,12))
        header_h = int(mh*0.16); header = pg.Surface((mw, header_h)); header.fill((120,0,20)); canvas.blit(header,(0,0))
        title_size = max(8, header_h-2)
        canvas.blit(make_text_surf("Cardinals", title_size, (250,250,250)), (1,-2))
        r = make_text_surf("3", title_size, (250,250,250)); canvas.blit(r, (mw-r.get_width()-2, -2))
        mid_h = int(mh*0.12); mid=pg.Surface((mw,mid_h)); mid.fill((5,5,7)); canvas.blit(mid, (0,header_h+1))
        canvas.blit(make_text_surf("Pirates", title_size, (255,200,40)), (1, header_h-2))
        cx,cy=int(mw*0.42), int(mh*0.64); size=int(mh*0.16)
        self.rot_square(canvas, cx-int(size*0.7), cy-int(size*0.6), size, (255,200,40), True)
        self.rot_square(canvas, cx, cy-int(size*0.9), size, (255,200,40), True)
        self.rot_square(canvas, cx+int(size*0.7), cy-int(size*0.6), size, (255,200,40), False)
        canvas.blit(make_text_surf("0-0", int(mh*0.10), (255,200,40)), (int(mw*0.72), int(mh*0.70)))
        canvas.blit(make_text_surf("□□□", int(mh*0.10), (255,200,40)), (int(mw*0.72), int(mh*0.82)))
        canvas.blit(make_text_surf("↟ 4", int(mh*0.12), (255,200,40)), (int(mw*0.80), int(mh*0.48)))
    def rot_square(self, surf, cx, cy, size, col, outline):
        s = pg.Surface((size,size), pg.SRCALPHA)
        if outline: pg.draw.rect(s, col, s.get_rect(), width=max(1,size//7))
        else: s.fill(col)
        surf.blit(pg.transform.rotate(s,45), s.get_rect(center=(cx,cy)))

class TickerDemo:
    def __init__(self, m, theme):
        self.m, self.theme = m, theme; self.x=0
        self.text = "  NOW PLAYING: SARAH PEDERZANI                           CLUB FAST                         "
        self.speed=18
    def draw(self, dt):
        bg, primary, accent, whiteish = self.theme
        canvas = self.m.canvas
        mw, mh = canvas.get_size()
        canvas.fill((0,0,10))

        band_h = int(mh*0.22)
        y0 = (mh - band_h) // 2   # <— center position

        band = pg.Surface((mw, band_h))
        band.fill((20,20,28))
        canvas.blit(band, (0, y0))

        surf = make_text_surf(self.text, band_h, primary)
        self.x -= dt * self.speed
        px = int(self.x * (band_h//2))
        canvas.blit(surf, (px % surf.get_width() - surf.get_width(), y0 - 2))
        canvas.blit(surf, ((px % surf.get_width()), y0 - 2))


# ---- Now Playing Watcher ----
class NowPlayingWatcher:
    def __init__(self, path_str):
        # resolve absolute path relative to script file if needed
        base = Path(path_str).expanduser()
        if not base.is_absolute():
            base = (Path(__file__).resolve().parent / base).resolve()
        self.path = base
        self.last_mtime = None
        self.value = "—"
        self.error = ""
        self.poll()  # initial read
        

    def _read_file(self):
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        return s
            return "—"  # empty file
        except FileNotFoundError:
            return "— (file not found)"
        except PermissionError:
            return "— (permission denied)"
        except Exception as e:
            return f"— (error: {e})"

    def poll(self):
        try:
            mt = self.path.stat().st_mtime
        except Exception:
            mt = None
        if mt != self.last_mtime:
            self.last_mtime = mt
            new_val = self._read_file()
            if new_val != self.value:
                self.value = new_val
                print(f"[NowPlaying] Updated: {self.value}")

# -------- Video Matrix --------
class VideoDemo:
    """
    Camera/video mapped to LED matrix + toggle header (NOW PLAYING ↔ artist).
    - Pass a NowPlayingWatcher as `watcher` (optional)
    - Toggle camera/file in your main (unchanged)
    - Mirror with 'm' (unchanged)
    Behavior:
      * Header shows 'NOW PLAYING' for 2s, then artist for 4s, loops.
      * Artist tries to shrink-to-fit; if still too long, it scrolls (no ellipsis).
      * Text uses a thin outline to pop on bright footage.
    """
    def __init__(self, matrix, theme, watcher=None, source_index=0, mirror=True):
        self.m = matrix
        self.theme = theme
        self.watch = watcher
        self.mirror = mirror
        self.source_index = source_index
        self.mode = "cam"     # "cam" or "file"
        self.cap = None

        # Now Playing state
        self.artist = (self.watch.value if self.watch else "—")
        self.scroll_px = 0.0  # marquee offset

        self._open_capture()

    # ---------- capture/source control ----------
    def _open_capture(self):
        if cv2 is None:
            print("[VideoDemo] OpenCV not available; video mode will show placeholder.")
            self.cap = None
            return
        if self.mode == "cam":
            self.cap = cv2.VideoCapture(self.source_index)
        else:
            self.cap = cv2.VideoCapture(VIDEO_FILE) if (VIDEO_FILE and os.path.exists(VIDEO_FILE)) else None
        if self.cap and not self.cap.isOpened():
            print("[VideoDemo] Could not open capture.")
            try: self.cap.release()
            except Exception: pass
            self.cap = None

    def cycle_source(self):
        if self.mode == "cam":
            self.source_index = (self.source_index + 1) % 2
            self._open_capture()
        else:
            self.mode = "cam"
            self._open_capture()

    def toggle_file(self):
        if VIDEO_FILE and os.path.exists(VIDEO_FILE):
            self.mode = "file" if self.mode != "file" else "cam"
            self._open_capture()
        else:
            print("[VideoDemo] No VIDEO_FILE configured/found; staying on camera.")

    def release(self):
        try:
            if self.cap: self.cap.release()
        except Exception:
            pass

    # ---------- now playing ----------
    def _update_now_playing(self):
        """Prefer watcher; otherwise use read_now_playing() if present."""
        try:
            if self.watch:
                self.watch.poll()
                new_val = self.watch.value
            else:
                new_val = read_now_playing()
        except NameError:
            new_val = None

        if new_val:
            new_val = new_val.strip() or "—"
            if new_val != self.artist:
                self.artist = new_val
                self.scroll_px = 0.0
                print(f"[VideoDemo] Now Playing: {self.artist}")

    # ---------- text helpers ----------
    @staticmethod
    def _make_font(px):
        try:
            return pg.font.SysFont("PressStart2P, PixelMplus10, Verdana, Arial, sans", px, bold=True)
        except Exception:
            return pg.font.SysFont(None, px, bold=True)

    def _render_shrink_to_fit(self, text, max_w, max_h, color, max_px, min_px=10):
        lo, hi = min_px, max_px
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            f = self._make_font(mid)
            s = f.render(text, True, color).convert_alpha()
            if s.get_width() <= max_w and s.get_height() <= max_h:
                best = s
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    @staticmethod
    def _blit_outline(target, surf, xy, outline_color=(0,0,0), thickness=2):
        x, y = xy
        for dx in (-thickness, 0, thickness):
            for dy in (-thickness, 0, thickness):
                if dx == 0 and dy == 0: 
                    continue
                shadow = surf.copy()
                shadow.fill((*outline_color, 255), special_flags=pg.BLEND_RGBA_MIN)
                target.blit(shadow, (x+dx, y+dy))
        target.blit(surf, (x, y))

    # ---------- placeholder ----------
    def _draw_placeholder(self, canvas):
        mw, mh = canvas.get_size()
        grid = pg.Surface((mw, mh))
        grid.fill((8,8,12))
        for y in range(0, mh, 4):
            pg.draw.line(grid, (10,10,16), (0,y), (mw,y))
        canvas.blit(grid, (0,0))
        f = self._make_font(int(mh*0.18))
        txt = f.render("No Camera / Video", True, (220,60,60)).convert_alpha()
        rect = txt.get_rect(center=(mw//2, mh//2))
        canvas.blit(txt, rect)

    # ---------- drawing ----------
    def draw(self, dt):
        self._update_now_playing()

        canvas = self.m.canvas
        mw, mh = canvas.get_size()
        canvas.fill((0,0,8))

        # --- video frame ---
        if self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                if self.mirror: frame = cv2.flip(frame, 1)
                small = cv2.resize(frame, (mw, mh), interpolation=cv2.INTER_AREA)
                small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                arr = np.transpose(small, (1,0,2))
                surf = pg.surfarray.make_surface(arr)
                canvas.blit(surf, (0,0))
            else:
                if self.mode == "file" and self.cap:  # loop files
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._draw_placeholder(canvas)
        else:
            self._draw_placeholder(canvas)

        # --- Toggle header: NOW PLAYING ↔ artist ---
        header_h = int(mh * 0.26)
        if len(self.theme) >= 4:
            bg, primary, accent, _whiteish = self.theme
        else:
            bg, primary, accent = self.theme

        band = pg.Surface((mw, header_h))
        band.fill((16,16,24))
        canvas.blit(band, (0, 0))

        cycle = 6.0
        phase = (time.time() % cycle)
        show_prefix = phase < 2.0  # 2s prefix, 4s artist

        font_px = int(header_h * 0.65)
        f = self._make_font(font_px)

        if show_prefix:
            surf = f.render("NOW PLAYING", True, (235,235,245)).convert_alpha()
            rect = surf.get_rect(center=(mw//2, header_h//2 - 2))
            self._blit_outline(canvas, surf, rect.topleft, outline_color=(0,0,0), thickness=2)
        else:
            artist_text = self.artist if self.artist else "—"
            # Try to fit exactly
            fit = self._render_shrink_to_fit(artist_text, mw-20, header_h-6, primary,
                                             max_px=int(header_h*0.70), min_px=10)
            if fit is not None:
                rect = fit.get_rect(center=(mw//2, header_h//2 - 2))
                self._blit_outline(canvas, fit, rect.topleft, outline_color=(0,0,0), thickness=2)
                self.scroll_px = 0.0
            else:
                # Marquee fallback (no ellipsis)
                px = max(10, int(header_h*0.62))
                f2 = self._make_font(px)
                base = f2.render(artist_text + "   ", True, primary).convert_alpha()
                self.scroll_px = (self.scroll_px + dt * (px*1.2)) % base.get_width()
                xscroll = 10 - int(self.scroll_px)
                yscroll = header_h//2 - base.get_height()//2
                self._blit_outline(canvas, base, (xscroll, yscroll), outline_color=(0,0,0), thickness=2)
                if xscroll + base.get_width() < mw-10:
                    self._blit_outline(canvas, base, (xscroll + base.get_width(), yscroll),
                                       outline_color=(0,0,0), thickness=2)






def main():
    pg.init()
    pg.display.set_caption("LED Matrix — Video + Now Playing")
    screen = pg.display.set_mode(WINDOW_START, pg.RESIZABLE | pg.DOUBLEBUF)
    clock = pg.time.Clock()

    matrix = LedMatrix(MATRIX_W, MATRIX_H)
    theme_idx = THEME_IDX; theme = THEMES[theme_idx]

    watcher = NowPlayingWatcher(NOW_PLAYING_FILE)

    mode = 3  # start in video mode for testing
    demo1 = ScoreboardDemo(matrix, theme)
    demo2 = TickerDemo(matrix, theme)
    demo3 = VideoDemo(matrix, theme, watcher, source_index=0, mirror=True)

    global ROUND_LEDS, GLOW, SCANLINE, SHOW_HUD, LED_SIZE, LED_GAP, BRIGHTNESS

    running=True; t0=time.time()
    while running:
        dt = clock.tick(FPS)/1000.0; t=time.time()-t0
        for e in pg.event.get():
            if e.type==pg.QUIT: running=False
            elif e.type==pg.VIDEORESIZE:
                screen = pg.display.set_mode((e.w, e.h), pg.RESIZABLE | pg.DOUBLEBUF)
            elif e.type==pg.KEYDOWN:
                if e.key in (pg.K_ESCAPE, pg.K_q): running=False
                elif e.key==pg.K_f: pg.display.toggle_fullscreen()
                elif e.key==pg.K_h: SHOW_HUD = not SHOW_HUD
                elif e.key==pg.K_1: mode=1
                elif e.key==pg.K_2: mode=2
                elif e.key==pg.K_3: mode=3
                elif e.key==pg.K_g: GLOW = not GLOW
                elif e.key==pg.K_s: SCANLINE = not SCANLINE
                elif e.key==pg.K_r: ROUND_LEDS = not ROUND_LEDS
                elif e.key==pg.K_LEFTBRACKET: LED_SIZE=max(4,LED_SIZE-1)
                elif e.key==pg.K_RIGHTBRACKET: LED_SIZE=min(32,LED_SIZE+1)
                elif e.key==pg.K_MINUS: LED_GAP=max(1,LED_GAP-1)
                elif e.key==pg.K_EQUALS: LED_GAP=min(12,LED_GAP+1)
                elif e.key==pg.K_COMMA: BRIGHTNESS=max(0.2, round(BRIGHTNESS-0.05,2))
                elif e.key==pg.K_PERIOD: BRIGHTNESS=min(1.0, round(BRIGHTNESS+0.05,2))
                elif e.key==pg.K_c:
                    theme_idx=(theme_idx+1)%len(THEMES); theme=THEMES[theme_idx]
                    demo1.theme=theme; demo2.theme=theme; demo3.theme=theme
                elif e.key==pg.K_m: demo3.mirror = not demo3.mirror
                elif e.key==pg.K_o: demo3.cycle_source()
                elif e.key==pg.K_v: demo3.toggle_file()
                elif e.key==pg.K_n: watcher.poll(); print("[NowPlaying] Manual reload")
                elif e.key==pg.K_p: print(f"[NowPlaying] Path: {watcher.path}")
                elif e.key == pg.K_w:  # toggle white-pixel camera mode
                    WHITE_MASK = not WHITE_MASK
                    print("[WhiteMask]", "ON" if WHITE_MASK else "OFF")
                elif e.key == pg.K_SEMICOLON:  # ; lower threshold
                    WHITE_THRESH = max(0.0, round(WHITE_THRESH - 0.02, 2))
                    print("[WhiteThresh]", WHITE_THRESH)
                elif e.key == pg.K_QUOTE:      # ' raise threshold
                    WHITE_THRESH = min(1.0, round(WHITE_THRESH + 0.02, 2))
                    print("[WhiteThresh]", WHITE_THRESH)


        if mode==1: demo1.draw(dt)
        elif mode==2: demo2.draw(dt)
        else: demo3.draw(dt)

        geom = matrix.compute_geom(screen.get_size())
        matrix.blit_to_leds(screen, theme, geom,
                            brightness=BRIGHTNESS, round_leds=ROUND_LEDS,
                            glow=GLOW, scanline=SCANLINE, t=t)

        if SHOW_HUD:
            try: font = pg.font.SysFont("Menlo", 16)
            except Exception: font = pg.font.SysFont(None, 16)
            lines = [
                f"Mode: {('Scoreboard','Ticker','Video')[mode-1]} | Window {geom.w}x{geom.h}  Matrix {MATRIX_W}x{MATRIX_H}",
                f"LED size {LED_SIZE}px gap {LED_GAP}px | Glow {'ON' if GLOW else 'OFF'} | Scanline {'ON' if SCANLINE else 'OFF'}",
                f"Theme {THEME_IDX+1}/{len(THEMES)} | Brightness {int(BRIGHTNESS*100)}%",
                f"Now Playing: {watcher.value}",
                f"File: {watcher.path}",
                "Keys: 1/2/3 modes | o cycle cam | v file/cam | m mirror | n reload | p print path",
                "[ ] size  - = gap  , . bright  g glow  s scanline  r round  c theme  f fullscreen  h HUD  q quit"
            ]
            y=8
            for L in lines:
                screen.blit(font.render(L, True, (230,230,235)), (10,y)); y+=20

        pg.display.flip()

    demo3.release(); pg.quit(); sys.exit()

if __name__ == "__main__":
    main()
