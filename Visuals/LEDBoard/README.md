LED Matrix Wall Emulator (Pygame)

A performance-ready dot-matrix LED wall emulator built with Pygame (and NumPy, optional OpenCV for camera/video). It renders a crisp LED grid with soft glow and scanline effects, multiple visual modes, and a “NOW PLAYING” header that auto-fits or toggles for clean stage visuals.

Designed for club LED walls (Speed Garage / UKG / Hardgroove vibes). Tested fullscreen at 1280×720 and higher.

Features

True LED look – round/square dots, per-pixel glow, scanline row refresh

Modes

Scoreboard demo (retro sports board)

Ticker – wide scrolling band (customizable position)

Video Matrix – webcam/capture card or looping video file, plus a strong NOW PLAYING header that:

auto-fits artist text,

falls back to a smooth marquee for long strings,

or toggles “NOW PLAYING” ↔ artist every few seconds for a clean, centered look

Now Playing from a text file – update the first line and the app refreshes live

Themeable – multiple palettes tuned for LED walls (darker backgrounds)

Performance friendly – scale matrix resolution & effects on the fly

Requirements

Python 3.9+ (works on 3.13 as used in testing)

Packages:

pip install pygame numpy
# Optional (needed for camera/video mode):
pip install opencv-python


macOS users: OpenCV will use AVFoundation by default. Homebrew ffmpeg is optional.

Files

led_board_emulator_now_playing.py — main app (all modes)

(optional) led_now_playing.py — minimal “Now Playing” only version

now_playing.txt — first non-blank line is displayed as the current artist/title

Quick Start

Set paths in the config block (top of led_board_emulator_now_playing.py):

# Absolute path recommended
NOW_PLAYING_FILE = "/Users/yourname/Desktop/Club Fast/now_playing.txt"

# Optional video file; press 'v' to toggle file/camera
VIDEO_FILE = "/Users/yourname/Videos/clip.mp4"


Create now_playing.txt and put a single line, e.g.

SARAH — Midnight Pressure (VIP)


Run:

python3 led_board_emulator_now_playing.py


Go fullscreen with f. Update now_playing.txt during the show—changes appear live.

Controls
Key	Action
1 / 2 / 3	Switch mode: Scoreboard / Ticker / Video Matrix
f	Toggle fullscreen
h	Toggle HUD
c	Cycle color theme
g	Toggle LED glow
s	Toggle scanline “row refresh”
r	Toggle round/square LEDs
, / .	Brightness − / +
[ / ]	LED size − / +
- / =	LED gap − / +
o	Cycle camera index (0 → 1 → …)
v	Toggle video file ↔ camera
m	Mirror camera/video
q / Esc	Quit

In the Video Matrix mode, the header is a clean toggle: NOW PLAYING ↔ Artist (2s / 4s). For very long names, it gracefully scrolls (no ellipsis).

Modes
1) Scoreboard Demo

Retro sports board with bases/inning UI. Great for testing palettes and scaling.

2) Ticker

Wide scrolling band with your message.
To center vertically, inside TickerDemo.draw() set:

y0 = (mh - band_h) // 2
canvas.blit(band, (0, y0))
# ...and draw the text relative to y0 instead of mh - band_h

3) Video Matrix (Camera / Video File)

Uses OpenCV to grab frames and quantize them into the LED grid.

NOW PLAYING header sits on top:

Toggle text: “NOW PLAYING” ↔ artist (clean, centered)

Auto-fit artist text; marquee fallback if it still doesn’t fit

Thin outline for visibility on bright footage

To use a video file, set VIDEO_FILE and press v. Press o to cycle capture devices (0/1/etc.) when in camera mode.

“Now Playing” Integration

The app reads the first non-blank line of NOW_PLAYING_FILE.

Use an absolute path (safest on macOS).

It polls the file’s modification time and updates automatically.

HUD (toggle with h) shows the resolved path and the current text.

Common workflow: have your DJ app or a small script update now_playing.txt whenever the track changes.

Configuration Highlights

At the top of led_board_emulator_now_playing.py:

WINDOW_START = (1280, 720)
MATRIX_W = 96      # LED columns
MATRIX_H = 54      # LED rows
LED_SIZE = 10      # base dot size (scales with window)
LED_GAP  = 2       # spacing between dots
BRIGHTNESS = 0.85  # master dimmer (0..1)

THEMES = [
    ((6,8,12), (255,200,40), (255,80,30)),   # bg, primary, accent
    ((4,6,10), (40,220,255), (255,80,180)),
    ...
]


Performance tips

Lower MATRIX_W/H (e.g., 80×45) for extra headroom.

Turn off GLOW or SCANLINE during heavy scenes.

Keep brightness moderate for camera-friendly recordings.

Troubleshooting

Blank or tiny dot only

Switch to a different mode with 1/2/3.

Ensure your window isn’t minimized; try f for fullscreen.

ModuleNotFoundError: cv2

Install OpenCV: pip install opencv-python.

Camera won’t open / wrong device

Press o to cycle indices (0/1/2…).

Close other apps using the camera.

On macOS, grant Terminal/iTerm camera permissions (System Settings → Privacy & Security → Camera).

Now Playing not updating

Check HUD path matches the file you’re editing.

Make sure you changed the first non-blank line and saved.

Use an absolute path in NOW_PLAYING_FILE.

Ellipsis / clipping

In this build, artist text fits or scrolls (no ellipsis). If you prefer a static fit only, raise the header height in code:

header_h = int(mh * 0.30)

Customization Ideas

White-Pixel Camera (silhouette effect): threshold luminance and light only white dots.

Audio-reactive tint: modulate the primary color on mic peaks.

OSC/HTTP control: expose an endpoint to update “Now Playing” from other apps.

(Ping me if you want any of these; the codebase is set up to add them cleanly.)

License

Use it freely for performances and shows. If you share it online, a credit back to this repo is appreciated 🎛️

Credits

Built with Pygame & NumPy.

Optional video/camera via OpenCV.

Palettes tuned for darker LED environments (club walls & stage cams).