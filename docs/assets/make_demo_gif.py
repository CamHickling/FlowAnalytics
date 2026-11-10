"""Draw docs/assets/flowanalytics-demo.gif: an animated diagram of the pipeline.

It is a diagram, not a screen recording: the pipeline's real output shows study
participants, so the review-video layout is drawn as labelled boxes with a
made-up heart-rate line. No data is read.

    python docs/assets/make_demo_gif.py      (needs Pillow; Windows fonts)
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "flowanalytics-demo.gif"
W, H = 960, 420
FPS = 12
NAVY, DEEP, BLUE, SKY, WHITE = (7, 26, 51), (10, 47, 107), (30, 111, 192), (142, 197, 240), (255, 255, 255)
DIM = (40, 64, 100)
FONTS = Path("C:/Windows/Fonts")

STAGES = [("Sync", "cameras lined up\nfrom their audio"),
          ("Correct", "GoPro lens\ndistortion removed"),
          ("Time", "cleaned narration\ngets WhisperX timing"),
          ("Compose", "one review video\nper trial")]


def font(size, style="Semibold Text"):
    f = ImageFont.truetype(str(FONTS / "SegUIVar.ttf"), size)
    f.set_variation_by_name(style)
    return f


def mix(a, b, p):
    return tuple(int(a[i] + (b[i] - a[i]) * p) for i in range(3))


def ease(p):
    p = min(max(p, 0.0), 1.0)
    return 1 - (1 - p) ** 3


def stages_frame(t):
    """0-4 s: the four stages light up one by one."""
    im = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(im)
    title = "From raw recordings to one review video"
    d.text(((W - d.textlength(title, font=font(30))) / 2, 34), title, font=font(30), fill=WHITE)
    bw, bh, gap = 190, 150, 34
    x0 = (W - (4 * bw + 3 * gap)) / 2
    y0 = 130
    for i, (name, sub) in enumerate(STAGES):
        p = ease((t - 0.3 - 0.8 * i) / 0.4)
        x = x0 + i * (bw + gap)
        d.rounded_rectangle((x, y0, x + bw, y0 + bh), 16, fill=mix(DEEP, BLUE, p),
                            outline=mix(DIM, SKY, p), width=2)
        d.text((x + 20, y0 + 18), f"{i + 1}", font=font(22), fill=mix(DIM, SKY, p))
        d.text((x + 20, y0 + 46), name, font=font(32, "Bold Display"), fill=mix(DIM, WHITE, p))
        d.multiline_text((x + 20, y0 + 92), sub, font=font(17, "Regular"), fill=mix(DIM, SKY, p),
                         spacing=4)
        if i < 3:
            ax = x + bw + 6
            q = ease((t - 0.7 - 0.8 * i) / 0.3)
            d.line((ax, y0 + bh / 2, ax + (gap - 12) * q, y0 + bh / 2), fill=SKY, width=4)
    foot = "Built for the UBC BAR Lab's Taekwondo flow study"
    d.text(((W - d.textlength(foot, font=font(18, "Regular"))) / 2, 340), foot,
           font=font(18, "Regular"), fill=SKY)
    return im


def review_frame(t):
    """4-9 s: the review video's layout, drawn as labelled boxes."""
    im = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(im)
    p = ease(t / 0.5)
    title = "One review video per trial"
    d.text(((W - d.textlength(title, font=font(30))) / 2, 24), title, font=font(30), fill=mix(NAVY, WHITE, p))
    L, R, T = 150, 810, 78
    mid = (L + R) / 2
    box = dict(radius=10, width=2)
    d.rounded_rectangle((L, T, mid - 4, T + 170), fill=DEEP, outline=SKY, **box)
    d.rounded_rectangle((mid + 4, T, R, T + 170), fill=DEEP, outline=SKY, **box)
    d.rounded_rectangle((R - 120, T + 92, R - 10, T + 160), fill=BLUE, outline=WHITE, **box)
    lab = font(17)
    d.text((L + 14, T + 12), "Front GoPro (lens-corrected)", font=lab, fill=WHITE)
    d.text((mid + 18, T + 12), "Side GoPro (lens-corrected)", font=lab, fill=WHITE)
    d.text((R - 108, T + 112), "Face cam", font=lab, fill=WHITE)
    # subtitles
    d.rounded_rectangle((L, T + 180, R, T + 222), fill=(14, 36, 70), outline=DIM, **box)
    sub = "Subtitles from the athlete's cleaned narration"
    d.text((L + 18, T + 189), sub, font=font(19, "Regular"), fill=mix(NAVY, WHITE, ease((t - 0.4) / 0.5)))
    # heart-rate graph with a moving marker
    gy0, gy1 = T + 232, T + 318
    d.rounded_rectangle((L, gy0, R, gy1), fill=(14, 36, 70), outline=DIM, **box)
    pts = [(L + 10 + k, gy1 - 18 - 40 * (0.5 + 0.35 * math.sin(k / 55) + 0.15 * math.sin(k / 13)))
           for k in range(0, R - L - 20, 3)]
    d.line(pts, fill=SKY, width=3)
    m = int(((t - 0.5) % 4.5) / 4.5 * (len(pts) - 1))
    mx, my = pts[m]
    d.line((mx, gy0 + 6, mx, gy1 - 6), fill=WHITE, width=2)
    d.ellipse((mx - 6, my - 6, mx + 6, my + 6), fill=WHITE)
    d.text((L + 14, gy0 + 6), "Heart rate", font=font(15), fill=SKY)
    note = "Diagram of the layout: the real videos show study participants, so they aren't shown here"
    d.text(((W - d.textlength(note, font=font(14, "Regular"))) / 2, 396), note,
           font=font(14, "Regular"), fill=DIM)
    return im


def main():
    frames = []
    for i in range(int(4.2 * FPS)):
        frames.append(stages_frame(i / FPS))
    for i in range(int(5.0 * FPS)):
        frames.append(review_frame(i / FPS))
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=int(1000 / FPS), loop=0,
                optimize=True, disposal=1)
    print(f"{OUT.name}: {len(frames)} frames, {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
