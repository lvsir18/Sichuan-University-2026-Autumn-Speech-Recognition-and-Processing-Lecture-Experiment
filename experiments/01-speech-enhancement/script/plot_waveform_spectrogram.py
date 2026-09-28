"""Create an SVG comparison of noisy, enhanced, and clean speech."""

from __future__ import annotations

import argparse
import base64
import math
import struct
import wave
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "presentation" / "assets"
TEST_DIR = ROOT / "VoiceBank-DEMAND" / "test"
ENHANCED_DIR = ROOT / "output" / "enhanced"
OUTPUT_DIR = ROOT / "output"
SAMPLE_ID = "p232_160.wav"
FFT_SIZE = 512
WINDOW_SIZE = 400
HOP_SIZE = 100
SPEC_WIDTH = 480
SPEC_HEIGHT = 108


def load_wav(path: Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)

    if sample_width == 1:
        values = [((value - 128) / 128.0) for value in raw]
    elif sample_width == 2:
        values = [value[0] / 32768.0 for value in struct.iter_unpack("<h", raw)]
    elif sample_width == 3:
        values = []
        for offset in range(0, len(raw), 3):
            value = int.from_bytes(raw[offset : offset + 3], "little", signed=False)
            if value & 0x800000:
                value -= 0x1000000
            values.append(value / 8388608.0)
    elif sample_width == 4:
        values = [value[0] / 2147483648.0 for value in struct.iter_unpack("<i", raw)]
    else:
        raise ValueError(f"不支持的 WAV 采样宽度：{sample_width} 字节（{path}）")

    if channels > 1:
        values = [sum(values[i : i + channels]) / channels for i in range(0, len(values), channels)]
    return sample_rate, values


def fft(values: list[complex]) -> list[complex]:
    """In-place radix-2 FFT; input length must be a power of two."""
    n = len(values)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            values[i], values[j] = values[j], values[i]

    length = 2
    while length <= n:
        angle = -2 * math.pi / length
        root = complex(math.cos(angle), math.sin(angle))
        half = length // 2
        for start in range(0, n, length):
            factor = 1 + 0j
            for offset in range(half):
                even = values[start + offset]
                odd = factor * values[start + offset + half]
                values[start + offset] = even + odd
                values[start + offset + half] = even - odd
                factor *= root
        length *= 2
    return values


def spectrogram(audio: list[float], sample_rate: int) -> tuple[list[list[float]], int]:
    starts = list(range(0, max(1, len(audio) - WINDOW_SIZE + 1), HOP_SIZE))
    bins = min(FFT_SIZE // 2 + 1, int(8000 * FFT_SIZE / sample_rate) + 1)
    columns: list[list[float]] = []
    window = [0.5 - 0.5 * math.cos(2 * math.pi * i / (WINDOW_SIZE - 1)) for i in range(WINDOW_SIZE)]
    for start in starts:
        frame = audio[start : start + WINDOW_SIZE]
        if len(frame) < WINDOW_SIZE:
            frame = frame + [0.0] * (WINDOW_SIZE - len(frame))
        transformed = [complex(frame[i] * window[i], 0) for i in range(WINDOW_SIZE)]
        transformed.extend([0j] * (FFT_SIZE - WINDOW_SIZE))
        values = fft(transformed)
        columns.append([abs(value) / FFT_SIZE for value in values[:bins]])
    return columns, bins


def color_for_db(db: float) -> tuple[int, int, int]:
    stops = [
        (-70.0, (35, 26, 52)),
        (-48.0, (91, 34, 75)),
        (-28.0, (165, 43, 54)),
        (-12.0, (231, 105, 43)),
        (0.0, (255, 220, 139)),
    ]
    value = max(-70.0, min(0.0, db))
    for (left_db, left), (right_db, right) in zip(stops, stops[1:]):
        if value <= right_db:
            ratio = (value - left_db) / (right_db - left_db)
            return tuple(round(a + ratio * (b - a)) for a, b in zip(left, right))
    return stops[-1][1]


def png_data(columns: list[list[float]], bins: int, reference: float) -> str:
    rows: list[bytes] = []
    for y in range(SPEC_HEIGHT):
        frequency_index = round((SPEC_HEIGHT - 1 - y) * (bins - 1) / (SPEC_HEIGHT - 1))
        row = bytearray([0])
        for x in range(SPEC_WIDTH):
            column_index = min(len(columns) - 1, round(x * (len(columns) - 1) / max(1, SPEC_WIDTH - 1)))
            magnitude = columns[column_index][frequency_index]
            db = 20 * math.log10(max(magnitude, 1e-10) / max(reference, 1e-10))
            row.extend(color_for_db(db))
        rows.append(bytes(row))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    raw = b"".join(rows)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">2I5B", SPEC_WIDTH, SPEC_HEIGHT, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, level=6))
    png += chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def write_comparison_png(
    output_path: Path,
    loaded: list[tuple[str, Path, str, int, list[float]]],
    audio_by_track: list[list[float]],
    spectra: list[tuple[list[list[float]], int]],
    sample_rate: int,
    duration: float,
    shared_peak: float,
    shared_spectral_peak: float,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("生成 PNG 需要 Pillow；请在当前环境安装 Pillow。") from exc

    def font(size: int, bold: bool = False):
        candidates = (
            [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"]
            if bold else
            [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc"]
        )
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
        return ImageFont.load_default()

    scale = 2
    width, height = 1600, 720
    wave_x, wave_w = 170, 690
    spec_x, spec_w = 950, 570
    row_top, row_step, wave_h = 90, 185, 120
    wave_label_font = font(22 * scale, bold=True)
    heading_font = font(22 * scale, bold=True)
    tick_font = font(13 * scale)
    image = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(image)
    draw.text((wave_x * scale, 28 * scale), "波形：振幅随时间变化", font=heading_font, fill="#b72b1a")
    draw.text((spec_x * scale, 28 * scale), "语谱图：不同频率的强弱", font=heading_font, fill="#b72b1a")
    draw.text(((width - 35) * scale, 34 * scale), f"样本：{loaded[0][1].stem}", font=tick_font, fill="#777777", anchor="ra")

    for index, ((label, _path, color, _sr, _samples), audio, (columns, bins)) in enumerate(
        zip(loaded, audio_by_track, spectra)
    ):
        y = (row_top + index * row_step) * scale
        spec_y = y + 4 * scale
        scaled_wave_x, scaled_wave_w, scaled_wave_h = wave_x * scale, wave_w * scale, wave_h * scale
        scaled_spec_x, scaled_spec_w = spec_x * scale, spec_w * scale
        scaled_spec_h = SPEC_HEIGHT * scale
        draw.text((35 * scale, y + 44 * scale), label, font=wave_label_font, fill=color)
        draw.rectangle((scaled_wave_x, y, scaled_wave_x + scaled_wave_w, y + scaled_wave_h), fill="white", outline="#ded9d5", width=scale)
        draw.rectangle((scaled_spec_x, spec_y, scaled_spec_x + scaled_spec_w, spec_y + scaled_spec_h), fill="white", outline="#ded9d5", width=scale)

        for second in range(math.ceil(duration) + 1):
            if second > duration:
                continue
            x = round((wave_x + second / duration * wave_w) * scale)
            spec_tick_x = round((spec_x + second / duration * spec_w) * scale)
            draw.line((x, y, x, y + scaled_wave_h), fill="#d7d2ce", width=scale)
            draw.line((spec_tick_x, spec_y, spec_tick_x, spec_y + scaled_spec_h), fill="#d7d2ce", width=scale)
            draw.text((x - 4 * scale, y + scaled_wave_h + 4 * scale), str(second), font=tick_font, fill="#777777")

        center_y = y + scaled_wave_h / 2
        draw.line((scaled_wave_x, center_y, scaled_wave_x + scaled_wave_w, center_y), fill="#bcb6b1", width=scale)
        upper, lower = waveform_envelope(audio, wave_w * scale)
        for column, (peak, trough) in enumerate(zip(upper, lower)):
            x = scaled_wave_x + column
            y_peak = center_y - peak / shared_peak * (scaled_wave_h * 0.45)
            y_trough = center_y - trough / shared_peak * (scaled_wave_h * 0.45)
            draw.line((x, y_peak, x, y_trough), fill=color, width=scale)

        spec_image = Image.new("RGB", (SPEC_WIDTH, SPEC_HEIGHT))
        for py in range(SPEC_HEIGHT):
            frequency_index = round((SPEC_HEIGHT - 1 - py) * (bins - 1) / (SPEC_HEIGHT - 1))
            for px in range(SPEC_WIDTH):
                column_index = min(len(columns) - 1, round(px * (len(columns) - 1) / max(1, SPEC_WIDTH - 1)))
                magnitude = columns[column_index][frequency_index]
                db = 20 * math.log10(max(magnitude, 1e-10) / max(shared_spectral_peak, 1e-10))
                spec_image.putpixel((px, py), color_for_db(db))
        image.paste(spec_image.resize((scaled_spec_w, scaled_spec_h), Image.Resampling.BILINEAR), (scaled_spec_x, spec_y))
        draw.rectangle((scaled_spec_x, spec_y, scaled_spec_x + scaled_spec_w, spec_y + scaled_spec_h), outline="#ded9d5", width=scale)
        for tick in (0, 2, 4, 6, 8):
            if tick > sample_rate / 2000:
                continue
            y_tick = round(spec_y + scaled_spec_h * (1 - tick / 8))
            draw.line((scaled_spec_x, y_tick, scaled_spec_x + scaled_spec_w, y_tick), fill="#d7d2ce", width=scale)
            draw.text((scaled_spec_x - 20 * scale, y_tick - 7 * scale), str(tick), font=tick_font, fill="#777777")

    draw.text(((wave_x + wave_w / 2 - 35) * scale, (height - 27) * scale), "时间（秒）", font=tick_font, fill="#777777")
    draw.text(((spec_x + spec_w / 2 - 36) * scale, (height - 27) * scale), "时间（秒）", font=tick_font, fill="#777777")
    axis_label = Image.new("RGBA", (100 * scale, 20 * scale), (255, 255, 255, 0))
    ImageDraw.Draw(axis_label).text((0, 0), "频率（kHz）", font=tick_font, fill="#777777")
    axis_label = axis_label.rotate(90, expand=True)
    image.paste(axis_label, ((spec_x - 55) * scale, (row_top + row_step + 5) * scale), axis_label)
    gradient_y = (height - 12) * scale
    for x in range(spec_w * scale):
        db = -70 + 70 * x / max(1, spec_w * scale - 1)
        draw.line((spec_x * scale + x, gradient_y, spec_x * scale + x, gradient_y + 6 * scale), fill=color_for_db(db))
    draw.text((spec_x * scale, (height - 34) * scale), "弱", font=tick_font, fill="#777777")
    draw.text(((spec_x + spec_w - 20) * scale, (height - 34) * scale), "强", font=tick_font, fill="#777777")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def waveform_envelope(audio: list[float], columns: int) -> tuple[list[float], list[float]]:
    """Keep each horizontal bucket's extrema so waveform peaks are not skipped."""
    upper: list[float] = []
    lower: list[float] = []
    for column in range(columns):
        start = column * len(audio) // columns
        end = max(start + 1, (column + 1) * len(audio) // columns)
        values = audio[start:end]
        upper.append(max(values))
        lower.append(min(values))
    return upper, lower


def make_svg(noisy_path: Path, enhanced_path: Path, clean_path: Path, output_path: Path) -> None:
    tracks = [
        ("带噪语音", noisy_path, "#777777"),
        ("增强语音", enhanced_path, "#b72b1a"),
        ("干净语音", clean_path, "#222222"),
    ]
    loaded = [(label, path, color, *load_wav(path)) for label, path, color in tracks]
    sample_rates = {item[3] for item in loaded}
    if len(sample_rates) != 1:
        raise ValueError(f"输入音频采样率不一致：{sorted(sample_rates)}")
    sample_rate = loaded[0][3]
    common_length = min(len(item[4]) for item in loaded)
    if common_length < WINDOW_SIZE:
        raise ValueError("音频太短，无法计算语谱图")
    audio_by_track = [item[4][:common_length] for item in loaded]
    duration = common_length / sample_rate
    shared_peak = max(max(abs(value) for value in audio) for audio in audio_by_track)
    shared_peak = max(shared_peak, 1e-8)

    spectra = [spectrogram(audio, sample_rate) for audio in audio_by_track]
    shared_spectral_peak = max(max(max(column) for column in columns) for columns, _ in spectra)
    shared_spectral_peak = max(shared_spectral_peak, 1e-10)

    width, height = 1600, 720
    wave_x, wave_w = 170, 690
    spec_x, spec_w = 950, 570
    row_top, row_step, wave_h = 90, 185, 120
    colors = [item[2] for item in loaded]
    rows = []
    for index, ((label, path, color, _, _), audio, (columns, bins)) in enumerate(zip(loaded, audio_by_track, spectra)):
        y = row_top + index * row_step
        wave_y = y
        spec_y = y + 4
        row_svg = [f'<text x="35" y="{y + 86}" class="row-label" fill="{color}">{xml_escape(label)}</text>']
        row_svg.append(f'<rect x="{wave_x}" y="{wave_y}" width="{wave_w}" height="{wave_h}" class="plot-bg"/>')
        row_svg.append(f'<rect x="{spec_x}" y="{spec_y}" width="{spec_w}" height="{SPEC_HEIGHT}" class="plot-bg"/>')

        for second in range(math.ceil(duration) + 1):
            if second > duration:
                continue
            x = wave_x + second / duration * wave_w
            row_svg.append(f'<line x1="{x:.1f}" y1="{wave_y}" x2="{x:.1f}" y2="{wave_y + wave_h}" class="grid"/>')
            row_svg.append(f'<line x1="{x:.1f}" y1="{spec_y}" x2="{x:.1f}" y2="{spec_y + SPEC_HEIGHT}" class="grid"/>')
            row_svg.append(f'<text x="{x:.1f}" y="{wave_y + wave_h + 18}" class="tick" text-anchor="middle">{second}</text>')

        center_y = wave_y + wave_h / 2
        row_svg.append(f'<line x1="{wave_x}" y1="{center_y:.1f}" x2="{wave_x + wave_w}" y2="{center_y:.1f}" class="axis"/>')
        row_svg.append(f'<text x="{wave_x - 14}" y="{wave_y + 8}" class="tick" text-anchor="end">+</text>')
        row_svg.append(f'<text x="{wave_x - 14}" y="{wave_y + wave_h - 2}" class="tick" text-anchor="end">−</text>')

        upper, lower = waveform_envelope(audio, wave_w)
        upper_points = []
        lower_points = []
        for column, (peak, trough) in enumerate(zip(upper, lower)):
            x = wave_x + column
            y_peak = center_y - peak / shared_peak * (wave_h * 0.45)
            y_trough = center_y - trough / shared_peak * (wave_h * 0.45)
            upper_points.append(f"{x:.1f},{y_peak:.1f}")
            lower_points.append(f"{x:.1f},{y_trough:.1f}")
        envelope_points = " ".join(upper_points + list(reversed(lower_points)))
        row_svg.append(f'<polygon points="{envelope_points}" fill="{color}" fill-opacity=".18"/>')
        row_svg.append(f'<polyline points="{" ".join(upper_points)}" fill="none" stroke="{color}" stroke-width=".7" vector-effect="non-scaling-stroke"/>')
        row_svg.append(f'<polyline points="{" ".join(lower_points)}" fill="none" stroke="{color}" stroke-width=".7" vector-effect="non-scaling-stroke"/>')

        image = png_data(columns, bins, shared_spectral_peak)
        row_svg.append(f'<image x="{spec_x}" y="{spec_y}" width="{spec_w}" height="{SPEC_HEIGHT}" preserveAspectRatio="none" href="{image}"/>')
        for tick in (0, 2, 4, 6, 8):
            if tick > sample_rate / 2000:
                continue
            y_tick = spec_y + SPEC_HEIGHT * (1 - tick / 8)
            row_svg.append(f'<line x1="{spec_x}" y1="{y_tick:.1f}" x2="{spec_x + spec_w}" y2="{y_tick:.1f}" class="grid"/>')
            row_svg.append(f'<text x="{spec_x - 12}" y="{y_tick + 4:.1f}" class="tick" text-anchor="end">{tick}</text>')
        row_svg.append(f'<text x="{spec_x + spec_w / 2}" y="{spec_y + SPEC_HEIGHT + 18}" class="tick" text-anchor="middle">时间（秒）</text>')
        rows.extend(row_svg)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text{{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#333}}
  .heading{{font-size:22px;font-weight:700;fill:#b72b1a}}
  .row-label{{font-size:22px;font-weight:700}}
  .tick{{font-size:13px;fill:#777}}
  .plot-bg{{fill:#fff;stroke:#ded9d5;stroke-width:1}}
  .grid{{stroke:#d7d2ce;stroke-width:1;stroke-dasharray:3 5;opacity:.8}}
  .axis{{stroke:#bcb6b1;stroke-width:1}}
</style>
<rect width="100%" height="100%" fill="#fff"/>
<text x="{wave_x}" y="55" class="heading">波形｜振幅随时间变化</text>
<text x="{spec_x}" y="55" class="heading">语谱图｜不同频率的强弱</text>
<text x="{width - 35}" y="55" class="tick" text-anchor="end">样本：{xml_escape(noisy_path.stem)}</text>
<text x="{wave_x + wave_w / 2}" y="{height - 25}" class="tick" text-anchor="middle">时间（秒）</text>
{''.join(rows)}
<text x="{spec_x - 55}" y="{row_top + row_step + 80}" class="tick" text-anchor="middle" transform="rotate(-90 {spec_x - 55} {row_top + row_step + 80})">频率（kHz）</text>
<defs><linearGradient id="scale" x1="0" x2="1"><stop offset="0%" stop-color="#231a34"/><stop offset="28%" stop-color="#5b224b"/><stop offset="58%" stop-color="#a52b36"/><stop offset="82%" stop-color="#e7692b"/><stop offset="100%" stop-color="#ffdc8b"/></linearGradient></defs>
<rect x="{spec_x}" y="{height - 12}" width="{spec_w}" height="7" fill="url(#scale)"/>
<text x="{spec_x}" y="{height - 18}" class="tick">弱</text><text x="{spec_x + spec_w}" y="{height - 18}" class="tick" text-anchor="end">强</text>
</svg>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    print(f"Saved comparison figure: {output_path}")
    png_path = output_path.with_suffix(".png")
    write_comparison_png(
        png_path,
        loaded,
        audio_by_track,
        spectra,
        sample_rate,
        duration,
        shared_peak,
        shared_spectral_peak,
    )
    print(f"Saved comparison figure: {png_path}")
    print(f"Sample rate: {sample_rate} Hz; duration: {duration:.3f} s; window: {WINDOW_SIZE}; hop: {HOP_SIZE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noisy", type=Path, default=TEST_DIR / "noisy" / SAMPLE_ID)
    parser.add_argument("--enhanced", type=Path, default=ENHANCED_DIR / SAMPLE_ID)
    parser.add_argument("--clean", type=Path, default=TEST_DIR / "clean" / SAMPLE_ID)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "waveform_spectrogram_comparison.svg")
    args = parser.parse_args()
    make_svg(args.noisy, args.enhanced, args.clean, args.output)


if __name__ == "__main__":
    main()
