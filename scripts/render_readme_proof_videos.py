#!/usr/bin/env python3
"""Generate verifier-backed README proof videos and validate media drift."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import readme_proof  # noqa: E402


ASSET_DIR = ROOT / "docs/assets"
MANIFEST_PATH = ASSET_DIR / "proof_media_manifest.json"
WIDTH = 1200
HEIGHT = 720
FPS = 10
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

COLORS = {
    "background": "#10131d",
    "terminal": "#171b2a",
    "chrome": "#22283a",
    "white": "#d8dee9",
    "muted": "#7f8aa3",
    "cyan": "#54d6df",
    "green": "#69e59a",
    "yellow": "#f3c969",
    "red": "#ff6b7a",
}


@dataclass(frozen=True)
class VideoSpec:
    slug: str
    command: str
    title: str
    sources: tuple[Path, ...]


SPECS = (
    VideoSpec(
        slug="proof_local",
        command="local",
        title="Local no-key verification",
        sources=(ROOT / "entroly/verify_claims.py",),
    ),
    VideoSpec(
        slug="proof_model_recovery",
        command="model-recovery",
        title="Model-triggered recovery quality",
        sources=(ROOT / "benchmarks/model_recovery.py", readme_proof.MODEL_REPORT),
    ),
    VideoSpec(
        slug="proof_restart_recovery",
        command="restart-recovery",
        title="Concurrent restart recovery",
        sources=(
            ROOT / "benchmarks/recovery_resilience.py",
            readme_proof.RECOVERY_REPORT,
            readme_proof.PRIOR_RECOVERY_REPORT,
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font_path() -> Path:
    candidates = (
        Path("C:/Windows/Fonts/CascadiaMono.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/System/Library/Fonts/SFNSMono.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError("No supported monospace font found")


def _capture(spec: VideoSpec) -> list[str]:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exit_code = readme_proof.main([spec.command])
    if exit_code != 0:
        raise RuntimeError(f"proof command failed: {spec.command}")
    return [ANSI_RE.sub("", line).rstrip() for line in stream.getvalue().splitlines()]


def _line_color(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("ENTROLY PROOF"):
        return COLORS["cyan"]
    if stripped.startswith("[PASS]") or "McNemar p" in line:
        return COLORS["green"]
    if stripped.startswith("$") or "worker error retained" in line:
        return COLORS["yellow"]
    if stripped.startswith(("Source:", "Scoped ", "One frozen ", "Run it ")):
        return COLORS["muted"]
    if set(stripped) == {"-"}:
        return COLORS["muted"]
    return COLORS["white"]


def _render_frame(
    *,
    spec: VideoSpec,
    lines: list[str],
    body_font: ImageFont.FreeTypeFont,
    chrome_font: ImageFont.FreeTypeFont,
) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
    draw = ImageDraw.Draw(image)
    terminal = (24, 22, WIDTH - 24, HEIGHT - 22)
    draw.rounded_rectangle(terminal, radius=16, fill=COLORS["terminal"])
    draw.rounded_rectangle(
        (24, 22, WIDTH - 24, 72),
        radius=16,
        fill=COLORS["chrome"],
    )
    draw.rectangle((24, 54, WIDTH - 24, 72), fill=COLORS["chrome"])
    for index, color in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        x = 50 + index * 28
        draw.ellipse((x, 39, x + 12, 51), fill=color)
    title_box = draw.textbbox((0, 0), spec.title, font=chrome_font)
    title_width = title_box[2] - title_box[0]
    draw.text(
        ((WIDTH - title_width) / 2, 36),
        spec.title,
        font=chrome_font,
        fill=COLORS["muted"],
    )

    y = 94
    line_height = 31
    for line in lines:
        draw.text(
            (54, y),
            line,
            font=body_font,
            fill=_line_color(line),
        )
        y += line_height
    return image


def _frames(spec: VideoSpec) -> tuple[list[Image.Image], list[int], list[str]]:
    lines = _capture(spec)
    font_path = _font_path()
    body_font = ImageFont.truetype(str(font_path), 19)
    chrome_font = ImageFont.truetype(str(font_path), 15)
    frames: list[Image.Image] = []
    durations: list[int] = []
    for visible in range(1, len(lines) + 1):
        frames.append(
            _render_frame(
                spec=spec,
                lines=lines[:visible],
                body_font=body_font,
                chrome_font=chrome_font,
            )
        )
        line = lines[visible - 1].strip()
        durations.append(420 if not line or line.startswith("$") else 240)
    durations[-1] = 2600
    return frames, durations, lines


def _write_gif(path: Path, frames: list[Image.Image], durations: list[int]) -> None:
    palette_frames = [
        frame.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
        for frame in frames
    ]
    palette_frames[0].save(
        path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=durations,
        disposal=2,
        loop=0,
        optimize=True,
    )


def _write_mp4(path: Path, frames: list[Image.Image], durations: list[int]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to generate MP4 proof videos")
    process = subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame, duration in zip(frames, durations, strict=True):
            copies = max(1, round(duration * FPS / 1000))
            payload = frame.tobytes()
            for _ in range(copies):
                process.stdin.write(payload)
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {path}")


def generate() -> int:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": "entroly.readme-proof-media.v1",
        "renderer": "scripts/render_readme_proof_videos.py",
        "font": _font_path().name,
        "videos": {},
    }
    videos: dict[str, object] = {}
    for spec in SPECS:
        frames, durations, lines = _frames(spec)
        gif_path = ASSET_DIR / f"{spec.slug}.gif"
        mp4_path = ASSET_DIR / f"{spec.slug}.mp4"
        png_path = ASSET_DIR / f"{spec.slug}.png"
        _write_gif(gif_path, frames, durations)
        _write_mp4(mp4_path, frames, durations)
        frames[-1].save(png_path, optimize=True)
        videos[spec.slug] = {
            "proof_command": f"python scripts/readme_proof.py {spec.command}",
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
                for path in spec.sources
            },
            "rendered_lines_sha256": hashlib.sha256(
                "\n".join(lines).encode("utf-8")
            ).hexdigest(),
            "outputs": {
                path.name: _sha256(path)
                for path in (gif_path, mp4_path, png_path)
            },
        }
        print(
            f"rendered {spec.slug}: "
            f"gif={gif_path.stat().st_size:,} "
            f"mp4={mp4_path.stat().st_size:,} "
            f"png={png_path.stat().st_size:,}"
        )
    manifest["videos"] = videos
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def verify() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    for spec in SPECS:
        record = manifest["videos"][spec.slug]
        expected_sources = record["source_sha256"]
        for source in spec.sources:
            relative = str(source.relative_to(ROOT)).replace("\\", "/")
            if expected_sources.get(relative) != _sha256(source):
                failures.append(f"source drift: {relative}")
        for filename, expected in record["outputs"].items():
            path = ASSET_DIR / filename
            if not path.exists() or _sha256(path) != expected:
                failures.append(f"media drift: {filename}")
        lines = _capture(spec)
        rendered_hash = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
        if rendered_hash != record["rendered_lines_sha256"]:
            failures.append(f"proof output drift: {spec.command}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print(f"verified {len(SPECS)} README proof videos and source bindings")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "verify"))
    args = parser.parse_args(argv)
    return generate() if args.action == "generate" else verify()


if __name__ == "__main__":
    raise SystemExit(main())
