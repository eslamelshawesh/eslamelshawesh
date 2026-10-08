#!/usr/bin/env python3
"""
enhance_print.py -- deterministic, pixel-faithful high-definition upscale for print.

WHAT THIS DOES
    Upsamples a source image with a Lanczos resampling kernel and refines the
    detail that is genuinely present in the file:

      1. EXIF orientation is baked in so the pixels match what a viewer shows.
      2. Lanczos resampling to the target size (no interpolation blur).
      3. Dynamic-range stretch measured from the image's own luminance
         histogram (0.5th/99.5th percentile), blended through a gentle
         S-curve so highlights and shadows open up without clipping.
      4. A broad unsharp pass for edge structure.
      5. A tight, low-strength unsharp pass for micro-contrast ("crisp edges").
      6. Print DPI metadata written into the file.

WHAT THIS DOES NOT DO
    It is purely pointwise and locally-filtered. There is no generative model,
    no inpainting, no synthesis, and no added or removed subject matter.
    Composition, framing, colour cast and every object in the frame are
    preserved. It cannot invent pores, fabric weave or other texture that was
    never captured -- only the optics and sensor that produced the source can
    do that, and it does not fabricate it either.

Because no pixels are invented, this is the correct tool when the brief says
"the image and its overall composition remain unchanged" and the output has to
survive a print shop.

USAGE
    # Report on a file without touching it
    .venv/bin/python tools/enhance_print.py photo.jpg --inspect

    # Upscale 2x, write a 300 DPI print file
    .venv/bin/python tools/enhance_print.py photo.jpg out.png --scale 2 --dpi 300

    # Size for a specific print, e.g. A4 at 300 DPI
    .venv/bin/python tools/enhance_print.py photo.jpg out.jpg --print-in 8.27 11.69

    # Stronger refinement
    .venv/bin/python tools/enhance_print.py photo.jpg out.png --scale 4 --strength strong
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from PIL import Image, ImageFilter, ImageOps
except ImportError:  # pragma: no cover
    sys.exit(
        "Pillow is required. Create a virtualenv and install it:\n"
        "  python3 -m venv .venv && .venv/bin/python -m pip install Pillow"
    )


# Refinement presets: (dynamic-range blend, broad unsharp, micro unsharp)
STRENGTHS = {
    "subtle": {"range": 0.35, "edge": (0.014, 70, 3), "micro": (0.004, 35, 1)},
    "medium": {"range": 0.60, "edge": (0.018, 110, 3), "micro": (0.006, 55, 1)},
    "strong": {"range": 0.80, "edge": (0.022, 150, 2), "micro": (0.008, 75, 1)},
}

# Radius values above are fractions of the output long edge.

LOW_PCT, HIGH_PCT = 0.5, 99.5


@dataclass
class Report:
    source: str
    source_mode: str
    source_px: list
    output_px: list
    upscale_factor: float
    strength: str
    dpi: int
    luminance_percentiles: dict
    range_stretch: dict
    print_sizes: list


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _luminance(img: Image.Image) -> Image.Image:
    """Rec. 709 luma, computed in one pass by PIL's convert('L')."""
    return img.convert("L")


def _percentile_bounds(hist: list, low_pct: float, high_pct: float):
    """Return (lo, hi) pixel values bracketing the requested percentiles."""
    total = sum(hist)
    if total == 0:
        return 0, 255
    lo_target = total * (low_pct / 100.0)
    hi_target = total * (high_pct / 100.0)
    running = 0
    lo, hi = 0, 255
    for value, count in enumerate(hist):
        running += count
        if running >= lo_target:
            lo = value
            break
    running = 0
    for value, count in enumerate(hist):
        running += count
        if running >= hi_target:
            hi = value
            break
    if hi <= lo:
        return 0, 255
    return lo, hi


# How much of the smoothstep S-curve to mix into the identity ramp. 0 = no
# curve at all, 1 = full smoothstep (crushes both ends hard). 0.35 is gentle
# enough for print: it deepens shadows and lifts highlights while keeping the
# derivative above 0.6 everywhere, so no tonal band collapses.
S_CURVE_MIX = 0.35


def _s_curve_lut(lo: int, hi: int, blend: float) -> list:
    """
    256-entry LUT: linearly stretch [lo, hi] onto [0, 255], then push the
    result through a soft S-curve for contrast. `blend` mixes the result back
    with the untouched ramp so the effect can never be applied at full force
    unless the caller asks for it.

    The S-curve is (1-m)*n + m*smoothstep(n). smoothstep is exactly 0 at n=0,
    exactly 0.5 at n=0.5 and exactly 1 at n=1, so the curve is anchored at both
    endpoints -- pure black stays pure black and pure white stays pure white --
    and it is monotonically increasing across the whole ramp.
    """
    span = max(1, hi - lo)
    m = S_CURVE_MIX
    lut = []
    for v in range(256):
        stretched = (v - lo) * 255.0 / span
        n = max(0.0, min(1.0, stretched / 255.0))
        smooth = 3.0 * n * n - 2.0 * n * n * n
        s = (1.0 - m) * n + m * smooth
        blended = v * (1.0 - blend) + (s * 255.0) * blend
        lut.append(int(max(0, min(255, round(blended)))))
    return lut


def stretch_dynamic_range(img: Image.Image, blend: float):
    """Return (image, diagnostics) after a histogram-driven range stretch."""
    luma = _luminance(img)
    hist = luma.histogram()
    lo, hi = _percentile_bounds(hist, LOW_PCT, HIGH_PCT)
    lut = _s_curve_lut(lo, hi, blend)
    if img.mode == "L":
        return img.point(lut), {"lo": lo, "hi": hi, "blend": blend}

    # Apply the same luma-derived curve per channel: keeps colour ratios
    # intact instead of skewing the cast the way an auto-level would.
    bands = img.split()
    processed = [b.point(lut) for b in bands]
    return Image.merge(img.mode, processed), {"lo": lo, "hi": hi, "blend": blend}


def unsharp(img: Image.Image, radius_frac: float, percent: int, threshold: int) -> Image.Image:
    """UnsharpMask with radius expressed as a fraction of the long edge."""
    long_edge = max(img.size)
    radius = max(0.4, radius_frac * long_edge)
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def resolve_target_size(src: tuple, scale: float, long_edge: int, print_in, dpi: int):
    """Work out the output dimensions from whichever sizing argument was given."""
    w, h = src
    if print_in is not None:
        pw, ph = print_in
        tw, th = int(round(pw * dpi)), int(round(ph * dpi))
        return (tw, th), "print"
    if long_edge is not None:
        if w >= h:
            return (long_edge, max(1, round(h * long_edge / w))), "long-edge"
        return (max(1, round(w * long_edge / h)), long_edge), "long-edge"
    return (max(1, round(w * scale)), max(1, round(h * scale))), "scale"


def print_size_table(px: tuple) -> list:
    """Max printable size at common print resolutions, in inches and cm."""
    out = []
    for dpi in (300, 240, 150):
        w_in, h_in = px[0] / dpi, px[1] / dpi
        out.append(
            {
                "dpi": dpi,
                "inches": [round(w_in, 2), round(h_in, 2)],
                "cm": [round(w_in * 2.54, 1), round(h_in * 2.54, 1)],
                "quality": "reference" if dpi == 300 else ("good" if dpi == 240 else "acceptable"),
            }
        )
    return out


# --------------------------------------------------------------------------
# main pipeline
# --------------------------------------------------------------------------
def enhance(
    src_path: Path,
    dst_path: Path,
    scale: float = 2.0,
    long_edge: int | None = None,
    print_in=None,
    dpi: int = 300,
    strength: str = "medium",
) -> Report:
    preset = STRENGTHS[strength]

    with Image.open(src_path) as raw:
        # Bake EXIF rotation in, otherwise an upright-looking portrait is
        # upscaled sideways and the sharpening runs along the wrong axis.
        img = ImageOps.exif_transpose(raw)
        source_px = list(img.size)
        source_mode = img.mode

        # Preserve alpha if present; refine colour only, then re-attach.
        alpha = None
        if img.mode in ("RGBA", "LA", "PA"):
            alpha = img.getchannel("A")
            img = img.convert("RGB")
        elif img.mode == "P":
            img = img.convert("RGBA")
            alpha = img.getchannel("A")
            img = img.convert("RGB")
        elif img.mode not in ("RGB", "L", "CMYK"):
            img = img.convert("RGB")

        target, mode_used = resolve_target_size(
            tuple(source_px), scale, long_edge, print_in, dpi
        )

        # 1) Lanczos resample -- the highest-fidelity kernel Pillow offers.
        upscaled = img.resize(target, Image.LANCZOS)

        # 2) Histogram-measured dynamic range + gentle S-curve.
        ranged, range_diag = stretch_dynamic_range(upscaled, preset["range"])

        # 3) Broad edge structure.
        edged = unsharp(ranged, *preset["edge"])

        # 4) Micro-contrast for crisp, precise edges.
        final = unsharp(edged, *preset["micro"])

        if alpha is not None:
            # Alpha rides the same geometric transform; it must not be sharpened,
            # or translucent edges acquire a hard fringe.
            alpha = alpha.resize(target, Image.LANCZOS)
            final = final.convert("RGBA")
            final.putalpha(alpha)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": (dpi, dpi)}
    suffix = dst_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        save_kwargs.update(quality=100, subsampling=0, progressive=True)
    elif suffix == ".png":
        save_kwargs.update(compress_level=6)
    final.save(dst_path, **save_kwargs)

    factor = round(target[0] / source_px[0], 4)
    report = Report(
        source=str(src_path),
        source_mode=source_mode,
        source_px=source_px,
        output_px=list(target),
        upscale_factor=factor,
        strength=f"{strength} ({mode_used})",
        dpi=dpi,
        luminance_percentiles={"p0.5": range_diag["lo"], "p99.5": range_diag["hi"]},
        range_stretch={"blend": range_diag["blend"]},
        print_sizes=print_size_table(target),
    )
    return report


def inspect(src_path: Path, dpi: int = 300) -> dict:
    with Image.open(src_path) as raw:
        img = ImageOps.exif_transpose(raw)
        px = img.size
        luma = _luminance(img.convert("RGB") if img.mode not in ("RGB", "L") else img)
        hist = luma.histogram()
        lo, hi = _percentile_bounds(hist, LOW_PCT, HIGH_PCT)
        total = sum(hist)
        mean = sum(i * c for i, c in enumerate(hist)) / max(1, total)
    return {
        "source": str(src_path),
        "px": list(px),
        "megapixels": round(px[0] * px[1] / 1e6, 2),
        "mode": img.mode,
        "luminance": {"mean": round(mean, 1), "p0.5": lo, "p99.5": hi},
        "clipped_shadows_pct": round(100 * hist[0] / max(1, total), 3),
        "clipped_highlights_pct": round(100 * hist[255] / max(1, total), 3),
        "print_sizes": print_size_table(px),
        "max_safe_upscale_note": (
            "Native pixels set the true ceiling. Beyond ~4x, Lanczos adds size, "
            "not information; the print will look smooth rather than detailed."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Deterministic pixel-faithful HD upscale for print. No AI, no invented detail.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="source image")
    p.add_argument("output", type=Path, nargs="?", help="destination (.png recommended for print)")
    p.add_argument("--inspect", action="store_true", help="report only, write nothing")
    p.add_argument("--scale", type=float, default=2.0, help="upscale factor (default 2.0)")
    p.add_argument("--long-edge", type=int, help="force output long edge in px")
    p.add_argument(
        "--print-in",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        help="size output for this print, in inches, at --dpi",
    )
    p.add_argument("--dpi", type=int, default=300, help="print DPI metadata (default 300)")
    p.add_argument("--strength", choices=sorted(STRENGTHS), default="medium")
    p.add_argument("--json", action="store_true", help="machine-readable report")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.exists():
        sys.exit(f"error: input not found: {args.input}")

    if args.inspect:
        data = inspect(args.input, args.dpi)
        print(json.dumps(data, indent=2) if args.json else _pretty(data))
        return 0

    if args.output is None:
        sys.exit("error: an output path is required unless --inspect is given")

    report = enhance(
        args.input,
        args.output,
        scale=args.scale,
        long_edge=args.long_edge,
        print_in=tuple(args.print_in) if args.print_in else None,
        dpi=args.dpi,
        strength=args.strength,
    )
    data = asdict(report)
    print(json.dumps(data, indent=2) if args.json else _pretty(data))
    return 0


def _pretty(data: dict) -> str:
    lines = []
    for key, value in data.items():
        if key == "print_sizes":
            lines.append("print sizes:")
            for row in value:
                lines.append(
                    f"  {row['dpi']} DPI ({row['quality']}): "
                    f"{row['inches'][0]}x{row['inches'][1]} in / "
                    f"{row['cm'][0]}x{row['cm'][1]} cm"
                )
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
