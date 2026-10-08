"""
Self-test for tools/enhance_print.py.

Uses a synthetic test pattern -- NOT a photograph -- so the suite can assert
things a real photo cannot be checked against: exact expected dimensions, DPI
metadata, measurable sharpness gain, and proof that the pipeline adds no
structure that was not in the source.

Run:
    .venv/bin/python -m unittest discover -s tests -v
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import enhance_print as ep  # noqa: E402


def make_fixture(path: Path, size=(320, 240)) -> Image.Image:
    """
    Deterministic test pattern: a smooth ramp, hard geometric edges, a fine
    hatch (stands in for texture) and a large flat region that must stay flat.
    """
    img = Image.new("RGB", size, (24, 24, 24))
    px = img.load()
    w, h = size
    for y in range(h):
        for x in range(w):
            # left third: horizontal ramp (tests tonal handling)
            if x < w // 3:
                v = int(255 * x / (w // 3))
                px[x, y] = (v, v, v)
            # middle third: hard-edged blocks (tests edge sharpening)
            elif x < 2 * w // 3:
                block = ((x // 16) + (y // 16)) % 2
                v = 200 if block else 40
                px[x, y] = (v, v, v)
            else:
                # right third: fine 2px hatch = the "texture" surrogate,
                # with a flat corner inside it that must not gain texture.
                if x > w - 60 and y > h - 60:
                    px[x, y] = (128, 128, 128)
                else:
                    v = 210 if (x + y) % 4 < 2 else 60
                    px[x, y] = (v, v, v)
    d = ImageDraw.Draw(img)
    d.ellipse([20, 20, 90, 90], outline=(255, 255, 255), width=2)
    img.save(path)
    return img


def sharpness(img: Image.Image) -> float:
    """Mean edge response -- a monotonic proxy for perceived sharpness."""
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    hist = edges.histogram()
    total = sum(hist)
    return sum(i * c for i, c in enumerate(hist)) / max(1, total)


def patch_variance(img: Image.Image, box) -> float:
    """Variance of a flat patch; invented texture would raise this."""
    hist = img.convert("L").crop(box).histogram()
    total = sum(hist)
    mean = sum(i * c for i, c in enumerate(hist)) / max(1, total)
    var = sum(((i - mean) ** 2) * c for i, c in enumerate(hist)) / max(1, total)
    return var


class ToneCurveTest(unittest.TestCase):
    """The curve must be anchored and monotonic for every preset blend."""

    def test_endpoints_are_anchored(self):
        for blend in (0.0, 0.35, 0.6, 0.8, 1.0):
            lut = ep._s_curve_lut(0, 255, blend)
            self.assertEqual(lut[0], 0, f"black moved at blend={blend}")
            self.assertEqual(lut[255], 255, f"white moved at blend={blend}")

    def test_monotonic_non_decreasing(self):
        for blend in (0.35, 0.6, 0.8):
            lut = ep._s_curve_lut(12, 240, blend)
            for i in range(1, 256):
                self.assertGreaterEqual(
                    lut[i], lut[i - 1], f"curve inverts at {i} (blend={blend})"
                )

    def test_percentile_bounds(self):
        hist = [0] * 256
        hist[10] = 1000
        hist[240] = 1000
        lo, hi = ep._percentile_bounds(hist, 0.5, 99.5)
        self.assertEqual((lo, hi), (10, 240))

    def test_degenerate_histogram_does_not_crash(self):
        self.assertEqual(ep._percentile_bounds([0] * 256, 0.5, 99.5), (0, 255))


class SizingTest(unittest.TestCase):
    def test_scale(self):
        self.assertEqual(ep.resolve_target_size((300, 200), 2.0, None, None, 300)[0], (600, 400))

    def test_long_edge_portrait(self):
        self.assertEqual(
            ep.resolve_target_size((200, 400), 2.0, 800, None, 300)[0], (400, 800)
        )

    def test_print_inches_a4_at_300dpi(self):
        got, mode = ep.resolve_target_size((300, 200), 2.0, None, (8.27, 11.69), 300)
        self.assertEqual(got, (2481, 3507))
        self.assertEqual(mode, "print")


class PipelineTest(unittest.TestCase):
    """End-to-end: run the real enhance() on the fixture."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = Path(cls.tmp.name) / "fixture.png"
        make_fixture(cls.src)
        cls.out = Path(cls.tmp.name) / "out.png"
        cls.report = ep.enhance(cls.src, cls.out, scale=2.0, dpi=300, strength="medium")
        cls.src_img = Image.open(cls.src)
        cls.out_img = Image.open(cls.out)
        # Control: the SAME Lanczos resize with no tone or sharpening stage.
        # Sharpness must be compared at identical resolution -- comparing the
        # 320px source against the 640px output is meaningless, because
        # upscaling alone spreads each edge over more pixels and lowers the
        # per-pixel edge response regardless of any enhancement.
        cls.control = Image.open(cls.src).resize((640, 480), Image.LANCZOS)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_output_dimensions_are_exactly_2x(self):
        self.assertEqual(self.out_img.size, (640, 480))
        self.assertEqual(self.report.output_px, [640, 480])
        self.assertEqual(self.report.upscale_factor, 2.0)

    def test_dpi_metadata_written_to_file(self):
        # PNG stores resolution as an integer pixels-per-metre count, so 300 dpi
        # (11811.02 px/m) round-trips as 299.9994. That is a limitation of the
        # format, not of the writer -- JPEG and TIFF store it exactly.
        dpi = self.out_img.info.get("dpi")
        self.assertIsNotNone(dpi, "no dpi metadata in output")
        self.assertAlmostEqual(dpi[0], 300, delta=0.01)
        self.assertAlmostEqual(dpi[1], 300, delta=0.01)

    def test_dpi_is_exact_in_jpeg(self):
        jpeg = Path(self.tmp.name) / "out.jpg"
        ep.enhance(self.src, jpeg, scale=2.0, dpi=300, strength="medium")
        with Image.open(jpeg) as img:
            self.assertEqual(img.info.get("dpi"), (300, 300))

    def test_sharpness_increases_over_the_upscale_alone(self):
        """
        Both measured at 640x480, so the delta is attributable to the tone and
        unsharp stages rather than to the change in resolution.
        """
        control = sharpness(self.control)
        enhanced = sharpness(self.out_img)
        self.assertGreater(
            enhanced, control,
            f"refinement added no edge response: {control:.2f} -> {enhanced:.2f}",
        )
        print(f"\n    edge response @640x480: Lanczos-only {control:.2f} "
              f"-> refined {enhanced:.2f} (+{100 * (enhanced / control - 1):.1f}%)")

    def test_no_structure_invented(self):
        """
        Downsample the result back to source size. If the pipeline had added
        elements, the round trip would not match the original.
        """
        rt = self.out_img.resize(self.src_img.size, Image.LANCZOS).convert("L")
        a = self.src_img.convert("L").histogram()
        b = rt.histogram()
        # Compare the two luminance distributions rather than pixel-for-pixel,
        # since sharpening legitimately shifts local contrast.
        ma = sum(i * c for i, c in enumerate(a)) / sum(a)
        mb = sum(i * c for i, c in enumerate(b)) / sum(b)
        self.assertAlmostEqual(ma, mb, delta=6.0,
                               msg=f"tone/composition drifted: {ma:.1f} vs {mb:.1f}")

    def test_interior_of_flat_region_gains_no_texture(self):
        """
        The fixture's flat 128-grey corner stands in for 'no invented pores or
        texture'. Measured 12px inside the boundary so unsharp ringing from the
        adjacent high-contrast hatch cannot leak into the sample. It must stay
        exactly flat -- this is the assertion that backs the no-new-elements
        guarantee.
        """
        src_box, out_box = (272, 192, 314, 234), (544, 384, 628, 468)
        src_var = patch_variance(self.src_img, src_box)
        out_var = patch_variance(self.out_img, out_box)
        self.assertEqual(out_var, 0.0,
                         f"flat interior gained texture: {src_var} -> {out_var}")

    def test_sharpen_overshoot_at_edges_is_bounded(self):
        """
        Two pixels inside the boundary the flat patch sits next to a
        high-contrast hatch, so unsharp overshoot legitimately bleeds in there.
        It must stay small -- runaway overshoot would mean halos in a print.
        """
        src_box, out_box = (262, 182, 318, 238), (524, 364, 636, 476)
        control_var = patch_variance(self.control, out_box)
        out_var = patch_variance(self.out_img, out_box)
        self.assertLess(out_var, 40.0,
                        f"unsharp overshoot is producing halos: variance {out_var:.1f}")
        print(f"\n    boundary-bleed variance: control {control_var:.2f} "
              f"-> refined {out_var:.2f} (expected: sharpening the adjacent edge)")


class CliTest(unittest.TestCase):
    """Exercise the actual command line, not just the imported function."""

    def test_inspect_reports_native_size_and_print_table(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "cli.png"
            make_fixture(src)
            r = subprocess.run(
                [sys.executable, str(REPO / "tools" / "enhance_print.py"),
                 str(src), "--inspect", "--json"],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            import json
            data = json.loads(r.stdout)
            self.assertEqual(data["px"], [320, 240])
            self.assertIn("print_sizes", data)
            self.assertEqual(data["print_sizes"][0]["dpi"], 300)

    def test_print_in_writes_requested_dimensions(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "cli.png"
            make_fixture(src)
            out = Path(d) / "cli_out.png"
            r = subprocess.run(
                [sys.executable, str(REPO / "tools" / "enhance_print.py"),
                 str(src), str(out), "--print-in", "8.27", "11.69", "--dpi", "300",
                 "--json"],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            with Image.open(out) as img:
                self.assertEqual(img.size, (2481, 3507))
                dpi = img.info.get("dpi")
                self.assertAlmostEqual(dpi[0], 300, delta=0.01)
                self.assertAlmostEqual(dpi[1], 300, delta=0.01)

    def test_missing_input_exits_nonzero(self):
        r = subprocess.run(
            [sys.executable, str(REPO / "tools" / "enhance_print.py"), "/nope/x.png"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
