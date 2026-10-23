"""Physical layout of a v2 ring marker.

Everything here is in millimetres. The pipeline rectifies the camera feed
to a metric top-down view before detection, so millimetres -- not pixels --
are the natural unit for the marker definition, and the same geometry
object drives both the printable artwork and the decoder. In v1 those two
were defined separately and silently drifted apart, which is how an entire
print run ended up mirrored relative to what the decoder expected.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RingGeometry:
    """Radii are measured from the centre of the token base.

    Layout, outward from the centre:

        0            -> clear_r    miniature keep-out, nothing printed
        clear_r      -> code_inner white guard ring
        code_inner   -> code_outer the angular code band
        code_outer   -> det_inner  white gap
        det_inner    -> r_out      solid black detection ring

    The solid detection ring is the geometric anchor. Its INNER edge is
    what the detector locks onto, because that edge has printed white on
    one side and printed ink on the other, so its contrast is fixed by the
    artwork and is independent of the table surface, the lighting, or
    whatever the token happens to be sitting on. The outer silhouette is
    deliberately not used for that: its contrast depends on the background
    and therefore changes from table to table.
    """
    base_d_mm: float = 40.0
    clear_frac: float = 0.70          # fraction of DIAMETER kept clear
    det_ring_mm: float = 1.5
    gap_mm: float = 0.5
    guard_mm: float = 0.5
    n_chips: int = 20

    @property
    def r_out(self) -> float:
        return self.base_d_mm / 2.0

    @property
    def clear_r(self) -> float:
        return self.r_out * self.clear_frac

    @property
    def det_inner(self) -> float:
        return self.r_out - self.det_ring_mm

    @property
    def code_outer(self) -> float:
        return self.det_inner - self.gap_mm

    @property
    def code_inner(self) -> float:
        return self.clear_r + self.guard_mm

    @property
    def code_band_mm(self) -> float:
        return self.code_outer - self.code_inner

    def validate(self) -> None:
        if self.code_band_mm <= 0:
            raise ValueError(
                f"no code band left: code_inner={self.code_inner:.2f}mm >= "
                f"code_outer={self.code_outer:.2f}mm. Reduce clear_frac, "
                f"det_ring_mm, gap_mm or guard_mm, or use a larger base.")
        if self.code_band_mm < 1.5:
            raise ValueError(
                f"code band is only {self.code_band_mm:.2f}mm wide. Measured "
                f"behaviour below ~1.5mm is not just degraded but unsafe: it "
                f"starts producing confident WRONG ids rather than refusing.")

    def describe(self) -> str:
        return (f"base {self.base_d_mm:.0f}mm | clear <{self.clear_r:.1f}mm | "
                f"code {self.code_inner:.1f}-{self.code_outer:.1f}mm "
                f"({self.code_band_mm:.1f}mm, {self.n_chips} chips) | "
                f"det ring {self.det_inner:.1f}-{self.r_out:.1f}mm")

    # --- pixel helpers -------------------------------------------------
    def px(self, px_per_mm: float) -> dict:
        """All radii in pixels for a given metric scale."""
        return {
            "r_out": self.r_out * px_per_mm,
            "det_inner": self.det_inner * px_per_mm,
            "code_outer": self.code_outer * px_per_mm,
            "code_inner": self.code_inner * px_per_mm,
            "clear_r": self.clear_r * px_per_mm,
        }

    def min_px_per_mm(self) -> float:
        """Below roughly this scale the code band stops being resolvable.

        Derived from the sweep in docs/VALIDATION.md: decoding held at 100%
        down to 2.4 px/mm (an 8.4px band) and the band needs a handful of
        pixels across to survive radial averaging, so this is the point to
        warn the operator at rather than a hard cliff.
        """
        return 8.0 / self.code_band_mm
