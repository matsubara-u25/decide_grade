"""
instances.py

Data classes for simulating knot and concentrated-knot features of sawn lumber.

The class layout follows the uploaded instances_attribute.pdf:
- values marked with '-' in the PDF are constructor inputs,
- values marked with '/' are derived and stored after calling derive_features().

Coordinate convention
---------------------
length_* values represent positions in the lumber-length direction.
width_* values represent positions in the surface-width direction.
The constructor position values are assumed to be in image coordinates, and are
converted to millimetres using Surface.length_ratio and Surface.width_ratio.
For a pure simulation that already uses mm coordinates, set Surface.length_px to
Lumber.lumber_length_mm and Surface.width_px to Surface.width_mm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import math


KnotRegion = Literal["edge_upper", "edge_lower", "center"]
SurfaceClass = Literal["side_wide", "side_narrow"]
LumberPurpose = Literal["ko_1", "ko_2", "otsu"]
GradeLabel = Literal["1", "2", "3", "out"]


CKR_WINDOW_LENGTH_MM = 150.0
ELONGATED_RATIO_THRESHOLD = 2.5
ELONGATED_DIAMETER_FACTOR = 0.5
NORMAL_DIAMETER_FACTOR = 1.0
EPSILON_MM = 1e-6


@dataclass
class Knot:
    """One knot instance on a surface."""

    # ---- constructor inputs ----
    knot_id: str
    lumber_id: str
    surface_id: str

    length_max_pos: float
    length_min_pos: float
    width_max_pos: float
    width_min_pos: float

    long_diam_length: float
    long_diam_width: float
    short_diam_length: float
    short_diam_width: float

    center_point_length: float
    center_point_width: float

    # is_alive: bool = True  #造作用の生き/死に。構造用は不要

    # ---- derived values ----
    length_max_pos_mm: Optional[float] = None
    length_min_pos_mm: Optional[float] = None
    width_max_pos_mm: Optional[float] = None
    width_min_pos_mm: Optional[float] = None
    center_point_length_mm: Optional[float] = None
    center_point_width_mm: Optional[float] = None

    length_mm: Optional[float] = None
    width_mm: Optional[float] = None

    long_diam_length_mm: Optional[float] = None
    long_diam_width_mm: Optional[float] = None
    short_diam_length_mm: Optional[float] = None
    short_diam_width_mm: Optional[float] = None

    long_diam_mm: Optional[float] = None
    short_diam_mm: Optional[float] = None

    is_elongated: Optional[bool] = None

    # JAS diameter after the 2.5x long/short-diameter correction.
    jas_diameter: Optional[float] = None

    region: Optional[KnotRegion] = None

    # Concentrated-knot members for two candidate windows.
    # plus: window whose left end just includes this knot's upper/right length end.
    # minus: window whose right end just includes this knot's lower/left length end.
    ck_member_plus: list[str] = field(default_factory=list)
    ck_member_minus: list[str] = field(default_factory=list)
    ck_member_plus_regional: list[str] = field(default_factory=list)
    ck_member_minus_regional: list[str] = field(default_factory=list)

    # Maximum concentrated-knot diameter sums for this base knot.
    max_ck: Optional[float] = None
    max_ck_regional: Optional[float] = None

    def derive_features(
        self,
        length_ratio: float,
        width_ratio: float,
        surface_width_mm: float,
    ) -> None:
        """Derive knot-level features from input positions.

        Parameters
        ----------
        length_ratio:
            mm per input unit in the lumber-length direction.
        width_ratio:
            mm per input unit in the surface-width direction.
        surface_width_mm:
            Width of the parent surface in mm, used to classify region.
        """
        self.length_max_pos_mm = self.length_max_pos * length_ratio
        self.length_min_pos_mm = self.length_min_pos * length_ratio
        self.width_max_pos_mm = self.width_max_pos * width_ratio
        self.width_min_pos_mm = self.width_min_pos * width_ratio
        self.center_point_length_mm = self.center_point_length * length_ratio
        self.center_point_width_mm = self.center_point_width * width_ratio
        self.long_diam_length_mm = self.long_diam_length * length_ratio
        self.long_diam_width_mm = self.long_diam_width * width_ratio
        self.short_diam_length_mm = self.short_diam_length * length_ratio
        self.short_diam_width_mm = self.short_diam_width * width_ratio

        self.length_mm = max(0.0, self.length_max_pos_mm - self.length_min_pos_mm)
        self.width_mm = max(0.0, self.width_max_pos_mm - self.width_min_pos_mm)

        self._derive_long_short_diameter()  #要変更
        self._derive_jas_diameter()
        self.region = self._classify_region(surface_width_mm)

    def _derive_long_short_diameter(self) -> None:
        """Culculate long/short diameters."""
        
        long_length = self.long_diam_length_mm
        long_width = self.long_diam_width_mm
        short_length = self.short_diam_length_mm
        short_width = self.short_diam_width_mm
        self.long_diam_mm = math.sqrt(long_length ** 2 + long_width ** 2)
        self.short_diam_mm = math.sqrt(short_length ** 2 + short_width ** 2)

    def _derive_jas_diameter(self) -> None:
        """Derive JAS knot diameter.

        In this implementation, width_mm is treated as the raw JAS measurement
        direction diameter. If long_diam_mm / short_diam_mm >= 2.5, the raw
        diameter is multiplied by 0.5.
        """
        raw_diameter = self.width_mm or 0.0
        short_diam = self.short_diam_mm or 0.0
        long_diam = self.long_diam_mm or 0.0

        if short_diam <= 0.0:
            self.is_elongated = False
        else:
            self.is_elongated = (long_diam / short_diam) >= ELONGATED_RATIO_THRESHOLD

        factor = ELONGATED_DIAMETER_FACTOR if self.is_elongated else NORMAL_DIAMETER_FACTOR
        self.jas_diameter = raw_diameter * factor

    def _classify_region(self, surface_width_mm: float) -> KnotRegion:
        """Classify the knot into lower edge, upper edge, or center.

        edge_lower is the low-coordinate edge region, and edge_upper is the
        high-coordinate edge region in the surface-width direction.
        """
        center_width = self.center_point_width_mm
        if center_width is None:
            raise ValueError("center_point_width_mm must be derived before region classification")

        lower_boundary = surface_width_mm * 0.25
        upper_boundary = surface_width_mm * 0.75

        if center_width <= lower_boundary:
            return "edge_lower"
        if center_width >= upper_boundary:
            return "edge_upper"
        return "center"

    def get_diameter(self) -> float:
        if self.jas_diameter is None:
            raise ValueError("jas_diameter has not been derived yet")
        return self.jas_diameter

    def get_region(self) -> KnotRegion:
        if self.region is None:
            raise ValueError("region has not been derived yet")
        return self.region

    def get_length_max_pos_mm(self) -> float:
        if self.length_max_pos_mm is None:
            raise ValueError("length_max_pos_mm has not been derived yet")
        return self.length_max_pos_mm

    def get_length_min_pos_mm(self) -> float:
        if self.length_min_pos_mm is None:
            raise ValueError("length_min_pos_mm has not been derived yet")
        return self.length_min_pos_mm

    def get_width_max_pos_mm(self) -> float:
        if self.width_max_pos_mm is None:
            raise ValueError("width_max_pos_mm has not been derived yet")
        return self.width_max_pos_mm

    def get_width_min_pos_mm(self) -> float:
        if self.width_min_pos_mm is None:
            raise ValueError("width_min_pos_mm has not been derived yet")
        return self.width_min_pos_mm


@dataclass
class SideSurface:
    """One surface instance of a lumber."""

    # ---- constructor inputs ----
    surface_id: str
    lumber_id: str
    width_mm: float
    length_px: float
    width_px: float
    surface_class: SurfaceClass = "side_wide"
    knots: list[Knot] = field(default_factory=list)

    # ---- derived values ----
    length_ratio: Optional[float] = None
    width_ratio: Optional[float] = None
    window_length: float = CKR_WINDOW_LENGTH_MM

    max_knot_id: Optional[str] = None
    max_edge_knot_id: Optional[str] = None
    max_center_knot_id: Optional[str] = None

    max_knot_ratio: Optional[float] = None
    max_edge_knot_ratio: Optional[float] = None
    max_center_knot_ratio: Optional[float] = None

    max_ck_member: list[str] = field(default_factory=list)
    max_edge_ck_member: list[str] = field(default_factory=list)
    max_center_ck_member: list[str] = field(default_factory=list)
    max_ckr: Optional[float] = None
    max_edge_ckr: Optional[float] = None
    max_center_ckr: Optional[float] = None

    def derive_features(self, lumber_length_mm: float) -> None:
        """Derive surface-level features.

        Parameters
        ----------
        lumber_length_mm:
            Length of the parent lumber. It is used instead of storing the same
            length redundantly in each Surface.
        """
        self._derive_scale(lumber_length_mm)
        self._derive_knot_features()
        self._derive_knot_ratio_features()
        self._derive_concentrated_knot_features()

    def _derive_scale(self, lumber_length_mm: float) -> None:
        if self.length_px <= 0:
            raise ValueError("length_px must be positive")
        if self.width_px <= 0:
            raise ValueError("width_px must be positive")

        self.length_ratio = lumber_length_mm / self.length_px
        self.width_ratio = self.width_mm / self.width_px

    def _derive_knot_features(self) -> None:
        if self.length_ratio is None or self.width_ratio is None:
            raise ValueError("length_ratio and width_ratio must be derived first")

        for knot in self.knots:
            knot.derive_features(
                length_ratio=self.length_ratio,
                width_ratio=self.width_ratio,
                surface_width_mm=self.width_mm,
            )

    def _derive_knot_ratio_features(self) -> None:
        max_all = (0.0, None)
        max_edge = (0.0, None)
        max_center = (0.0, None)

        for knot in self.knots:
            if knot.jas_diameter is None or knot.region is None:
                continue

            ratio = 100.0 * knot.jas_diameter / self.width_mm
            if not ratio == 100.0:
                if ratio > max_all[0]:
                    max_all = (ratio, knot.knot_id)

                if knot.region in ("edge_upper", "edge_lower") and ratio > max_edge[0]:
                    max_edge = (ratio, knot.knot_id)

                if knot.region == "center" and ratio > max_center[0]:
                    max_center = (ratio, knot.knot_id)

        self.max_knot_ratio, self.max_knot_id = max_all
        self.max_edge_knot_ratio, self.max_edge_knot_id = max_edge
        self.max_center_knot_ratio, self.max_center_knot_id = max_center

    def _derive_concentrated_knot_features(self) -> None:
        max_all_ck_sum = 0.0
        max_all_ck_member: list[str] = []
        max_edge_ck_member: list[str] = []
        max_center_ck_member: list[str] = []
        max_edge_ck_sum = 0.0
        max_center_ck_sum = 0.0

        for base_knot in self.knots:
            
            plus_window = self._make_plus_window(base_knot)
            minus_window = self._make_minus_window(base_knot)

            plus_sum, plus_members = self._sum_window_members(plus_window)
            minus_sum, minus_members = self._sum_window_members(minus_window)

            base_knot.ck_member_plus = plus_members
            base_knot.ck_member_minus = minus_members
            base_knot.max_ck = max(plus_sum, minus_sum)

            if plus_sum >= minus_sum:
                current_members = plus_members
            else:
                current_members = minus_members

            if base_knot.max_ck > max_all_ck_sum:
                max_all_ck_sum = base_knot.max_ck
                max_all_ck_member = current_members

            plus_regional_sum, plus_regional_members = self._sum_window_members(
                plus_window,
                region=base_knot.region,
            )
            minus_regional_sum, minus_regional_members = self._sum_window_members(
                minus_window,
                region=base_knot.region,
            )

            base_knot.ck_member_plus_regional = plus_regional_members
            base_knot.ck_member_minus_regional = minus_regional_members
            base_knot.max_ck_regional = max(plus_regional_sum, minus_regional_sum)

            if plus_regional_sum >= minus_regional_sum:
                current__regional_members = plus_regional_members
            else:
                current_regional_members = minus_regional_members

            if base_knot.region in ("edge_upper", "edge_lower"):
                if base_knot.max_ck_regional > max_edge_ck_sum:
                    max_edge_ck_sum = base_knot.max_ck_regional
                    max_edge_ck_member = current_regional_members
            elif base_knot.region == "center":
                if base_knot.max_ck_regional > max_edge_ck_sum:
                    max_center_ck_sum = base_knot.max_ck_regional
                    max_center_ck_member = current_regional_members

        self.max_ck_member = max_all_ck_member
        self.max_edge_ck_member = max_edge_ck_member
        self.max_center_ck_member = max_center_ck_member
        self.max_ckr = 100.0 * max_all_ck_sum / self.width_mm
        self.max_edge_ckr = 100.0 * max_edge_ck_sum / self.width_mm
        self.max_center_ckr = 100.0 * max_center_ck_sum / self.width_mm

    def _make_plus_window(self, knot: Knot) -> tuple[float, float]:
        """Window whose left end just includes knot.length_max_pos_mm."""
        if knot.length_max_pos_mm is None:
            raise ValueError("knot length positions must be derived first")

        lumber_length_mm = self.length_px * (self.length_ratio or 0.0)
        max_start = max(lumber_length_mm - self.window_length, 0.0)
        start = min(max(knot.length_max_pos_mm - EPSILON_MM, 0.0), max_start)
        end = start + self.window_length
        return start, end

    def _make_minus_window(self, knot: Knot) -> tuple[float, float]:
        """Window whose right end just includes knot.length_min_pos_mm."""
        if knot.length_min_pos_mm is None:
            raise ValueError("knot length positions must be derived first")

        lumber_length_mm = self.length_px * (self.length_ratio or 0.0)
        max_start = max(lumber_length_mm - self.window_length, 0.0)
        start = min(max(knot.length_min_pos_mm - self.window_length + EPSILON_MM, 0.0), max_start)
        end = start + self.window_length
        return start, end

    def _sum_window_members(
        self,
        window: tuple[float, float],
        region: Optional[KnotRegion] = None,
    ) -> tuple[float, list[str]]:
        """Sum JAS diameters of knots overlapping a window.

        If region is specified, only knots in the same region are included.
        This treats edge_upper and edge_lower as separate regions. If later you
        decide to combine both edges, change this region filter.
        """
        start, end = window
        diameter_sum = 0.0
        member_ids: list[str] = []

        for knot in self.knots:
            if knot.jas_diameter is None:
                continue
            if knot.length_min_pos_mm is None or knot.length_max_pos_mm is None:
                continue
            if region is not None and knot.region != region:
                continue

            overlaps = knot.length_max_pos_mm >= start and knot.length_min_pos_mm <= end
            if overlaps:
                diameter_sum += knot.jas_diameter
                member_ids.append(knot.knot_id)

        return diameter_sum, member_ids

    def get_max_knot_ratio(self) -> float:
        return self.max_knot_ratio or 0.0

    def get_edge_max_kr(self) -> float:
        return self.max_edge_knot_ratio or 0.0

    def get_center_max_kr(self) -> float:
        return self.max_center_knot_ratio or 0.0

    def get_max_ckr(self) -> float:
        return self.max_ckr or 0.0

    def get_edge_max_ckr(self) -> float:
        return self.max_edge_ckr or 0.0

    def get_center_max_ckr(self) -> float:
        return self.max_center_ckr or 0.0


@dataclass
class Lumber:
    """One lumber instance containing multiple surfaces."""

    # ---- constructor inputs ----
    lumber_id: str
    lumber_length_mm: float
    lumber_purpose: LumberPurpose
    side_surfaces: list[SideSurface] = field(default_factory=list)

    # ---- derived values ----
    wide_max_kr_surface_id: Optional[str] = None
    wide_edge_max_kr_surface_id: Optional[str] = None
    wide_center_max_kr_surface_id: Optional[str] = None
    narrow_max_kr_surface_id: Optional[str] = None

    wide_max_kr: Optional[float] = None
    wide_edge_max_kr: Optional[float] = None
    wide_center_max_kr: Optional[float] = None
    narrow_max_kr: Optional[float] = None

    wide_max_ckr_surface_id: Optional[str] = None
    wide_edge_max_ckr_surface_id: Optional[str] = None
    wide_center_max_ckr_surface_id: Optional[str] = None
    narrow_max_ckr_surface_id: Optional[str] = None

    wide_max_ckr: Optional[float] = None
    wide_edge_max_ckr: Optional[float] = None
    wide_center_max_ckr: Optional[float] = None
    narrow_max_ckr: Optional[float] = None

    use_narrow_surface_rules: bool = True
    use_regional_rules: bool = True

    grade: Optional[GradeLabel] = None

    def derive_features(self) -> None:
        self._assign_surface_classes()

        for surface in self.side_surfaces:
            surface.derive_features(self.lumber_length_mm)

        self._derive_lumber_max_features()
    #     self.select_features()

    def _assign_surface_classes(self) -> None:
        if not self.surfaces:
            return

        widths = [surface.width_mm for surface in self.surfaces]

        max_width = max(widths)
        min_width = min(widths)

        if max_width == min_width:
            for surface in self.surfaces:
                surface.surface_class = "side_wide"
            return

        for surface in self.surfaces:
            if surface.width_mm == max_width:
                surface.surface_class = "side_wide"
            elif surface.width_mm == min_width:
                surface.surface_class = "side_narrow"
            else:
                raise ValueError(
                    f"Unexpected surface width: {surface.width_mm}. "
                    f"Only two side widths are expected: {min_width} and {max_width}."
                )

    def _derive_lumber_max_features() -> None:
        wide_surfaces = [
            surface for surface in self.surfaces
            if surface.surface_class == "side_wide"
        ]

        narrow_surfaces = [
            surface for surface in self.surfaces
            if surface.surface_class == "side_narrow"
        ]

        if not self.use_narrow_surface_rules:
            narrow_surfaces = []

        self.wide_max_kr, self.wide_max_kr_surface_id = self._max_surface_attr(
            wide_surfaces,
            "max_knot_ratio",
        )

        self.wide_edge_max_kr, self.wide_edge_max_kr_surface_id = self._max_surface_attr(
            wide_surfaces,
            "max_edge_knot_ratio",
        )

        self.wide_center_max_kr, self.wide_center_max_kr_surface_id = self._max_surface_attr(
            wide_surfaces,
            "max_center_knot_ratio",
        )

        self.narrow_max_kr, self.narrow_max_kr_surface_id = self._max_surface_attr(
            narrow_surfaces,
            "max_knot_ratio",
        )

        self.wide_max_ckr, self.wide_max_ckr_surface_id = self._max_surface_attr(
            wide_surfaces,
            "max_ckr",
        )

        self.wide_edge_max_ckr, self.wide_edge_max_ckr_surface_id = self._max_surface_attr(
            wide_surfaces,
            "max_edge_ckr",
        )

        self.wide_center_max_ckr, self.wide_center_max_ckr_surface_id = self._max_surface_attr(
            wide_surfaces,
            "max_center_ckr",
        )

        self.narrow_max_ckr, self.narrow_max_ckr_surface_id = self._max_surface_attr(
            narrow_surfaces,
            "max_ckr",
        )

    def _max_surface_attr(
        self,
        surfaces: list[SideSurface],
        attr_name: str,
    ) -> tuple[float, Optional[str]]:
        """
        指定した材面属性の最大値と、その surface_id を返す。
        """
        max_value = 0.0
        max_surface_id = None

        for surface in surfaces:
            value = getattr(surface, attr_name)

            if value is None:
                continue

            if value > max_value:
                max_value = value
                max_surface_id = surface.surface_id

        return max_value, max_surface_id

    # def select_features(self) -> None:
    #     wide_surfaces = [s for s in self.surfaces if s.surface_class == "side_wide"]
    #     narrow_surfaces = [s for s in self.surfaces if s.surface_class == "side_narrow"]

    #     if not self.use_narrow_surface_rules:
    #         narrow_surfaces = []

    #     self.wide_max_kr, self.wide_max_kr_surface_id = self._max_surface_attr(
    #         wide_surfaces,
    #         "max_knot_ratio",
    #     )
    #     self.wide_edge_max_kr, self.wide_edge_max_kr_surface_id = self._max_surface_attr(
    #         wide_surfaces,
    #         "max_edge_knot_ratio",
    #     )
    #     self.wide_center_max_kr, self.wide_center_max_kr_surface_id = self._max_surface_attr(
    #         wide_surfaces,
    #         "max_center_knot_ratio",
    #     )
    #     self.narrow_max_kr, self.narrow_max_kr_surface_id = self._max_surface_attr(
    #         narrow_surfaces,
    #         "max_knot_ratio",
    #     )

    #     self.wide_max_ckr, self.wide_max_ckr_surface_id = self._max_surface_attr(
    #         wide_surfaces,
    #         "max_ckr",
    #     )
    #     self.wide_edge_max_ckr, self.wide_edge_max_ckr_surface_id = self._max_surface_attr(
    #         wide_surfaces,
    #         "max_edge_ckr",
    #     )
    #     self.wide_center_max_ckr, self.wide_center_max_ckr_surface_id = self._max_surface_attr(
    #         wide_surfaces,
    #         "max_center_ckr",
    #     )
    #     self.narrow_max_ckr, self.narrow_max_ckr_surface_id = self._max_surface_attr(
    #         narrow_surfaces,
    #         "max_ckr",
    #     )

    def _max_surface_attr(
        self,
        surfaces: Sequence[SideSurface],
        attr_name: str,
    ) -> tuple[float, Optional[str]]:
        max_value = 0.0
        max_surface_id: Optional[str] = None

        for surface in surfaces:
            value = getattr(surface, attr_name)
            if value is None:
                continue
            if value > max_value:
                max_value = float(value)
                max_surface_id = surface.surface_id

        return max_value, max_surface_id

    def decide_grade(
        self,
        rules: Optional[dict[str, tuple[float, float, float]]] = None,
    ) -> Optional[GradeLabel]:
        """Decide grade from derived features.

        Parameters
        ----------
        rules:
            Optional threshold dictionary. Keys should correspond to feature
            names such as 'wide_edge_max_kr' or 'narrow_max_ckr'. Each value is
            a tuple of thresholds for grade 1, grade 2, and grade 3.

        Returns
        -------
        Optional grade label. If rules is None, features are derived but grade
        is left as None.
        """
        self.derive_features()

        if rules is None:
            self.grade = None
            return self.grade

        component_grades: list[GradeLabel] = []
        for feature_name, thresholds in rules.items():
            value = getattr(self, feature_name)
            if value is None:
                continue
            component_grades.append(_judge_leq_threshold(float(value), thresholds))

        self.grade = _worst_grade(component_grades) if component_grades else None
        return self.grade


def _judge_leq_threshold(value: float, thresholds: tuple[float, float, float]) -> GradeLabel:
    grade_1, grade_2, grade_3 = thresholds
    if value <= grade_1:
        return "1"
    if value <= grade_2:
        return "2"
    if value <= grade_3:
        return "3"
    return "out"


def _worst_grade(grades: Sequence[GradeLabel]) -> GradeLabel:
    rank = {"1": 1, "2": 2, "3": 3, "out": 4}
    return max(grades, key=lambda grade: rank[grade])
