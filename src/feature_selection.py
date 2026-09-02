# feature_selection.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


try:
    from instances import Lumber
except ImportError:
    from .instances import Lumber


JASClass = str
FeatureName = str


SHORT_SIDE_THRESHOLD_MM = 36.0
DIMENSION_TOLERANCE_MM = 1e-6


LUMBER_PURPOSE_TO_JAS_CLASS: dict[str, JASClass] = {
    "ko_1": "甲Ⅰ",
    "ko_2": "甲Ⅱ",
    "otsu": "乙",
}


@dataclass(frozen=True)
class SectionDimensions:
    """Cross-section dimensions estimated from side-surface widths."""

    short_side_mm: float
    long_side_mm: float
    is_square: bool
    has_narrow_surface: bool


@dataclass(frozen=True)
class JudgmentFeature:
    """A selected feature used for JAS grade judgment.

    This corresponds to one row in the lumber judgment feature table.
    """

    lumber_id: str
    lumber_purpose: str
    jas_class: JASClass

    # This should match the feature name used in the JAS rule table.
    feature_name: FeatureName

    value: float
    unit: str = "%"

    # Source feature names from the Lumber object.
    source_features: tuple[str, ...] = field(default_factory=tuple)

    # Source values before selection.
    source_values: dict[str, float] = field(default_factory=dict)

    # Feature actually selected when max() or conditional selection is used.
    selected_source_feature: Optional[str] = None

    # Explanation of why this feature was selected.
    selection_reason: str = ""


class FeatureSelectionError(Exception):
    """Raised when judgment features cannot be selected."""
    pass


def select_judgment_features(
    lumber: Lumber,
    *,
    derive: bool = True,
) -> list[JudgmentFeature]:
    """Select judgment features from a Lumber object.

    Parameters
    ----------
    lumber:
        Lumber instance whose derived features are used.

    derive:
        If True, call lumber.derive_features() before selecting features.
        Set False if derived features have already been calculated.

    Returns
    -------
    list[JudgmentFeature]
        Selected judgment features for the lumber purpose and dimensions.
    """
    if derive:
        lumber.derive_features()

    jas_class = _get_jas_class(lumber)
    section = _get_section_dimensions(lumber)

    if jas_class == "甲Ⅰ":
        return _select_ko1_features(lumber, jas_class, section)

    if jas_class == "甲Ⅱ":
        return _select_ko2_features(lumber, jas_class, section)

    if jas_class == "乙":
        return _select_otsu_features(lumber, jas_class, section)

    raise FeatureSelectionError(f"Unsupported JAS class: {jas_class}")


def _select_ko1_features(
    lumber: Lumber,
    jas_class: JASClass,
    section: SectionDimensions,
) -> list[JudgmentFeature]:
    """Select judgment features for 甲Ⅰ."""
    return [
        _select_max_knot_ratio_for_ko1_or_otsu(
            lumber=lumber,
            jas_class=jas_class,
            section=section,
        ),
        _select_max_ckr_for_ko1_or_otsu(
            lumber=lumber,
            jas_class=jas_class,
            section=section,
        ),
    ]


def _select_otsu_features(
    lumber: Lumber,
    jas_class: JASClass,
    section: SectionDimensions,
) -> list[JudgmentFeature]:
    """Select judgment features for 乙."""
    return [
        _select_max_knot_ratio_for_ko1_or_otsu(
            lumber=lumber,
            jas_class=jas_class,
            section=section,
        ),
        _select_max_ckr_for_ko1_or_otsu(
            lumber=lumber,
            jas_class=jas_class,
            section=section,
        ),
    ]


def _select_ko2_features(
    lumber: Lumber,
    jas_class: JASClass,
    section: SectionDimensions,
) -> list[JudgmentFeature]:
    """Select judgment features for 甲Ⅱ.

    甲Ⅱでは、狭い材面、広い材面の材縁部、広い材面の中央部を分ける。
    正方形断面などで狭い材面がない場合は、狭い材面項目をスキップする。
    """
    features: list[JudgmentFeature] = []

    if section.has_narrow_surface:
        features.append(
            _make_direct_feature(
                lumber=lumber,
                jas_class=jas_class,
                feature_name="狭い材面の節",
                source_attr="narrow_max_kr",
                selection_reason="甲Ⅱで狭い材面が存在するため、狭い材面の最大節径比を参照する。",
            )
        )
        features.append(
            _make_direct_feature(
                lumber=lumber,
                jas_class=jas_class,
                feature_name="狭い材面の集中節",
                source_attr="narrow_max_ckr",
                selection_reason="甲Ⅱで狭い材面が存在するため、狭い材面の集中節径比を参照する。",
            )
        )

    features.extend(
        [
            _make_direct_feature(
                lumber=lumber,
                jas_class=jas_class,
                feature_name="広い材面材縁部の節",
                source_attr="wide_edge_max_kr",
                selection_reason="甲Ⅱでは広い材面の材縁部の節を参照する。",
            ),
            _make_direct_feature(
                lumber=lumber,
                jas_class=jas_class,
                feature_name="広い材面中央部の節",
                source_attr="wide_center_max_kr",
                selection_reason="甲Ⅱでは広い材面の中央部の節を参照する。",
            ),
            _make_direct_feature(
                lumber=lumber,
                jas_class=jas_class,
                feature_name="広い材面材縁部の集中節",
                source_attr="wide_edge_max_ckr",
                selection_reason="甲Ⅱでは広い材面の材縁部の集中節径比を参照する。",
            ),
            _make_direct_feature(
                lumber=lumber,
                jas_class=jas_class,
                feature_name="広い材面中央部の集中節",
                source_attr="wide_center_max_ckr",
                selection_reason="甲Ⅱでは広い材面の中央部の集中節径比を参照する。",
            ),
        ]
    )

    return features


def _select_max_knot_ratio_for_ko1_or_otsu(
    *,
    lumber: Lumber,
    jas_class: JASClass,
    section: SectionDimensions,
) -> JudgmentFeature:
    """Select max knot ratio for 甲Ⅰ or 乙 with dimension-dependent scope."""
    if section.short_side_mm < SHORT_SIDE_THRESHOLD_MM:
        return _make_direct_feature(
            lumber=lumber,
            jas_class=jas_class,
            feature_name="節",
            source_attr="wide_max_kr",
            selection_reason=(
                f"{jas_class}で木口短辺が{SHORT_SIDE_THRESHOLD_MM:g}mm未満のため、"
                "広い材面の最大節径比のみを参照する。"
            ),
        )

    return _make_max_feature(
        lumber=lumber,
        jas_class=jas_class,
        feature_name="節",
        source_attrs=("wide_max_kr", "narrow_max_kr"),
        selection_reason=(
            f"{jas_class}で木口短辺が{SHORT_SIDE_THRESHOLD_MM:g}mm以上のため、"
            "広い材面と狭い材面を含むすべての材面の最大値を参照する。"
        ),
    )


def _select_max_ckr_for_ko1_or_otsu(
    *,
    lumber: Lumber,
    jas_class: JASClass,
    section: SectionDimensions,
) -> JudgmentFeature:
    """Select max concentrated-knot ratio for 甲Ⅰ or 乙.

    ここでは最大節径比と同じ寸法条件を集中節径比にも適用している。
    もし集中節径比について別の寸法条件を使う場合は、この関数だけ修正する。
    """
    if section.short_side_mm < SHORT_SIDE_THRESHOLD_MM:
        return _make_direct_feature(
            lumber=lumber,
            jas_class=jas_class,
            feature_name="集中節",
            source_attr="wide_max_ckr",
            selection_reason=(
                f"{jas_class}で木口短辺が{SHORT_SIDE_THRESHOLD_MM:g}mm未満のため、"
                "広い材面の集中節径比のみを参照する。"
            ),
        )

    return _make_max_feature(
        lumber=lumber,
        jas_class=jas_class,
        feature_name="集中節",
        source_attrs=("wide_max_ckr", "narrow_max_ckr"),
        selection_reason=(
            f"{jas_class}で木口短辺が{SHORT_SIDE_THRESHOLD_MM:g}mm以上のため、"
            "広い材面と狭い材面を含むすべての材面の最大集中節径比を参照する。"
        ),
    )


def _make_direct_feature(
    *,
    lumber: Lumber,
    jas_class: JASClass,
    feature_name: FeatureName,
    source_attr: str,
    selection_reason: str,
) -> JudgmentFeature:
    """Create a judgment feature directly from one Lumber attribute."""
    value = _get_float_attr(lumber, source_attr)

    return JudgmentFeature(
        lumber_id=lumber.lumber_id,
        lumber_purpose=lumber.lumber_purpose,
        jas_class=jas_class,
        feature_name=feature_name,
        value=value,
        source_features=(source_attr,),
        source_values={source_attr: value},
        selected_source_feature=source_attr,
        selection_reason=selection_reason,
    )


def _make_max_feature(
    *,
    lumber: Lumber,
    jas_class: JASClass,
    feature_name: FeatureName,
    source_attrs: tuple[str, ...],
    selection_reason: str,
) -> JudgmentFeature:
    """Create a judgment feature by taking max() of multiple Lumber attributes."""
    source_values = {
        attr_name: _get_float_attr(lumber, attr_name)
        for attr_name in source_attrs
    }

    selected_source_feature = max(source_values, key=source_values.get)
    value = source_values[selected_source_feature]

    return JudgmentFeature(
        lumber_id=lumber.lumber_id,
        lumber_purpose=lumber.lumber_purpose,
        jas_class=jas_class,
        feature_name=feature_name,
        value=value,
        source_features=source_attrs,
        source_values=source_values,
        selected_source_feature=selected_source_feature,
        selection_reason=selection_reason,
    )


def _get_jas_class(lumber: Lumber) -> JASClass:
    """Return JAS class name from lumber.lumber_purpose."""
    try:
        return LUMBER_PURPOSE_TO_JAS_CLASS[lumber.lumber_purpose]
    except KeyError as error:
        raise FeatureSelectionError(
            f"Unknown lumber_purpose: {lumber.lumber_purpose}"
        ) from error


def _get_section_dimensions(lumber: Lumber) -> SectionDimensions:
    """Estimate cross-section dimensions from side-surface widths."""
    if not lumber.side_surfaces:
        raise FeatureSelectionError("lumber.side_surfaces is empty")

    widths = [float(surface.width_mm) for surface in lumber.side_surfaces]

    short_side_mm = min(widths)
    long_side_mm = max(widths)

    is_square = abs(long_side_mm - short_side_mm) <= DIMENSION_TOLERANCE_MM

    has_narrow_surface = any(
        surface.surface_class == "side_narrow"
        for surface in lumber.side_surfaces
    )

    return SectionDimensions(
        short_side_mm=short_side_mm,
        long_side_mm=long_side_mm,
        is_square=is_square,
        has_narrow_surface=has_narrow_surface,
    )


def _get_float_attr(lumber: Lumber, attr_name: str) -> float:
    """Read a numeric attribute from Lumber.

    None is treated as 0.0 because non-applicable values should not worsen
    the grade by themselves.
    """
    if not hasattr(lumber, attr_name):
        raise FeatureSelectionError(
            f"Lumber does not have feature attribute: {attr_name}"
        )

    value = getattr(lumber, attr_name)

    if value is None:
        return 0.0

    return float(value)


def judgment_features_to_rows(
    judgment_features: list[JudgmentFeature],
) -> list[dict[str, object]]:
    """Convert selected judgment features to table-like rows."""
    rows: list[dict[str, object]] = []

    for feature in judgment_features:
        rows.append(
            {
                "lumber_id": feature.lumber_id,
                "lumber_purpose": feature.lumber_purpose,
                "jas_class": feature.jas_class,
                "feature": feature.feature_name,
                "value": feature.value,
                "unit": feature.unit,
                "source_features": ",".join(feature.source_features),
                "source_values": feature.source_values,
                "selected_source_feature": feature.selected_source_feature,
                "selection_reason": feature.selection_reason,
            }
        )

    return rows