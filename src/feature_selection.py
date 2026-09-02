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


LUMBER_PURPOSE_TO_JAS_CLASS: dict[str, JASClass] = {
    "ko_1": "甲Ⅰ",
    "ko_2": "甲Ⅱ",
    "otsu": "乙",
}


@dataclass(frozen=True)
class JudgmentFeature:
    """A feature value selected for JAS grade judgment.

    This object corresponds to one row in the lumber judgment feature table.
    """

    lumber_id: str
    lumber_purpose: str
    jas_class: JASClass

    # Feature name should match the `feature` column in jas_definition_structure.csv.
    feature_name: FeatureName

    # Selected value used for grade judgment.
    value: float

    unit: str = "%"

    # Source feature names from the Lumber object.
    source_features: tuple[str, ...] = field(default_factory=tuple)

    # Source values before selection.
    source_values: dict[str, float] = field(default_factory=dict)

    # If the value is selected by max(), this stores the chosen source feature.
    selected_source_feature: Optional[str] = None


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

    try:
        jas_class = LUMBER_PURPOSE_TO_JAS_CLASS[lumber.lumber_purpose]
    except KeyError as error:
        raise FeatureSelectionError(
            f"Unknown lumber_purpose: {lumber.lumber_purpose}"
        ) from error

    if jas_class == "甲Ⅰ":
        return _select_ko1_features(lumber, jas_class)

    if jas_class == "甲Ⅱ":
        return _select_ko2_features(lumber, jas_class)

    if jas_class == "乙":
        return _select_otsu_features(lumber, jas_class)

    raise FeatureSelectionError(f"Unsupported JAS class: {jas_class}")


def _select_ko1_features(lumber: Lumber, jas_class: JASClass) -> list[JudgmentFeature]:
    """Select judgment features for 甲Ⅰ."""
    return [
        _make_max_feature(
            lumber=lumber,
            jas_class=jas_class,
            feature_name="節",
            source_attrs=("wide_max_kr", "narrow_max_kr"),
        ),
        _make_max_feature(
            lumber=lumber,
            jas_class=jas_class,
            feature_name="集中節",
            source_attrs=("wide_max_ckr", "narrow_max_ckr"),
        ),
    ]


def _select_ko2_features(lumber: Lumber, jas_class: JASClass) -> list[JudgmentFeature]:
    """Select judgment features for 甲Ⅱ.

    For rectangular sections:
        - narrow surface values are used for narrow-surface items.
        - wide-surface edge/center values are used for wide-surface items.

    For square sections:
        all side surfaces are assigned as side_wide in instances.py,
        so narrow-surface items are skipped.
    """
    features: list[JudgmentFeature] = []

    has_narrow = _has_narrow_surfaces(lumber)

    if has_narrow:
        features.append(
            _make_direct_feature(
                lumber=lumber,
                jas_class=jas_class,
                feature_name="狭い材面の節",
                source_attr="narrow_max_kr",
            )
        )
        features.append(
            _make_direct_feature(
                lumber=lumber,
                jas_class=jas_class,
                feature_name="狭い材面の集中節",
                source_attr="narrow_max_ckr",
            )
        )

    features.extend(
        [
            _make_direct_feature(
                lumber=lumber,
                jas_class=jas_class,
                feature_name="材縁部(広い材面)の節",
                source_attr="wide_edge_max_kr",
            ),
            _make_direct_feature(
                lumber=lumber,
                jas_class=jas_class,
                feature_name="中央部(広い材面)の節",
                source_attr="wide_center_max_kr",
            ),
            _make_direct_feature(
                lumber=lumber,
                jas_class=jas_class,
                feature_name="材縁部(広い材面)の集中節",
                source_attr="wide_edge_max_ckr",
            ),
            _make_direct_feature(
                lumber=lumber,
                jas_class=jas_class,
                feature_name="中央部(広い材面)の集中節",
                source_attr="wide_center_max_ckr",
            ),
        ]
    )

    return features


def _select_otsu_features(lumber: Lumber, jas_class: JASClass) -> list[JudgmentFeature]:
    """Select judgment features for 乙."""
    return [
        _make_max_feature(
            lumber=lumber,
            jas_class=jas_class,
            feature_name="節",
            source_attrs=("wide_max_kr", "narrow_max_kr"),
        ),
        _make_max_feature(
            lumber=lumber,
            jas_class=jas_class,
            feature_name="集中節",
            source_attrs=("wide_max_ckr", "narrow_max_ckr"),
        ),
    ]


def _make_direct_feature(
    *,
    lumber: Lumber,
    jas_class: JASClass,
    feature_name: FeatureName,
    source_attr: str,
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
    )


def _make_max_feature(
    *,
    lumber: Lumber,
    jas_class: JASClass,
    feature_name: FeatureName,
    source_attrs: tuple[str, ...],
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
    )


def _get_float_attr(lumber: Lumber, attr_name: str) -> float:
    """Read a numeric attribute from Lumber.

    None is treated as 0.0 because missing/non-applicable measured values
    should not worsen the grade by themselves.
    """
    if not hasattr(lumber, attr_name):
        raise FeatureSelectionError(
            f"Lumber does not have feature attribute: {attr_name}"
        )

    value = getattr(lumber, attr_name)

    if value is None:
        return 0.0

    return float(value)


def _has_narrow_surfaces(lumber: Lumber) -> bool:
    """Return whether this lumber has side_narrow surfaces."""
    return any(
        surface.surface_class == "side_narrow"
        for surface in lumber.side_surfaces
    )


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
                "selected_source_feature": feature.selected_source_feature,
            }
        )

    return rows