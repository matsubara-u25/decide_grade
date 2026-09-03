# grading.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


try:
    from instances import Lumber
    from feature_selection import JudgmentFeature, select_judgment_features
    from rules import (
        GradeLabel,
        Thresholds,
        get_thresholds,
        judge_by_thresholds,
        worst_grade,
    )
except ImportError:
    from .instances import Lumber
    from .feature_selection import JudgmentFeature, select_judgment_features
    from .rules import (
        GradeLabel,
        Thresholds,
        get_thresholds,
        judge_by_thresholds,
        worst_grade,
    )


@dataclass(frozen=True)
class ItemJudgment:
    """Judgment result for one JAS item."""

    lumber_id: str
    lumber_purpose: str
    jas_class: str

    feature_name: str
    value: float
    unit: str

    thresholds: Thresholds
    item_grade: GradeLabel

    source_features: tuple[str, ...] = field(default_factory=tuple)
    source_values: dict[str, float] = field(default_factory=dict)
    selected_source_feature: Optional[str] = None
    selection_reason: str = ""


@dataclass(frozen=True)
class LumberJudgment:
    """Final judgment result for one lumber."""

    lumber_id: str
    lumber_purpose: str
    jas_class: str

    item_judgments: list[ItemJudgment]
    final_grade: GradeLabel


class GradingError(Exception):
    """Raised when lumber grading fails."""
    pass


def grade_lumber(
    lumber: Lumber,
    *,
    derive: bool = True,
    inclusive: bool = True,
) -> LumberJudgment:
    """Judge final lumber grade.

    Parameters
    ----------
    lumber:
        Lumber object to judge.

    derive:
        If True, call lumber.derive_features() through feature_selection.
        If derived features are already calculated, set this to False.

    inclusive:
        If True, use <= threshold comparison.
        If False, use < threshold comparison.

    Returns
    -------
    LumberJudgment
        Item-level judgments and final lumber grade.
    """
    judgment_features = select_judgment_features(lumber, derive=derive)

    if not judgment_features:
        raise GradingError("No judgment features were selected.")

    item_judgments = judge_selected_features(
        judgment_features,
        inclusive=inclusive,
    )

    final_grade = worst_grade(
        [item.item_grade for item in item_judgments]
    )

    first_feature = judgment_features[0]

    return LumberJudgment(
        lumber_id=lumber.lumber_id,
        lumber_purpose=lumber.lumber_purpose,
        jas_class=first_feature.jas_class,
        item_judgments=item_judgments,
        final_grade=final_grade,
    )


def judge_selected_features(
    judgment_features: list[JudgmentFeature],
    *,
    inclusive: bool = True,
) -> list[ItemJudgment]:
    """Judge item grades from selected judgment features."""
    item_judgments: list[ItemJudgment] = []

    for feature in judgment_features:
        item_judgments.append(
            judge_one_feature(
                feature,
                inclusive=inclusive,
            )
        )

    return item_judgments


def judge_one_feature(
    judgment_feature: JudgmentFeature,
    *,
    inclusive: bool = True,
) -> ItemJudgment:
    """Judge one JAS item from one selected feature."""
    try:
        thresholds = get_thresholds(
            judgment_feature.jas_class,
            judgment_feature.feature_name,
        )
    except KeyError as error:
        raise GradingError(
            "No threshold found for "
            f"jas_class={judgment_feature.jas_class}, "
            f"feature_name={judgment_feature.feature_name}"
        ) from error

    item_grade = judge_by_thresholds(
        judgment_feature.value,
        thresholds,
        inclusive=inclusive,
    )

    return ItemJudgment(
        lumber_id=judgment_feature.lumber_id,
        lumber_purpose=judgment_feature.lumber_purpose,
        jas_class=judgment_feature.jas_class,
        feature_name=judgment_feature.feature_name,
        value=judgment_feature.value,
        unit=judgment_feature.unit,
        thresholds=thresholds,
        item_grade=item_grade,
        source_features=judgment_feature.source_features,
        source_values=judgment_feature.source_values,
        selected_source_feature=judgment_feature.selected_source_feature,
        selection_reason=judgment_feature.selection_reason,
    )


def item_judgment_to_row(item: ItemJudgment) -> dict[str, object]:
    """Convert one item judgment to a table row."""
    grade_1_threshold, grade_2_threshold, grade_3_threshold = item.thresholds

    return {
        "lumber_id": item.lumber_id,
        "lumber_purpose": item.lumber_purpose,
        "jas_class": item.jas_class,
        "feature": item.feature_name,
        "value": item.value,
        "unit": item.unit,
        "grade_1_threshold": grade_1_threshold,
        "grade_2_threshold": grade_2_threshold,
        "grade_3_threshold": grade_3_threshold,
        "item_grade": item.item_grade,
        "source_features": ",".join(item.source_features),
        "source_values": item.source_values,
        "selected_source_feature": item.selected_source_feature,
        "selection_reason": item.selection_reason,
    }


def lumber_judgment_to_rows(
    lumber_judgment: LumberJudgment,
) -> list[dict[str, object]]:
    """Convert lumber judgment result to table-like rows."""
    rows: list[dict[str, object]] = []

    for item in lumber_judgment.item_judgments:
        row = item_judgment_to_row(item)
        row["final_grade"] = lumber_judgment.final_grade
        rows.append(row)

    return rows


def summarize_lumber_judgment(
    lumber_judgment: LumberJudgment,
) -> dict[str, object]:
    """Create one summary row for final lumber grade."""
    return {
        "lumber_id": lumber_judgment.lumber_id,
        "lumber_purpose": lumber_judgment.lumber_purpose,
        "jas_class": lumber_judgment.jas_class,
        "final_grade": lumber_judgment.final_grade,
        "num_items": len(lumber_judgment.item_judgments),
    }
