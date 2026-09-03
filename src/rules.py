# rules.py

from __future__ import annotations

from typing import Literal


JASClass = Literal["甲Ⅰ", "甲Ⅱ", "乙"]
GradeLabel = Literal["1級", "2級", "3級", "等外"]
LumberPurpose = Literal["ko_1", "ko_2", "otsu"]

# Thresholds are ordered as:
# (grade_1_threshold, grade_2_threshold, grade_3_threshold)
Thresholds = tuple[float, float, float]


LUMBER_PURPOSE_TO_JAS_CLASS: dict[LumberPurpose, JASClass] = {
    "ko_1": "甲Ⅰ",
    "ko_2": "甲Ⅱ",
    "otsu": "乙",
}


JAS_THRESHOLDS: dict[tuple[JASClass, str], Thresholds] = {
    # ---- 甲Ⅰ ----
    ("甲Ⅰ", "節"): (20.0, 40.0, 60.0),
    ("甲Ⅰ", "集中節"): (30.0, 60.0, 90.0),

    # ---- 甲Ⅱ ----
    ("甲Ⅱ", "狭い材面の節"): (20.0, 40.0, 60.0),
    ("甲Ⅱ", "広い材面材縁部の節"): (15.0, 25.0, 35.0),
    ("甲Ⅱ", "広い材面中央部の節"): (30.0, 40.0, 70.0),

    ("甲Ⅱ", "狭い材面の集中節"): (30.0, 60.0, 90.0),
    ("甲Ⅱ", "広い材面材縁部の集中節"): (20.0, 40.0, 50.0),
    ("甲Ⅱ", "広い材面中央部の集中節"): (45.0, 60.0, 90.0),

    # ---- 乙 ----
    ("乙", "節"): (30.0, 40.0, 70.0),
    ("乙", "集中節"): (45.0, 60.0, 90.0),
}


JAS_FEATURES_BY_CLASS: dict[JASClass, tuple[str, ...]] = {
    "甲Ⅰ": (
        "節",
        "集中節",
    ),
    "甲Ⅱ": (
        "狭い材面の節",
        "広い材面材縁部の節",
        "広い材面中央部の節",
        "狭い材面の集中節",
        "広い材面材縁部の集中節",
        "広い材面中央部の集中節",
    ),
    "乙": (
        "節",
        "集中節",
    ),
}


GRADE_RANK: dict[GradeLabel, int] = {
    "1級": 1,
    "2級": 2,
    "3級": 3,
    "等外": 4,
}


def get_jas_class(lumber_purpose: LumberPurpose) -> JASClass:
    """Return JAS class from internal lumber purpose."""
    return LUMBER_PURPOSE_TO_JAS_CLASS[lumber_purpose]


def get_thresholds(jas_class: JASClass, feature_name: str) -> Thresholds:
    """Return thresholds for one JAS judgment item."""
    key = (jas_class, feature_name)

    if key not in JAS_THRESHOLDS:
        raise KeyError(
            f"No JAS thresholds found for jas_class={jas_class}, "
            f"feature_name={feature_name}"
        )

    return JAS_THRESHOLDS[key]


def judge_by_thresholds(
    value: float,
    thresholds: Thresholds,
    *,
    inclusive: bool = True,
) -> GradeLabel:
    """Judge one item grade from value and thresholds.

    Parameters
    ----------
    value:
        Feature value to judge.

    thresholds:
        Thresholds ordered as 1級, 2級, 3級.

    inclusive:
        If True, use <= comparison.
        If False, use < comparison.
    """
    grade_1, grade_2, grade_3 = thresholds

    if inclusive:
        if value <= grade_1:
            return "1級"
        if value <= grade_2:
            return "2級"
        if value <= grade_3:
            return "3級"
    else:
        if value < grade_1:
            return "1級"
        if value < grade_2:
            return "2級"
        if value < grade_3:
            return "3級"

    return "等外"


def worst_grade(grades: list[GradeLabel]) -> GradeLabel:
    """Return the worst grade from item grades."""
    if not grades:
        raise ValueError("grades is empty")

    return max(grades, key=lambda grade: GRADE_RANK[grade])