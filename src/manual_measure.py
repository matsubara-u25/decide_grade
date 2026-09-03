"""
manual_measure.py

Interactive manual measurement tool for surface images of sawn lumber.

Default input directory:
    decide_grade/data/images/wood_joined_cropped/

Default outputs:
    decide_grade/data/input/surface_info.csv
    decide_grade/data/annotations/manual_measure/knot_measurements.csv
    decide_grade/data/annotations/manual_measure/annotation_detail.jsonl
    decide_grade/data/annotations/manual_measure/preview/

Controls
--------
Left click:
    Add a polygon point, or add an ellipse-fit point in fit-point mode.

Left click near the first polygon point:
    Close the polygon and fit an ellipse automatically.

Right click or u:
    Undo the last point in the current mode.

Enter:
    - polygon mode: close polygon if it has 3 or more points
    - fit mode: fit ellipse from selected fit points
    - review mode: save the current knot

f:
    In review mode, select only the elliptical arc points to refit the ellipse.
    Use this for truncated knots whose boundary includes a straight cut line.

a:
    In review mode, fit the ellipse again using the polygon boundary.

s:
    Save the current knot in review mode.

r:
    Reset the current unsaved knot.

n / p:
    Next / previous image.

h / j / k / l:
    Pan left / down / up / right.

+ or = / -:
    Zoom in / out.

0:
    Fit the full image to the display window.

q or Esc:
    Quit.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

KNOT_CSV_COLUMNS = [
    "lumber_id",
    "surface_id",
    "knot_id",
    "image_file",
    "length_min_pos",
    "length_max_pos",
    "width_min_pos",
    "width_max_pos",
    "center_point_length",
    "center_point_width",
    "long_diam_length",
    "long_diam_width",
    "short_diam_length",
    "short_diam_width",
    "ellipse_method",
]

SURFACE_INFO_COLUMNS = [
    "lumber_id",
    "surface_id",
    "image_file",
    "surface_width_mm",
    "lumber_length_mm",
    "length_px",
    "width_px",
]


@dataclass
class SurfaceInfo:
    lumber_id: str
    surface_id: str
    image_file: str
    surface_width_mm: str
    lumber_length_mm: str
    length_px: int
    width_px: int


@dataclass
class EllipseResult:
    method: str
    center: tuple[float, float]
    long_axis_endpoints: tuple[tuple[float, float], tuple[float, float]]
    short_axis_endpoints: tuple[tuple[float, float], tuple[float, float]]
    long_diam_length: float
    long_diam_width: float
    short_diam_length: float
    short_diam_width: float
    angle_deg: float
    long_diam_px: float
    short_diam_px: float


@dataclass
class AnnotationRecord:
    lumber_id: str
    surface_id: str
    knot_id: str
    image_file: str
    polygon_points: list[list[float]]
    ellipse_fit_points: list[list[float]]
    ellipse_method: str
    bbox: dict[str, float]
    ellipse: dict[str, object]
    created_at: str
    comment: str = ""


def project_root_from_this_file() -> Path:
    # If this file is placed at decide_grade/src/manual_measure.py,
    # parents[1] is decide_grade/.
    return Path(__file__).resolve().parents[1]


def relpath_string(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_optional_float(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    return str(float(value))


def ensure_csv_header(path: Path, columns: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def append_csv_row(path: Path, columns: list[str], row: dict[str, object]) -> None:
    ensure_csv_header(path, columns)
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow({col: row.get(col, "") for col in columns})


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_existing_knot_counts(knot_csv_path: Path) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    if not knot_csv_path.exists():
        return counts

    with knot_csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("lumber_id", ""), row.get("surface_id", ""))
            counts[key] = counts.get(key, 0) + 1
    return counts


def read_existing_surface_info(surface_info_path: Path) -> dict[str, SurfaceInfo]:
    infos: dict[str, SurfaceInfo] = {}
    if not surface_info_path.exists():
        return infos

    with surface_info_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_file = row.get("image_file", "")
            if not image_file:
                continue
            infos[image_file] = SurfaceInfo(
                lumber_id=row.get("lumber_id", ""),
                surface_id=row.get("surface_id", ""),
                image_file=image_file,
                surface_width_mm=row.get("surface_width_mm", ""),
                lumber_length_mm=row.get("lumber_length_mm", ""),
                length_px=int(float(row.get("length_px", 0) or 0)),
                width_px=int(float(row.get("width_px", 0) or 0)),
            )
    return infos


def prompt_surface_info(
    image_path: Path,
    image: np.ndarray,
    project_root: Path,
    default_lumber_id: Optional[str],
    no_prompt: bool,
) -> SurfaceInfo:
    h, w = image.shape[:2]
    image_file = relpath_string(image_path, project_root)
    default_surface_id = image_path.stem
    lumber_id = default_lumber_id or "L001"
    surface_id = default_surface_id
    surface_width_mm = ""
    lumber_length_mm = ""

    if not no_prompt:
        print("\nSurface metadata")
        print(f"  image_file: {image_file}")
        value = input(f"  lumber_id [{lumber_id}]: ").strip()
        if value:
            lumber_id = value
        value = input(f"  surface_id [{surface_id}]: ").strip()
        if value:
            surface_id = value
        surface_width_mm = parse_optional_float(input("  surface_width_mm [blank OK]: "))
        lumber_length_mm = parse_optional_float(input("  lumber_length_mm [blank OK]: "))

    return SurfaceInfo(
        lumber_id=lumber_id,
        surface_id=surface_id,
        image_file=image_file,
        surface_width_mm=surface_width_mm,
        lumber_length_mm=lumber_length_mm,
        length_px=w,
        width_px=h,
    )


def polygon_bbox(points: list[tuple[float, float]]) -> dict[str, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return {
        "length_min_pos": float(min(xs)),
        "length_max_pos": float(max(xs)),
        "width_min_pos": float(min(ys)),
        "width_max_pos": float(max(ys)),
    }


def sample_polyline(points: list[tuple[float, float]], closed: bool, step_px: float = 4.0) -> np.ndarray:
    if len(points) < 2:
        return np.array(points, dtype=np.float32)

    dense: list[tuple[float, float]] = []
    n = len(points)
    edge_count = n if closed else n - 1
    for i in range(edge_count):
        p1 = np.array(points[i], dtype=np.float32)
        p2 = np.array(points[(i + 1) % n], dtype=np.float32)
        dist = float(np.linalg.norm(p2 - p1))
        steps = max(1, int(dist / step_px))
        for j in range(steps):
            t = j / steps
            p = p1 * (1.0 - t) + p2 * t
            dense.append((float(p[0]), float(p[1])))
    return np.array(dense, dtype=np.float32)


def ellipse_from_cv2_fit(points: np.ndarray, method: str) -> EllipseResult:
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    (cx, cy), (axis_a, axis_b), angle_deg = cv2.fitEllipse(pts)

    theta = np.deg2rad(angle_deg)
    vec_a = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    vec_b = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)

    if axis_a >= axis_b:
        long_len = float(axis_a)
        short_len = float(axis_b)
        long_vec = vec_a
        short_vec = vec_b
        long_angle = float(angle_deg)
    else:
        long_len = float(axis_b)
        short_len = float(axis_a)
        long_vec = vec_b
        short_vec = vec_a
        long_angle = float((angle_deg + 90.0) % 180.0)

    center = np.array([cx, cy], dtype=np.float64)
    long_delta = long_vec * long_len
    short_delta = short_vec * short_len

    long_p1 = center - long_delta / 2.0
    long_p2 = center + long_delta / 2.0
    short_p1 = center - short_delta / 2.0
    short_p2 = center + short_delta / 2.0

    return EllipseResult(
        method=method,
        center=(float(cx), float(cy)),
        long_axis_endpoints=((float(long_p1[0]), float(long_p1[1])), (float(long_p2[0]), float(long_p2[1]))),
        short_axis_endpoints=((float(short_p1[0]), float(short_p1[1])), (float(short_p2[0]), float(short_p2[1]))),
        long_diam_length=abs(float(long_delta[0])),
        long_diam_width=abs(float(long_delta[1])),
        short_diam_length=abs(float(short_delta[0])),
        short_diam_width=abs(float(short_delta[1])),
        angle_deg=long_angle,
        long_diam_px=long_len,
        short_diam_px=short_len,
    )


def ellipse_from_min_area_rect(points: np.ndarray, method: str) -> EllipseResult:
    rect = cv2.minAreaRect(points.astype(np.float32))
    (cx, cy), (w, h), angle_deg = rect

    theta = np.deg2rad(angle_deg)
    vec_w = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    vec_h = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)

    if w >= h:
        long_len = float(w)
        short_len = float(h)
        long_vec = vec_w
        short_vec = vec_h
        long_angle = float(angle_deg)
    else:
        long_len = float(h)
        short_len = float(w)
        long_vec = vec_h
        short_vec = vec_w
        long_angle = float((angle_deg + 90.0) % 180.0)

    center = np.array([cx, cy], dtype=np.float64)
    long_delta = long_vec * long_len
    short_delta = short_vec * short_len

    long_p1 = center - long_delta / 2.0
    long_p2 = center + long_delta / 2.0
    short_p1 = center - short_delta / 2.0
    short_p2 = center + short_delta / 2.0

    return EllipseResult(
        method=method,
        center=(float(cx), float(cy)),
        long_axis_endpoints=((float(long_p1[0]), float(long_p1[1])), (float(long_p2[0]), float(long_p2[1]))),
        short_axis_endpoints=((float(short_p1[0]), float(short_p1[1])), (float(short_p2[0]), float(short_p2[1]))),
        long_diam_length=abs(float(long_delta[0])),
        long_diam_width=abs(float(long_delta[1])),
        short_diam_length=abs(float(short_delta[0])),
        short_diam_width=abs(float(short_delta[1])),
        angle_deg=long_angle,
        long_diam_px=long_len,
        short_diam_px=short_len,
    )


def fit_ellipse_from_points(
    points: list[tuple[float, float]],
    *,
    method: str,
    closed: bool,
) -> EllipseResult:
    if len(points) < 3:
        raise ValueError("At least 3 points are required.")

    if method == "fit_ellipse":
        fit_points = sample_polyline(points, closed=closed, step_px=4.0)
    else:
        fit_points = np.array(points, dtype=np.float32)

    if len(fit_points) >= 5:
        return ellipse_from_cv2_fit(fit_points, method=method)

    return ellipse_from_min_area_rect(fit_points, method=f"{method}_min_area_rect_fallback")


def point_to_list(point: tuple[float, float]) -> list[float]:
    return [float(point[0]), float(point[1])]


def ellipse_to_dict(ellipse: EllipseResult) -> dict[str, object]:
    return {
        "method": ellipse.method,
        "center": point_to_list(ellipse.center),
        "long_axis_endpoints": [
            point_to_list(ellipse.long_axis_endpoints[0]),
            point_to_list(ellipse.long_axis_endpoints[1]),
        ],
        "short_axis_endpoints": [
            point_to_list(ellipse.short_axis_endpoints[0]),
            point_to_list(ellipse.short_axis_endpoints[1]),
        ],
        "long_diam_length": ellipse.long_diam_length,
        "long_diam_width": ellipse.long_diam_width,
        "short_diam_length": ellipse.short_diam_length,
        "short_diam_width": ellipse.short_diam_width,
        "angle_deg": ellipse.angle_deg,
        "long_diam_px": ellipse.long_diam_px,
        "short_diam_px": ellipse.short_diam_px,
    }


class ManualMeasureApp:
    def __init__(
        self,
        *,
        project_root: Path,
        image_dir: Path,
        annotation_dir: Path,
        surface_info_path: Path,
        default_lumber_id: Optional[str],
        no_prompt_surface_info: bool,
        display_width: int,
        display_height: int,
    ) -> None:
        self.project_root = project_root
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.surface_info_path = surface_info_path
        self.default_lumber_id = default_lumber_id
        self.no_prompt_surface_info = no_prompt_surface_info
        self.display_width = display_width
        self.display_height = display_height

        self.knot_csv_path = annotation_dir / "knot_measurements.csv"
        self.detail_jsonl_path = annotation_dir / "annotation_detail.jsonl"
        self.preview_dir = annotation_dir / "preview"
        self.backup_dir = annotation_dir / "backups"

        for path in [annotation_dir, self.preview_dir, self.backup_dir, surface_info_path.parent]:
            path.mkdir(parents=True, exist_ok=True)

        self.image_paths = self._find_images()
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        ensure_csv_header(self.knot_csv_path, KNOT_CSV_COLUMNS)
        ensure_csv_header(self.surface_info_path, SURFACE_INFO_COLUMNS)

        self.surface_infos = read_existing_surface_info(self.surface_info_path)
        self.knot_counts = read_existing_knot_counts(self.knot_csv_path)

        self.index = 0
        self.image: Optional[np.ndarray] = None
        self.image_path: Optional[Path] = None
        self.surface_info: Optional[SurfaceInfo] = None
        self.annotations_by_image: dict[str, list[AnnotationRecord]] = {}

        self.zoom = 1.0
        self.view_x = 0.0
        self.view_y = 0.0

        self.mode = "polygon"  # polygon, review, fit
        self.polygon_points: list[tuple[float, float]] = []
        self.fit_points: list[tuple[float, float]] = []
        self.current_ellipse: Optional[EllipseResult] = None
        self.current_ellipse_fit_points: list[tuple[float, float]] = []
        self.current_method = "fit_ellipse"
        self.message = ""

        self.window_name = "manual_measure"

    def _find_images(self) -> list[Path]:
        return [
            p for p in sorted(self.image_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        self._load_image(self.index)

        while True:
            frame = self._render()
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKeyEx(30)
            if key == -1:
                continue
            if self._handle_key(key):
                break

        cv2.destroyAllWindows()

    def _load_image(self, index: int) -> None:
        self.index = max(0, min(index, len(self.image_paths) - 1))
        self.image_path = self.image_paths[self.index]
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise RuntimeError(f"Failed to read image: {self.image_path}")

        image_file = relpath_string(self.image_path, self.project_root)
        if image_file in self.surface_infos:
            self.surface_info = self.surface_infos[image_file]
        else:
            self.surface_info = prompt_surface_info(
                self.image_path,
                self.image,
                self.project_root,
                self.default_lumber_id,
                self.no_prompt_surface_info,
            )
            append_csv_row(self.surface_info_path, SURFACE_INFO_COLUMNS, asdict(self.surface_info))
            self.surface_infos[image_file] = self.surface_info

        self._reset_current()
        self._fit_to_window()
        self.message = f"Loaded {self.image_path.name} ({self.index + 1}/{len(self.image_paths)})"

    def _reset_current(self) -> None:
        self.mode = "polygon"
        self.polygon_points = []
        self.fit_points = []
        self.current_ellipse = None
        self.current_ellipse_fit_points = []
        self.current_method = "fit_ellipse"

    def _fit_to_window(self) -> None:
        assert self.image is not None
        h, w = self.image.shape[:2]
        self.zoom = min(self.display_width / w, self.display_height / h)
        self.zoom = min(self.zoom, 1.0)
        if self.zoom <= 0:
            self.zoom = 1.0
        self.view_x = 0.0
        self.view_y = 0.0

    def _clamp_view(self) -> None:
        assert self.image is not None
        h, w = self.image.shape[:2]
        viewport_w = self.display_width / self.zoom
        viewport_h = self.display_height / self.zoom
        self.view_x = max(0.0, min(self.view_x, max(0.0, w - viewport_w)))
        self.view_y = max(0.0, min(self.view_y, max(0.0, h - viewport_h)))

    def _screen_to_image(self, x: int, y: int) -> tuple[float, float]:
        return self.view_x + x / self.zoom, self.view_y + y / self.zoom

    def _image_to_screen(self, point: tuple[float, float]) -> tuple[int, int]:
        x = int(round((point[0] - self.view_x) * self.zoom))
        y = int(round((point[1] - self.view_y) * self.zoom))
        return x, y

    def _point_inside_image(self, point: tuple[float, float]) -> bool:
        assert self.image is not None
        h, w = self.image.shape[:2]
        return 0 <= point[0] < w and 0 <= point[1] < h

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return

        point = self._screen_to_image(x, y)
        if not self._point_inside_image(point):
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            self._undo_point()
            return

        if self.mode == "polygon":
            self._add_polygon_point(point)
        elif self.mode == "fit":
            self.fit_points.append(point)
            self.message = f"Fit points: {len(self.fit_points)}"
        elif self.mode == "review":
            self.message = "Review mode. Press s/Enter to save, f to refit, r to reset."

    def _add_polygon_point(self, point: tuple[float, float]) -> None:
        if len(self.polygon_points) >= 3:
            first = self.polygon_points[0]
            screen_first = np.array(self._image_to_screen(first), dtype=np.float64)
            screen_point = np.array(self._image_to_screen(point), dtype=np.float64)
            if float(np.linalg.norm(screen_first - screen_point)) <= 12.0:
                self._close_polygon_and_fit()
                return

        self.polygon_points.append(point)
        self.message = f"Polygon points: {len(self.polygon_points)}"

    def _close_polygon_and_fit(self) -> None:
        if len(self.polygon_points) < 3:
            self.message = "Need at least 3 polygon points."
            return
        try:
            ellipse = fit_ellipse_from_points(
                self.polygon_points,
                method="fit_ellipse",
                closed=True,
            )
        except Exception as error:
            self.message = f"Ellipse fitting failed: {error}"
            return

        self.current_ellipse = ellipse
        self.current_ellipse_fit_points = list(self.polygon_points)
        self.current_method = ellipse.method
        self.mode = "review"
        self.message = "Polygon closed. Press s/Enter to save, f to select arc points, a to auto-fit again."

    def _fit_selected_points(self) -> None:
        if len(self.fit_points) < 3:
            self.message = "Need at least 3 fit points."
            return
        try:
            ellipse = fit_ellipse_from_points(
                self.fit_points,
                method="selected_arc_fit",
                closed=False,
            )
        except Exception as error:
            self.message = f"Selected-arc fit failed: {error}"
            return

        self.current_ellipse = ellipse
        self.current_ellipse_fit_points = list(self.fit_points)
        self.current_method = ellipse.method
        self.mode = "review"
        self.message = "Selected-arc ellipse fitted. Press s/Enter to save."

    def _undo_point(self) -> None:
        if self.mode == "polygon" and self.polygon_points:
            self.polygon_points.pop()
            self.message = f"Polygon points: {len(self.polygon_points)}"
        elif self.mode == "fit" and self.fit_points:
            self.fit_points.pop()
            self.message = f"Fit points: {len(self.fit_points)}"
        else:
            self.message = "Nothing to undo."

    def _handle_key(self, key: int) -> bool:
        # Normalize common ASCII keys.
        char = chr(key & 0xFF) if 0 <= (key & 0xFF) < 256 else ""

        if key in (27,) or char == "q":
            return True

        if char in ("+", "="):
            self._zoom(1.25)
        elif char in ("-", "_"):
            self._zoom(1 / 1.25)
        elif char == "0":
            self._fit_to_window()
        elif char == "h":
            self._pan(-0.20, 0.0)
        elif char == "l":
            self._pan(0.20, 0.0)
        elif char == "k":
            self._pan(0.0, -0.20)
        elif char == "j":
            self._pan(0.0, 0.20)
        elif char == "u":
            self._undo_point()
        elif char == "r":
            self._reset_current()
            self.message = "Current knot reset."
        elif char == "n":
            self._next_image()
        elif char == "p":
            self._previous_image()
        elif char == "f":
            if self.mode == "review":
                self.mode = "fit"
                self.fit_points = []
                self.message = "Fit mode: click ellipse-arc points, then press Enter."
        elif char == "a":
            if self.mode == "review":
                self._close_polygon_and_fit()
        elif char == "s":
            if self.mode == "review":
                self._save_current_annotation()
        elif key in (10, 13):
            if self.mode == "polygon":
                self._close_polygon_and_fit()
            elif self.mode == "fit":
                self._fit_selected_points()
            elif self.mode == "review":
                self._save_current_annotation()
        else:
            self.message = "Keys: q quit, n/p image, h/j/k/l pan, +/- zoom, r reset, s save, f refit."

        return False

    def _zoom(self, factor: float) -> None:
        assert self.image is not None
        old_zoom = self.zoom
        center_x = self.view_x + self.display_width / (2 * old_zoom)
        center_y = self.view_y + self.display_height / (2 * old_zoom)
        self.zoom = max(0.02, min(self.zoom * factor, 20.0))
        self.view_x = center_x - self.display_width / (2 * self.zoom)
        self.view_y = center_y - self.display_height / (2 * self.zoom)
        self._clamp_view()
        self.message = f"zoom: {self.zoom:.3f}"

    def _pan(self, dx_viewport_fraction: float, dy_viewport_fraction: float) -> None:
        self.view_x += dx_viewport_fraction * self.display_width / self.zoom
        self.view_y += dy_viewport_fraction * self.display_height / self.zoom
        self._clamp_view()

    def _next_image(self) -> None:
        if self.mode != "polygon" or self.polygon_points:
            self.message = "Unsaved/current knot exists. Press r to reset before changing image."
            return
        if self.index + 1 >= len(self.image_paths):
            self.message = "Already at last image."
            return
        self._load_image(self.index + 1)

    def _previous_image(self) -> None:
        if self.mode != "polygon" or self.polygon_points:
            self.message = "Unsaved/current knot exists. Press r to reset before changing image."
            return
        if self.index <= 0:
            self.message = "Already at first image."
            return
        self._load_image(self.index - 1)

    def _generate_knot_id(self) -> str:
        assert self.surface_info is not None
        key = (self.surface_info.lumber_id, self.surface_info.surface_id)
        current = self.knot_counts.get(key, 0) + 1
        self.knot_counts[key] = current
        return f"{self.surface_info.surface_id}_K{current:03d}"

    def _save_current_annotation(self) -> None:
        if self.current_ellipse is None:
            self.message = "No ellipse to save."
            return
        if len(self.polygon_points) < 3:
            self.message = "No closed polygon to save."
            return
        assert self.image_path is not None
        assert self.surface_info is not None

        knot_id = self._generate_knot_id()
        image_file = relpath_string(self.image_path, self.project_root)
        bbox = polygon_bbox(self.polygon_points)
        ellipse = self.current_ellipse

        csv_row = {
            "lumber_id": self.surface_info.lumber_id,
            "surface_id": self.surface_info.surface_id,
            "knot_id": knot_id,
            "image_file": image_file,
            "length_min_pos": bbox["length_min_pos"],
            "length_max_pos": bbox["length_max_pos"],
            "width_min_pos": bbox["width_min_pos"],
            "width_max_pos": bbox["width_max_pos"],
            "center_point_length": ellipse.center[0],
            "center_point_width": ellipse.center[1],
            "long_diam_length": ellipse.long_diam_length,
            "long_diam_width": ellipse.long_diam_width,
            "short_diam_length": ellipse.short_diam_length,
            "short_diam_width": ellipse.short_diam_width,
            "ellipse_method": ellipse.method,
        }
        append_csv_row(self.knot_csv_path, KNOT_CSV_COLUMNS, csv_row)

        record = AnnotationRecord(
            lumber_id=self.surface_info.lumber_id,
            surface_id=self.surface_info.surface_id,
            knot_id=knot_id,
            image_file=image_file,
            polygon_points=[point_to_list(p) for p in self.polygon_points],
            ellipse_fit_points=[point_to_list(p) for p in self.current_ellipse_fit_points],
            ellipse_method=ellipse.method,
            bbox=bbox,
            ellipse=ellipse_to_dict(ellipse),
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        append_jsonl(self.detail_jsonl_path, asdict(record))

        self.annotations_by_image.setdefault(image_file, []).append(record)
        self._save_preview()
        self._reset_current()
        self.message = f"Saved {knot_id}"

    def _save_preview(self) -> None:
        assert self.image is not None
        assert self.image_path is not None
        image_file = relpath_string(self.image_path, self.project_root)
        canvas = self.image.copy()
        for record in self.annotations_by_image.get(image_file, []):
            self._draw_record_on_image(canvas, record)
        out_path = self.preview_dir / f"{self.image_path.stem}_annotated.png"
        cv2.imwrite(str(out_path), canvas)

    def _draw_record_on_image(self, image: np.ndarray, record: AnnotationRecord) -> None:
        polygon = np.array(record.polygon_points, dtype=np.int32)
        if len(polygon) >= 2:
            cv2.polylines(image, [polygon], True, (0, 0, 255), 2)
        ellipse = record.ellipse
        self._draw_ellipse_dict(image, ellipse, scale=1.0, offset=(0.0, 0.0))
        if polygon.size > 0:
            p = tuple(polygon[0])
            cv2.putText(image, record.knot_id, (int(p[0]), int(p[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    def _render(self) -> np.ndarray:
        assert self.image is not None
        h, w = self.image.shape[:2]
        self._clamp_view()
        x0 = int(max(0, np.floor(self.view_x)))
        y0 = int(max(0, np.floor(self.view_y)))
        x1 = int(min(w, np.ceil(self.view_x + self.display_width / self.zoom)))
        y1 = int(min(h, np.ceil(self.view_y + self.display_height / self.zoom)))
        crop = self.image[y0:y1, x0:x1]
        if crop.size == 0:
            canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        else:
            resized = cv2.resize(crop, None, fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_AREA)
            canvas = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
            rh, rw = resized.shape[:2]
            canvas[: min(rh, self.display_height), : min(rw, self.display_width)] = resized[: self.display_height, : self.display_width]

        self._draw_overlay(canvas)
        return canvas

    def _draw_overlay(self, canvas: np.ndarray) -> None:
        assert self.image_path is not None
        image_file = relpath_string(self.image_path, self.project_root)

        for record in self.annotations_by_image.get(image_file, []):
            self._draw_record_on_canvas(canvas, record)

        self._draw_points(canvas, self.polygon_points, (0, 255, 255), closed=False)
        if self.mode == "fit":
            self._draw_points(canvas, self.fit_points, (255, 255, 0), closed=False)
        if self.current_ellipse is not None:
            self._draw_ellipse_result_on_canvas(canvas, self.current_ellipse)

        status = f"{self.image_path.name} | mode={self.mode} | {self.message}"
        cv2.rectangle(canvas, (0, 0), (self.display_width, 34), (0, 0, 0), -1)
        cv2.putText(canvas, status[:180], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    def _draw_points(
        self,
        canvas: np.ndarray,
        points: list[tuple[float, float]],
        color: tuple[int, int, int],
        closed: bool,
    ) -> None:
        if not points:
            return
        screen_pts = [self._image_to_screen(p) for p in points]
        for p in screen_pts:
            cv2.circle(canvas, p, 4, color, -1)
        if len(screen_pts) >= 2:
            cv2.polylines(canvas, [np.array(screen_pts, dtype=np.int32)], closed, color, 2)
        if len(screen_pts) >= 1:
            cv2.circle(canvas, screen_pts[0], 8, (0, 0, 255), 2)

    def _draw_record_on_canvas(self, canvas: np.ndarray, record: AnnotationRecord) -> None:
        points = [(float(x), float(y)) for x, y in record.polygon_points]
        self._draw_points(canvas, points, (0, 0, 255), closed=True)
        self._draw_ellipse_dict(canvas, record.ellipse, scale=self.zoom, offset=(self.view_x, self.view_y))

    def _draw_ellipse_result_on_canvas(self, canvas: np.ndarray, ellipse: EllipseResult) -> None:
        ellipse_dict = ellipse_to_dict(ellipse)
        self._draw_ellipse_dict(canvas, ellipse_dict, scale=self.zoom, offset=(self.view_x, self.view_y))

    def _draw_ellipse_dict(
        self,
        image: np.ndarray,
        ellipse: dict[str, object],
        *,
        scale: float,
        offset: tuple[float, float],
    ) -> None:
        center = ellipse["center"]
        long_endpoints = ellipse["long_axis_endpoints"]
        short_endpoints = ellipse["short_axis_endpoints"]
        long_diam_px = float(ellipse["long_diam_px"])
        short_diam_px = float(ellipse["short_diam_px"])
        angle_deg = float(ellipse["angle_deg"])

        def transform(p: list[float] | tuple[float, float]) -> tuple[int, int]:
            return (
                int(round((float(p[0]) - offset[0]) * scale)),
                int(round((float(p[1]) - offset[1]) * scale)),
            )

        c = transform(center)  # type: ignore[arg-type]
        axes = (max(1, int(round(long_diam_px * scale / 2))), max(1, int(round(short_diam_px * scale / 2))))
        cv2.ellipse(image, c, axes, angle_deg, 0, 360, (0, 255, 0), 2)
        cv2.circle(image, c, 4, (0, 255, 0), -1)
        lp1 = transform(long_endpoints[0])  # type: ignore[index]
        lp2 = transform(long_endpoints[1])  # type: ignore[index]
        sp1 = transform(short_endpoints[0])  # type: ignore[index]
        sp2 = transform(short_endpoints[1])  # type: ignore[index]
        cv2.line(image, lp1, lp2, (255, 0, 255), 2)
        cv2.line(image, sp1, sp2, (255, 255, 0), 2)


def build_arg_parser() -> argparse.ArgumentParser:
    default_root = project_root_from_this_file()
    parser = argparse.ArgumentParser(description="Manual knot measurement tool.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=default_root / "data" / "images" / "wood_joined_cropped",
        help="Directory containing surface images.",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=default_root / "data" / "annotations" / "manual_measure",
        help="Directory to save annotation outputs.",
    )
    parser.add_argument(
        "--surface-info-path",
        type=Path,
        default=default_root / "data" / "input" / "surface_info.csv",
        help="Path to surface_info.csv.",
    )
    parser.add_argument(
        "--lumber-id",
        type=str,
        default=None,
        help="Default lumber_id used when prompting is disabled or left blank.",
    )
    parser.add_argument(
        "--no-prompt-surface-info",
        action="store_true",
        help="Do not prompt for surface metadata. Defaults are used instead.",
    )
    parser.add_argument("--display-width", type=int, default=1600)
    parser.add_argument("--display-height", type=int, default=900)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = project_root_from_this_file()
    app = ManualMeasureApp(
        project_root=project_root,
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        surface_info_path=args.surface_info_path,
        default_lumber_id=args.lumber_id,
        no_prompt_surface_info=args.no_prompt_surface_info,
        display_width=args.display_width,
        display_height=args.display_height,
    )
    app.run()


if __name__ == "__main__":
    main()
