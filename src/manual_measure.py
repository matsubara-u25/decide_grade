# manual_measure.py

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data" / "images" 
SURFACE_INFO_PATH = PROJECT_ROOT / "data" / "input" / "surface_info.csv"

ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations" / "manual_measure"
KNOT_MEASUREMENTS_PATH = ANNOTATION_DIR / "knot_measurements.csv"
ANNOTATION_DETAIL_PATH = ANNOTATION_DIR / "annotation_detail.jsonl"
PREVIEW_DIR = ANNOTATION_DIR / "preview"


# =============================================================================
# CSV columns
# =============================================================================

SURFACE_INFO_COLUMNS = [
    "lumber_id",
    "surface_id",
    "image_file",
    "surface_width_mm",
    "lumber_length_mm",
    "length_px",
    "width_px",
    "created_at",
]

KNOT_MEASUREMENT_COLUMNS = [
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
    "is_truncated",
    "created_at",
]


# =============================================================================
# Constants
# =============================================================================

CLOSE_POINT_RADIUS_SCREEN_PX = 12
POLYGON_POINT_RADIUS = 4
FIT_POINT_RADIUS = 4

AUTO_DENSIFY_STEP_PX = 2.0


# =============================================================================
# Utilities
# =============================================================================

def ensure_dirs() -> None:
    for path in [
        IMAGE_DIR,
        SURFACE_INFO_PATH.parent,
        ANNOTATION_DIR,
        PREVIEW_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def path_to_storable_string(path: Path) -> str:
    """Store a path relative to the project root if possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def read_image_bgr(path: Path) -> np.ndarray:
    """Read image with Japanese/Unicode path support."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    return image


def write_image(path: Path, image_bgr: np.ndarray) -> None:
    """Write image with Japanese/Unicode path support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".png"
    success, buffer = cv2.imencode(suffix, image_bgr)

    if not success:
        raise ValueError(f"Failed to encode image: {path}")

    buffer.tofile(str(path))


def append_csv_rows(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)

        if write_header:
            writer.writeheader()

        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def upsert_surface_info(row: dict[str, object]) -> None:
    """Insert or replace one surface_info row by lumber_id + surface_id."""
    SURFACE_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    if SURFACE_INFO_PATH.exists() and SURFACE_INFO_PATH.stat().st_size > 0:
        with SURFACE_INFO_PATH.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    lumber_id = str(row["lumber_id"])
    surface_id = str(row["surface_id"])

    rows = [
        existing
        for existing in rows
        if not (
            existing.get("lumber_id") == lumber_id
            and existing.get("surface_id") == surface_id
        )
    ]

    rows.append({column: str(row.get(column, "")) for column in SURFACE_INFO_COLUMNS})

    with SURFACE_INFO_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SURFACE_INFO_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_existing_knots(lumber_id: str, surface_id: str) -> int:
    if not KNOT_MEASUREMENTS_PATH.exists():
        return 0

    count = 0

    with KNOT_MEASUREMENTS_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("lumber_id") == lumber_id and row.get("surface_id") == surface_id:
                count += 1

    return count


def round_float(value: float, ndigits: int = 3) -> float:
    return round(float(value), ndigits)


# =============================================================================
# Geometry
# =============================================================================

Point = tuple[float, float]


def distance(p1: Point, p2: Point) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def densify_polyline(points: list[Point], closed: bool = True, step: float = 2.0) -> list[Point]:
    """Add points along polygon edges for more stable ellipse fitting."""
    if len(points) < 2:
        return points[:]

    result: list[Point] = []
    n = len(points)

    edge_count = n if closed else n - 1

    for i in range(edge_count):
        p1 = points[i]
        p2 = points[(i + 1) % n]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)

        segments = max(1, int(length / step))

        for j in range(segments):
            t = j / segments
            result.append((p1[0] + dx * t, p1[1] + dy * t))

    return result


def ellipse_from_cv2_rect(
    rect: tuple[tuple[float, float], tuple[float, float], float],
    method: str,
) -> dict[str, object]:
    """Convert OpenCV ellipse/minAreaRect output to measurement values."""
    (cx, cy), (axis_w, axis_h), angle_deg = rect

    theta_w = math.radians(angle_deg)
    theta_h = theta_w + math.pi / 2.0

    if axis_w >= axis_h:
        long_len = axis_w
        short_len = axis_h
        long_theta = theta_w
        short_theta = theta_h
    else:
        long_len = axis_h
        short_len = axis_w
        long_theta = theta_h
        short_theta = theta_w

    long_dx = long_len * math.cos(long_theta)
    long_dy = long_len * math.sin(long_theta)
    short_dx = short_len * math.cos(short_theta)
    short_dy = short_len * math.sin(short_theta)

    long_p1 = (cx - long_dx / 2.0, cy - long_dy / 2.0)
    long_p2 = (cx + long_dx / 2.0, cy + long_dy / 2.0)

    short_p1 = (cx - short_dx / 2.0, cy - short_dy / 2.0)
    short_p2 = (cx + short_dx / 2.0, cy + short_dy / 2.0)

    return {
        "method": method,
        "center": [round_float(cx), round_float(cy)],
        "long_axis_length": round_float(long_len),
        "short_axis_length": round_float(short_len),
        "long_axis_endpoints": [
            [round_float(long_p1[0]), round_float(long_p1[1])],
            [round_float(long_p2[0]), round_float(long_p2[1])],
        ],
        "short_axis_endpoints": [
            [round_float(short_p1[0]), round_float(short_p1[1])],
            [round_float(short_p2[0]), round_float(short_p2[1])],
        ],
        "long_diam_length": round_float(long_dx),
        "long_diam_width": round_float(long_dy),
        "short_diam_length": round_float(short_dx),
        "short_diam_width": round_float(short_dy),
        "cv2_angle_deg": round_float(angle_deg),
    }


def fit_ellipse_from_points(points: list[Point], method_hint: str) -> dict[str, object]:
    """Fit an ellipse. Fallback to minAreaRect if fitEllipse fails."""
    if len(points) < 2:
        raise ValueError("At least 2 points are required for axis estimation.")

    pts = np.array(points, dtype=np.float32)

    if len(points) >= 5:
        try:
            rect = cv2.fitEllipse(pts.reshape(-1, 1, 2))
            return ellipse_from_cv2_rect(rect, method_hint)
        except cv2.error:
            pass

    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    return ellipse_from_cv2_rect(rect, f"{method_hint}_min_area_rect_fallback")


def bbox_from_polygon(points: list[Point]) -> dict[str, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return {
        "length_min_pos": round_float(min(xs)),
        "length_max_pos": round_float(max(xs)),
        "width_min_pos": round_float(min(ys)),
        "width_max_pos": round_float(max(ys)),
    }


# =============================================================================
# Preview drawing
# =============================================================================

def draw_preview(
    image_bgr: np.ndarray,
    details: list[dict[str, object]],
) -> np.ndarray:
    preview = image_bgr.copy()

    for detail in details:
        knot_id = str(detail["knot_id"])
        polygon_points = detail.get("polygon_points", [])
        ellipse = detail.get("ellipse", {})

        if polygon_points:
            pts = np.array(polygon_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(preview, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        try:
            center = ellipse["center"]
            long_ep = ellipse["long_axis_endpoints"]
            short_ep = ellipse["short_axis_endpoints"]

            center_pt = (int(round(center[0])), int(round(center[1])))

            long_p1 = tuple(int(round(v)) for v in long_ep[0])
            long_p2 = tuple(int(round(v)) for v in long_ep[1])
            short_p1 = tuple(int(round(v)) for v in short_ep[0])
            short_p2 = tuple(int(round(v)) for v in short_ep[1])

            long_len = distance(long_ep[0], long_ep[1])
            short_len = distance(short_ep[0], short_ep[1])
            angle = math.degrees(
                math.atan2(long_ep[1][1] - long_ep[0][1], long_ep[1][0] - long_ep[0][0])
            )

            if long_len > 0 and short_len > 0:
                cv2.ellipse(
                    preview,
                    center_pt,
                    (int(round(long_len / 2.0)), int(round(short_len / 2.0))),
                    angle,
                    0,
                    360,
                    color=(0, 255, 0),
                    thickness=2,
                )

            cv2.line(preview, long_p1, long_p2, (255, 0, 0), 2)
            cv2.line(preview, short_p1, short_p2, (0, 255, 255), 2)
            cv2.circle(preview, center_pt, 4, (255, 255, 255), -1)
            cv2.putText(
                preview,
                knot_id,
                (center_pt[0] + 5, center_pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        except Exception:
            continue

    return preview


# =============================================================================
# Tkinter app
# =============================================================================

class ManualMeasureApp:
    def __init__(self, root: tk.Tk) -> None:
        ensure_dirs()

        self.root = root
        self.root.title("Manual Knot Measurement Tool")
        self.root.geometry("1280x800")

        self.image_path: Path | None = None
        self.image_bgr: np.ndarray | None = None
        self.image_rgb: np.ndarray | None = None
        self.image_width_px: int = 0
        self.image_height_px: int = 0

        self.scale: float = 1.0
        self.tk_image: ImageTk.PhotoImage | None = None

        self.current_polygon: list[Point] = []
        self.current_fit_points: list[Point] = []
        self.current_polygon_closed: bool = False

        self.current_knot_rows: list[dict[str, object]] = []
        self.current_detail_records: list[dict[str, object]] = []
        self.next_knot_index: int = 1

        self.lumber_id_var = tk.StringVar()
        self.surface_id_var = tk.StringVar()
        self.surface_width_var = tk.StringVar()
        self.lumber_length_var = tk.StringVar()
        self.image_path_var = tk.StringVar()

        self.selected_arc_mode_var = tk.BooleanVar(value=False)
        self.is_truncated_var = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="Select a surface image.")

        self.selection_frame: ttk.Frame | None = None
        self.annotation_frame: ttk.Frame | None = None
        self.canvas: tk.Canvas | None = None

        self._show_selection_frame()

    # -------------------------------------------------------------------------
    # Selection screen
    # -------------------------------------------------------------------------

    def _show_selection_frame(self) -> None:
        self._clear_root()

        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        self.selection_frame = frame

        ttk.Label(frame, text="Surface image selection", font=("Arial", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 20)
        )

        ttk.Label(frame, text="Image file").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.image_path_var, width=80).grid(
            row=1, column=1, sticky="ew", padx=5
        )
        ttk.Button(frame, text="Browse...", command=self._select_image).grid(row=1, column=2)

        ttk.Label(frame, text="lumber_id").grid(row=2, column=0, sticky="w", pady=(15, 0))
        ttk.Entry(frame, textvariable=self.lumber_id_var, width=30).grid(
            row=2, column=1, sticky="w", pady=(15, 0)
        )

        ttk.Label(frame, text="surface_id").grid(row=3, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.surface_id_var, width=30).grid(
            row=3, column=1, sticky="w"
        )

        ttk.Label(frame, text="surface_width_mm").grid(row=4, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.surface_width_var, width=30).grid(
            row=4, column=1, sticky="w"
        )

        ttk.Label(frame, text="lumber_length_mm").grid(row=5, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.lumber_length_var, width=30).grid(
            row=5, column=1, sticky="w"
        )

        ttk.Button(
            frame,
            text="Start measurement",
            command=self._start_annotation,
        ).grid(row=6, column=1, sticky="w", pady=25)

        ttk.Label(
            frame,
            textvariable=self.status_var,
            foreground="blue",
        ).grid(row=7, column=0, columnspan=3, sticky="w")

        frame.columnconfigure(1, weight=1)

    def _select_image(self) -> None:
        initial_dir = IMAGE_DIR if IMAGE_DIR.exists() else PROJECT_ROOT

        file_path = filedialog.askopenfilename(
            title="Select surface image",
            initialdir=str(initial_dir),
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        self.image_path = Path(file_path)
        self.image_path_var.set(str(self.image_path))

        if not self.surface_id_var.get():
            self.surface_id_var.set(self.image_path.stem)

    def _start_annotation(self) -> None:
        try:
            image_path = Path(self.image_path_var.get()).expanduser()

            if not image_path.exists():
                raise ValueError("Image file does not exist.")

            lumber_id = self.lumber_id_var.get().strip()
            surface_id = self.surface_id_var.get().strip()

            if not lumber_id:
                raise ValueError("lumber_id is required.")
            if not surface_id:
                raise ValueError("surface_id is required.")

            surface_width_mm = float(self.surface_width_var.get())
            lumber_length_mm = float(self.lumber_length_var.get())

            if surface_width_mm <= 0:
                raise ValueError("surface_width_mm must be positive.")
            if lumber_length_mm <= 0:
                raise ValueError("lumber_length_mm must be positive.")

            self.image_path = image_path
            self.image_bgr = read_image_bgr(image_path)
            self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)

            self.image_height_px, self.image_width_px = self.image_bgr.shape[:2]

            self.current_polygon = []
            self.current_fit_points = []
            self.current_polygon_closed = False
            self.current_knot_rows = []
            self.current_detail_records = []
            self.next_knot_index = count_existing_knots(lumber_id, surface_id) + 1

            self._show_annotation_frame()

        except Exception as error:
            messagebox.showerror("Error", str(error))

    # -------------------------------------------------------------------------
    # Annotation screen
    # -------------------------------------------------------------------------

    def _show_annotation_frame(self) -> None:
        self._clear_root()

        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)
        self.annotation_frame = frame

        control = ttk.Frame(frame, padding=8)
        control.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(
            control,
            text="Manual Measurement",
            font=("Arial", 13, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(control, text=f"lumber_id: {self.lumber_id_var.get()}").pack(anchor="w")
        ttk.Label(control, text=f"surface_id: {self.surface_id_var.get()}").pack(anchor="w")
        ttk.Label(control, text=f"image: {Path(self.image_path_var.get()).name}").pack(anchor="w")
        ttk.Label(control, text=f"size(px): {self.image_width_px} x {self.image_height_px}").pack(
            anchor="w", pady=(0, 10)
        )

        ttk.Checkbutton(
            control,
            text="Use selected ellipse arc points",
            variable=self.selected_arc_mode_var,
        ).pack(anchor="w", pady=(8, 2))

        ttk.Checkbutton(
            control,
            text="Truncated / cut knot",
            variable=self.is_truncated_var,
        ).pack(anchor="w", pady=(0, 8))

        ttk.Button(control, text="Save current knot", command=self._save_current_knot).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(control, text="Undo point", command=self._undo_point).pack(fill=tk.X, pady=2)
        ttk.Button(control, text="Cancel current knot", command=self._cancel_current_knot).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(control, text="Undo last saved knot", command=self._undo_last_saved_knot).pack(
            fill=tk.X, pady=2
        )

        ttk.Separator(control).pack(fill=tk.X, pady=10)

        ttk.Button(control, text="Zoom in", command=lambda: self._change_zoom(1.25)).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(control, text="Zoom out", command=lambda: self._change_zoom(0.8)).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(control, text="Fit to window", command=self._fit_to_window).pack(
            fill=tk.X, pady=2
        )

        ttk.Separator(control).pack(fill=tk.X, pady=10)

        ttk.Button(control, text="Finish this surface", command=self._finish_surface).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(control, text="Back without saving", command=self._back_without_saving).pack(
            fill=tk.X, pady=2
        )

        ttk.Label(
            control,
            textvariable=self.status_var,
            foreground="blue",
            wraplength=260,
        ).pack(anchor="w", pady=(15, 0))

        canvas_frame = ttk.Frame(frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, background="black")
        hbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)

        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")

        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas.bind("<Button-1>", self._on_left_click)

        self.root.update_idletasks()
        self._fit_to_window()
        self.status_var.set(
            "Click polygon points. Click the first point again to close the knot."
        )

    def _fit_to_window(self) -> None:
        if self.image_rgb is None:
            return

        screen_w = max(self.root.winfo_width() - 330, 400)
        screen_h = max(self.root.winfo_height() - 80, 300)

        scale_w = screen_w / self.image_width_px
        scale_h = screen_h / self.image_height_px

        self.scale = min(1.0, max(0.05, min(scale_w, scale_h)))
        self._render_image()

    def _change_zoom(self, factor: float) -> None:
        self.scale = max(0.03, min(5.0, self.scale * factor))
        self._render_image()

    def _render_image(self) -> None:
        if self.canvas is None or self.image_rgb is None:
            return

        display_w = max(1, int(self.image_width_px * self.scale))
        display_h = max(1, int(self.image_height_px * self.scale))

        image = Image.fromarray(self.image_rgb)

        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS

        image = image.resize((display_w, display_h), resample)
        self.tk_image = ImageTk.PhotoImage(image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags=("image",))
        self.canvas.configure(scrollregion=(0, 0, display_w, display_h))

        self._redraw_overlay()

    def _image_to_canvas(self, point: Point) -> Point:
        return point[0] * self.scale, point[1] * self.scale

    def _canvas_to_image(self, event: tk.Event) -> Point:
        if self.canvas is None:
            return 0.0, 0.0

        x = self.canvas.canvasx(event.x) / self.scale
        y = self.canvas.canvasy(event.y) / self.scale

        x = max(0.0, min(float(self.image_width_px - 1), x))
        y = max(0.0, min(float(self.image_height_px - 1), y))

        return x, y

    def _redraw_overlay(self) -> None:
        if self.canvas is None:
            return

        self.canvas.delete("overlay")

        # Saved knots
        for detail in self.current_detail_records:
            self._draw_detail_overlay(detail)

        # Current polygon
        if self.current_polygon:
            for i, point in enumerate(self.current_polygon):
                x, y = self._image_to_canvas(point)
                self.canvas.create_oval(
                    x - POLYGON_POINT_RADIUS,
                    y - POLYGON_POINT_RADIUS,
                    x + POLYGON_POINT_RADIUS,
                    y + POLYGON_POINT_RADIUS,
                    fill="red",
                    outline="white",
                    tags=("overlay",),
                )

                if i > 0:
                    x0, y0 = self._image_to_canvas(self.current_polygon[i - 1])
                    self.canvas.create_line(x0, y0, x, y, fill="red", width=2, tags=("overlay",))

            if self.current_polygon_closed and len(self.current_polygon) >= 3:
                x0, y0 = self._image_to_canvas(self.current_polygon[-1])
                x1, y1 = self._image_to_canvas(self.current_polygon[0])
                self.canvas.create_line(x0, y0, x1, y1, fill="red", width=2, tags=("overlay",))

        # Current selected ellipse-fit points
        for point in self.current_fit_points:
            x, y = self._image_to_canvas(point)
            self.canvas.create_oval(
                x - FIT_POINT_RADIUS,
                y - FIT_POINT_RADIUS,
                x + FIT_POINT_RADIUS,
                y + FIT_POINT_RADIUS,
                fill="cyan",
                outline="black",
                tags=("overlay",),
            )

    def _draw_detail_overlay(self, detail: dict[str, object]) -> None:
        if self.canvas is None:
            return

        polygon = detail.get("polygon_points", [])
        knot_id = str(detail.get("knot_id", ""))

        if polygon:
            canvas_points: list[float] = []

            for p in polygon:
                x, y = self._image_to_canvas((float(p[0]), float(p[1])))
                canvas_points.extend([x, y])

            self.canvas.create_polygon(
                canvas_points,
                outline="orange",
                fill="",
                width=2,
                tags=("overlay",),
            )

        ellipse = detail.get("ellipse", {})

        try:
            center = ellipse["center"]
            long_ep = ellipse["long_axis_endpoints"]
            short_ep = ellipse["short_axis_endpoints"]

            cx, cy = self._image_to_canvas((float(center[0]), float(center[1])))
            long_p1 = self._image_to_canvas((float(long_ep[0][0]), float(long_ep[0][1])))
            long_p2 = self._image_to_canvas((float(long_ep[1][0]), float(long_ep[1][1])))
            short_p1 = self._image_to_canvas((float(short_ep[0][0]), float(short_ep[0][1])))
            short_p2 = self._image_to_canvas((float(short_ep[1][0]), float(short_ep[1][1])))

            self.canvas.create_line(*long_p1, *long_p2, fill="blue", width=2, tags=("overlay",))
            self.canvas.create_line(*short_p1, *short_p2, fill="yellow", width=2, tags=("overlay",))
            self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="white", tags=("overlay",))
            self.canvas.create_text(
                cx + 12,
                cy - 12,
                text=knot_id,
                fill="white",
                anchor=tk.W,
                tags=("overlay",),
            )
        except Exception:
            return

    def _on_left_click(self, event: tk.Event) -> None:
        point = self._canvas_to_image(event)

        if self.current_polygon_closed:
            if self.selected_arc_mode_var.get():
                self.current_fit_points.append(point)
                self.status_var.set(
                    f"Ellipse arc points: {len(self.current_fit_points)}. "
                    "Click Save current knot when finished."
                )
                self._redraw_overlay()
            return

        if len(self.current_polygon) >= 3:
            first_point = self.current_polygon[0]
            threshold = CLOSE_POINT_RADIUS_SCREEN_PX / self.scale

            if distance(point, first_point) <= threshold:
                self.current_polygon_closed = True

                if self.selected_arc_mode_var.get():
                    self.status_var.set(
                        "Polygon closed. Select at least 5 ellipse arc points, "
                        "then click Save current knot."
                    )
                    self._redraw_overlay()
                else:
                    self._save_current_knot()

                return

        self.current_polygon.append(point)
        self._redraw_overlay()

    # -------------------------------------------------------------------------
    # Knot operations
    # -------------------------------------------------------------------------

    def _save_current_knot(self) -> None:
        try:
            if not self.current_polygon_closed:
                if len(self.current_polygon) < 3:
                    raise ValueError("At least 3 polygon points are required.")
                self.current_polygon_closed = True

            if len(self.current_polygon) < 3:
                raise ValueError("At least 3 polygon points are required.")

            if self.selected_arc_mode_var.get():
                if len(self.current_fit_points) < 5:
                    raise ValueError(
                        "At least 5 ellipse arc points are required in selected-arc mode."
                    )
                fit_points = self.current_fit_points[:]
                fit_method = "selected_arc_fit"
                stored_fit_points = self.current_fit_points[:]
            else:
                fit_points = densify_polyline(
                    self.current_polygon,
                    closed=True,
                    step=AUTO_DENSIFY_STEP_PX,
                )
                fit_method = "fit_ellipse"
                stored_fit_points = self.current_polygon[:]

            ellipse = fit_ellipse_from_points(fit_points, fit_method)
            bbox = bbox_from_polygon(self.current_polygon)

            lumber_id = self.lumber_id_var.get().strip()
            surface_id = self.surface_id_var.get().strip()
            image_file = path_to_storable_string(Path(self.image_path_var.get()))

            knot_id = f"K{self.next_knot_index:03d}"
            self.next_knot_index += 1

            created_at = now_string()

            row = {
                "lumber_id": lumber_id,
                "surface_id": surface_id,
                "knot_id": knot_id,
                "image_file": image_file,
                "length_min_pos": bbox["length_min_pos"],
                "length_max_pos": bbox["length_max_pos"],
                "width_min_pos": bbox["width_min_pos"],
                "width_max_pos": bbox["width_max_pos"],
                "center_point_length": ellipse["center"][0],
                "center_point_width": ellipse["center"][1],
                "long_diam_length": ellipse["long_diam_length"],
                "long_diam_width": ellipse["long_diam_width"],
                "short_diam_length": ellipse["short_diam_length"],
                "short_diam_width": ellipse["short_diam_width"],
                "ellipse_method": ellipse["method"],
                "is_truncated": str(bool(self.is_truncated_var.get())),
                "created_at": created_at,
            }

            detail = {
                "lumber_id": lumber_id,
                "surface_id": surface_id,
                "knot_id": knot_id,
                "image_file": image_file,
                "polygon_points": [
                    [round_float(x), round_float(y)]
                    for x, y in self.current_polygon
                ],
                "ellipse_fit_points": [
                    [round_float(x), round_float(y)]
                    for x, y in stored_fit_points
                ],
                "ellipse_method": ellipse["method"],
                "bbox": bbox,
                "ellipse": ellipse,
                "is_truncated": bool(self.is_truncated_var.get()),
                "created_at": created_at,
                "comment": "",
            }

            self.current_knot_rows.append(row)
            self.current_detail_records.append(detail)

            self._cancel_current_knot(reset_truncated=True)

            self.status_var.set(f"Saved {knot_id}. Continue clicking next knot.")

        except Exception as error:
            messagebox.showerror("Save current knot failed", str(error))

    def _undo_point(self) -> None:
        if self.current_polygon_closed and self.current_fit_points:
            self.current_fit_points.pop()
        elif self.current_polygon:
            self.current_polygon.pop()
            self.current_polygon_closed = False

        self._redraw_overlay()

    def _cancel_current_knot(self, reset_truncated: bool = False) -> None:
        self.current_polygon = []
        self.current_fit_points = []
        self.current_polygon_closed = False

        if reset_truncated:
            self.is_truncated_var.set(False)

        self._redraw_overlay()

    def _undo_last_saved_knot(self) -> None:
        if not self.current_knot_rows:
            messagebox.showinfo("Undo", "No saved knot in this surface session.")
            return

        removed = self.current_knot_rows.pop()
        self.current_detail_records.pop()
        self.next_knot_index = max(1, self.next_knot_index - 1)

        self.status_var.set(f"Removed {removed.get('knot_id')}.")
        self._redraw_overlay()

    # -------------------------------------------------------------------------
    # Surface operations
    # -------------------------------------------------------------------------

    def _finish_surface(self) -> None:
        try:
            if self.current_polygon:
                proceed = messagebox.askyesno(
                    "Unsaved current knot",
                    "There is an unfinished knot. Finish surface without saving it?",
                )
                if not proceed:
                    return

            lumber_id = self.lumber_id_var.get().strip()
            surface_id = self.surface_id_var.get().strip()
            image_file = path_to_storable_string(Path(self.image_path_var.get()))
            surface_width_mm = float(self.surface_width_var.get())
            lumber_length_mm = float(self.lumber_length_var.get())

            created_at = now_string()

            surface_row = {
                "lumber_id": lumber_id,
                "surface_id": surface_id,
                "image_file": image_file,
                "surface_width_mm": surface_width_mm,
                "lumber_length_mm": lumber_length_mm,
                "length_px": self.image_width_px,
                "width_px": self.image_height_px,
                "created_at": created_at,
            }

            upsert_surface_info(surface_row)
            append_csv_rows(
                KNOT_MEASUREMENTS_PATH,
                KNOT_MEASUREMENT_COLUMNS,
                self.current_knot_rows,
            )
            append_jsonl(ANNOTATION_DETAIL_PATH, self.current_detail_records)

            if self.image_bgr is not None:
                preview = draw_preview(self.image_bgr, self.current_detail_records)
                preview_path = PREVIEW_DIR / f"{lumber_id}_{surface_id}_annotated.png"
                write_image(preview_path, preview)

            messagebox.showinfo(
                "Saved",
                f"Saved surface {surface_id}.\n"
                f"Knots: {len(self.current_knot_rows)}",
            )

            self._prepare_next_surface()

        except Exception as error:
            messagebox.showerror("Finish surface failed", str(error))

    def _prepare_next_surface(self) -> None:
        self.image_path = None
        self.image_bgr = None
        self.image_rgb = None
        self.image_width_px = 0
        self.image_height_px = 0

        self.current_polygon = []
        self.current_fit_points = []
        self.current_polygon_closed = False
        self.current_knot_rows = []
        self.current_detail_records = []

        self.image_path_var.set("")
        self.surface_id_var.set("")
        self.status_var.set("Select next surface image.")

        self._show_selection_frame()

    def _back_without_saving(self) -> None:
        proceed = messagebox.askyesno(
            "Back without saving",
            "Current surface annotations will be discarded. Continue?",
        )

        if proceed:
            self._prepare_next_surface()

    # -------------------------------------------------------------------------
    # Common
    # -------------------------------------------------------------------------

    def _clear_root(self) -> None:
        for child in self.root.winfo_children():
            child.destroy()


def main() -> None:
    root = tk.Tk()
    ManualMeasureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()