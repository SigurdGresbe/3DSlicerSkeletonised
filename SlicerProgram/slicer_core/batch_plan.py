import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


class BatchPlanError(ValueError):
    """Raised when a planned parameter batch CSV cannot be parsed."""


@dataclass(frozen=True)
class BatchPlanRow:
    index: int
    source_row_number: int
    label: str
    overrides: Dict[str, Any]
    raw_values: Dict[str, str]


@dataclass(frozen=True)
class BatchPlan:
    path: Path
    rows: List[BatchPlanRow]
    ignored_headers: List[str]


def _normalize_header(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _parse_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise BatchPlanError(f"Could not parse boolean value {value!r}.")


def _parse_intlike(value: str) -> int:
    text = str(value).strip().lower().replace(" ", "")
    multiplier = 1
    if text.endswith("k"):
        multiplier = 1000
        text = text[:-1]
    return int(round(float(text) * multiplier))


_COLUMN_SPECS: Dict[str, Dict[str, Any]] = {
    "label": {
        "aliases": ("label", "trial", "name", "run"),
        "parser": str,
    },
    "stl_file": {
        "aliases": ("stl_file", "mesh", "mesh_path"),
        "parser": str,
    },
    "knn_k": {
        "aliases": ("k", "knn_k", "knn", "k_value"),
        "parser": _parse_intlike,
    },
    "overlap_factor": {
        "aliases": ("of", "overlap_factor", "overlap", "overlapfactor"),
        "parser": float,
    },
    "centreline_samples": {
        "aliases": ("samples", "centreline_samples", "sample_count"),
        "parser": _parse_intlike,
    },
    "line_method": {
        "aliases": ("line_method",),
        "parser": str,
    },
    "sphere_generation_method": {
        "aliases": ("sphere_generation_method", "sphere_method"),
        "parser": str,
    },
    "plane_spacing": {
        "aliases": ("plane_spacing", "plane_spacing_mm"),
        "parser": float,
    },
    "shell_point_spacing": {
        "aliases": ("shell_point_spacing", "shell_spacing", "shell_point_spacing_mm"),
        "parser": float,
    },
    "start_z": {
        "aliases": ("start_z",),
        "parser": float,
    },
    "start_z_tolerance": {
        "aliases": ("start_z_tolerance",),
        "parser": float,
    },
    "sphere_min_diameter": {
        "aliases": ("sphere_min_diameter", "min_diameter", "dmin"),
        "parser": float,
    },
    "sphere_max_diameter": {
        "aliases": ("sphere_max_diameter", "max_diameter", "dmax"),
        "parser": float,
    },
    "spline_s": {
        "aliases": ("spline_s", "spline_smoothing"),
        "parser": float,
    },
    "centreline_extension_length": {
        "aliases": ("centreline_extension_length", "extension_length"),
        "parser": float,
    },
    "tree_sphere_graph_k": {
        "aliases": ("tree_sphere_graph_k", "tree_graph_k"),
        "parser": _parse_intlike,
    },
    "graph_strategy": {
        "aliases": ("graph_strategy",),
        "parser": str,
    },
    "max_point_distance_from_centreline": {
        "aliases": ("max_point_distance_from_centreline", "max_point_distance"),
        "parser": float,
    },
    "adaptive_plane_spacing": {
        "aliases": ("adaptive_plane_spacing", "adaptive_spacing"),
        "parser": _parse_bool,
    },
    "adaptive_spacing_min_factor": {
        "aliases": ("adaptive_spacing_min_factor", "adaptive_min_factor"),
        "parser": float,
    },
    "outer_corner_spacing_safety": {
        "aliases": ("outer_corner_spacing_safety", "corner_safety"),
        "parser": float,
    },
    "enable_nurbs": {
        "aliases": ("enable_nurbs",),
        "parser": _parse_bool,
    },
    "export_csv": {
        "aliases": ("export_csv",),
        "parser": _parse_bool,
    },
    "export_layers_json": {
        "aliases": ("export_layers_json", "export_json"),
        "parser": _parse_bool,
    },
}


_ALIASES_TO_CANONICAL: Dict[str, str] = {}
for canonical_name, spec in _COLUMN_SPECS.items():
    for alias in spec["aliases"]:
        _ALIASES_TO_CANONICAL[_normalize_header(alias)] = canonical_name


def _coerce_value(parser: Callable[[str], Any], value: str, header: str, row_number: int) -> Any:
    try:
        return parser(value)
    except BatchPlanError:
        raise
    except Exception as exc:
        raise BatchPlanError(
            f"Invalid value {value!r} for column {header!r} on CSV row {row_number}: {exc}"
        ) from exc


def load_batch_plan_csv(path: Path) -> BatchPlan:
    path = Path(path)
    if not path.is_file():
        raise BatchPlanError(f"Batch plan CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise BatchPlanError("Batch plan CSV is missing a header row.")

        normalized_headers: List[Tuple[str, str]] = [
            (header, _normalize_header(header)) for header in reader.fieldnames if header is not None
        ]
        ignored_headers = [
            original
            for original, normalized in normalized_headers
            if normalized and normalized not in _ALIASES_TO_CANONICAL
        ]

        rows: List[BatchPlanRow] = []
        for row_number, row in enumerate(reader, start=2):
            cleaned_raw = {
                str(key): str(value).strip()
                for key, value in row.items()
                if key is not None and value is not None and str(value).strip()
            }
            if not cleaned_raw:
                continue

            first_value = next(iter(cleaned_raw.values()), "")
            if first_value.startswith("#"):
                continue

            overrides: Dict[str, Any] = {}
            label = ""

            for original_header, normalized_header in normalized_headers:
                if not original_header:
                    continue
                raw_value = row.get(original_header)
                if raw_value is None:
                    continue
                raw_value = str(raw_value).strip()
                if not raw_value:
                    continue

                canonical_name = _ALIASES_TO_CANONICAL.get(normalized_header)
                if canonical_name is None:
                    continue

                parser = _COLUMN_SPECS[canonical_name]["parser"]
                parsed_value = _coerce_value(parser, raw_value, original_header, row_number)
                if canonical_name == "label":
                    label = str(parsed_value).strip()
                else:
                    overrides[canonical_name] = parsed_value

            if not overrides:
                continue

            run_index = len(rows) + 1
            if not label:
                label = f"Run {run_index}"

            rows.append(
                BatchPlanRow(
                    index=run_index,
                    source_row_number=row_number,
                    label=label,
                    overrides=overrides,
                    raw_values=cleaned_raw,
                )
            )

    if not rows:
        raise BatchPlanError(
            "Batch plan CSV did not contain any runnable rows. "
            "Add at least one row with parameter columns such as k, of, or samples."
        )

    return BatchPlan(path=path, rows=rows, ignored_headers=ignored_headers)
