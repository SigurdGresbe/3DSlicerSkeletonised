from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SlicerConfig:
    stl_file: Path = PACKAGE_ROOT / "mesh" / "curve1.stl"
    log_dir: Path = PACKAGE_ROOT / "logs"
    output_dir: Path = PACKAGE_ROOT / "outputs"
    plot_filename_prefix: str = "ortho_plot"
