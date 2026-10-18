from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WATERMARK_CONCAT_ROOT = PROJECT_ROOT.parent
DIFFUSETRACE_ROOT = WATERMARK_CONCAT_ROOT / "reference_weights"


@dataclass(frozen=True)
class LocalPaths:
    project_root: Path = PROJECT_ROOT
    diffusetrace_root: Path = DIFFUSETRACE_ROOT
    sd21_model: Path = Path("weights/stable-diffusion-2-1-base")
    sdxl_model: Path = DIFFUSETRACE_ROOT / "weights" / "stable-diffusion-xl-base-1.0"
    prompts: Path = DIFFUSETRACE_ROOT / "data" / "sd_prompts"
    coco_val2017: Path = DIFFUSETRACE_ROOT / "data" / "val2017"
    seed_images: Path = PROJECT_ROOT / "data" / "seed_images"
    runs: Path = PROJECT_ROOT / "runs"


PATHS = LocalPaths()


def discover_local_models() -> dict[str, Path]:
    """Return locally available diffusion model folders without downloading."""
    candidates = {
        "sd21": PATHS.sd21_model,
        "sdxl": PATHS.sdxl_model,
    }
    return {
        name: path
        for name, path in candidates.items()
        if (path / "model_index.json").exists()
    }


def ensure_project_dirs() -> None:
    for path in (PATHS.seed_images, PATHS.runs):
        path.mkdir(parents=True, exist_ok=True)
