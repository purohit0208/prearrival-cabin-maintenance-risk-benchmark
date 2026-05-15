from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "synthetic_intact_llm_v2_realistic"
RUN_NAME = "RUN_20260302_093458"


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DATA_DIR = Path(os.environ["INTACT_DATA_DIR"]) if "INTACT_DATA_DIR" in os.environ else _first_existing(
    [
        PROJECT_ROOT / "data" / DATASET_NAME,
        LEGACY_ROOT / "INTACT" / "data" / DATASET_NAME,
    ]
)

RUN_DIR = Path(os.environ["INTACT_RUN_DIR"]) if "INTACT_RUN_DIR" in os.environ else _first_existing(
    [
        PROJECT_ROOT / "runs" / RUN_NAME,
        LEGACY_ROOT / "INTACT" / "runs" / RUN_NAME,
    ]
)

OUT_DIR = Path(os.environ["INTACT_OUT_DIR"]) if "INTACT_OUT_DIR" in os.environ else PROJECT_ROOT / "outputs"
