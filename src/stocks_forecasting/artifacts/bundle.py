"""Model artifact bundle management."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BundlePaths:
    """Paths for a model artifact bundle."""

    root: Path
    manifest_path: Path
    config_snapshot_path: Path
    metrics_path: Path
    model_dir: Path
    preprocessing_dir: Path


def create_bundle_paths(*, root_dir: Path, bundle_name: str, version: str) -> BundlePaths:
    """Create bundle paths for a model version."""
    root = root_dir / bundle_name / version
    return BundlePaths(
        root=root,
        manifest_path=root / "manifest.json",
        config_snapshot_path=root / "config.snapshot.json",
        metrics_path=root / "metrics.json",
        model_dir=root / "model",
        preprocessing_dir=root / "preprocessing",
    )


def now_utc_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write dict to JSON file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

