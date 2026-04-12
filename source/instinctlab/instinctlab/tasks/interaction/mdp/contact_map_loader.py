from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

__all__ = [
    "SparseContactPartMetadata",
    "SparseContactMapMetadata",
    "load_sparse_contact_map_directory",
    "load_sparse_contact_map_file",
]


@dataclass(frozen=True)
class SparseContactPartMetadata:
    """Sparse contact metadata for one object part in object-local frame."""

    exists: bool
    points_local: torch.Tensor
    center_local: torch.Tensor
    normal_local: torch.Tensor | None = None


@dataclass(frozen=True)
class SparseContactMapMetadata:
    """Sparse contact metadata for one object family."""

    object_key: str
    usd_basename: str
    robot_links: tuple[str, ...]
    object_parts_order: tuple[str, ...]
    parts: dict[str, SparseContactPartMetadata]
    relation: torch.Tensor
    exists_mask: torch.Tensor


def load_sparse_contact_map_directory(metadata_dir: str | Path) -> dict[str, SparseContactMapMetadata]:
    """Load every sparse contact metadata file in a directory keyed by object_key."""
    directory = Path(metadata_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Sparse contact metadata directory does not exist: {directory}")

    metadata_by_key: dict[str, SparseContactMapMetadata] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
            continue
        metadata = load_sparse_contact_map_file(path)
        if metadata.object_key in metadata_by_key:
            raise ValueError(f"Duplicate sparse contact metadata object_key='{metadata.object_key}' in {directory}")
        metadata_by_key[metadata.object_key] = metadata

    if len(metadata_by_key) == 0:
        raise ValueError(f"No sparse contact metadata files found in: {directory}")
    return metadata_by_key


def load_sparse_contact_map_file(path: str | Path) -> SparseContactMapMetadata:
    """Load one sparse contact metadata file."""
    file_path = Path(path)
    raw = _read_raw_metadata(file_path)

    object_key = str(raw["object_key"])
    usd_basename = Path(raw["usd_basename"]).name
    robot_links = tuple(str(name) for name in raw["robot_links"])
    object_parts_order = tuple(str(name) for name in raw["object_parts_order"])

    parts_raw = raw["parts"]
    missing_parts = [name for name in object_parts_order if name not in parts_raw]
    if missing_parts:
        raise ValueError(f"Metadata '{file_path}' is missing parts: {missing_parts}")

    parts: dict[str, SparseContactPartMetadata] = {}
    exists_mask = []
    for part_name in object_parts_order:
        part_raw = parts_raw[part_name]
        exists = bool(part_raw["exists"])
        points_local = _as_points_tensor(part_raw.get("points_local", []), file_path=file_path, field=part_name)
        center_local = _as_vector_tensor(part_raw["center_local"], file_path=file_path, field=f"{part_name}.center_local")
        normal_local_raw = part_raw.get("normal_local", None)
        normal_local = (
            _as_vector_tensor(normal_local_raw, file_path=file_path, field=f"{part_name}.normal_local")
            if normal_local_raw is not None
            else None
        )
        if exists and points_local.shape[0] == 0:
            raise ValueError(f"Metadata '{file_path}' part '{part_name}' is marked exists=True but has no points.")
        parts[part_name] = SparseContactPartMetadata(
            exists=exists,
            points_local=points_local,
            center_local=center_local,
            normal_local=normal_local,
        )
        exists_mask.append(exists)

    relation = torch.as_tensor(raw["relation"], dtype=torch.int64)
    expected_shape = (len(robot_links), len(object_parts_order))
    if tuple(relation.shape) != expected_shape:
        raise ValueError(
            f"Metadata '{file_path}' relation shape mismatch: expected {expected_shape}, got {tuple(relation.shape)}."
        )
    if not torch.isin(relation, torch.tensor([-1, 0, 1], dtype=relation.dtype)).all():
        raise ValueError(f"Metadata '{file_path}' relation must only contain -1, 0, 1.")

    return SparseContactMapMetadata(
        object_key=object_key,
        usd_basename=usd_basename,
        robot_links=robot_links,
        object_parts_order=object_parts_order,
        parts=parts,
        relation=relation,
        exists_mask=torch.tensor(exists_mask, dtype=torch.bool),
    )


def _read_raw_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Sparse contact metadata file does not exist: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text())
    raise ValueError(f"Unsupported sparse contact metadata extension: {path.suffix}")


def _as_points_tensor(raw_value: Any, file_path: Path, field: str) -> torch.Tensor:
    points = torch.as_tensor(raw_value, dtype=torch.float32)
    if points.numel() == 0:
        return torch.zeros((0, 3), dtype=torch.float32)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(
            f"Metadata '{file_path}' field '{field}.points_local' must have shape (K, 3), got {tuple(points.shape)}."
        )
    return points


def _as_vector_tensor(raw_value: Any, file_path: Path, field: str) -> torch.Tensor:
    vector = torch.as_tensor(raw_value, dtype=torch.float32)
    if vector.shape != (3,):
        raise ValueError(f"Metadata '{file_path}' field '{field}' must have shape (3,), got {tuple(vector.shape)}.")
    return vector
