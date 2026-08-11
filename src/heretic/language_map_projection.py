"""Frozen three-dimensional projections for text-free geometry packages."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from safetensors.torch import save_file
import torch
from torch import Tensor


@dataclass(frozen=True)
class ProjectionBasis:
    centers: Tensor
    components: Tensor
    explained_variance: Tensor
    seed: int
    algorithm: str


def _canonicalize_component_signs(components: Tensor) -> Tensor:
    canonical = components.clone()
    for layer in range(canonical.shape[0]):
        for component in range(canonical.shape[1]):
            vector = canonical[layer, component]
            pivot = int(torch.argmax(torch.abs(vector)))
            if float(vector[pivot]) < 0.0:
                canonical[layer, component].neg_()
    return canonical


def fit_frozen_projection(
    residuals: Tensor,
    *,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> ProjectionBasis:
    """Fit one immutable three-component PCA basis per model layer."""

    if residuals.ndim != 3:
        raise ValueError("residuals must have shape [rows, layers, hidden]")
    rows, layers, hidden = map(int, residuals.shape)
    if min(rows, hidden) < 3:
        raise ValueError("at least three rows and hidden dimensions are required")
    target = torch.device(device)
    centers = torch.empty((layers, hidden), dtype=torch.float32)
    components = torch.empty((layers, 3, hidden), dtype=torch.float32)
    explained = torch.empty((layers, 3), dtype=torch.float32)
    exact = rows * hidden <= 2_000_000

    for layer in range(layers):
        matrix = residuals[:, layer, :].detach().to(target, torch.float32)
        center = matrix.mean(dim=0)
        centered = matrix - center
        if exact:
            _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
            layer_components = vh[:3]
            layer_singular = singular[:3]
        else:
            devices = [target.index or 0] if target.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed + layer)
                if target.type == "cuda":
                    torch.cuda.manual_seed_all(seed + layer)
                _, singular, vectors = torch.pca_lowrank(
                    centered,
                    q=min(8, rows, hidden),
                    center=False,
                    niter=4,
                )
            layer_components = vectors[:, :3].T
            layer_singular = singular[:3]
        centers[layer] = center.cpu()
        components[layer] = layer_components.cpu()
        explained[layer] = (layer_singular.square() / max(rows - 1, 1)).cpu()

    components = _canonicalize_component_signs(components)
    return ProjectionBasis(
        centers=centers.contiguous(),
        components=components.contiguous(),
        explained_variance=explained.contiguous(),
        seed=int(seed),
        algorithm="exact_svd" if exact else "torch_pca_lowrank_q8_niter4",
    )


def project_residuals(residuals: Tensor, basis: ProjectionBasis) -> Tensor:
    """Project residuals with an existing basis without fitting new axes."""

    if residuals.ndim != 3:
        raise ValueError("residuals must have shape [rows, layers, hidden]")
    if tuple(residuals.shape[1:]) != (
        int(basis.centers.shape[0]),
        int(basis.centers.shape[1]),
    ):
        raise ValueError("residual shape does not match projection basis")
    centered = residuals.detach().to(torch.float32) - basis.centers.unsqueeze(0)
    return torch.einsum("rlh,lch->rlc", centered, basis.components).contiguous()


def retained_shift_ratio(
    reference: Tensor,
    candidate: Tensor,
    basis: ProjectionBasis,
) -> Tensor:
    """Return how much full-space movement remains visible in frozen 3D."""

    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate residual shapes differ")
    full = torch.linalg.vector_norm(candidate.float() - reference.float(), dim=-1)
    projected = torch.linalg.vector_norm(
        project_residuals(candidate, basis) - project_residuals(reference, basis),
        dim=-1,
    )
    ratio = torch.where(full > 1e-12, projected / full, torch.ones_like(full))
    return ratio.clamp_(0.0, 1.0 + 1e-6)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _public_index(index: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for position, row in enumerate(index):
        direction = str(row.get("direction_class", "")).lower()
        if direction not in {"safe", "unsafe"}:
            raise ValueError("base index direction must be safe or unsafe")
        output.append(
            {
                "index": position,
                "canonical_id": str(row["canonical_id"]),
                "row_id": str(row["row_id"]),
                "language": str(row["language"]),
                "group": "A" if direction == "safe" else "B",
                "category_id": str(row["category_id"]),
            }
        )
    return output


def write_projection_package(
    *,
    index: list[dict[str, object]],
    residuals: Tensor,
    output_dir: Path,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Atomically create the immutable base of a 3D geometry package."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        basis = fit_frozen_projection(residuals, seed=seed, device=device)
        points = project_residuals(residuals, basis).cpu().numpy().astype("<f4")
        if not bool(np.isfinite(points).all()):
            raise ValueError("projected points contain non-finite values")

        basis_path = temporary / "projection_basis.safetensors"
        save_file(
            {
                "centers": basis.centers,
                "components": basis.components,
                "explained_variance": basis.explained_variance,
            },
            str(basis_path),
            metadata={
                "schema_version": "1",
                "seed": str(seed),
                "algorithm": basis.algorithm,
            },
        )
        points_path = temporary / "base_points.f32"
        points.tofile(points_path)
        index_path = temporary / "base_index.json"
        _write_json(index_path, _public_index(index))

        files = {}
        for path in (basis_path, points_path, index_path):
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "PASS",
            "rows": int(residuals.shape[0]),
            "layers": int(residuals.shape[1]),
            "hidden_size": int(residuals.shape[2]),
            "shape": [int(residuals.shape[0]), int(residuals.shape[1]), 3],
            "projection": {
                "dimensions": 3,
                "basis_scope": "original_model_only",
                "seed": int(seed),
                "algorithm": basis.algorithm,
            },
            "files": files,
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
