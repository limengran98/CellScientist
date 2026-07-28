from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .data import TaskData
from .schemas import CandidateConfig, ContractError


def _clean_with_train_median(
    values: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cleaned = np.asarray(values, dtype=np.float32).copy()
    cleaned[~np.isfinite(cleaned)] = np.nan
    medians = np.nanmedian(cleaned[train_indices], axis=0)
    medians[~np.isfinite(medians)] = 0.0
    missing = np.where(np.isnan(cleaned))
    if missing[0].size:
        cleaned[missing] = medians[missing[1]]
    return cleaned, medians.astype(np.float32)


def _hashed_smiles_features(
    smiles: Sequence[object],
    dimension: int,
    ngram_min: int,
    ngram_max: int,
) -> np.ndarray:
    output = np.zeros((len(smiles), dimension), dtype=np.float32)
    cache: Dict[str, np.ndarray] = {}
    for row, raw in enumerate(smiles):
        text = str(raw)
        if text in cache:
            output[row] = cache[text]
            continue
        vector = np.zeros(dimension, dtype=np.float32)
        padded = f"^{text}$"
        for width in range(ngram_min, ngram_max + 1):
            for start in range(max(0, len(padded) - width + 1)):
                token = padded[start : start + width]
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                integer = int.from_bytes(digest, "little", signed=False)
                index = integer % dimension
                sign = 1.0 if ((integer >> 8) & 1) else -1.0
                vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        cache[text] = vector
        output[row] = vector
    return output


@dataclass
class FittedLowRankRidge:
    candidate: CandidateConfig
    pre_medians: np.ndarray
    target_medians: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    x_pca: PCA
    y_pca: PCA
    ridge: Ridge
    smiles_hash_dim: int
    smiles_ngram_min: int
    smiles_ngram_max: int

    def _input_matrix(self, data: TaskData, indices: np.ndarray) -> np.ndarray:
        pre = np.asarray(data.morphology_pre[indices], dtype=np.float32).copy()
        pre[~np.isfinite(pre)] = np.nan
        missing = np.where(np.isnan(pre))
        if missing[0].size:
            pre[missing] = self.pre_medians[missing[1]]
        blocks = [pre]
        if self.candidate.input_mode in {"pre_dose", "pre_smiles_dose"}:
            dose = np.log10(np.maximum(data.dose[indices], 0.0) + 1e-6).reshape(-1, 1)
            blocks.append(dose.astype(np.float32))
        if self.candidate.input_mode in {"pre_smiles", "pre_smiles_dose"}:
            blocks.append(
                _hashed_smiles_features(
                    data.smiles[indices],
                    self.smiles_hash_dim,
                    self.smiles_ngram_min,
                    self.smiles_ngram_max,
                )
            )
        matrix = np.concatenate(blocks, axis=1).astype(np.float32)
        return (matrix - self.x_mean) / self.x_scale

    def predict(self, data: TaskData, indices: Sequence[int]) -> np.ndarray:
        index_array = np.asarray(indices, dtype=np.int64)
        standardized = self._input_matrix(data, index_array)
        x_latent = self.x_pca.transform(standardized)
        y_latent = self.ridge.predict(x_latent)
        target = self.y_pca.inverse_transform(y_latent).astype(np.float32)
        if self.candidate.target_mode == "delta":
            pre = np.asarray(data.morphology_pre[index_array], dtype=np.float32).copy()
            pre[~np.isfinite(pre)] = np.nan
            missing = np.where(np.isnan(pre))
            if missing[0].size:
                pre[missing] = self.pre_medians[missing[1]]
            target = target + pre
        return np.asarray(target, dtype=np.float32)


def _torch_device(backend: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ContractError(
            "The torch backend requires PyTorch to be installed"
        ) from exc
    if backend == "torch_cpu":
        return torch.device("cpu")
    if backend == "torch_cuda":
        if not torch.cuda.is_available():
            raise ContractError(
                "model.backend=torch_cuda requires a visible CUDA device"
            )
        return torch.device("cuda")
    raise ContractError(f"Unsupported torch backend: {backend}")


def _torch_pca(
    matrix: Any,
    components: int,
    solver: str = "exact",
    niter: int = 4,
) -> tuple[Any, Any, Any]:
    """PCA used by the CPU and CUDA torch backends."""
    import torch

    mean = torch.mean(matrix, dim=0)
    centered = matrix - mean
    if solver == "randomized":
        _, _, basis = torch.pca_lowrank(
            matrix,
            q=components,
            center=True,
            niter=niter,
        )
        latent = centered @ basis
        return mean, basis, latent
    if solver != "exact":
        raise ContractError(f"Unsupported torch PCA solver: {solver}")
    denominator = max(int(matrix.shape[0]) - 1, 1)
    covariance = centered.T @ centered / float(denominator)
    _, eigenvectors = torch.linalg.eigh(covariance)
    basis = torch.flip(eigenvectors[:, -components:], dims=(1,))
    latent = centered @ basis
    return mean, basis, latent


@dataclass
class FittedTorchLowRankRidge:
    candidate: CandidateConfig
    pre_medians: np.ndarray
    target_medians: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    x_pca_mean: np.ndarray
    x_components: np.ndarray
    y_pca_mean: np.ndarray
    y_components: np.ndarray
    ridge_coef: np.ndarray
    ridge_intercept: np.ndarray
    smiles_hash_dim: int
    smiles_ngram_min: int
    smiles_ngram_max: int
    backend: str

    def _input_matrix(self, data: TaskData, indices: np.ndarray) -> np.ndarray:
        pre = np.asarray(data.morphology_pre[indices], dtype=np.float32).copy()
        pre[~np.isfinite(pre)] = np.nan
        missing = np.where(np.isnan(pre))
        if missing[0].size:
            pre[missing] = self.pre_medians[missing[1]]
        blocks = [pre]
        if self.candidate.input_mode in {"pre_dose", "pre_smiles_dose"}:
            dose = np.log10(np.maximum(data.dose[indices], 0.0) + 1e-6).reshape(
                -1, 1
            )
            blocks.append(dose.astype(np.float32))
        if self.candidate.input_mode in {"pre_smiles", "pre_smiles_dose"}:
            blocks.append(
                _hashed_smiles_features(
                    data.smiles[indices],
                    self.smiles_hash_dim,
                    self.smiles_ngram_min,
                    self.smiles_ngram_max,
                )
            )
        matrix = np.concatenate(blocks, axis=1).astype(np.float32)
        return (matrix - self.x_mean) / self.x_scale

    def predict(self, data: TaskData, indices: Sequence[int]) -> np.ndarray:
        import torch

        index_array = np.asarray(indices, dtype=np.int64)
        standardized = self._input_matrix(data, index_array)
        device = _torch_device(self.backend)
        with torch.no_grad():
            x = torch.as_tensor(standardized, dtype=torch.float32, device=device)
            x_pca_mean = torch.as_tensor(
                self.x_pca_mean, dtype=torch.float32, device=device
            )
            x_components = torch.as_tensor(
                self.x_components, dtype=torch.float32, device=device
            )
            ridge_coef = torch.as_tensor(
                self.ridge_coef, dtype=torch.float32, device=device
            )
            ridge_intercept = torch.as_tensor(
                self.ridge_intercept, dtype=torch.float32, device=device
            )
            y_components = torch.as_tensor(
                self.y_components, dtype=torch.float32, device=device
            )
            y_pca_mean = torch.as_tensor(
                self.y_pca_mean, dtype=torch.float32, device=device
            )
            x_latent = (x - x_pca_mean) @ x_components
            y_latent = x_latent @ ridge_coef + ridge_intercept
            target = y_latent @ y_components.T + y_pca_mean
            target_array = target.cpu().numpy().astype(np.float32)
        if self.candidate.target_mode == "delta":
            pre = np.asarray(
                data.morphology_pre[index_array], dtype=np.float32
            ).copy()
            pre[~np.isfinite(pre)] = np.nan
            missing = np.where(np.isnan(pre))
            if missing[0].size:
                pre[missing] = self.pre_medians[missing[1]]
            target_array = target_array + pre
        return np.asarray(target_array, dtype=np.float32)


FittedModel = Union[FittedLowRankRidge, FittedTorchLowRankRidge]


def _fit_sklearn_candidate(
    data: TaskData,
    candidate: CandidateConfig,
    model_config: Mapping[str, Any],
    seed: int,
) -> FittedLowRankRidge:
    train = np.asarray(data.partitions.train, dtype=np.int64)
    pre, pre_medians = _clean_with_train_median(data.morphology_pre, train)
    post, target_medians = _clean_with_train_median(data.morphology_post, train)

    input_blocks = [pre]
    if candidate.input_mode in {"pre_dose", "pre_smiles_dose"}:
        dose = np.log10(np.maximum(data.dose, 0.0) + 1e-6).reshape(-1, 1)
        input_blocks.append(dose.astype(np.float32))
    if candidate.input_mode in {"pre_smiles", "pre_smiles_dose"}:
        input_blocks.append(
            _hashed_smiles_features(
                data.smiles,
                int(model_config["smiles_hash_dim"]),
                int(model_config["smiles_ngram_min"]),
                int(model_config["smiles_ngram_max"]),
            )
        )
    x_all = np.concatenate(input_blocks, axis=1).astype(np.float32)
    x_mean = np.mean(x_all[train], axis=0, dtype=np.float64).astype(np.float32)
    x_scale = np.std(x_all[train], axis=0, dtype=np.float64).astype(np.float32)
    x_scale[~np.isfinite(x_scale) | (x_scale < 1e-6)] = 1.0
    x_train = (x_all[train] - x_mean) / x_scale

    if candidate.target_mode == "delta":
        y_all = post - pre
    else:
        y_all = post
    y_train = y_all[train]

    x_components = min(
        candidate.x_components,
        x_train.shape[0] - 1,
        x_train.shape[1],
    )
    y_components = min(
        candidate.y_components,
        y_train.shape[0] - 1,
        y_train.shape[1],
    )
    if x_components < 2 or y_components < 2:
        raise ContractError("Insufficient samples/features for low-rank predictor")

    x_pca = PCA(
        n_components=x_components,
        svd_solver="randomized",
        random_state=seed,
    )
    y_pca = PCA(
        n_components=y_components,
        svd_solver="randomized",
        random_state=seed + 1,
    )
    x_latent = x_pca.fit_transform(x_train)
    y_latent = y_pca.fit_transform(y_train)
    ridge = Ridge(alpha=float(candidate.alpha))
    ridge.fit(x_latent, y_latent)
    return FittedLowRankRidge(
        candidate=candidate,
        pre_medians=pre_medians,
        target_medians=target_medians,
        x_mean=x_mean,
        x_scale=x_scale,
        x_pca=x_pca,
        y_pca=y_pca,
        ridge=ridge,
        smiles_hash_dim=int(model_config["smiles_hash_dim"]),
        smiles_ngram_min=int(model_config["smiles_ngram_min"]),
        smiles_ngram_max=int(model_config["smiles_ngram_max"]),
    )


def _fit_torch_candidate(
    data: TaskData,
    candidate: CandidateConfig,
    model_config: Mapping[str, Any],
    seed: int,
    backend: str,
) -> FittedTorchLowRankRidge:
    import torch

    device = _torch_device(backend)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    pca_solver = str(model_config.get("pca_solver", "exact"))
    pca_niter = int(model_config.get("pca_niter", 4))
    if pca_niter < 0:
        raise ContractError("model.pca_niter must be non-negative")

    train = np.asarray(data.partitions.train, dtype=np.int64)
    pre, pre_medians = _clean_with_train_median(data.morphology_pre, train)
    post, target_medians = _clean_with_train_median(data.morphology_post, train)

    input_blocks = [pre]
    if candidate.input_mode in {"pre_dose", "pre_smiles_dose"}:
        dose = np.log10(np.maximum(data.dose, 0.0) + 1e-6).reshape(-1, 1)
        input_blocks.append(dose.astype(np.float32))
    if candidate.input_mode in {"pre_smiles", "pre_smiles_dose"}:
        input_blocks.append(
            _hashed_smiles_features(
                data.smiles,
                int(model_config["smiles_hash_dim"]),
                int(model_config["smiles_ngram_min"]),
                int(model_config["smiles_ngram_max"]),
            )
        )
    x_all = np.concatenate(input_blocks, axis=1).astype(np.float32)
    x_mean = np.mean(x_all[train], axis=0, dtype=np.float64).astype(np.float32)
    x_scale = np.std(x_all[train], axis=0, dtype=np.float64).astype(np.float32)
    x_scale[~np.isfinite(x_scale) | (x_scale < 1e-6)] = 1.0
    x_train = (x_all[train] - x_mean) / x_scale

    if candidate.target_mode == "delta":
        y_all = post - pre
    else:
        y_all = post
    y_train = y_all[train]

    x_component_count = min(
        candidate.x_components,
        x_train.shape[0] - 1,
        x_train.shape[1],
    )
    y_component_count = min(
        candidate.y_components,
        y_train.shape[0] - 1,
        y_train.shape[1],
    )
    if x_component_count < 2 or y_component_count < 2:
        raise ContractError("Insufficient samples/features for low-rank predictor")

    # Float64 keeps the covariance eigendecomposition and ridge solve stable.
    # The fitted arrays are stored as float32 to keep cache files compact.
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_train, dtype=torch.float64, device=device)
        y_tensor = torch.as_tensor(y_train, dtype=torch.float64, device=device)
        x_pca_mean, x_basis, x_latent = _torch_pca(
            x_tensor,
            x_component_count,
            solver=pca_solver,
            niter=pca_niter,
        )
        y_pca_mean, y_basis, y_latent = _torch_pca(
            y_tensor,
            y_component_count,
            solver=pca_solver,
            niter=pca_niter,
        )
        x_latent_mean = torch.mean(x_latent, dim=0)
        y_latent_mean = torch.mean(y_latent, dim=0)
        centered_x = x_latent - x_latent_mean
        centered_y = y_latent - y_latent_mean
        identity = torch.eye(
            x_component_count,
            dtype=torch.float64,
            device=device,
        )
        gram = centered_x.T @ centered_x + float(candidate.alpha) * identity
        coefficient = torch.linalg.solve(gram, centered_x.T @ centered_y)
        intercept = y_latent_mean - x_latent_mean @ coefficient

        def as_numpy(value: Any) -> np.ndarray:
            return value.detach().cpu().numpy().astype(np.float32)

        fitted = FittedTorchLowRankRidge(
            candidate=candidate,
            pre_medians=pre_medians,
            target_medians=target_medians,
            x_mean=x_mean,
            x_scale=x_scale,
            x_pca_mean=as_numpy(x_pca_mean),
            x_components=as_numpy(x_basis),
            y_pca_mean=as_numpy(y_pca_mean),
            y_components=as_numpy(y_basis),
            ridge_coef=as_numpy(coefficient),
            ridge_intercept=as_numpy(intercept),
            smiles_hash_dim=int(model_config["smiles_hash_dim"]),
            smiles_ngram_min=int(model_config["smiles_ngram_min"]),
            smiles_ngram_max=int(model_config["smiles_ngram_max"]),
            backend=backend,
        )
    del x_tensor, y_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return fitted


def fit_candidate(
    data: TaskData,
    candidate: CandidateConfig,
    model_config: Mapping[str, Any],
    seed: int,
) -> FittedModel:
    backend = str(model_config.get("backend", "sklearn"))
    if backend == "sklearn":
        return _fit_sklearn_candidate(data, candidate, model_config, seed)
    if backend in {"torch_cpu", "torch_cuda"}:
        return _fit_torch_candidate(
            data,
            candidate,
            model_config,
            seed,
            backend,
        )
    raise ContractError(f"Unsupported model backend: {backend}")


class ModelCache:
    def __init__(self, root: Path, protocol_hash: str) -> None:
        self.root = root
        self.protocol_hash = protocol_hash

    def _path(
        self,
        task_id: str,
        initialization_seed: int,
        candidate: CandidateConfig,
    ) -> Path:
        return (
            self.root
            / self.protocol_hash[:16]
            / task_id
            / f"seed-{initialization_seed}"
            / f"{candidate.candidate_id}.joblib"
        )

    def fit_or_load(
        self,
        data: TaskData,
        candidate: CandidateConfig,
        model_config: Mapping[str, Any],
        seed: int,
    ) -> tuple[FittedModel, bool, float]:
        path = self._path(data.spec.task_id, seed, candidate)
        if path.is_file():
            started = time.monotonic()
            payload = joblib.load(path)
            if isinstance(payload, dict):
                model = payload.get("model")
                canonical_fit_seconds = float(payload.get("fit_seconds", 0.0))
            else:
                model = payload
                canonical_fit_seconds = time.monotonic() - started
            if not isinstance(
                model,
                (FittedLowRankRidge, FittedTorchLowRankRidge),
            ):
                raise ContractError(f"Unexpected cached object: {path}")
            return model, True, canonical_fit_seconds
        started = time.monotonic()
        model = fit_candidate(data, candidate, model_config, seed)
        fit_seconds = time.monotonic() - started
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(
            path.suffix + f".tmp-{os.getpid()}-{time.time_ns()}"
        )
        joblib.dump(
            {
                "schema_version": 2,
                "model": model,
                "fit_seconds": fit_seconds,
            },
            temporary,
            compress=3,
        )
        try:
            os.replace(temporary, path)
        except FileNotFoundError:
            if not path.is_file():
                raise
        finally:
            if temporary.exists():
                temporary.unlink()
        return model, False, fit_seconds
