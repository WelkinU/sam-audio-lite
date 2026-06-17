"""Configuration loading and validation for sam-audio-lite."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import yaml

_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Short model names -> Hugging Face repo ids (all lowercase on the Hub).
MODEL_REPOS: dict[str, str] = {
    "sam-audio-small": "facebook/sam-audio-small",
    "sam-audio-base": "facebook/sam-audio-base",
    "sam-audio-large": "facebook/sam-audio-large",
    "sam-audio-small-tv": "facebook/sam-audio-small-tv",
    "sam-audio-base-tv": "facebook/sam-audio-base-tv",
    "sam-audio-large-tv": "facebook/sam-audio-large-tv",
}


@dataclass
class ModelCfg:
    name: str = "sam-audio-small"
    device: str = "auto"
    dtype: str = "bfloat16"


@dataclass
class OptimizationCfg:
    strip_vision_encoder: bool = True
    strip_visual_ranker: bool = True
    strip_text_ranker: bool = True
    strip_span_predictor: bool = False


@dataclass
class InferenceCfg:
    predict_spans: bool = False
    reranking_candidates: int = 1


@dataclass
class ChunkingCfg:
    enabled: bool = True
    max_duration_without_chunking: float = 60.0
    chunk_duration: float = 25.0
    overlap_duration: float = 2.0


@dataclass
class AppConfig:
    model: ModelCfg = field(default_factory=ModelCfg)
    optimization: OptimizationCfg = field(default_factory=OptimizationCfg)
    inference: InferenceCfg = field(default_factory=InferenceCfg)
    chunking: ChunkingCfg = field(default_factory=ChunkingCfg)

    @property
    def resolved_device(self) -> str:
        if self.model.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.model.device

    @property
    def torch_dtype(self) -> torch.dtype:
        # CPU does not support bfloat16/float16 matmul well; force float32.
        if self.resolved_device == "cpu":
            return torch.float32
        return _DTYPES[self.model.dtype]

    @property
    def repo_id(self) -> str:
        return MODEL_REPOS.get(self.model.name, self.model.name)


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        warnings.warn(f"Ignoring unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: v for k, v in data.items() if k in known})


def _validate(cfg: AppConfig) -> AppConfig:
    """Apply cross-field validation and warn about incompatible combinations."""
    if cfg.model.dtype not in _DTYPES:
        raise ValueError(
            f"Unknown dtype '{cfg.model.dtype}'. Choose from {sorted(_DTYPES)}."
        )
    if cfg.model.device not in ("auto", "cuda", "cpu"):
        raise ValueError(f"Unknown device '{cfg.model.device}'. Use auto|cuda|cpu.")

    rankers_stripped = cfg.optimization.strip_text_ranker and cfg.optimization.strip_visual_ranker
    if cfg.inference.reranking_candidates > 1 and rankers_stripped:
        warnings.warn(
            "reranking_candidates > 1 requires rankers, but both rankers are stripped. "
            "Clamping reranking_candidates to 1."
        )
        cfg.inference.reranking_candidates = 1

    if cfg.inference.predict_spans and cfg.optimization.strip_span_predictor:
        warnings.warn(
            "predict_spans is enabled but strip_span_predictor is true. "
            "Disabling predict_spans."
        )
        cfg.inference.predict_spans = False

    if cfg.inference.reranking_candidates < 1:
        cfg.inference.reranking_candidates = 1

    if cfg.chunking.overlap_duration >= cfg.chunking.chunk_duration:
        raise ValueError("chunking.overlap_duration must be smaller than chunk_duration.")

    return cfg


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate an :class:`AppConfig` from a YAML file."""
    raw: dict[str, Any] = {}
    p = Path(path)
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        warnings.warn(f"Config file '{p}' not found; using defaults.")

    cfg = AppConfig(
        model=_build(ModelCfg, raw.get("model", {})),
        optimization=_build(OptimizationCfg, raw.get("optimization", {})),
        inference=_build(InferenceCfg, raw.get("inference", {})),
        chunking=_build(ChunkingCfg, raw.get("chunking", {})),
    )
    return _validate(cfg)
