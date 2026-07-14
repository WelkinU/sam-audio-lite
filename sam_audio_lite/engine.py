"""SAM-Audio inference engine: load, strip for low VRAM, and separate audio."""

from __future__ import annotations

import contextlib
import gc
import io
import os
import types
from typing import Callable

import torch
import torchaudio

from sam_audio_lite._compat import apply_hub_compat
from sam_audio_lite.chunking import merge_with_crossfade, split_into_chunks
from sam_audio_lite.config import AppConfig

# Anchor: (sign, start_seconds, end_seconds), e.g. ("+", 6.3, 7.0).
Anchor = tuple[str, float, float]

# Progress callback: (fraction 0-1, description)
ProgressFn = Callable[[float, str], None]


def _read_audio_file(path: str) -> tuple[torch.Tensor, int]:
    """Load audio without TorchCodec.

    ``torchaudio.load`` in 2.11+ delegates to TorchCodec, which is brittle on
    Windows. We use ``soundfile`` (WAV/FLAC/OGG) with a ``pydub`` fallback.
    """
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        # soundfile: [samples, channels] -> torch: [channels, samples]
        return torch.from_numpy(data.T.copy()), int(sr)
    except Exception:
        pass

    try:
        import numpy as np
        from pydub import AudioSegment

        seg = AudioSegment.from_file(path)
        scale = float(1 << (8 * seg.sample_width - 1))
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / scale
        if seg.channels > 1:
            samples = samples.reshape(-1, seg.channels).mean(axis=1)
        waveform = torch.from_numpy(samples).unsqueeze(0)
        return waveform, seg.frame_rate
    except Exception as exc:
        raise RuntimeError(
            f"Could not load audio file '{path}'. "
            "Supported via soundfile: WAV, FLAC, OGG. "
            "Other formats need FFmpeg on PATH (pydub fallback)."
        ) from exc


def _empty_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_local_source(source: str) -> bool:
    return os.path.isdir(source)


def _hf_is_cached(repo_id: str) -> bool:
    """Return True if the model is already present in the HuggingFace local cache."""
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore
        return try_to_load_from_cache(repo_id, "config.json") is not None
    except Exception:
        return False


@contextlib.contextmanager
def _intercept_tqdm(progress_fn: ProgressFn, offset: float, span: float, label: str):
    """Temporarily replace tqdm classes so HF downloads forward byte-level progress."""
    import tqdm as _tqdm_mod
    import tqdm.auto as _tqdm_auto_mod

    _orig = _tqdm_mod.tqdm
    _orig_auto = _tqdm_auto_mod.tqdm

    class _ProgressTqdm(_orig):  # type: ignore[valid-type]
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("file", io.StringIO())  # suppress console noise
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            super().update(n)
            if self.total and self.total > 0:
                frac = min(self.n / self.total, 1.0)
                mb_done = self.n / 1_000_000
                mb_total = self.total / 1_000_000
                progress_fn(
                    offset + frac * span,
                    f"{label} — {mb_done:.0f} / {mb_total:.0f} MB",
                )

    _tqdm_mod.tqdm = _ProgressTqdm  # type: ignore[attr-defined]
    _tqdm_auto_mod.tqdm = _ProgressTqdm  # type: ignore[attr-defined]
    # Also patch huggingface_hub's own tqdm wrapper when present.
    _hf_tqdm_mod = None
    _orig_hf_tqdm = None
    try:
        import huggingface_hub.utils._tqdm as _hf_tqdm_mod  # type: ignore
        _orig_hf_tqdm = _hf_tqdm_mod.tqdm
        _hf_tqdm_mod.tqdm = _ProgressTqdm
    except (ImportError, AttributeError):
        pass
    try:
        yield
    finally:
        _tqdm_mod.tqdm = _orig
        _tqdm_auto_mod.tqdm = _orig_auto
        if _hf_tqdm_mod is not None and _orig_hf_tqdm is not None:
            _hf_tqdm_mod.tqdm = _orig_hf_tqdm


class SamAudioEngine:
    """Loads a SAM-Audio model, optionally strips heavy components, and separates audio."""

    def __init__(self, config: AppConfig, progress_fn: ProgressFn | None = None):
        apply_hub_compat()
        self.config = config
        self.device = config.resolved_device
        self.dtype = config.torch_dtype
        self._stripped: list[str] = []
        self._progress_fn: ProgressFn = progress_fn or (lambda _v, _d: None)

        self.model, self.processor = self._load()
        self.sample_rate: int = int(self.processor.audio_sampling_rate)

    # -- loading & optimisation -------------------------------------------------

    def _load(self):
        from sam_audio import SAMAudio, SAMAudioProcessor

        source = self._resolve_source()
        skip_kwargs = self._skip_kwargs()
        pf = self._progress_fn
        model_name = self.config.model.name

        is_local = _is_local_source(source)
        needs_download = not is_local and not _hf_is_cached(source)

        # Phase 1: weights
        if needs_download:
            pf(0.0, f"Downloading {model_name} weights from HuggingFace…")
            with _intercept_tqdm(pf, 0.0, 1.0, f"Downloading {model_name}"):
                model = SAMAudio.from_pretrained(source, **skip_kwargs)
        else:
            pf(0.0, f"Loading {model_name} weights…")
            with _intercept_tqdm(pf, 0.0, 1.0, f"Loading {model_name}"):
                model = SAMAudio.from_pretrained(source, **skip_kwargs)
        pf(1.0, "Weights loaded.")

        # Phase 2: processor
        if needs_download:
            pf(0.0, f"Downloading {model_name} processor…")
            with _intercept_tqdm(pf, 0.0, 1.0, "Downloading processor"):
                processor = SAMAudioProcessor.from_pretrained(source)
        else:
            pf(0.0, f"Loading {model_name} processor…")
            with _intercept_tqdm(pf, 0.0, 1.0, "Loading processor"):
                processor = SAMAudioProcessor.from_pretrained(source)
        pf(1.0, "Processor loaded.")

        # Phase 3: optimise + move to device
        pf(0.0, "Applying optimizations…")
        self._apply_optimizations(model)
        pf(0.5, f"Moving model to {self.device}…")
        model = model.eval().to(self.device, self.dtype)
        _empty_cache()
        pf(1.0, "Model ready.")
        return model, processor

    def _skip_kwargs(self) -> dict:
        """Config overrides that stop stripped sub-models from being built or downloaded.

        ``SAMAudio._from_pretrained`` overrides any config key passed as a keyword
        argument, so setting these to ``None`` makes the model skip constructing the
        corresponding ranker / span predictor entirely (avoiding multi-GB downloads
        for components we would otherwise immediately strip).
        """
        opt = self.config.optimization
        skip: dict[str, object] = {}
        if opt.strip_visual_ranker:
            skip["visual_ranker"] = None
        if opt.strip_text_ranker:
            skip["text_ranker"] = None
        if opt.strip_span_predictor:
            skip["span_predictor"] = None
        return skip

    def _resolve_source(self) -> str:
        return self.config.repo_id

    def _apply_optimizations(self, model) -> None:
        opt = self.config.optimization

        if opt.strip_vision_encoder and getattr(model, "vision_encoder", None) is not None:
            dim = getattr(model.vision_encoder, "dim", 1024)
            model._vision_encoder_dim = dim
            del model.vision_encoder
            model.vision_encoder = None

            def _video_features_lite(self, video, audio_features):
                b, t, _ = audio_features.shape
                return audio_features.new_zeros(b, self._vision_encoder_dim, t)

            model._get_video_features = types.MethodType(_video_features_lite, model)
            self._stripped.append("vision_encoder")

        # Rankers / span predictor are usually skipped at construction (see
        # _skip_kwargs); free them here if any were still built, and record them
        # as stripped regardless so status reporting stays accurate.
        if opt.strip_visual_ranker:
            if getattr(model, "visual_ranker", None) is not None:
                del model.visual_ranker
            model.visual_ranker = None
            self._stripped.append("visual_ranker")

        if opt.strip_text_ranker:
            if getattr(model, "text_ranker", None) is not None:
                del model.text_ranker
            model.text_ranker = None
            self._stripped.append("text_ranker")

        if opt.strip_span_predictor:
            for attr in ("span_predictor", "span_predictor_transform"):
                if getattr(model, attr, None) is not None:
                    delattr(model, attr)
            self._stripped.append("span_predictor")

        _empty_cache()

    @property
    def stripped_components(self) -> list[str]:
        return list(self._stripped)

    # -- audio io ---------------------------------------------------------------

    def _load_audio(self, path: str) -> torch.Tensor:
        waveform, sr = _read_audio_file(path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        if waveform.shape[0] > 1:  # downmix to mono
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    # -- separation -------------------------------------------------------------

    def _run(self, audio: str | torch.Tensor, description: str, anchors: list[Anchor] | None, predict_spans: bool):
        """Run the model on a single audio input (path or tensor)."""
        if isinstance(audio, str):
            audio = self._load_audio(audio)
        batch_anchors = [list(anchors)] if anchors else None
        batch = self.processor(
            descriptions=[description],
            audios=[audio],
            anchors=batch_anchors,
        ).to(self.device)

        autocast = torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype)
        with torch.inference_mode(), autocast:
            result = self.model.separate(
                batch,
                predict_spans=predict_spans,
                reranking_candidates=self.config.inference.reranking_candidates,
            )
        target = result.target[0].float().cpu()
        residual = result.residual[0].float().cpu()
        del batch, result
        _empty_cache()
        return target, residual

    def separate(
        self,
        audio_path: str,
        description: str,
        anchors: list[Anchor] | None = None,
        predict_spans: bool | None = None,
        progress_fn: ProgressFn | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Separate ``description`` from ``audio_path``.

        Returns ``(target, residual, sample_rate)`` where target/residual are
        ``[1, samples]`` tensors. When ``anchors`` (time spans) are given, the
        clip is always processed in a single pass so the timestamps stay valid.
        """
        pf: ProgressFn = progress_fn or (lambda _v, _d: None)

        if predict_spans is None:
            predict_spans = self.config.inference.predict_spans

        waveform = self._load_audio(audio_path)
        duration = waveform.shape[-1] / self.sample_rate
        chunk_cfg = self.config.chunking

        use_chunking = (
            chunk_cfg.enabled
            and not anchors  # absolute spans are invalid once the clip is split
            and duration > chunk_cfg.max_duration_without_chunking
        )

        if not use_chunking:
            pf(0.1, "Running model…")
            target, residual = self._run(waveform, description, anchors, predict_spans)
            pf(1.0, "Inference complete.")
            return self._as_2d(target), self._as_2d(residual), self.sample_rate

        chunks = split_into_chunks(
            waveform, self.sample_rate, chunk_cfg.chunk_duration, chunk_cfg.overlap_duration
        )
        valid_chunks = [c for c in chunks if c.shape[-1] >= self.sample_rate]
        n_chunks = len(valid_chunks)
        targets, residuals = [], []
        for i, chunk in enumerate(valid_chunks):
            pf(i / max(n_chunks, 1), f"Processing chunk {i + 1} / {n_chunks}…")
            t, r = self._run(chunk, description, None, predict_spans)
            targets.append(self._as_2d(t))
            residuals.append(self._as_2d(r))

        pf(0.97, "Merging chunks…")
        target = merge_with_crossfade(targets, self.sample_rate, chunk_cfg.overlap_duration)
        residual = merge_with_crossfade(residuals, self.sample_rate, chunk_cfg.overlap_duration)
        pf(1.0, "Inference complete.")
        return target.clamp(-1, 1), residual.clamp(-1, 1), self.sample_rate

    @staticmethod
    def _as_2d(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0) if t.dim() == 1 else t

    def peak_vram_gb(self) -> float | None:
        if self.device.startswith("cuda"):
            return torch.cuda.max_memory_allocated() / 1024**3
        return None
