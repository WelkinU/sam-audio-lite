"""Audio chunking helpers for processing long clips without running out of memory.

Long audio is split into overlapping chunks; the separated chunks are then
recombined with a linear crossfade over the overlapping region so there are no
audible seams.
"""

from __future__ import annotations

import torch


def split_into_chunks(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_duration: float,
    overlap_duration: float,
) -> list[torch.Tensor]:
    """Split a ``[channels, samples]`` waveform into overlapping chunks."""
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    stride = max(1, chunk_samples - overlap_samples)

    total = waveform.shape[-1]
    chunks: list[torch.Tensor] = []
    start = 0
    while start < total:
        end = min(start + chunk_samples, total)
        chunks.append(waveform[..., start:end])
        if end >= total:
            break
        start += stride
    return chunks


def merge_with_crossfade(
    chunks: list[torch.Tensor],
    sample_rate: int,
    overlap_duration: float,
) -> torch.Tensor:
    """Merge ``[channels, samples]`` chunks, crossfading the overlap regions."""
    overlap_samples = int(overlap_duration * sample_rate)

    def as_2d(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0) if t.dim() == 1 else t

    processed = [as_2d(c) for c in chunks]
    if len(processed) == 1:
        return processed[0]

    result = processed[0]
    for nxt in processed[1:]:
        actual = min(overlap_samples, result.shape[-1], nxt.shape[-1])
        if actual <= 0:
            result = torch.cat([result, nxt], dim=-1)
            continue

        fade_out = torch.linspace(1.0, 0.0, actual, device=result.device, dtype=result.dtype)
        fade_in = torch.linspace(0.0, 1.0, actual, device=nxt.device, dtype=nxt.dtype)

        head = result[..., :-actual]
        crossfaded = result[..., -actual:] * fade_out + nxt[..., :actual] * fade_in
        tail = nxt[..., actual:]
        result = torch.cat([head, crossfaded, tail], dim=-1)

    return result
