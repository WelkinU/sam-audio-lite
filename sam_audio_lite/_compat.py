"""Compatibility shims for running SAM-Audio against newer dependencies.

The installed ``sam_audio`` package was written against ``huggingface_hub`` 0.x,
whose ``ModelHubMixin.from_pretrained`` forwarded ``proxies`` and
``resume_download`` to ``_from_pretrained``. ``huggingface_hub`` >= 1.0 dropped
those keyword arguments, so SAM-Audio's ``BaseModel._from_pretrained`` (which
still declares them as required keyword-only parameters) raises ``TypeError``.

Calling :func:`apply_hub_compat` once, before any model is loaded, patches
``BaseModel._from_pretrained`` to supply sane defaults so loading works on
current ``huggingface_hub`` versions.
"""

from __future__ import annotations

_applied = False


def apply_hub_compat() -> None:
    """Patch SAM-Audio's ``_from_pretrained`` to tolerate huggingface_hub >= 1.0."""
    global _applied
    if _applied:
        return

    from sam_audio.model.base import BaseModel

    original = BaseModel._from_pretrained.__func__  # underlying function

    def _from_pretrained(
        cls,
        *,
        model_id,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=False,
        local_files_only=False,
        token=None,
        map_location="cpu",
        strict=True,
        revision=None,
        **model_kwargs,
    ):
        return original(
            cls,
            model_id=model_id,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            map_location=map_location,
            strict=strict,
            revision=revision,
            **model_kwargs,
        )

    BaseModel._from_pretrained = classmethod(_from_pretrained)
    _applied = True
