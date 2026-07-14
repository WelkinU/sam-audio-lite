"""Compatibility shims for running SAM-Audio against newer dependencies.

The installed ``sam_audio`` package was written against ``huggingface_hub`` 0.x,
whose ``ModelHubMixin.from_pretrained`` forwarded ``proxies`` and
``resume_download`` to ``_from_pretrained``. ``huggingface_hub`` >= 1.0 dropped
those keyword arguments, so SAM-Audio's ``BaseModel._from_pretrained`` (which
still declares them as required keyword-only parameters) raises ``TypeError``.

Calling :func:`apply_hub_compat` once, before any model is loaded, patches
``BaseModel._from_pretrained`` to supply sane defaults so loading works on
current ``huggingface_hub`` versions.

On Windows, TorchCodec often fails to load (FFmpeg shared DLLs + PyTorch ABI).
When vision prompting is disabled we stub TorchCodec at import time and load
audio with ``torchaudio`` instead (see :func:`apply_runtime_compat`).
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
import types
from pathlib import Path

_applied = False
_runtime_applied = False
_torchcodec_stubbed = False


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_ffmpeg_bin() -> Path | None:
    """Return the bin directory of a bundled FFmpeg *shared* build, if present."""
    tools = _project_root() / "tools" / "ffmpeg"
    if not tools.is_dir():
        return None
    for child in sorted(tools.iterdir()):
        candidate = child / "bin"
        if candidate.is_dir() and any(candidate.glob("avcodec-*.dll")):
            return candidate
    return None


def prepare_windows_dll_paths() -> None:
    """Expose bundled FFmpeg shared libraries to the Windows DLL loader."""
    if sys.platform != "win32":
        return
    ffmpeg_bin = _find_ffmpeg_bin()
    if ffmpeg_bin is None:
        return
    bin_str = str(ffmpeg_bin)
    os.environ["PATH"] = bin_str + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(bin_str)


def patch_windows_asyncio() -> None:
    """Silence benign connection-reset noise from the Windows proactor event loop.

    Gradio's file upload closes HTTP connections abruptly; Python 3.12's default
    ``ProactorEventLoop`` then logs ``ConnectionResetError`` in a background
    callback even though the upload succeeded.
    """
    if sys.platform != "win32":
        return
    from asyncio.proactor_events import _ProactorBasePipeTransport

    _orig = _ProactorBasePipeTransport._call_connection_lost

    def _call_connection_lost(self, exc):  # type: ignore[no-untyped-def]
        try:
            _orig(self, exc)
        except (ConnectionResetError, ConnectionAbortedError):
            pass

    _ProactorBasePipeTransport._call_connection_lost = _call_connection_lost  # type: ignore[method-assign]


def _clear_torchcodec_modules() -> None:
    """Remove torchcodec (and submodules) from ``sys.modules``."""
    for name in list(sys.modules):
        if name == "torchcodec" or name.startswith("torchcodec."):
            del sys.modules[name]


def _torchcodec_import_ok() -> bool:
    """Return True when the real TorchCodec package imports successfully."""
    prepare_windows_dll_paths()
    _clear_torchcodec_modules()
    try:
        importlib.import_module("torchcodec.decoders")
        return True
    except Exception:
        return False


def _install_torchcodec_stub() -> None:
    """Register a minimal ``torchcodec`` package so SAM-Audio can import."""
    global _torchcodec_stubbed
    if _torchcodec_stubbed:
        return

    _clear_torchcodec_modules()

    def _stub_module(name: str, *, is_package: bool = False) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(
            name,
            loader=None,
            is_package=is_package,
        )
        if is_package:
            mod.__path__ = []
        return mod

    root = _stub_module("torchcodec", is_package=True)
    decoders = _stub_module("torchcodec.decoders")
    encoders = _stub_module("torchcodec.encoders")
    samplers = _stub_module("torchcodec.samplers")
    transforms = _stub_module("torchcodec.transforms")

    class _StubDecoder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "TorchCodec is unavailable in this environment. "
                "Pass pre-loaded audio tensors to the processor (sam-audio-lite does this "
                "automatically) or install FFmpeg full-shared and re-run setup."
            )

    decoders.AudioDecoder = _StubDecoder
    decoders.VideoDecoder = _StubDecoder
    root.decoders = decoders
    root.encoders = encoders
    root.samplers = samplers
    root.transforms = transforms

    for name, mod in (
        ("torchcodec", root),
        ("torchcodec.decoders", decoders),
        ("torchcodec.encoders", encoders),
        ("torchcodec.samplers", samplers),
        ("torchcodec.transforms", transforms),
    ):
        sys.modules[name] = mod
    _torchcodec_stubbed = True


def _remove_torchcodec_stub() -> None:
    """Drop stub modules so transformers can probe TorchCodec availability safely."""
    global _torchcodec_stubbed
    if not _torchcodec_stubbed:
        return
    _clear_torchcodec_modules()
    _torchcodec_stubbed = False


def apply_runtime_compat() -> None:
    """Prepare the process environment and ensure ``torchcodec`` can be imported."""
    global _runtime_applied
    if _runtime_applied:
        return

    if not _torchcodec_import_ok():
        _install_torchcodec_stub()
    _runtime_applied = True


def apply_hub_compat() -> None:
    """Patch SAM-Audio's ``_from_pretrained`` to tolerate huggingface_hub >= 1.0."""
    global _applied
    if _applied:
        return

    apply_runtime_compat()

    from sam_audio.model.base import BaseModel

    # transformers probes torchcodec via importlib.find_spec; a stub breaks that.
    _remove_torchcodec_stub()

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
