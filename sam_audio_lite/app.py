"""Gradio web UI for sam-audio-lite (text + span prompting)."""

from __future__ import annotations

import gc

import gradio as gr
import torch

from sam_audio_lite.config import load_config
from sam_audio_lite.engine import ProgressFn, SamAudioEngine

CONFIG = load_config("config.yaml")

# Models exposed in the UI (key -> name used by the engine/config).
UI_MODELS = {
    "sam-audio-small": "sam-audio-small",
    "sam-audio-base": "sam-audio-base",
    "sam-audio-large": "sam-audio-large",
    "sam-audio-small-tv": "sam-audio-small-tv",
    "sam-audio-base-tv": "sam-audio-base-tv",
    "sam-audio-large-tv": "sam-audio-large-tv",
}

_engine: SamAudioEngine | None = None
_engine_model: str | None = None


def get_engine(model_name: str, progress_fn: ProgressFn | None = None) -> SamAudioEngine:
    """Return a cached engine, reloading only when the model changes."""
    global _engine, _engine_model
    if _engine is not None and _engine_model == model_name:
        return _engine

    if _engine is not None:  # free the previous model first
        del _engine
        _engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cfg = load_config("config.yaml")
    cfg.model.name = model_name
    _engine = SamAudioEngine(cfg, progress_fn=progress_fn)
    _engine_model = model_name
    return _engine


def _parse_anchors(rows) -> list[tuple[str, float, float]]:
    """Convert dataframe rows into ``(sign, start, end)`` anchors, skipping blanks."""
    if hasattr(rows, "values"):  # pandas DataFrame from gr.Dataframe
        rows = rows.values.tolist()
    anchors: list[tuple[str, float, float]] = []
    for row in rows or []:
        if row is None or all(c in (None, "") for c in row):
            continue
        sign = str(row[0]).strip() or "+"
        if sign not in ("+", "-"):
            sign = "+"
        try:
            start, end = float(row[1]), float(row[2])
        except (TypeError, ValueError):
            continue
        if end > start:
            anchors.append((sign, start, end))
    return anchors


def separate(model_label, audio_path, prompt, chunk_duration, predict_spans, spans, progress=gr.Progress()):
    if not audio_path:
        return None, None, "Please upload an audio file."
    if not prompt or not prompt.strip():
        return None, None, "Please enter a text prompt."

    try:
        model_name = UI_MODELS[model_label]
        engine_is_cached = _engine is not None and _engine_model == model_name

        if engine_is_cached:
            # Model already loaded — inference owns the full bar.
            engine = get_engine(model_name)
        else:
            # Each loading phase resets the bar to 0 → 100% on its own.
            def _load_pf(frac: float, desc: str) -> None:
                progress(frac, desc=desc)

            engine = get_engine(model_name, progress_fn=_load_pf)

        # Reset bar for inference phase.
        progress(0.0, desc="Running inference…")

        def _inf_pf(frac: float, desc: str) -> None:
            progress(frac, desc=desc)

        engine.config.chunking.chunk_duration = float(chunk_duration)

        anchors = _parse_anchors(spans)
        target, residual, sr = engine.separate(
            audio_path,
            prompt.strip(),
            anchors=anchors or None,
            predict_spans=bool(predict_spans),
            progress_fn=_inf_pf,
        )

        progress(1.0, desc="Done.")

        notes = []
        if anchors:
            notes.append(f"{len(anchors)} span(s)")
        if predict_spans:
            notes.append("auto-spans")
        suffix = f" ({', '.join(notes)})" if notes else ""
        vram = engine.peak_vram_gb()
        vram_note = f" | peak VRAM {vram:.2f} GB" if vram else ""
        target_audio = (sr, target.squeeze(0).numpy())
        residual_audio = (sr, residual.squeeze(0).numpy())
        return target_audio, residual_audio, f"Isolated '{prompt}' with {model_label}{suffix}{vram_note}"
    except Exception as exc:  # surface errors in the UI
        import traceback

        traceback.print_exc()
        return None, None, f"Error: {exc}"


def build_ui() -> gr.Blocks:
    rankers_stripped = (
        CONFIG.optimization.strip_text_ranker and CONFIG.optimization.strip_visual_ranker
    )
    with gr.Blocks(title="sam-audio-lite") as demo:
        gr.Markdown(
            "# sam-audio-lite\n"
            "Isolate any sound from audio with a text prompt (and optional time spans). "
            "Use lowercase noun/verb phrases, e.g. `man speaking`, `dog barking`, `piano`."
        )
        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=list(UI_MODELS.keys()),
                    value="Small",
                    label="Model",
                )
                input_audio = gr.Audio(label="Audio file", type="filepath")
                text_prompt = gr.Textbox(
                    label="Text prompt",
                    placeholder="e.g. man speaking, dog barking, piano",
                )
                with gr.Accordion("Time spans (optional)", open=False):
                    gr.Markdown(
                        "Restrict the target to specific time ranges. "
                        "`sign` is `+` (include) or `-` (exclude). "
                        "Spans process the whole clip in one pass (no chunking)."
                    )
                    spans = gr.Dataframe(
                        headers=["sign", "start (s)", "end (s)"],
                        datatype=["str", "number", "number"],
                        row_count=(1, "dynamic"),
                        column_count=(3, "fixed"),
                        label="Spans",
                    )
                with gr.Accordion("Advanced", open=False):
                    chunk_duration = gr.Slider(
                        minimum=5,
                        maximum=60,
                        value=CONFIG.chunking.chunk_duration,
                        step=5,
                        label="Chunk duration (s)",
                        info=f"Clips longer than {CONFIG.chunking.max_duration_without_chunking:.0f}s "
                        "are split automatically.",
                    )
                    predict_spans = gr.Checkbox(
                        value=CONFIG.inference.predict_spans,
                        label="Auto-predict spans from prompt",
                        interactive=not CONFIG.optimization.strip_span_predictor,
                        info="Disabled when the span predictor is stripped in config.yaml."
                        if CONFIG.optimization.strip_span_predictor
                        else None,
                    )
                run_btn = gr.Button("Isolate sound", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("### Results")
                output_target = gr.Audio(label="Isolated sound (target)")
                output_residual = gr.Audio(label="Background (residual)")
                status = gr.Markdown("")

        stripped = ", ".join(
            c for c, on in [
                ("vision", CONFIG.optimization.strip_vision_encoder),
                ("visual ranker", CONFIG.optimization.strip_visual_ranker),
                ("text ranker", CONFIG.optimization.strip_text_ranker),
                ("span predictor", CONFIG.optimization.strip_span_predictor),
            ] if on
        ) or "none"
        gr.Markdown(
            f"_Device: {CONFIG.resolved_device} | dtype: {CONFIG.model.dtype} | "
            f"reranking: {CONFIG.inference.reranking_candidates} | stripped: {stripped}_"
            + ("\n\n_Rankers stripped: reranking is fixed at 1 candidate._" if rankers_stripped else "")
        )

        run_btn.click(
            fn=separate,
            inputs=[model_selector, input_audio, text_prompt, chunk_duration, predict_spans, spans],
            outputs=[output_target, output_residual, status],
        )
    return demo


def main() -> None:
    build_ui().launch(inbrowser = True)


if __name__ == "__main__":
    main()
