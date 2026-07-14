# sam-audio-lite

A lightweight Gradio UI around Meta's [SAM-Audio](https://github.com/facebookresearch/sam-audio) model for text-prompted audio source separation. Describe the sound you want (e.g. *"acoustic guitar"*, *"crowd noise"*) and the app isolates it from the mix, returning both the target and residual tracks.

Low-VRAM friendly — the vision encoder, rankers, and span predictor are stripped by default, leaving only the audio separation path.

---

## Requirements

- Python 3.11+
- CUDA-capable GPU (recommended; CPU works but is slow)
- **GTX 10xx / Pascal GPUs** (e.g. GTX 1070): this project pins PyTorch **CUDA 12.6** builds (`cu126`), which still support compute capability 6.1. Newer `cu128+` PyTorch wheels drop Pascal support.
- A free [HuggingFace](https://huggingface.co) account with model access (see below)
- **Windows only:** FFmpeg *full-shared* (with DLLs) for TorchCodec — `scripts\setup.bat` downloads this automatically into `tools/ffmpeg/`

---

## HuggingFace access token

The SAM-Audio weights are gated on HuggingFace and require a token even for free access.

1. **Create an account** at https://huggingface.co/join
2. **Accept the licence** on each model page you plan to use, e.g.:
   - https://huggingface.co/facebook/sam-audio-small
   - https://huggingface.co/facebook/sam-audio-base
   - https://huggingface.co/facebook/sam-audio-large
   
   Click *"Agree and access repository"* — access is granted instantly.
3. **Create a read token** at https://huggingface.co/settings/tokens → *New token* → *Read*.  
   It looks like `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`.

---

## Quickstart

### Windows — automated setup

Get the code locally then run `scripts\setup.bat`

The script installs Git and [uv](https://docs.astral.sh/uv/) if missing, runs `uv sync` to create the virtual environment, then walks you through saving your HuggingFace token.

Once done, launch the Gradio GUI with `uv run run_gradio.py`

### Manual setup (Windows / Linux)

```bash
cd sam-audio-lite

# Install uv if you don't have it: https://docs.astral.sh/uv/
uv sync

# Save your HuggingFace token (one-time)
uv run python scripts/setup_hf_token.py

# Launch the app
uv run run_gradio.py
```

The first run downloads the selected model into the HuggingFace cache (`~/.cache/huggingface/hub/`). Subsequent runs load from cache instantly. You can change the cache location with the `HF_HOME` environment variable.

---

## Configuration

Edit `config.yaml` to change the model, device, or chunking behaviour:

```yaml
model:
  name: sam-audio-base  # sam-audio-small | sam-audio-base | sam-audio-large
                        # sam-audio-small-tv | sam-audio-base-tv | sam-audio-large-tv
  device: auto          # auto | cuda | cpu
  dtype: bfloat16

optimization:
  strip_vision_encoder: true   # saves VRAM; disables visual prompting
  strip_visual_ranker: true
  strip_text_ranker: true
  strip_span_predictor: true   # set false to enable predict_spans

chunking:
  enabled: true
  max_duration_without_chunking: 60  # seconds
  chunk_duration: 30
  overlap_duration: 2
```
