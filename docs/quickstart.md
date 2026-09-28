# Quickstart

Install, fetch models, and run your first meeting note — in under five minutes.

## 1. Install

```bash
git clone https://github.com/luoyao-appleaccount/meetnote.git
cd meetnote
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

All dependencies are pure CPU: faster-whisper (transcription + Silero VAD + PyAV audio decoding), sherpa-onnx (diarization), FastAPI/uvicorn (web), numpy, sounddevice (mic), requests (optional LLM), mcp (optional MCP server).

> **Tip**: if you already have a faster-whisper environment (e.g. the one used by whisper-handoff), these requirements are a superset — install only what's missing.

## 2. Download models

```bash
bash scripts/download_models.sh
```

Downloads (~50 MB, from hf-mirror by default) into `models/`:

| File | Role |
|---|---|
| `model.onnx` | pyannote segmentation 3.0 (speech/segment boundaries) |
| `3dspeaker_zh.onnx` | 3D-Speaker CAM++ voiceprint embeddings (zh) |

Whisper models are not downloaded here: on first launch they load from the local Hugging Face cache, and fall back to hf-mirror automatically. Prefer offline: put whisper `small`/`medium`/`large-v3` under `~/.cache/huggingface/hub` yourself.

## 3. Run the web app

```bash
python server.py
#  会记 MeetNote · http://localhost:7861
```

| Feature | How |
|---|---|
| Live recording | 新建会议 → pick a speaker-count option (自动 or preset) → start; speakers are auto-colored in the transcript |
| In-meeting marks | Hover a transcript line → ⭐ key point / ✓ todo |
| Import audio | 导入录音 → pick wav/mp3/m4a/mp4 → background transcription + diarization |
| Overview | 一句话总结 / 要点 / 决策 / 行动项 / 说话人统计与主题词 |
| Re-transcribe | Meeting detail → switch to `medium` / `large-v3`; versions coexist |
| Action hub | Cross-meeting todos, check-off, manual add, trace back to source meeting |
| Export | Markdown minutes / wav recording |

## 4. Optional: LLM-enhanced summary

The default summarizer is a local rules engine. To use an LLM (any OpenAI-compatible endpoint), set all three variables; on failure it silently falls back to the rules engine:

```bash
export MEETNOTE_LLM_BASE_URL="https://api.deepseek.com/v1"
export MEETNOTE_LLM_API_KEY="sk-..."
export MEETNOTE_LLM_MODEL="deepseek-chat"
```

## Platform notes

- **macOS**: mic permission required for the terminal; the code pins an explicit blocksize to avoid the known 14-frame callback issue.
- **Linux**: needs PortAudio (`libportaudio2`) for `sounddevice`.
- **Windows**: fully supported via the same venv steps; WSL also works.

## Data layout

```
data/
├── meetings.db        # SQLite: meetings / segments / actions
├── recordings/        # imported + recorded wav
└── benchmarks.db      # diarization benchmark results (after running benchmark_diar.py)
```

Redirect with `MEETNOTE_DATA_DIR=/your/path`. Redirect the models dir with `MEETNOTE_MODELS_DIR=/your/path`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| 麦克风无声音 / blocksize 报错 | check `python -c "import sounddevice"`; install PortAudio on Linux |
| 导入格式不支持 | PyAV covers common formats; install ffmpeg for exotic ones |
| whisper 下载慢 | hf-mirror is on by default; or pre-place models in the HF cache |
| 说话人分错 | preset the count when known (most stable); auto mode is two-stage and anti-fragmentation (see [benchmark](benchmark.md)) |
| Port 7861 busy | change `port=` at the bottom of `server.py` |
