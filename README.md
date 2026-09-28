<div align="center">
  <img src="imgs/logo.svg" alt="MeetNote logo" width="140" />
  <h1>MeetNote 会记</h1>
  <p><b>Local-first meeting notes</b> — record, transcribe, diarize, and summarize meetings entirely on your machine.</p>
</div>

<p align="center">
  <a href="https://github.com/luoyao-appleaccount/meetnote/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/luoyao-appleaccount/meetnote?style=flat-square" /></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square" /></a>
  <a href="#platforms"><img alt="macOS" src="https://img.shields.io/badge/macOS-supported-blue.svg" /></a>
  <a href="#platforms"><img alt="Linux" src="https://img.shields.io/badge/Linux-supported-blue.svg" /></a>
  <a href="#platforms"><img alt="Windows" src="https://img.shields.io/badge/Windows-supported-blue.svg" /></a>
  <a href="docs/mcp.md"><img alt="Claude Code" src="https://img.shields.io/badge/Claude_Code-supported-blueviolet.svg" /></a>
  <a href="docs/mcp.md"><img alt="MCP" src="https://img.shields.io/badge/MCP-supported-blueviolet.svg" /></a>
  <a href="#privacy"><img alt="100% local" src="https://img.shields.io/badge/100%25-local-green?style=flat-square" /></a>
</p>

<p align="center">
  English | <a href="docs/i18n/README.zh-CN.md">简体中文</a>
</p>

---

## What is MeetNote?

MeetNote is a self-hosted meeting notes application. It records microphone input or imports audio/video files, transcribes them with faster-whisper, separates speakers with sherpa-onnx voiceprint models, and extracts a structured summary — one-liner, key points, decisions, action items, and speaker statistics — using a pure local rules engine (or an optional LLM). No cloud, no API tokens, no data leaving your machine.

It ships in three forms on one codebase:

- **Web app** — a FastAPI server with a single-page UI (live recording, system-audio capture for meeting apps via BlackHole, in-meeting marks ⭐/✓, action-item hub, re-transcription with bigger models, Markdown export).
- **SDK / CLI** — `import sdk` or `python sdk.py diarize meeting.wav` for scripts and agents.
- **MCP server** — expose transcription and diarization as tools for Claude Code, Claude Desktop, WorkBuddy, and other MCP clients.
- **OpenAI-compatible API** — `POST /v1/audio/transcriptions` so any OpenAI-API-aware tool can transcribe through MeetNote locally.

## Benchmark

Speaker diarization is the hardest part of a meeting-notes pipeline. The stock sherpa-onnx auto clustering severely **over-splits** — it turned a confirmed 4-speaker meeting into 7 speakers and a 23-minute conference-room recording into **49** "speakers". Threshold tuning alone cannot fix it.

MeetNote replaces the clustering stage with a two-stage pipeline validated on a local benchmark suite (real recordings, TTS dialogues, official sherpa test audio, and synthetic 1–5 speaker mixes with exact ground truth):

| Stage | What it does |
|---|---|
| pyannote segmentation + **Silero VAD re-split** | The segmentation model merges turns across short silences in long/repetitive audio; VAD cuts them back apart |
| Voiceprint extraction (3dspeaker) | Per-segment embeddings |
| Average-linkage clustering | Greedy merge at cosine similarity ≥ 0.7 |
| Core-cluster reassignment | Fragments (< 2s or < 2% of speech) are absorbed into the nearest core cluster |

Results on Chinese content (the primary use case):

| Test case | Ground truth | Stock clustering | MeetNote |
|---|---|---|---|
| Real 4-speaker meeting (user-confirmed) | 4 | 7 | **4** ✓ |
| 23-min conference-room recording | unknown (1 main + side talkers) | 49 | **3** |
| TTS two-person dialogues ×2 | 2 | 1 | **2** ✓ |
| sherpa official 4-speaker Chinese | 4 | 7 | **4** ✓ |
| Synthetic Chinese single speaker | 1 | 1 | **1** ✓ (attribution 100%) |
| Synthetic Chinese two speakers | 2 | 1 | **2** ✓ (attribution 99.9%) |
| Synthetic mixed CN/EN five speakers | 5 | 4 | **5** ✓ |

**Known limit**: English-only audio is constrained by the segmentation stage itself (measured oracle ceiling 52–68% attribution on the synthetic cases), not by clustering — see [docs/benchmark.md](docs/benchmark.md). Chinese meetings, the primary use case, are fully covered.

Every case is reproducible: `python benchmark_diar.py` writes all results into `data/benchmarks.db` (SQLite) and prints a comparison table.

## Why MeetNote?

### The problem with general meeting-notes tools

- **Cloud dependency** — meeting audio is among the most sensitive data; uploading it to a SaaS transcription API is a hard no in many organizations.
- **Speaker count is a guess** — most diarization pipelines force you to preset the speaker count, or fall back to threshold clustering that fragments voices. Real meetings don't come with a labeled roster.
- **Preset-only clustering** — even when the count is known, a fixed clustering threshold breaks down across acoustic conditions (real speech, TTS, distant mics, noisy venues).

### Core design: verified models, defensive engineering

The model stack is a **measured** combination that runs smoothly on CPU int8 — no GPU required:

| Model | Role | Why |
|---|---|---|
| faster-whisper `small` (int8) | Live transcription | Real-time on CPU; `medium`/`large-v3` for offline re-transcription |
| Silero VAD (built into faster-whisper) | Voice detection | Skips silence, drastically reduces hallucination |
| sherpa-onnx + 3dspeaker voiceprint | Diarization | Zero license, zero token (vs. pyannote's HF-auth requirement) |
| Local rules engine (optional LLM) | Summary extraction | Structured oneliner / points / decisions / actions without any API |

Hard-won pitfalls are handled in code: offline-first model loading (hf-mirror fallback), explicit mic blocksize on macOS, hallucination filtering, and the diarization over-split fix above.

## How to Use

### Prerequisites

- **Python >= 3.10**
- Microphone access (for live recording) — macOS needs mic permission for the terminal

### Quick Start (Web app)

```bash
git clone https://github.com/luoyao-appleaccount/meetnote.git
cd meetnote
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash scripts/download_models.sh        # voiceprint + segmentation models (~50 MB)
python server.py                       # open http://localhost:7861
```

First launch loads whisper `small` from the local HF cache or downloads it via hf-mirror automatically.

### CLI

```bash
python sdk.py diarize meeting.wav              # auto speaker count
python sdk.py diarize meeting.wav -n 4         # preset count (most stable)
python sdk.py transcribe meeting.wav --model medium --json
```

### Python SDK

```python
import sdk
segs = sdk.diarize("meeting.wav")   # [{"start","end","speaker","text"}]
```

### MCP server (for AI agents)

```bash
claude mcp add meetnote -s user -- /path/to/venv/bin/python /path/to/meetnote/mcp_server.py
```

Two tools are exposed: `transcribe_audio(path, model)` and `diarize_audio(path, num_speakers, model)`. Works with any MCP client — see [docs/mcp.md](docs/mcp.md) for Claude Desktop / WorkBuddy JSON config.

## Documentation

- [Quickstart](docs/quickstart.md) — install, models, web app walkthrough
- [SDK & CLI](docs/sdk.md) — full API reference and usage patterns
- [MCP server](docs/mcp.md) — agent integration (Claude Code / Desktop, WorkBuddy, generic MCP)
- [Benchmark](docs/benchmark.md) — diarization test suite, metrics, known limits, how to re-run

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and guidelines.

## License

[Apache-2.0](LICENSE) — Copyright 2026 MeetNote Contributors
