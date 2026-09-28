# Contributing

Thanks for your interest in MeetNote! Here is how to get set up and what we look for in contributions.

## Development setup

```bash
git clone https://github.com/luoyao-appleaccount/meetnote.git
cd meetnote
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash scripts/download_models.sh
python server.py            # http://localhost:7861
```

## Areas that welcome contributions

- **Diarization** — see [docs/benchmark.md](docs/benchmark.md). The known ceiling is English/segmentation-bound; a better segmentation model (or a language-aware one) is the highest-impact improvement. Any clustering change must be validated with `python benchmark_diar.py` and the results added to the docs table.
- **LLM summarizer** — any OpenAI-compatible endpoint works today; provider-specific quirks and prompt improvements are welcome.
- **Frontend** — the single-page UI in `static/index.html` is deliberately dependency-free; keep it that way.
- **MCP / agent integrations** — new tools (e.g. summarize-a-meeting) and more client examples.

## Guidelines

- Keep it local-first: no telemetry, no forced cloud calls, offline fallbacks everywhere.
- Keep the dependency list short (see `requirements.txt`); hand-rolled code is preferred over heavy frameworks for small tasks.
- Code comments in Chinese, doc/README in English + Chinese (see `docs/i18n/`).
- Before submitting: run `python benchmark_diar.py` if you touched `engine.py`, and smoke-test `python sdk.py diarize` on a short wav.

## Pull requests

1. Fork and create a feature branch.
2. Keep changes focused; describe what and why.
3. Reference the benchmark run id (printed by `benchmark_diar.py`) when changing diarization behavior.

## License

By contributing, you agree that your contributions will be licensed under the [Apache-2.0](LICENSE) license of this project.
