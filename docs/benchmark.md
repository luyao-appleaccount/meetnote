# Benchmark: Speaker Diarization

A reproducible test suite for the diarization pipeline. Run it after any change to `engine.py` clustering logic:

```bash
python benchmark_diar.py
```

Results are appended to `data/benchmarks.db` (SQLite, tables `runs` / `cases`) and a comparison table is printed.

## Test cases

| Kind | Cases | Ground truth |
|---|---|---|
| `real` | user recordings (`data/recordings/*.wav` when present) | speaker count (user-confirmed where available) |
| `tts` | TTS dialogues built from single-voice clips | speaker count from clip construction |
| `official` | sherpa official test waves in `models/*.wav` (downloaded by `scripts/download_models.sh`) | speaker count per file name |
| `synthetic` | 1–5 speaker mixes stitched from single-speaker clips with exact timelines (`data/benchmark_audio/`) | per-frame speaker labels — enables attribution accuracy |

External single-speaker clips (TTS voices, English interview clips) are optional: point `MEETNOTE_BENCH_AUDIO_DIR` at a folder with `d1.wav`…`d6.wav` etc., and the corresponding cases are added automatically. Without them the suite degrades gracefully to the cases whose assets exist.

## Metrics

| Metric | What it measures |
|---|---|
| `detected` | number of speakers found by the algorithm |
| `expected` | ground-truth count (blank = unknown) |
| `attribution accuracy` | frame-level agreement with ground-truth labels, after optimal cluster↔speaker mapping (Hungarian). Synthetic cases only |
| `secs_old/secs_new` | wall-clock time for the stock clustering vs. the MeetNote pipeline |

## Reference results (run 2, 2026-09-28)

| Case | Expected | Stock sherpa | MeetNote |
|---|---|---|---|
| real_four_speakers | 4 | 7 | **4** ✓ |
| real_meeting_23min | ? | 49 | 3 |
| tts_dialog / tts_dialog2 | 2 / 2 | 1 / 1 | **2 / 2** ✓ |
| official_0-four-speakers-zh | 4 | 7 | **4** ✓ |
| syn_zh_1spk | 1 | 1 | **1** ✓ (100%) |
| syn_zh_2spk | 2 | 1 | **2** ✓ (99.9%) |
| syn_mix_5spk | 5 | 4 | **5** ✓ |
| syn_mix_4spk | 4 | 4 | 5 |
| syn_en_2spk / 3spk | 2 / 3 | 2 / 3 | 3 / 3 |
| official_*-two-speakers-en | 2×3 | 2 / 5 / 4 | 3 / 8 / 3 |

## Known limits

- **English audio is segmentation-bound, not clustering-bound.** The pyannote segmentation model merges English turns across gaps; the measured oracle ceiling (perfect clustering on the given segments) is 52–68% attribution on the synthetic English cases, and `official_2-two-speakers-en` is extreme noise (whisper transcribes nothing). Fixing this requires a segmentation model change, not clustering parameters.
- **`real_meeting_23min` ground truth is unknown** (conference room: one main speaker plus side talkers). Detected = 3 looks plausible; treat as informational.
- **Very similar synthetic voices** (same TTS engine, different voice IDs) sit near the 0.7 merge threshold; preset `num_speakers` removes the ambiguity entirely.

## How the pipeline is structured (what the benchmark validates)

1. pyannote segmentation (boundaries only; cluster labels discarded)
2. Silero VAD re-split — segmentation merges turns across short silences in long/repetitive audio
3. 3dspeaker voiceprint per segment
4. average-linkage clustering, merge threshold 0.7 (see `AUTO_SIM_THR` in `engine.py`)
5. core-cluster reassignment: fragments (< 2 s or < 2% of speech) join the nearest core centroid

Preset-count path (`num_speakers > 0`) bypasses 4–5 and uses sherpa-onnx clustering directly.
