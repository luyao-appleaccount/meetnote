# SDK & CLI

`sdk.py` is a dependency-light entry point for scripts and agents: no server, no web UI, same engine.

## CLI

```bash
# transcribe only (returns timed text)
python sdk.py transcribe meeting.wav
python sdk.py transcribe meeting.m4a --model medium --json

# transcribe + diarize
python sdk.py diarize meeting.wav              # -1 = auto speaker count (default)
python sdk.py diarize meeting.wav -n 4         # preset count — most stable when known
```

Human output:

```
[00:00] 说话人1 大家好,我是产品经理小美。今天由我来介绍项目的整体进度。
[00:07] 说话人2 我是研发负责人王强,我先补充一下技术这边的情况。
```

`--json` emits the raw segment list.

## Python API

```python
import sdk

# [{"start": 0.0, "end": 6.9, "text": "..."}]
segs = sdk.transcribe("meeting.wav", model_size="small")
segs = sdk.transcribe("meeting.wav", engine="funasr")        # 快速中文引擎

# [{"start", "end", "speaker": "说话人1", "text"}]
segs = sdk.diarize("meeting.wav")                    # auto
segs = sdk.diarize("meeting.wav", num_speakers=4,    # preset
                   model_size="medium")
segs = sdk.diarize("meeting.wav", engine="funasr")   # 转写用 funasr,分离仍走 whisper
```

### Model choice

| Model | Speed | Accuracy | Use |
|---|---|---|---|
| `small` (int8) | real-time on CPU | good for everyday zh meetings | default, live recording |
| `medium` | slower | clearly better | offline re-transcription |
| `large-v3` | slowest | best | important meetings |

### Engine (转写引擎)

| Engine | 说明 |
|---|---|
| `whisper` (默认) | faster-whisper,支持中英等多语种,分离管线使用它 |
| `funasr` | 阿里 FunASR ONNX 版(paraformer-zh + ct-punc 标点,零 torch):中文汉字准确率与 whisper 持平、速度约快一个量级、加载 ~0.6s;仅覆盖转写,分离仍走 whisper |

### num_speakers

| Value | Behavior |
|---|---|
| `-1` (default) | auto: pyannote segmentation + Silero VAD re-split + voiceprint clustering + core reassignment (see [benchmark](benchmark.md)) |
| `> 0` | preset: sherpa-onnx clustering with fixed cluster count — most stable when you know the roster |

### Under the hood

`sdk.py` calls into `engine.py`:

```python
engine.retranscribe_file(path, model_size)            # -> [{"start","end","text"}]
engine.diarize_audio(audio, results, sd_diar=None)    # -> [{"start","end","speaker","text"}]
engine.build_diarizer(num_speakers)                   # sherpa diarizer (preset path)
```

`diarize_audio` with `sd_diar=None` runs the auto pipeline; pass a built diarizer to force the preset path. All processing is local — audio never leaves the machine.

## Examples

See [examples/sdk_example.py](../examples/sdk_example.py) for a runnable end-to-end script (batch process a folder, export JSON + Markdown).
