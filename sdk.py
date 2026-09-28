"""MeetNote SDK:给 agent / 脚本调用的本地转写 + 说话人分离 API。

用法(Python):
    import sdk
    segs = sdk.diarize("meeting.wav")                # 自动人数
    segs = sdk.diarize("meeting.wav", num_speakers=4)  # 预设人数
    text = sdk.transcribe("meeting.wav")

用法(CLI):
    python sdk.py transcribe meeting.wav [--model small]
    python sdk.py diarize meeting.wav [-n 4] [--model small] [--json]

返回段格式: {"start": 秒, "end": 秒, "speaker": "说话人1", "text": "..."}
(transcribe 返回的段没有 speaker 字段)
依赖与 server.py 相同,全部本地运行,音频不出机器。
"""
import argparse
import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))


def _decode(path):
    from faster_whisper.audio import decode_audio
    import engine
    return decode_audio(str(path), sampling_rate=engine.SAMPLE_RATE)


def transcribe(path, model_size="small", engine="whisper"):
    """整段音频转文字,返回 [{"start","end","text"}]。

    engine: whisper(默认) / funasr(paraformer-zh 快速中文,带标点)
    """
    import engine
    if engine == "funasr":
        return engine.transcribe_funasr_file(str(path))
    return engine.retranscribe_file(str(path), model_size)


def diarize(path, num_speakers=-1, model_size="small", engine="whisper"):
    """转写 + 说话人分离,返回 [{"start","end","speaker","text"}]。

    num_speakers: -1 自动人数(自研抗过拆聚类);>0 预设人数(sherpa 聚类,最稳)。
    engine: whisper(默认) / funasr(快速中文);分离管线始终使用 whisper。
    """
    import engine
    audio = _decode(path)
    results = [(s["start"], s["end"], s["text"]) for s in transcribe(path, model_size, engine)]
    sd = engine.build_diarizer(num_speakers) if num_speakers and num_speakers > 0 else None
    return engine.diarize_audio(audio, results, sd)


def main():
    ap = argparse.ArgumentParser(description="MeetNote 本地转写/说话人分离 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("transcribe", "diarize"):
        p = sub.add_parser(name)
        p.add_argument("audio", help="音频/视频文件路径")
        p.add_argument("--model", default="small", help="whisper 模型(small/medium/large-v3)")
        p.add_argument("--engine", default="whisper", choices=("whisper", "funasr"),
                       help="转写引擎: whisper(默认) / funasr(快速中文)")
        p.add_argument("--json", action="store_true", help="输出 JSON")
    sub.choices["diarize"].add_argument("-n", "--num-speakers", type=int, default=-1,
                                        help="说话人数,-1=自动")
    args = ap.parse_args()

    if args.cmd == "transcribe":
        segs = transcribe(args.audio, args.model, args.engine)
    else:
        segs = diarize(args.audio, args.num_speakers, args.model, args.engine)

    if args.json:
        print(json.dumps(segs, ensure_ascii=False, indent=2))
    else:
        from engine import fmt_mmss
        for s in segs:
            spk = f"{s.get('speaker')} " if s.get("speaker") else ""
            print(f"[{fmt_mmss(s['start'])}] {spk}{s['text']}")


if __name__ == "__main__":
    main()
    # sherpa-onnx 在解释器退出阶段的析构有已知 crash(recursive_mutex),打完结果直接退
    import os
    sys.stdout.flush()
    os._exit(0)
