"""MeetNote SDK 端到端示例:批量转写一个目录,导出 JSON + Markdown。

用法:
    python examples/sdk_example.py /path/to/audio_dir [--num-speakers 4] [--model small]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import sdk

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".mp4", ".mov", ".ogg"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("-n", "--num-speakers", type=int, default=-1)
    ap.add_argument("--model", default="small")
    args = ap.parse_args()

    files = [p for p in sorted(Path(args.folder).iterdir()) if p.suffix.lower() in AUDIO_EXTS]
    if not files:
        print(f"目录里没有音频文件: {args.folder}")
        return 1

    out_dir = Path(args.folder) / "meetnote_out"
    out_dir.mkdir(exist_ok=True)
    for p in files:
        print(f"处理 {p.name} ...")
        segs = sdk.diarize(str(p), num_speakers=args.num_speakers, model_size=args.model)
        (out_dir / f"{p.stem}.json").write_text(
            json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [f"# {p.stem}", ""]
        for s in segs:
            t = int(s["start"])
            lines.append(f"[{t // 60:02d}:{t % 60:02d}] {s['speaker']} {s['text']}")
        (out_dir / f"{p.stem}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"完成,输出在 {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
