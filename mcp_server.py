"""MeetNote MCP server:把本地转写/说话人分离暴露给 agent(Claude Code、Claude Desktop、WorkBuddy 等支持 MCP 的客户端)。

接入(Claude Code):
    claude mcp add meetnote -- /Users/hik/whisper-test/venv/bin/python /Users/hik/wly/meetnote-app/mcp_server.py

接入(Claude Desktop / 其他 MCP 客户端的 JSON 配置):
    {
      "mcpServers": {
        "meetnote": {
          "command": "/Users/hik/whisper-test/venv/bin/python",
          "args": ["/Users/hik/wly/meetnote-app/mcp_server.py"]
        }
      }
    }

工具:
    transcribe_audio(path, model)          音频/视频 → 带时间轴文字
    diarize_audio(path, num_speakers, model) 音频/视频 → 带说话人的逐句文字
全部本地运行,音频不出机器。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("meetnote")


def _fmt(segs):
    from engine import fmt_mmss
    lines = []
    for s in segs:
        spk = f"{s.get('speaker')} " if s.get("speaker") else ""
        lines.append(f"[{fmt_mmss(s['start'])}] {spk}{s['text']}")
    return "\n".join(lines)


@mcp.tool()
def transcribe_audio(path: str, model: str = "small") -> str:
    """把本地音频/视频文件转写成带时间轴的文字(不出本机)。

    path: 音频/视频文件的绝对路径(wav/mp3/m4a/mp4 等)
    model: whisper 模型,small(快) / medium / large-v3(慢但准)
    """
    import sdk
    segs = sdk.transcribe(path, model)
    return _fmt(segs)


@mcp.tool()
def diarize_audio(path: str, num_speakers: int = -1, model: str = "small") -> str:
    """把本地音频/视频文件转写成带说话人标注的逐句文字(不出本机)。

    path: 音频/视频文件的绝对路径
    num_speakers: 说话人数,-1=自动;知道人数时填具体数字更准
    model: whisper 模型,small(快) / medium / large-v3(慢但准)
    """
    import sdk
    segs = sdk.diarize(path, num_speakers=num_speakers, model_size=model)
    n = len({s["speaker"] for s in segs})
    return f"检测到 {n} 位说话人:\n" + _fmt(segs)


if __name__ == "__main__":
    mcp.run()
