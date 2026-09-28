# MCP Server

Expose local transcription and speaker diarization as MCP tools, so AI agents (Claude Code, Claude Desktop, WorkBuddy, Cursor, ...) can process meeting audio without the audio ever leaving your machine.

## Tools

| Tool | Params | Returns |
|---|---|---|
| `transcribe_audio` | `path` (absolute path), `model` = small/medium/large-v3 (default small) | timed transcript text |
| `diarize_audio` | `path`, `num_speakers` = -1 auto / N preset, `model` | speaker-labeled transcript, e.g. `检测到 4 位说话人:` + lines |

Input can be any local audio/video file: wav / mp3 / m4a / aac / flac / mp4 / mov.

## Claude Code

```bash
claude mcp add meetnote -s user -- /path/to/venv/bin/python /path/to/meetnote/mcp_server.py
```

`-s user` makes it available in every project; `claude mcp list` shows health. Remove with `claude mcp remove meetnote`.

## Claude Desktop / WorkBuddy / generic MCP clients

Add to the client's MCP config (Claude Desktop: `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "meetnote": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": ["/absolute/path/to/meetnote/mcp_server.py"]
    }
  }
}
```

A ready-to-copy version lives at [examples/mcp_config.json](../examples/mcp_config.json) — edit the two paths.

## Notes

- Requires `mcp` in the same environment: `pip install "mcp<2"`. **Pin below 2.x** — the 2.x release renamed FastMCP to MCPServer and changed APIs.
- First call loads whisper + voiceprint models (~a minute); subsequent calls are warm.
- Server stdout must stay clean: the tool results are pure text by design.

## Agent usage pattern

Typical flow for an agent that receives an audio file:

1. `diarize_audio(path="/tmp/meeting.wav")` → speaker-labeled transcript
2. If quality is insufficient and the roster size is known, retry with `num_speakers=N`
3. For important meetings, re-run with `model="large-v3"`
4. Feed the transcript to the agent's own summarization — or use the web app's rules/LLM summarizer
