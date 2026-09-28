<div align="center">
  <img src="../../imgs/logo.svg" alt="MeetNote logo" width="140" />
  <h1>MeetNote 会记</h1>
  <p><b>本地优先的会议纪要</b> — 录制、转写、说话人分离、纪要提取，全部在本机完成。</p>
</div>

<p align="center">
  <a href="https://github.com/luoyao-appleaccount/meetnote/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/luoyao-appleaccount/meetnote?style=flat-square" /></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square" /></a>
  <a href="#平台"><img alt="macOS" src="https://img.shields.io/badge/macOS-supported-blue.svg" /></a>
  <a href="#平台"><img alt="Linux" src="https://img.shields.io/badge/Linux-supported-blue.svg" /></a>
  <a href="#平台"><img alt="Windows" src="https://img.shields.io/badge/Windows-supported-blue.svg" /></a>
  <a href="../mcp.md"><img alt="Claude Code" src="https://img.shields.io/badge/Claude_Code-supported-blueviolet.svg" /></a>
  <a href="../mcp.md"><img alt="MCP" src="https://img.shields.io/badge/MCP-supported-blueviolet.svg" /></a>
  <a href="#隐私"><img alt="100% 本地" src="https://img.shields.io/badge/100%25-本地-green?style=flat-square" /></a>
</p>

<p align="center">
  <a href="../../README.md">English</a> | 简体中文
</p>

---

## MeetNote 是什么？

MeetNote 是一个自托管的会议纪要应用：录制麦克风或导入音视频文件，用 faster-whisper 转写、sherpa-onnx 声纹模型做说话人分离，再用纯本地规则引擎（可选 LLM）提取结构化纪要——一句话总结、要点、决策、行动项、说话人统计。不上云、不要 token、录音与文字不出机器。

同一套代码提供三种形态：

- **Web 应用** — FastAPI 服务 + 单页前端（实时录制、会中标记 ⭐/✓、行动项中心、大模型重转、Markdown 导出）。
- **SDK / CLI** — `import sdk` 或 `python sdk.py diarize meeting.wav`，给脚本和 agent 用。
- **MCP server** — 把转写和分离暴露成工具，供 Claude Code、Claude Desktop、WorkBuddy 等 MCP 客户端调用。

## 基准测试

说话人分离是会议纪要链路里最难的一环。sherpa-onnx 自带的自动聚类会严重**过拆**——实测把一场确认 4 人的会议拆成 7 个"说话人"，把一段 23 分钟的大会场录音拆成 **49** 个。光调阈值救不回来。

MeetNote 把聚类环节换成了两段式管线，并在本地基准集（真实录音、TTS 对白、sherpa 官方测试音频、1~5 人合成拼接且边界全已知）上验证：

| 环节 | 做法 |
|---|---|
| pyannote 分段 + **Silero VAD 二次切分** | 分段模型在长音频/重复节奏下会把短静音两侧的多人轮次并成一段，VAD 再切回来 |
| 逐段提取 3dspeaker 声纹 | 归一化嵌入 |
| 平均 linkage 聚类 | 余弦相似度 ≥ 0.7 才合并 |
| 核心簇重指派 | 碎片段（<2 秒或 <2% 人声时长）归入声纹最近的核心簇 |

中文内容（主要使用场景）实测结果：

| 用例 | 真实人数 | sherpa 原版聚类 | MeetNote |
|---|---|---|---|
| 真实四人会议（用户确认） | 4 | 7 | **4** ✓ |
| 23 分钟大会场录音 | 未知（1 主讲 + 插话） | 49 | **3** |
| TTS 双人对白 ×2 | 2 | 1 | **2** ✓ |
| sherpa 官方四人中文 | 4 | 7 | **4** ✓ |
| 合成中文单人 | 1 | 1 | **1** ✓（归属 100%） |
| 合成中文双人 | 2 | 1 | **2** ✓（归属 99.9%） |
| 合成中英混合五人 | 5 | 4 | **5** ✓ |

**已知边界**：纯英文音频受分段模型本身限制（合成用例实测归属准确率上限 52~68%），不是聚类问题——见 [docs/benchmark.md](../benchmark.md)。中文会议（主要场景）全覆盖。

全部用例可复现：`python benchmark_diar.py` 会把结果写入 `data/benchmarks.db`（SQLite）并打印对比表。

## 为什么是 MeetNote？

### 通用会议纪要工具的痛点

- **云依赖** — 会议录音是最敏感的数据之一，很多组织严禁上传到 SaaS 转写接口。
- **人数靠猜** — 多数分离管线要么强制预设人数，要么用固定阈值聚类导致碎片化。真实会议不会自带花名册。
- **固定阈值不稳** — 即使知道人数，固定阈值在真人语音、TTS、远场麦克风、嘈杂会场等不同声学条件下都会崩。

### 核心设计：实测验证的模型组合 + 防御式工程

模型组合经过实测，CPU int8 即可流畅运行，无需 GPU：

| 模型 | 角色 | 理由 |
|---|---|---|
| faster-whisper `small` (int8) | 实时转写 | CPU 实时；`medium`/`large-v3` 供离线精转 |
| Silero VAD（faster-whisper 内置） | 人声检测 | 静音段不送转写，显著减少幻觉 |
| sherpa-onnx + 3dspeaker 声纹 | 说话人分离 | 零授权零 token（对比 pyannote 需 HF 认证） |
| 本地规则引擎（可选 LLM） | 纪要提取 | 不依赖任何 API 的结构化输出 |

实测踩过的坑都写进了代码：离线优先加载模型（hf-mirror 兜底）、macOS 显式 blocksize、幻觉句过滤，以及上面的分离过拆修复。

## 使用方法

### 环境要求

- **Python >= 3.10**
- 麦克风权限（实时录制用；macOS 需给终端授权）

### 快速开始（Web 应用）

```bash
git clone https://github.com/luoyao-appleaccount/meetnote.git
cd meetnote
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash scripts/download_models.sh        # 声纹 + 分段模型（约 50 MB）
python server.py                       # 浏览器打开 http://localhost:7861
```

首次启动自动从本地 HF 缓存读取 whisper `small`，没有则走 hf-mirror 下载。

### CLI

```bash
python sdk.py diarize meeting.wav              # 自动人数
python sdk.py diarize meeting.wav -n 4         # 预设人数（最稳）
python sdk.py transcribe meeting.wav --model medium --json
```

### Python SDK

```python
import sdk
segs = sdk.diarize("meeting.wav")   # [{"start","end","speaker","text"}]
```

### MCP server（给 AI agent 用）

```bash
claude mcp add meetnote -s user -- /path/to/venv/bin/python /path/to/meetnote/mcp_server.py
```

暴露两个工具：`transcribe_audio(path, model)` 与 `diarize_audio(path, num_speakers, model)`。任何 MCP 客户端可用，Claude Desktop / WorkBuddy 的 JSON 配置见 [docs/mcp.md](../mcp.md)。

## 文档

- [快速开始](../quickstart.md) — 安装、模型下载、Web 应用全流程
- [SDK & CLI](../sdk.md) — API 参考与用法
- [MCP server](../mcp.md) — agent 接入（Claude Code / Desktop、WorkBuddy、通用 MCP）
- [基准测试](../benchmark.md) — 测试集、指标、已知边界、复现方法

## 参与贡献

欢迎贡献，开发环境与规范见 [CONTRIBUTING.md](../../CONTRIBUTING.md)。

## License

[Apache-2.0](../../LICENSE) — Copyright 2026 MeetNote Contributors
