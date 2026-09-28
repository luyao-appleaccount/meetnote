# Roadmap · 竞品强项落地计划

按 PR 拆分。每个 PR 独立可合、可独立验收；顺序按依赖关系排列（PR1/PR2 相互独立可并行）。

| PR | 主题 | 来源竞品 | 规模 | 状态 |
|---|---|---|---|---|
| [PR1](#pr1-funasr-中文引擎) | FunASR 中文引擎 | 阿里 FunASR / jkinco-listen-open | 中 | ✅ 已落地 |
| [PR2](#pr2-parakeet-英文引擎) | Parakeet 英文引擎 | meetily 同款 | 中 | 待开始 |
| [PR3](#pr3-说话人命名与声纹注册) | 说话人命名/声纹注册 | TalkTrack / 主流会议产品 | 中 | 待开始 |
| [PR4](#pr4-实时转写-websocket-与实时分离) | 实时 WebSocket + 实时分离 | Vexa / WhisperLiveKit | 大(拆两个 PR) | 待开始 |

通用规范（每个 PR 都适用）：

- 改动 `engine.py` 的 PR 必须跑 `python benchmark_diar.py`，把新 run_id 写进 PR 描述；
- 新模型一律进 `scripts/download_models.sh`，二进制不入 git；
- 文档同步：README / docs/ 下对应文档；
- 不破坏 local-first：所有新路径默认离线可用，联网仅为下载模型。

---

## PR1 FunASR 中文引擎

**目标**：中文转写质量提升（标点、顺滑、术语），作为可选 ASR 引擎与 whisper 并存。

**落地结果（2026-09-28）**：采用 `funasr-onnx`（阿里官方 ONNX 导出，零 torch）的 paraformer-zh + ct-punc 标点组合，int8 量化。实测结论：**汉字准确率与 whisper 持平、标点一致、速度约快一个量级、加载仅 ~0.6s**——定位为「快速中文引擎」而非「更准引擎」（准确率天花板仍是 whisper medium/large-v3）。分离管线保持 whisper 不变。

**改动**：

- `engine.py`：`transcribe_funasr_file()`（paraformer + ct-punc + Silero VAD 分段给时间戳）
- `sdk.py`：`transcribe()`/`diarize()` 增加 `engine="whisper"|"funasr"`；CLI 增加 `--engine`
- `server.py`：`/v1/audio/transcriptions` 增加 `engine` 字段
- `scripts/download_models.sh`：新增 paraformer-onnx / punc-onnx（ModelScope，int8，约 570MB）
- `requirements.txt`：`funasr-onnx`；`numpy` 收紧为 `==1.26.4`（funasr-onnx 硬性要求，faster-whisper/sherpa 均兼容）

**验收**：已实测 d2/dialog2 转写与分离，标点/时间戳/说话人交替均正确；whisper 默认路径不受影响。

**遗留说明**：SenseVoice（多语种、自带标点）的 ONNX 版 ModelScope 上只有 936MB 的 PyTorch `model.pt`，需 torch 才能导出，与「轻依赖」路线冲突，未采用；若后续接受 torch 依赖可再评估。

## PR2 Parakeet 英文引擎

**目标**：补上 benchmark 已知的英文转写短板（ASR 层）。

**背景**：Parakeet TDT 是 meetily 同款引擎，sherpa-onnx 原生支持——**零新增依赖**（本项目已依赖 sherpa-onnx），模型走 `scripts/download_models.sh`。

**改动**：

- `engine.py`：`load_parakeet()` / `transcribe_parakeet(audio)`，与 whisper/paraformer 并列
- 与 PR1 共用引擎抽象层（建议 PR1 先合，本 PR 只加新引擎实现）
- `scripts/download_models.sh`：新增 parakeet-tdt int8 模型（~300MB）

**验收**：

1. `official_1/3-two-speakers-en` 转写对比，parakeet 英文错误率明显低于 whisper small
2. 英文分离用例复测：**注意**——分离上限由分段模型决定（docs/benchmark.md 已记录），本 PR 只改善 ASR，不改动分离指标预期；如实更新 benchmark 表
3. 中英文自动选择或显式参数（`engine` 传 parakeet）均可，默认行为保持 whisper

**风险**：模型大（int8 ~300MB）；中文场景无收益——定位为英文专用可选引擎。

## PR3 说话人命名与声纹注册

**目标**：把"说话人1"变成"张三"，跨会议自动复用。

**改动**：

- `db.py`：新增 `speakers` 表（id, name, embedding BLOB, created_at）；meetings 的段落 speaker 字段存 `speaker_id` 或名字引用
- `engine.py`：暴露逐簇质心声纹（`_auto_diar` 返回簇质心），供注册/匹配
- `server.py`：`POST /api/meetings/{mid}/rename-speaker`（改名+保存声纹）、`GET /api/speakers`
- 新会议分离完成后：每个簇质心与已注册声纹比对，相似度 ≥ 阈值自动命名，未命中显示"说话人N"等待用户命名
- `static/index.html`：详情页说话人改名交互 + 已有声纹库提示

**验收**：

1. 同一人的两段录音（如 d1 与 d3）分别开会，第二次自动显示第一次命的名字
2. 改名后导出 Markdown 使用新名字
3. 注册声纹不影响自动人数结果（仅后处理命名）

**风险**：声纹匹配阈值需与聚类阈值区分（同一人 min 0.43 的实测边界，见 docs/benchmark.md）；跨设备/远场录音匹配率会下降——验收用同设备素材。

## PR4 实时 WebSocket + 实时分离

**目标**：会中边录边看，实时给出说话人标注。规模大，拆两个 PR。

### PR4a 实时转写 WebSocket

- `server.py`：`/ws/live` WebSocket 端点，把现有 `pollLive` 轮询改为推送
- `static/index.html`：`pollLive` 改 WebSocket 订阅，断线自动回退轮询
- **验收**：会中段落延迟 < 3s；无内存泄漏（连续录 1h）

### PR4b 实时说话人分离

- `engine.py`：滚动窗口增量聚类——每 20–30s 对新增段落提取声纹，与已建簇做在线合并（复用 `_average_linkage` 的阈值体系），会议结束时用离线 `_auto_diar` 全量重算覆盖
- **验收**：
  1. 会中显示"说话人N"（允许临时候选），结束后离线重算保证与现状同精度
  2. benchmark 不回归（实时路径不改变离线路径结果）

**风险**：实时聚类可能中途拆错人名造成 UI 跳动——4b 的应对是"临时标注 + 结束重算覆盖"，UI 需明确区分"实时猜测"与"最终结果"。
