"""实时转写引擎:复用 live_transcribe 已验证的逻辑,封装成可 start/stop 的类。

(基于原 whisper-handoff/scripts/engine.py,仅做两处改动:
  1. 模型路径默认指向本目录 models/,可用环境变量 MEETNOTE_MODELS_DIR 覆盖;
  2. 新增 diarize_audio() 静态方法,供"导入音频"对文件做说话人分离。)

供 server.py 调用:
  engine = TranscribeEngine(on_segment=lambda s,e,t: ...)
  engine.load_models()
  engine.start()          # 开始录音+实时转写
  audio = engine.stop()   # 停止,返回完整录音
  engine.save_wav(path)   # 保存录音
  segs = engine.diarize() # 说话人分离(返回带 speaker 的段)
"""
import queue
import re
import threading
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
PROMPT = "以下是普通话的句子。"
BLOCK_S = 0.3

MODELS_DIR = Path(__file__).parent / "models"


def _seg_model_path():
    import os
    base = Path(os.environ.get("MEETNOTE_MODELS_DIR", str(MODELS_DIR)))
    return base / "model.onnx"


def _emb_model_path():
    import os
    base = Path(os.environ.get("MEETNOTE_MODELS_DIR", str(MODELS_DIR)))
    return base / "3dspeaker_zh.onnx"


def load_whisper(model_size):
    """优先本地缓存加载(不联网);未缓存的模型再联网下载。"""
    try:
        return WhisperModel(model_size, device="cpu", compute_type="int8", local_files_only=True)
    except Exception:
        return WhisperModel(model_size, device="cpu", compute_type="int8")


def fmt_mmss(seconds):
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def retranscribe_file(wav_path, model_size="medium"):
    """对录音文件整体重转(用于 medium/large 高质量版),返回段列表。"""
    from faster_whisper.audio import decode_audio
    model = load_whisper(model_size)
    audio = decode_audio(str(wav_path), sampling_rate=SAMPLE_RATE)
    segments, _ = model.transcribe(
        audio, language="zh", vad_filter=True, beam_size=5, initial_prompt=PROMPT
    )
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


# ---------- FunASR 中文引擎(可选) ----------
# funasr-onnx(阿里官方 ONNX 导出,零 torch)的 paraformer-zh + ct-punc 标点:
# 汉字准确率与 whisper 持平,速度约快一个量级,加载仅 ~0.6s。仅覆盖转写,分离仍走 whisper。
_FUNASR = None


def _load_funasr():
    global _FUNASR
    if _FUNASR is None:
        from funasr_onnx import Paraformer, CT_Transformer
        asr_dir = MODELS_DIR / "paraformer-onnx"
        punc_dir = MODELS_DIR / "punc-onnx"
        if not (asr_dir / "model_quant.onnx").exists():
            raise FileNotFoundError(f"缺少 FunASR 模型,请先运行 scripts/download_models.sh: {asr_dir}")
        _FUNASR = (Paraformer(str(asr_dir), quantize=True),
                   CT_Transformer(str(punc_dir), quantize=True))
    return _FUNASR


def transcribe_funasr_file(wav_path):
    """FunASR 快速中文转写(paraformer-zh + ct-punc 标点),Silero VAD 分段给时间戳。"""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    asr, punc = _load_funasr()
    audio = decode_audio(str(wav_path), sampling_rate=SAMPLE_RATE)
    ts = get_speech_timestamps(
        audio, VadOptions(threshold=0.5, min_silence_duration_ms=500,
                          min_speech_duration_ms=250, speech_pad_ms=100))
    segs = []
    for v in ts:
        chunk = audio[v["start"]:v["end"]]
        res = asr(chunk)
        text = res[0]["preds"][0] if res else ""
        if not text.strip():
            continue
        text = punc(text)[0]
        segs.append({"start": v["start"] / SAMPLE_RATE,
                     "end": v["end"] / SAMPLE_RATE, "text": text})
    return segs


def build_diarizer(num_speakers=-1):
    """构建 sherpa-onnx 离线说话人分离器(零授权,不需要 HF token)。"""
    import sherpa_onnx
    seg = str(_seg_model_path())
    emb = str(_emb_model_path())
    if not Path(seg).exists() or not Path(emb).exists():
        raise FileNotFoundError(f"缺少说话人分离模型: {seg} / {emb}")
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=seg, window_shift_ratio=0.1,
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=0.6,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    return sherpa_onnx.OfflineSpeakerDiarization(config)


# ---------- 自动人数模式 ----------
# sherpa FastClustering 在自动模式下严重过拆(实测 4 人音频拆出 7 个、23 分钟会议拆出 49 个),
# 阈值怎么调都救不回。改为只用 sherpa 做分段,聚类自己做(benchmark_diar.py 全量用例验证):
#   1. pyannote 分段 → Silero VAD 二次切分(分段模型在长音频/重复节奏下会把多人轮次并成一段)
#   2. 逐段提取 3dspeaker 声纹 → 平均 linkage 按 AUTO_SIM_THR 合并
#   3. 碎片段(总时长过短的簇)归入声纹最近的核心簇
# 固定阈值能成立的前提是同人/异人相似度 gap 足够大(实测真人中文: 同人 min 0.43 / 异人 max 0.26);
# 已知边界: 英文音频的分段模型合并轮次较多,归属准确率受分段上限制约。
AUTO_SIM_THR = 0.7          # 平均 linkage 合并阈值(余弦相似度)
AUTO_CORE_MIN_S = 2.0       # 核心簇最短总时长(秒)
AUTO_CORE_MIN_RATIO = 0.02  # 核心簇总时长占全部人声比例下限

_boundary_diar = None
_emb_extractor = None


def _boundary_diarizer():
    """分段器(聚类标签丢弃,只用分段边界),模块级缓存避免重复建 ONNX 会话。"""
    global _boundary_diar
    if _boundary_diar is None:
        _boundary_diar = build_diarizer(-1)
    return _boundary_diar


def _vad_split(audio, segs):
    """用 Silero VAD 把分段结果再切一刀:分段模型可能把短静音两侧的不同说话人并成一段。"""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    ts = get_speech_timestamps(
        audio, VadOptions(threshold=0.5, min_silence_duration_ms=400,
                          min_speech_duration_ms=200, speech_pad_ms=80))
    out = []
    for s, e in segs:
        for v in ts:
            a = max(s, v["start"] / SAMPLE_RATE)
            b = min(e, v["end"] / SAMPLE_RATE)
            if b - a >= 0.3:
                out.append((a, b))
    return out or segs


def _embed_segments(audio, segs):
    """逐段提取 3dspeaker 声纹并归一化。"""
    import sherpa_onnx
    global _emb_extractor
    if _emb_extractor is None:
        _emb_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(_emb_model_path())))
    min_len = int(0.3 * SAMPLE_RATE)
    out = []
    for s, e in segs:
        chunk = audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]
        if len(chunk) < min_len:  # 边界取整导致的过短段,补零凑够模型输入
            chunk = np.pad(chunk, (0, min_len - len(chunk)))
        stream = _emb_extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=chunk)
        stream.input_finished()
        v = np.asarray(_emb_extractor.compute(stream), dtype=np.float32)
        out.append(v / (np.linalg.norm(v) + 1e-9))
    return np.stack(out)


def _average_linkage(X, sim_thr):
    """贪心平均 linkage:反复合并簇间平均相似度最高且 >= sim_thr 的两个簇。"""
    n = len(X)
    S = X @ X.T
    np.fill_diagonal(S, -1.0)
    members = [[i] for i in range(n)]
    active = np.ones(n, dtype=bool)
    idx = np.arange(n)
    while True:
        M = np.where(active[:, None] & active[None, :], S, -1.0)
        i, j = divmod(int(np.argmax(M)), n)
        if i == j or M[i, j] < sim_thr:
            break
        la, lb = len(members[i]), len(members[j])
        w = active & (idx != i) & (idx != j)
        S[i, :] = np.where(w, (S[i, :] * la + S[j, :] * lb) / (la + lb), -1.0)
        S[:, i] = S[i, :]
        S[j, :] = S[:, j] = -1.0
        members[i] += members[j]
        members[j] = []
        active[j] = False
    labels = np.empty(n, dtype=int)
    cid = 0
    for m in members:
        for x in m:
            labels[x] = cid
        cid += bool(m)
    return labels


def _auto_diar(audio):
    return _auto_diar_full(audio)[0]


def _auto_diar_full(audio):
    """自动分离,返回 ([(start, end, cluster_id)], {cluster_id: 归一化质心})。"""
    segs = [(r.start, r.end)
            for r in _boundary_diarizer().process(audio).sort_by_start_time()]
    segs = _vad_split(audio, segs)
    if len(segs) < 2:
        return [(s, e, 0) for s, e in segs], {}
    X = _embed_segments(audio, segs)
    labels = _average_linkage(X, AUTO_SIM_THR)
    # 筛核心簇(总时长过短的是碎片),碎片段按声纹相似度归入最近核心簇
    durs = np.array([e - s for s, e in segs])
    core_min = max(AUTO_CORE_MIN_S, AUTO_CORE_MIN_RATIO * durs.sum())
    cdur = {}
    for l, d in zip(labels, durs):
        cdur[l] = cdur.get(l, 0.0) + d
    core = sorted(l for l, c in cdur.items() if c >= core_min) or sorted(cdur)
    C = np.stack([X[labels == l].mean(axis=0) for l in core])
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    final = []
    for l, x in zip(labels, X):
        final.append(l if l in core else core[int(np.argmax(C @ x))])
    cents = {}
    for l in set(final):
        v = X[np.array(final) == l].mean(axis=0)
        cents[l] = v / (np.linalg.norm(v) + 1e-9)
    return [(s, e, l) for (s, e), l in zip(segs, final)], cents


def diarize_with_centroids(audio, results, sd_diar=None):
    """说话人分离,一次计算同时返回段落与每个说话人的质心声纹。

    返回: (aligned, centroids)
      aligned:   [{'start','end','speaker','text'}]
      centroids: {'说话人N': 归一化 float32 数组}
    """
    if len(audio) == 0:
        return [], {}
    if sd_diar is not None:
        diar = [(r.start, r.end, r.speaker)
                for r in sd_diar.process(audio).sort_by_start_time()]
        segs = [(s, e) for s, e, _ in diar]
        X = _embed_segments(audio, segs) if segs else None
        spk_ids = sorted({s for _, _, s in diar})
        spk_map = {sid: f"说话人{i + 1}" for i, sid in enumerate(spk_ids)}
        cents = {}
        if X is not None and len(X):
            for sid in spk_ids:
                idx = [k for k, (_, _, s) in enumerate(diar) if s == sid]
                v = X[idx].mean(axis=0)
                cents[spk_map[sid]] = v / (np.linalg.norm(v) + 1e-9)
    else:
        diar, cents_raw = _auto_diar_full(audio)
        ids = sorted(cents_raw)
        spk_map = {sid: f"说话人{i + 1}" for i, sid in enumerate(ids)}
        cents = {spk_map[sid]: cents_raw[sid] for sid in ids}
    aligned = []
    for start, end, text in sorted(results, key=lambda r: r[0]):
        speaker = _best_speaker(diar, start, end, spk_map)
        aligned.append({"start": start, "end": end, "speaker": speaker, "text": text})
    return aligned, cents


def best_speaker_match(emb, known_embs, threshold=0.65):
    """在已知声纹库 known_embs {name: np.ndarray} 里找最匹配的名字,低于阈值返回 None。

    阈值偏保守:误配(把甲认成乙)比漏配(显示"说话人N")代价大得多;
    实测真人同人质心 ~0.6-0.9、异人 ~0.26,TTS 相似音色异人 ~0.57,故取 0.65。
    """
    best, bs = None, -1.0
    n = np.linalg.norm(emb) + 1e-9
    for name, k in known_embs.items():
        s = float(np.dot(emb, k) / (n * (np.linalg.norm(k) + 1e-9)))
        if s > bs:
            bs, best = s, name
    return best if bs >= threshold else None


def diarize_audio(audio, results, sd_diar=None):
    """对一段完整音频 + 已有转写结果做说话人分离。

    audio:   float32 numpy 数组(16k mono)
    results: [(start, end, text)] 转写段
    sd_diar: 预设人数时传入 build_diarizer(n) 的实例;不传(自动人数)走 _auto_diar
    返回:    [{'start','end','speaker','text'}]
    """
    return diarize_with_centroids(audio, results, sd_diar)[0]


def _best_speaker(diar, start, end, spk_map):
    best, best_overlap = None, 0.0
    for ds, de, spk in diar:
        overlap = max(0.0, min(end, de) - max(start, ds))
        if overlap > best_overlap:
            best_overlap, best = overlap, spk
    return spk_map.get(best, "说话人?")


def list_input_devices():
    """列出可用音频输入设备,供 UI/agent 选择(如 BlackHole 用于录会议软件的声音)。"""
    import sounddevice as sd
    default = sd.default.device
    if isinstance(default, (tuple, list)):
        default = default[0]
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append({"index": i, "name": d["name"], "default": i == default})
    return out


class TranscribeEngine:
    def __init__(self, model_size="small", num_speakers=-1, on_segment=None):
        self.model_size = model_size
        self.num_speakers = num_speakers
        self.on_segment = on_segment      # fn(start, end, text) 实时回调
        self.results = []                 # (start, end, text)
        self.audio_frames = []
        self.total_samples = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self.model = None
        self.sd_diar = None
        self.device = None

    def load_models(self, with_diarization=True):
        if self.model is None:
            self.model = load_whisper(self.model_size)
        if with_diarization and self.sd_diar is None:
            self.sd_diar = build_diarizer(self.num_speakers)

    def start(self, device=None):
        """开始录音+实时转写。device: 音频输入设备名(如 "BlackHole 2ch"),None=系统默认麦克风。"""
        if self.model is None:
            self.load_models()
        if device:
            # 同步校验设备存在,让 start() 调用方 fail-fast;否则无效设备名要到后台线程才报错
            import sounddevice as sd
            if device not in {d["name"] for d in sd.query_devices()}:
                raise ValueError(f"音频输入设备不存在: {device}")
        self.device = device
        self.results = []
        self.audio_frames = []
        self.total_samples = 0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import sounddevice as sd
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        pending = np.zeros(0, dtype=np.float32)
        offset = 0
        done_end = 0
        tq = queue.Queue()

        VAD_EVERY = int(2.0 * SAMPLE_RATE)
        END_SILENCE = int(0.8 * SAMPLE_RATE)
        OVERLAP = int(0.5 * SAMPLE_RATE)
        MAX_SEG = int(12 * SAMPLE_RATE)

        def worker():
            while True:
                item = tq.get()
                if item is None:
                    return
                seg_start, audio = item
                segments, _ = self.model.transcribe(
                    audio, language="zh", vad_filter=False,
                    beam_size=5, initial_prompt=PROMPT,
                )
                text = "".join(s.text.strip() for s in segments).strip()
                if not text:
                    continue
                core = text.replace("以下是普通话的句子", "").replace("请使用简体中文转写", "")
                if not re.sub(r"[。，,、\s]", "", core):
                    continue
                end = seg_start + len(audio) / SAMPLE_RATE
                with self._lock:
                    self.results.append((seg_start, end, text))
                if self.on_segment:
                    self.on_segment(seg_start, end, text)

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        def run_vad(final=False):
            nonlocal pending, offset, done_end
            if len(pending) < int(0.5 * SAMPLE_RATE):
                return
            ts = get_speech_timestamps(
                pending,
                VadOptions(threshold=0.5,
                           min_silence_duration_ms=500 if final else 800,
                           min_speech_duration_ms=250, speech_pad_ms=100),
            )
            for seg in ts:
                s = offset + seg["start"]
                e = offset + seg["end"]
                if e <= done_end:
                    continue
                too_long = (e - s) > MAX_SEG
                if final or too_long or e + END_SILENCE < offset + len(pending):
                    tq.put((s / SAMPLE_RATE, pending[seg["start"]:seg["end"]]))
                    done_end = e
            if done_end > offset:
                new_offset = max(done_end - OVERLAP, offset)
                pending = pending[new_offset - offset:]
                offset = new_offset

        q = queue.Queue()

        def callback(indata, frames, tinfo, status):
            q.put(indata[:, 0].copy())

        # 坑3:macOS 上必须显式指定 blocksize,否则回调块只有 14 帧,CPU 被拖满
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            device=self.device,
                            blocksize=int(SAMPLE_RATE * BLOCK_S), callback=callback):
            while not self._stop_event.is_set():
                try:
                    block = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                samples = block.astype(np.float32).ravel()
                with self._lock:
                    self.total_samples += len(samples)
                    self.audio_frames.append(samples)
                pending = np.concatenate([pending, samples])
                if len(pending) >= VAD_EVERY + END_SILENCE:
                    run_vad()

        run_vad(final=True)
        tq.put(None)
        worker_thread.join()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=30)
        return self._full_audio()

    def elapsed(self):
        with self._lock:
            return self.total_samples / SAMPLE_RATE

    def _full_audio(self):
        with self._lock:
            if not self.audio_frames:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self.audio_frames)

    def save_wav(self, path):
        audio = self._full_audio()
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
        return str(path)

    def diarize(self):
        """对完整录音做说话人分离,返回 [{'start','end','speaker','text'}]。"""
        audio = self._full_audio()
        if self.num_speakers > 0:
            if self.sd_diar is None:
                self.load_models(with_diarization=True)
            return diarize_audio(audio, self.results, self.sd_diar)
        return diarize_audio(audio, self.results)
