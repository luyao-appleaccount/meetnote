"""说话人分离基准测试:真实音频 + 合成多人音频,新旧算法对比,结果存 data/benchmarks.db。

运行:  python benchmark_diar.py
可选:  MEETNOTE_BENCH_AUDIO_DIR=/path/to/clips   # 提供 d1-d6.wav 等单说话人素材,启用 tts/syn_zh 用例
用例:
  real_*   真实录音(人数为用户确认/未知)
  tts_*    TTS 合成对话(人数由素材构成确定)
  syn_*    单说话人片段拼接(人数与逐段边界全部已知,可算归属准确率)
指标:
  检测人数 vs 期望人数;syn_* 用例额外计算逐帧说话人归属准确率(Hungarian 最优映射)。
"""
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from faster_whisper.audio import decode_audio

import engine

SR = 16000
APP = Path(__file__).parent
BENCH_DB = APP / "data" / "benchmarks.db"
SYNTH_DIR = APP / "data" / "benchmark_audio"
# 外部单说话人素材(可选):目录里放 d1.wav..d6.wav 与 dialog.wav 等即可启用 tts/syn 用例
EXT_AUDIO = Path(os.environ.get("MEETNOTE_BENCH_AUDIO_DIR", "") or "/nonexistent")
try:
    import gradio as _g
    GRADIO = Path(_g.__file__).parent
except Exception:
    GRADIO = Path("/nonexistent")


def load(p):
    return decode_audio(str(p), sampling_rate=SR)


def _norm(a, rms=0.1):
    r = float(np.sqrt((a ** 2).mean()))
    return (a * (rms / r)).astype(np.float32) if r > 1e-6 else a


def synth_mix(clips, gap=0.8):
    """clips: [(speaker_idx, audio)] → (混合音频, [(start,end,spk)]), 片段响度归一。"""
    out, gt, t = [], [], 0.0
    silence = np.zeros(int(gap * SR), dtype=np.float32)
    for spk, a in clips:
        a = _norm(a)
        out.append(a)
        gt.append((t, t + len(a) / SR, spk))
        t += len(a) / SR
        out.append(silence)
        t += gap
    return np.concatenate(out), gt


def save_wav(path, audio):
    import wave
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---------- 评估 ----------

def _frame_labels(segs, n):
    lab = np.full(n, -1, dtype=np.int64)
    for s, e, spk in segs:
        lab[int(s * 100):min(int(e * 100), n)] = spk
    return lab


def attribution_acc(gt, det, duration):
    """逐帧归属准确率:仅在 GT 有人声的帧上统计,簇→人映射用 Hungarian 最优。"""
    from scipy.optimize import linear_sum_assignment
    n = int(duration * 100)
    g, d = _frame_labels(gt, n), _frame_labels(det, n)
    mask = g >= 0
    if mask.sum() == 0:
        return None
    gspk = sorted(set(g[mask].tolist()))
    dspk = sorted(set(d[d >= 0].tolist()))
    if not dspk:
        return 0.0
    C = np.zeros((len(gspk), len(dspk)))
    for i, gs in enumerate(gspk):
        for j, ds in enumerate(dspk):
            C[i, j] = ((g == gs) & (d == ds)).sum()
    ri, ci = linear_sum_assignment(-C)
    correct = sum(((g == gspk[i]) & (d == dspk[j])).sum() for i, j in zip(ri, ci))
    return float(correct / mask.sum())


def run_old(audio):
    diar = engine.build_diarizer(-1)
    return [(r.start, r.end, r.speaker) for r in diar.process(audio).sort_by_start_time()]


def run_new(audio):
    return engine._auto_diar(audio)


def n_speakers(segs):
    return len({s for _, _, s in segs})


# ---------- 用例 ----------

def _add_if(cases, name, kind, path, expected, note, gt=None):
    """素材存在才加入用例,保证仓库克隆后仍可降级运行。"""
    p = Path(path)
    if p.exists():
        cases.append(dict(name=name, kind=kind, path=str(p), expected=expected,
                          note=note, gt=gt))
    return cases


def build_cases():
    cases = []
    _add_if(cases, "real_four_speakers", "real",
            APP / "data/recordings/meeting_2.wav", 4,
            "用户上传四人对话(已确认 4 人)")
    _add_if(cases, "real_meeting_23min", "real",
            APP / "data/recordings/meeting_1.wav", -1,
            "大会场录音:1 主讲 + 疑似旁人插话,人数未知")
    _add_if(cases, "tts_dialog", "tts", EXT_AUDIO / "dialog.wav", 2,
            "TTS 双人对白(d1-d6 拼接)")
    _add_if(cases, "tts_dialog2", "tts", EXT_AUDIO / "dialog2.wav", 2,
            "TTS 双人对白第二版")
    _add_if(cases, "tts_meeting_m4a", "tts", EXT_AUDIO / "meeting.m4a", -1,
            "TTS 会议片段,人数未知")
    # sherpa 官方测试音频(models/,由 scripts/download_models.sh 下载)
    OFFICIAL = APP / "models"
    for f, n in [("0-four-speakers-zh.wav", 4), ("1-two-speakers-en.wav", 2),
                 ("2-two-speakers-en.wav", 2), ("3-two-speakers-en.wav", 2)]:
        _add_if(cases, f"official_{f[:-4]}", "official", OFFICIAL / f, n,
                "sherpa 官方测试音频" + ("(强噪声,whisper 都转不出字)" if f.startswith("2-") else ""))
    d = {}
    for i in range(1, 7):
        p = EXT_AUDIO / f"d{i}.wav"
        if p.exists():
            try:
                d[i] = load(p)
            except Exception:
                pass
    cate_p = GRADIO / "media_assets/audio/cate_blanch.mp3"
    heath_p = GRADIO / "media_assets/audio/heath_ledger.mp3"
    rec1_p = GRADIO / "media_assets/audio/recording1.wav"
    syn = []
    if all(i in d for i in (1, 3, 5)):
        syn.append(("syn_zh_1spk", [(0, d[1]), (0, d[3]), (0, d[5])],
                    "中文 TTS 单人 × 3 段(同一声纹)"))
        if all(i in d for i in range(1, 7)):
            syn.append(("syn_zh_2spk",
                        [(0, d[1]), (1, d[2]), (0, d[3]), (1, d[4]), (0, d[5]), (1, d[6])],
                        "中文 TTS 2 人 × 6 轮"))
    if all(Path(p).exists() for p in (cate_p, heath_p)):
        cate = load(cate_p)
        heath = load(heath_p)
        syn.append(("syn_en_2spk", [(0, cate), (1, heath), (0, cate), (1, heath)],
                    "英文真人 2 人 × 4 轮(分段模型对英文合并轮次,上限低)"))
        if rec1_p.exists():
            rec1 = load(rec1_p)
            syn.append(("syn_en_3spk",
                        [(0, cate), (1, heath), (2, rec1), (0, cate), (1, heath), (2, rec1)],
                        "英文真人 3 人 × 6 轮(分段上限低)"))
            if all(i in d for i in (1, 2, 3)):
                syn.append(("syn_mix_4spk",
                            [(0, d[1]), (1, cate), (2, d[2]), (3, heath), (0, d[3]), (1, cate)],
                            "中英混合 4 人 × 6 轮"))
                syn.append(("syn_mix_5spk",
                            [(0, d[1]), (1, cate), (2, d[2]), (3, heath), (4, rec1),
                             (0, d[3]), (2, d[4]), (1, cate)],
                            "中英混合 5 人 × 8 轮"))
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    for name, clips, note in syn:
        audio, gt = synth_mix(clips)
        path = SYNTH_DIR / f"{name}.wav"
        save_wav(path, audio)
        cases.append(dict(name=name, kind="synthetic", path=str(path),
                          expected=len({s for s, _ in clips}), note=note, gt=gt))
    return cases


def init_db():
    BENCH_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(BENCH_DB))
    conn.execute("""CREATE TABLE IF NOT EXISTS runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, note TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS cases(
        id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER,
        name TEXT, kind TEXT, duration REAL, expected INTEGER,
        detected_old INTEGER, detected_new INTEGER,
        acc_old REAL, acc_new REAL, secs_old REAL, secs_new REAL, note TEXT)""")
    conn.commit()
    return conn


def main():
    cases = build_cases()
    conn = init_db()
    cur = conn.execute("INSERT INTO runs(created_at, note) VALUES(?,?)",
                       (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "old=sherpa FastClustering(thr=0.6) new=自研平均linkage+核心簇重指派"))
    run_id = cur.lastrowid
    rows = []
    for c in cases:
        audio = load(c["path"])
        dur = len(audio) / SR
        t0 = time.time()
        old = run_old(audio)
        t1 = time.time()
        new = run_new(audio)
        t2 = time.time()
        acc_old = attribution_acc(c["gt"], old, dur) if c.get("gt") else None
        acc_new = attribution_acc(c["gt"], new, dur) if c.get("gt") else None
        conn.execute(
            "INSERT INTO cases(run_id,name,kind,duration,expected,detected_old,detected_new,"
            "acc_old,acc_new,secs_old,secs_new,note) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, c["name"], c["kind"], round(dur, 1), c["expected"],
             n_speakers(old), n_speakers(new), acc_old, acc_new,
             round(t1 - t0, 1), round(t2 - t1, 1), c["note"]))
        conn.commit()
        rows.append((c, dur, old, new, acc_old, acc_new, t1 - t0, t2 - t1))
        print(f"[done] {c['name']}", flush=True)
    conn.close()

    print(f"\n基准测试 run_id={run_id},结果已写入 {BENCH_DB}\n")
    print("| 用例 | 时长 | 期望人数 | 旧算法人数 | 新算法人数 | 旧归属准确率 | 新归属准确率 |")
    print("|---|---|---|---|---|---|---|")
    for c, dur, old, new, ao, an, _, _ in rows:
        fmt = lambda v: f"{v:.1%}" if v is not None else "-"
        exp = c["expected"] if c["expected"] > 0 else "?"
        print(f"| {c['name']} | {dur:.0f}s | {exp} | {n_speakers(old)} | "
              f"{n_speakers(new)} | {fmt(ao)} | {fmt(an)} |")


if __name__ == "__main__":
    main()
