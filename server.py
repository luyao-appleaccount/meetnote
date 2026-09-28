"""会记 MeetNote · 本地会议纪要服务(FastAPI)。

运行:
    python server.py
    浏览器打开 http://localhost:7861

依赖均在原 whisper-handoff 的 venv 中(fastapi/uvicorn/faster-whisper/sherpa-onnx/sounddevice),
无需额外安装。数据(数据库/录音)默认存于本项目 data/ 目录,可用 MEETNOTE_DATA_DIR 重定向。
"""
import json
import os
import threading
import wave
from datetime import datetime
from pathlib import Path

# 国内网络环境:默认走 HF 镜像(已在 README 坑2验证),直连顺畅可注释
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
import summarizer

APP_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("MEETNOTE_DATA_DIR", str(APP_DIR / "data")))
RECORDINGS = DATA_DIR / "recordings"
RECORDINGS.mkdir(parents=True, exist_ok=True)
DB_DIR_NOTE = str(DATA_DIR)

# ---------- 引擎可用性(缺依赖时 UI 仍可浏览历史) ----------
try:
    import engine as eng_mod
    ENGINE_OK = True
    ENGINE_ERR = ""
except Exception as e:  # pragma: no cover
    eng_mod = None
    ENGINE_OK = False
    ENGINE_ERR = str(e)

db.init_db()

app = FastAPI(title="会记 MeetNote")

state = {
    "engine": None,          # TranscribeEngine 实例
    "recording": False,
    "live": [],              # [(start, end, text)] 实时段
    "model_ready": False,
    "model_msg": "未加载",
    "cur_title": "",
    "cur_started": None,
    "processing": {},        # mid -> {"step": str, "msg": str}
}

SUMMARY_STYLE = "balanced"


def on_seg(start, end, text):
    state["live"].append((start, end, text))


def ensure_engine(num_speakers):
    if state["engine"] is None or state["engine"].num_speakers != num_speakers:
        state["engine"] = eng_mod.TranscribeEngine(
            model_size="small", num_speakers=num_speakers, on_segment=on_seg)
        state["model_ready"] = False
    return state["engine"]


def preload():
    if not ENGINE_OK:
        state["model_msg"] = f"引擎不可用:{ENGINE_ERR}"
        return
    try:
        ensure_engine(-1).load_models()
        state["model_ready"] = True
        state["model_msg"] = "就绪(whisper small + 说话人分离已加载)"
    except Exception as e:
        state["model_msg"] = f"模型加载失败:{e}"


threading.Thread(target=preload, daemon=True).start()


# ---------- 模型 ----------

class StartReq(BaseModel):
    title: str = ""
    num_speakers: int = -1   # -1 = 自动
    device: str = None       # 音频输入设备名(None=默认麦克风;BlackHole 可录会议软件声音)


class StopReq(BaseModel):
    stars: list = []         # 会中标记的重点文本
    todos: list = []         # 会中标记的待办文本
    tags: list = []


@app.get("/api/state")
def api_state():
    elapsed = 0.0
    if state["recording"] and state["engine"]:
        elapsed = state["engine"].elapsed()
    return {
        "engine_ok": ENGINE_OK,
        "engine_msg": ENGINE_ERR,
        "model_ready": state["model_ready"],
        "model_msg": state["model_msg"],
        "recording": state["recording"],
        "elapsed": elapsed,
        "cur_title": state["cur_title"],
        "llm_enabled": summarizer._llm_enabled(),
        "data_dir": DB_DIR_NOTE,
    }


@app.get("/api/audio-devices")
def api_audio_devices():
    """列出可用音频输入设备(BlackHole 等虚拟设备可采集会议软件声音)。"""
    if not ENGINE_OK:
        return []
    try:
        return eng_mod.list_input_devices()
    except Exception:
        return []


@app.post("/api/start")
def api_start(req: StartReq):
    if not ENGINE_OK:
        raise HTTPException(500, f"转写引擎不可用:{ENGINE_ERR}")
    if state["recording"]:
        raise HTTPException(409, "已在录制中")
    eng = ensure_engine(req.num_speakers)
    if not state["model_ready"]:
        eng.load_models()
        state["model_ready"] = True
    state["live"] = []
    state["cur_title"] = req.title.strip()
    state["cur_started"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        eng.start(device=req.device)
    except Exception as e:
        state["model_msg"] = f"无法打开麦克风:{e}"
        raise HTTPException(500, f"无法开始录音:{e}(服务器需要能访问麦克风/音频输入设备)")
    state["recording"] = True
    return {"ok": True, "title": state["cur_title"]}


@app.get("/api/live")
def api_live(after: int = 0):
    segs = state["live"][after:]
    return {
        "recording": state["recording"],
        "elapsed": state["engine"].elapsed() if (state["recording"] and state["engine"]) else 0,
        "total": len(state["live"]),
        "segments": [{"start": s, "end": e, "text": t} for s, e, t in segs],
    }


def _pipeline(mid, results, audio, num_speakers, source="small", stars=None, todos=None):
    """停止/导入后的统一流水线:保存录音 → 说话人分离 → 摘要 → 行动项入库。"""
    steps = state["processing"]
    try:
        steps[mid] = {"step": "save", "msg": "保存录音"}
        wav = RECORDINGS / f"meeting_{mid}.wav"
        if audio is not None and len(audio):
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            with wave.open(str(wav), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                w.writeframes(pcm.tobytes())
            db.update_meeting_wav(mid, str(wav))

        steps[mid] = {"step": "diarize", "msg": "说话人分离中"}
        if num_speakers and num_speakers > 0:
            diar = eng_mod.build_diarizer(num_speakers)
            segs = eng_mod.diarize_audio(audio, results, diar)
        else:
            segs = eng_mod.diarize_audio(audio, results)

        db.add_segments(mid, segs, source)

        steps[mid] = {"step": "summary", "msg": "生成纪要概览"}
        duration = max((e for _, e, _ in results), default=0) if isinstance(results, list) else 0
        m = db.get_meeting(mid)
        summary = summarizer.build(segs, duration=m.get("duration") or duration,
                                   style=SUMMARY_STYLE, title=m["title"])
        db.set_meeting_summary(mid, summary)
        for a in summary.get("actions", []):
            db.add_action(mid, a["text"], a.get("owner", ""), a.get("due", ""))
        db.set_meeting_marks(mid, {"stars": stars or [], "todos": todos or []})
        db.set_meeting_status(mid, "done")
        steps[mid] = {"step": "done", "msg": "完成"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        steps[mid] = {"step": "error", "msg": f"处理失败:{e}"}
        db.set_meeting_status(mid, "error")


@app.post("/api/stop")
def api_stop(req: StopReq):
    if not state["recording"]:
        raise HTTPException(409, "当前没有进行中的会议")
    eng = state["engine"]
    eng.stop()
    state["recording"] = False
    title = state["cur_title"] or f"会议 {datetime.now().strftime('%m-%d %H:%M')}"
    duration = eng.elapsed()
    results = list(eng.results)
    mid = db.create_meeting(title, "", duration, eng.num_speakers,
                            status="processing", tags=req.tags)
    state["cur_title"] = ""
    threading.Thread(target=_pipeline, daemon=True,
                     args=(mid, results, eng._full_audio(), eng.num_speakers,
                           "small", req.stars, req.todos)).start()
    return {"ok": True, "meeting_id": mid}


@app.post("/api/import")
async def api_import(file: UploadFile = File(...), num_speakers: int = Form(-1),
                     title: str = Form("")):
    """导入已有录音/视频文件,自动转写 + 说话人分离,生成纪要。"""
    if not ENGINE_OK:
        raise HTTPException(500, f"转写引擎不可用:{ENGINE_ERR}")
    suffix = Path(file.filename or "audio").suffix or ".wav"
    raw = RECORDINGS / f"import_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    raw.write_bytes(await file.read())
    title = (title or Path(file.filename).stem).strip()
    mid = db.create_meeting(title, str(raw), 0, num_speakers, status="processing", tags=["导入"])

    def work():
        try:
            state["processing"][mid] = {"step": "transcribe", "msg": "音频转写中(首次加载模型较慢)"}
            from faster_whisper.audio import decode_audio
            audio = decode_audio(str(raw), sampling_rate=16000)
            segs_raw = eng_mod.retranscribe_file(str(raw), "small")
            results = [(s["start"], s["end"], s["text"]) for s in segs_raw]
            duration = len(audio) / 16000
            db.set_meeting_duration(mid, duration)
            _pipeline(mid, results, audio, num_speakers, "small")
        except Exception as e:
            import traceback
            traceback.print_exc()
            state["processing"][mid] = {"step": "error", "msg": f"导入失败:{e}"}
            db.set_meeting_status(mid, "error")

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "meeting_id": mid}


# ---------- OpenAI 兼容转写端点(给通用 agent/工具调用) ----------

@app.post("/v1/audio/transcriptions")
async def v1_transcriptions(file: UploadFile = File(...), model: str = Form("small"),
                            response_format: str = Form("json"), engine: str = Form("whisper")):
    """OpenAI /v1/audio/transcriptions 兼容接口:本地 whisper 转写。

    model: small/medium/large-v3;engine: whisper(默认)/funasr(快速中文);
    response_format: json(带 segments)/text(纯文本)
    """
    if not ENGINE_OK:
        raise HTTPException(500, f"转写引擎不可用:{ENGINE_ERR}")
    suffix = Path(file.filename or "audio").suffix or ".wav"
    tmp = RECORDINGS / f"openai_tmp_{datetime.now().strftime('%H%M%S_%f')}{suffix}"
    try:
        tmp.write_bytes(await file.read())
        if model not in ("small", "medium", "large-v3"):
            model = "small"
        from starlette.concurrency import run_in_threadpool
        if engine == "funasr":
            segs = await run_in_threadpool(eng_mod.transcribe_funasr_file, str(tmp))
        else:
            segs = await run_in_threadpool(eng_mod.retranscribe_file, str(tmp), model)
    finally:
        tmp.unlink(missing_ok=True)
    text = " ".join(s["text"].strip() for s in segs)
    if response_format == "text":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(text)
    return {"text": text, "segments": segs}


# ---------- 会议 ----------

@app.get("/api/meetings")
def api_meetings():
    rows = db.list_meetings()
    for r in rows:
        if r.get("summary"):
            try:
                r["summary"] = json.loads(r["summary"])
            except Exception:
                r["summary"] = None
        r["processing"] = state["processing"].get(r["id"])
    return rows


@app.get("/api/meetings/{mid}")
def api_meeting(mid: int):
    m = db.get_meeting(mid)
    if not m:
        raise HTTPException(404, "会议不存在")
    m["sources"] = db.get_sources(mid)
    m["actions"] = db.get_actions(mid)
    m["processing"] = state["processing"].get(mid)
    if m.get("summary"):
        try:
            m["summary"] = json.loads(m["summary"])
        except Exception:
            m["summary"] = None
    if m.get("marks"):
        try:
            m["marks"] = json.loads(m["marks"])
        except Exception:
            m["marks"] = None
    m["tags"] = [t for t in (m.get("tags") or "").split(",") if t]
    return m


@app.get("/api/meetings/{mid}/segments")
def api_segments(mid: int, source: str = None):
    return db.get_segments(mid, source)


class PatchMeeting(BaseModel):
    title: str = None
    tags: list = None


@app.patch("/api/meetings/{mid}")
def api_patch_meeting(mid: int, req: PatchMeeting):
    if not db.get_meeting(mid):
        raise HTTPException(404, "会议不存在")
    db.update_meeting(mid, req.title, req.tags)
    return {"ok": True}


@app.delete("/api/meetings/{mid}")
def api_delete_meeting(mid: int):
    db.delete_meeting(mid)
    state["processing"].pop(mid, None)
    return {"ok": True}


class SumReq(BaseModel):
    style: str = "balanced"


@app.post("/api/meetings/{mid}/summarize")
def api_summarize(mid: int, req: SumReq):
    m = db.get_meeting(mid)
    if not m:
        raise HTTPException(404, "会议不存在")
    segs = db.get_segments(mid, "small") or db.get_segments(mid)
    summary = summarizer.build(segs, m.get("duration") or 0, req.style, m["title"])
    db.set_meeting_summary(mid, summary)
    return summary


class RetransReq(BaseModel):
    model: str = "medium"   # medium / large-v3


@app.post("/api/meetings/{mid}/retranscribe")
def api_retranscribe(mid: int, req: RetransReq):
    m = db.get_meeting(mid)
    if not m:
        raise HTTPException(404, "会议不存在")
    if not m.get("wav_path") or not Path(m["wav_path"]).exists():
        raise HTTPException(400, "找不到录音文件,无法重转")

    def work():
        try:
            state["processing"][mid] = {"step": "retranscribe", "msg": f"{req.model} 重转中(首次需下载模型)"}
            segs = eng_mod.retranscribe_file(m["wav_path"], req.model)
            db.delete_segments_source(mid, req.model)
            db.add_segments(mid, segs, req.model)
            state["processing"][mid] = {"step": "done", "msg": f"{req.model} 重转完成"}
        except Exception as e:
            state["processing"][mid] = {"step": "error", "msg": f"重转失败:{e}"}

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "msg": f"{req.model} 重转已开始"}


@app.get("/api/meetings/{mid}/audio")
def api_audio(mid: int):
    m = db.get_meeting(mid)
    if not m or not m.get("wav_path") or not Path(m["wav_path"]).exists():
        raise HTTPException(404, "音频文件不存在")
    return FileResponse(m["wav_path"], filename=Path(m["wav_path"]).name,
                        media_type="audio/wav")


@app.get("/api/meetings/{mid}/export")
def api_export(mid: int, source: str = None):
    m = db.get_meeting(mid)
    if not m:
        raise HTTPException(404, "会议不存在")
    sources = db.get_sources(mid)
    src = source if source in sources else (sources[0] if sources else None)
    segs = db.get_segments(mid, src)

    def mmss(s):
        s = int(s)
        return f"{s // 60:02d}:{s % 60:02d}"

    lines = [f"# {m['title']}", ""]
    if m.get("summary"):
        try:
            sm = json.loads(m["summary"])
            lines += [f"> {sm.get('oneliner', '')}", ""]
            if sm.get("points"):
                lines += ["## 关键要点", *[f"- {p}" for p in sm["points"]], ""]
            if sm.get("decisions"):
                lines += ["## 决策事项", *[f"- {d}" for d in sm["decisions"]], ""]
        except Exception:
            pass
    acts = db.get_actions(mid)
    if acts:
        lines += ["## 行动项"]
        for a in acts:
            owner = a.get("owner") or ""
            mark = "x" if a["status"] == "done" else " "
            lines.append(f"- [ {mark} ] {a['text']}" + (f"({owner})" if owner else ""))
        lines.append("")
    lines += [f"## 转写全文(版本 {src})", ""]
    lines += [f"[{mmss(s['start'])}] {s.get('speaker') or ''} {s['text']}".rstrip() for s in segs]
    out = RECORDINGS / f"meeting_{mid}_{src or 'all'}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return FileResponse(str(out), filename=out.name, media_type="text/markdown")


# ---------- 行动项 ----------

class ActionReq(BaseModel):
    text: str
    owner: str = ""
    due: str = ""


@app.get("/api/actions")
def api_actions():
    return db.all_actions()


@app.post("/api/meetings/{mid}/actions")
def api_add_action(mid: int, req: ActionReq):
    if not db.get_meeting(mid):
        raise HTTPException(404, "会议不存在")
    aid = db.add_action(mid, req.text, req.owner, req.due)
    return {"id": aid}


class ActionPatch(BaseModel):
    status: str = None
    text: str = None
    owner: str = None
    due: str = None


@app.patch("/api/actions/{aid}")
def api_patch_action(aid: int, req: ActionPatch):
    a = db.update_action(aid, req.status, req.text, req.owner, req.due)
    if not a:
        raise HTTPException(404, "行动项不存在")
    return a


@app.delete("/api/actions/{aid}")
def api_delete_action(aid: int):
    db.delete_action(aid)
    return {"ok": True}


# ---------- 前端 ----------

app.mount("/", StaticFiles(directory=str(APP_DIR / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print(f"\n  会记 MeetNote · http://localhost:7861\n  数据目录: {DATA_DIR}\n")
    uvicorn.run(app, host="127.0.0.1", port=7861, log_level="warning")
