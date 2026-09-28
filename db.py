"""SQLite 数据层:会议纪要、段落、行动项的存储。

(基于原 whisper-handoff/scripts/db.py 扩展:
  - meetings 新增 marks(会中标记 JSON)、summary(纪要概览 JSON)、tags
  - 新增 actions 表:跨会议行动项
  - 数据目录默认为项目内 data/,可用环境变量 MEETNOTE_DATA_DIR 指向旧目录 ~/whisper-app)

表:
  meetings(id, title, created_at, duration, status, wav_path, num_speakers,
           marks, summary, tags)         status ∈ {processing, done}
  segments(id, meeting_id, start, end, speaker, text, source)
  actions(id, meeting_id, text, owner, due, status, created_at)
"""
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("MEETNOTE_DATA_DIR", str(APP_DIR / "data")))
DB_PATH = DATA_DIR / "meetings.db"


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS meetings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TEXT,
            duration REAL,
            status TEXT,
            wav_path TEXT,
            num_speakers INTEGER
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS segments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id INTEGER,
            start REAL,
            end REAL,
            speaker TEXT,
            text TEXT,
            source TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS actions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id INTEGER,
            text TEXT,
            owner TEXT,
            due TEXT,
            status TEXT,
            created_at TEXT
        )"""
    )
    # migration:旧库升级(missing column 时补上)
    for col, ddl in [
        ("marks", "ALTER TABLE meetings ADD COLUMN marks TEXT"),
        ("summary", "ALTER TABLE meetings ADD COLUMN summary TEXT"),
        ("tags", "ALTER TABLE meetings ADD COLUMN tags TEXT"),
    ]:
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass  # 列已存在
    conn.execute("CREATE INDEX IF NOT EXISTS idx_seg_meeting ON segments(meeting_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_act_meeting ON actions(meeting_id)")
    conn.commit()
    conn.close()


# ---------- 会议 ----------

def create_meeting(title, wav_path, duration, num_speakers, status="processing", tags=None):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO meetings(title, created_at, duration, status, wav_path, num_speakers, tags) "
        "VALUES(?,?,?,?,?,?,?)",
        (title, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), duration, status,
         wav_path, num_speakers, ",".join(tags) if tags else ""),
    )
    mid = cur.lastrowid
    conn.commit()
    conn.close()
    return mid


def add_segments(meeting_id, segments, source):
    """segments: [{'start','end','speaker','text'}] 或 [(start,end,text)]。"""
    conn = get_conn()
    for seg in segments:
        if isinstance(seg, dict):
            conn.execute(
                "INSERT INTO segments(meeting_id,start,end,speaker,text,source) VALUES(?,?,?,?,?,?)",
                (meeting_id, seg["start"], seg["end"], seg.get("speaker", ""), seg["text"], source),
            )
        else:
            start, end, text = seg
            conn.execute(
                "INSERT INTO segments(meeting_id,start,end,speaker,text,source) VALUES(?,?,?,?,?,?)",
                (meeting_id, start, end, "", text, source),
            )
    conn.commit()
    conn.close()


def list_meetings():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, title, created_at, duration, status, num_speakers, tags, summary "
        "FROM meetings ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_meeting(meeting_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM meetings WHERE id=?", (meeting_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def set_meeting_status(meeting_id, status):
    conn = get_conn()
    conn.execute("UPDATE meetings SET status=? WHERE id=?", (status, meeting_id))
    conn.commit()
    conn.close()


def update_meeting_wav(meeting_id, wav_path):
    conn = get_conn()
    conn.execute("UPDATE meetings SET wav_path=? WHERE id=?", (wav_path, meeting_id))
    conn.commit()
    conn.close()


def set_meeting_duration(meeting_id, duration):
    conn = get_conn()
    conn.execute("UPDATE meetings SET duration=? WHERE id=?", (duration, meeting_id))
    conn.commit()
    conn.close()


def set_meeting_summary(meeting_id, summary: dict):
    conn = get_conn()
    conn.execute("UPDATE meetings SET summary=? WHERE id=?",
                 (json.dumps(summary, ensure_ascii=False), meeting_id))
    conn.commit()
    conn.close()


def set_meeting_marks(meeting_id, marks: dict):
    conn = get_conn()
    conn.execute("UPDATE meetings SET marks=? WHERE id=?",
                 (json.dumps(marks, ensure_ascii=False), meeting_id))
    conn.commit()
    conn.close()


def update_meeting(id_, title=None, tags=None):
    conn = get_conn()
    if title is not None:
        conn.execute("UPDATE meetings SET title=? WHERE id=?", (title, id_))
    if tags is not None:
        conn.execute("UPDATE meetings SET tags=? WHERE id=?", (",".join(tags), id_))
    conn.commit()
    conn.close()


def get_segments(meeting_id, source=None):
    conn = get_conn()
    if source:
        rows = conn.execute(
            "SELECT start, end, speaker, text FROM segments WHERE meeting_id=? AND source=? ORDER BY start",
            (meeting_id, source),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT start, end, speaker, text FROM segments WHERE meeting_id=? ORDER BY start",
            (meeting_id,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_sources(meeting_id):
    """返回该会议已有哪些转写版本(如 ['small','medium'])。"""
    conn = get_conn()
    rows = conn.execute(
        "SELECT DISTINCT source FROM segments WHERE meeting_id=?", (meeting_id,)
    ).fetchall()
    conn.close()
    return [r["source"] for r in rows]


def delete_segments_source(meeting_id, source):
    conn = get_conn()
    conn.execute("DELETE FROM segments WHERE meeting_id=? AND source=?", (meeting_id, source))
    conn.commit()
    conn.close()


def delete_meeting(meeting_id):
    conn = get_conn()
    conn.execute("DELETE FROM segments WHERE meeting_id=?", (meeting_id,))
    conn.execute("DELETE FROM actions WHERE meeting_id=?", (meeting_id,))
    conn.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))
    conn.commit()
    conn.close()


# ---------- 行动项 ----------

def add_action(meeting_id, text, owner="", due="", status="todo"):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO actions(meeting_id, text, owner, due, status, created_at) VALUES(?,?,?,?,?,?)",
        (meeting_id, text, owner, due, status, datetime.now().strftime("%Y-%m-%d %H:%M")),
    )
    aid = cur.lastrowid
    conn.commit()
    conn.close()
    return aid


def get_actions(meeting_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, meeting_id, text, owner, due, status, created_at FROM actions "
        "WHERE meeting_id=? ORDER BY id", (meeting_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def all_actions():
    conn = get_conn()
    rows = conn.execute(
        "SELECT a.id, a.meeting_id, a.text, a.owner, a.due, a.status, a.created_at, m.title "
        "FROM actions a LEFT JOIN meetings m ON a.meeting_id = m.id ORDER BY a.id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_action(action_id, status=None, text=None, owner=None, due=None):
    conn = get_conn()
    row = conn.execute("SELECT * FROM actions WHERE id=?", (action_id,)).fetchone()
    if not row:
        conn.close()
        return None
    row = dict(row)
    conn.execute(
        "UPDATE actions SET text=?, owner=?, due=?, status=? WHERE id=?",
        (text if text is not None else row["text"],
         owner if owner is not None else row["owner"],
         due if due is not None else row["due"],
         status if status is not None else row["status"],
         action_id),
    )
    conn.commit()
    conn.close()
    out = dict(row)
    out.update({"text": text or row["text"], "owner": owner or row["owner"],
                "due": due or row["due"], "status": status or row["status"]})
    return out


def delete_action(action_id):
    conn = get_conn()
    conn.execute("DELETE FROM actions WHERE id=?", (action_id,))
    conn.commit()
    conn.close()
