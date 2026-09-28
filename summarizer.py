"""纪要概览生成器。

两种模式:
  1. 规则版(默认,零依赖离线):从转写段落中提取要点 / 疑似决策句 / 疑似行动项 / 说话人统计。
  2. LLM 版(可选):设置环境变量后自动启用,效果更好。
     MEETNOTE_LLM_BASE_URL   OpenAI 兼容接口地址,如 https://api.xxx.com/v1
     MEETNOTE_LLM_API_KEY    API Key
     MEETNOTE_LLM_MODEL      模型名,如 gpt-4o-mini / deepseek-chat
"""
import json
import os
import re
from collections import Counter

DECISION_PAT = re.compile(
    r"(决定|确定了|确定了?要|拍板|一致同意|就这么定|敲定|确认采用|选定|通过了|达成一致|定下来)")
ACTION_PAT = re.compile(
    r"(我来|你来|他来|由[他她你我一 \u4e00-\u9fa5]{0,6}负责|需要[^\s。！？]{0,12}(跟进|确认|准备|输出|提交|整理|对接|排期)"
    r"|安排[^\s。！？]{0,10}(负责|跟进|去做)"
    r"|(下周|本周|下月|月底|周[一二三四五六日天]|今天|明天|后天|这周)[^\s。！？]{0,12}(前|之前|完成|给|发给|提交|发出来?))")
TIME_PAT = re.compile(
    r"((下|本|这)?(周[一二三四五六日天]?|月[初中底]?|今天|明天|后天|年底)[^\s。！？]{0,8}?(前|之前)?|(下|本|这)周|月底|年底)")
KEYWORDS = ("进度|数据|结论|问题|风险|方案|需求|用户|版本|上线|目标|计划|预算|时间|排期|指标|结果|功能|产品|客户|团队")


def _score_sentence(text):
    """给句子打"信息量"分,用于挑要点。"""
    n = len(text)
    score = 0.0
    if 12 <= n <= 80:
        score += 2
    elif 8 <= n < 12:
        score += 1
    if re.search(r"\d", text):
        score += 1.5
    hits = len(re.findall(KEYWORDS, text))
    score += min(hits * 0.8, 2.4)
    if DECISION_PAT.search(text) or ACTION_PAT.search(text):
        score += 1.2
    return score


def _top_words(texts, topn=6):
    """简单 bigram 词频,找讨论主题。"""
    cnt = Counter()
    stop = {"我们", "这个", "那个", "然后", "就是", "还是", "可以", "因为", "所以", "但是",
            "大家", "现在", "一下", "可能", "你们", "他们", "没有", "什么", "怎么", "如果",
            "以及", "接着", "其中", "另外", "关于", "对于", "通过", "进行", "开始", "结束",
            "一些", "很多", "非常", "特别", "一下", "时候", "时候", "问题", "一个", "这样"}
    for t in texts:
        clean = re.sub(r"[^\u4e00-\u9fa5]", "", t)
        for i in range(len(clean) - 1):
            bg = clean[i:i + 2]
            if bg not in stop:
                cnt[bg] += 1
    picked = []
    for w, _ in cnt.most_common(topn * 4):
        if any(w[0] in p or w[1] in p for p in picked):
            continue  # 跳过与已选词重叠的碎片
        picked.append(w)
        if len(picked) >= topn:
            break
    return picked


def extract(segments, duration=0, style="balanced", title=""):
    """规则版提取。segments: [{'start','end','speaker','text'}]"""
    texts = [s["text"] for s in segments]
    speakers = {}
    for s in segments:
        name = s.get("speaker") or "未知"
        d = speakers.setdefault(name, {"segs": 0, "secs": 0.0})
        d["segs"] += 1
        d["secs"] += (s["end"] - s["start"])

    decisions, actions, scored = [], [], []
    for s in segments:
        text = s["text"].strip()
        if not text:
            continue
        spk = s.get("speaker") or ""
        if DECISION_PAT.search(text) and len(text) >= 6:
            decisions.append(text)
        if ACTION_PAT.search(text) and len(text) >= 6:
            m = TIME_PAT.search(text)
            actions.append({
                "text": text,
                "owner": spk,
                "due": m.group(0) if m else "",
            })
        scored.append((_score_sentence(text), text))

    scored.sort(key=lambda x: -x[0])
    topn = 3 if style == "brief" else 5
    seen, points = set(), []
    for _, t in scored:
        key = t[:12]
        if key in seen:
            continue
        seen.add(key)
        points.append(t)
        if len(points) >= topn:
            break

    words = _top_words(texts)
    topic = "、".join(words[:3]) if words else "多主题讨论"
    n_spk = len(speakers)
    mm = int(duration // 60) or 1
    oneliner = (f"{n_spk} 位说话人参与、时长约 {mm} 分钟的会议，围绕「{topic}」展开，"
                f"识别出 {len(points)} 条要点、{len(decisions)} 项决策、{len(actions)} 项待办。")

    return {
        "method": "rule",
        "oneliner": oneliner,
        "points": points,
        "decisions": decisions[:5],
        "actions": actions[:8],
        "speakers": [{"name": k, "segs": v["segs"], "secs": round(v["secs"], 1)}
                     for k, v in sorted(speakers.items(), key=lambda x: -x[1]["secs"])],
    }


def _llm_enabled():
    return bool(os.environ.get("MEETNOTE_LLM_BASE_URL")
                and os.environ.get("MEETNOTE_LLM_API_KEY")
                and os.environ.get("MEETNOTE_LLM_MODEL"))


def extract_llm(segments, duration=0, style="balanced", title=""):
    """LLM 版提取(OpenAI 兼容接口)。失败时回退规则版。"""
    import requests
    base = os.environ["MEETNOTE_LLM_BASE_URL"].rstrip("/")
    transcript = "\n".join(
        f"[{int(s['start'])//60:02d}:{int(s['start'])%60:02d}] {s.get('speaker','')}: {s['text']}"
        for s in segments)
    detail = {"brief": "只要最核心的结论", "balanced": "要点+决策+待办",
              "detailed": "包含讨论过程"}[style]
    sys_prompt = (
        "你是专业的会议纪要助手。根据转写文本输出 JSON(不要 markdown 代码块),字段:\n"
        '{"oneliner":"一句话总结","points":["关键要点"],"decisions":["决策事项"],'
        '"actions":[{"text":"任务","owner":"负责人","due":"截止时间"}],\n'
        '"speakers":[{"name":"说话人","segs":段数,"secs":发言秒数}]}\n'
        f"风格:{detail}。若某类内容没有,给空数组。")
    try:
        r = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['MEETNOTE_LLM_API_KEY']}"},
            json={"model": os.environ["MEETNOTE_LLM_MODEL"],
                  "temperature": 0.2,
                  "messages": [{"role": "system", "content": sys_prompt},
                               {"role": "user", "content": f"会议标题:{title}\n时长:{int(duration)}秒\n\n{transcript}"}]},
            timeout=120,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```(json)?|```$", "", content, flags=re.M).strip()
        data = json.loads(content)
        data["method"] = "llm"
        data.setdefault("oneliner", "")
        data.setdefault("points", [])
        data.setdefault("decisions", [])
        data.setdefault("actions", [])
        data.setdefault("speakers", [])
        return data
    except Exception as e:
        print("LLM 摘要失败,回退规则版:", e)
        return extract(segments, duration, style, title)


def build(segments, duration=0, style="balanced", title=""):
    """入口:优先 LLM(已配置时),否则规则版。"""
    if _llm_enabled():
        return extract_llm(segments, duration, style, title)
    return extract(segments, duration, style, title)
