"""
ROCLING 2026 — 南島語言數詞進位制分析 pipeline（多 provider 版）
用法：
  python pipeline.py --model llama-3.3-70b-versatile           # 跑全部（接續上次進度）
  python pipeline.py --model llama-3.1-8b-instant --condition oneshot_noname
  python pipeline.py --dry-run --condition oneshot_name        # 只看 prompt，不呼叫 API
  python pipeline.py --model gemini-2.5-flash --summary        # 看既有結果統計
  python pipeline.py --probe groq                              # 列出該 provider 實際可用模型
  python pipeline.py --list                                    # 列出模型與條件

核心陣容（全免費、免綁卡）：
  Groq      llama-3.1-8b-instant (8B, 14400 RPD) / llama-3.3-70b-versatile (70B, 1000 RPD)
  Cerebras  qwen-3-32b (32B, 1M tokens/day)
  Gemini    gemini-2.0-flash (1500 RPD) / gemini-2.5-flash（閉源對照）
規模軸：8B → 32B → 70B（開源、參數量明確）＋ Gemini（商用閉源錨點）

API keys（依 provider 讀環境變數）：
  GROQ_API_KEY / CEREBRAS_API_KEY / GEMINI_API_KEY
  （ANTHROPIC_API_KEY / OPENAI_API_KEY 保留支援，未列入核心陣容）

條件命名：{zeroshot|oneshot}_{name|noname}[_search]
  zeroshot_noname   核心：純分析力基線（全 892 語言）
  oneshot_noname    核心：提示詞效果（全 892 語言）
  zeroshot_name     擴充：名稱效果（200 語言子集）
  zeroshot_noname_search  擴充：搜尋效果（200 語言子集，僅 Gemini 支援）
  zeroshot_name_search    選做：交互效果（200 語言子集，僅 Gemini 支援）
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path

# Windows 預設 cp950，遇非 ASCII 會炸。重綁為 UTF-8——但必須冪等：
# 匯入本模組的腳本若自己也包過一層，重複包裝會在舊 wrapper 被回收時關掉底層 buffer。
def _force_utf8(stream_name: str):
    stream = getattr(sys, stream_name)
    if getattr(stream, "encoding", "").lower().replace("-", "") == "utf8":
        return
    setattr(sys, stream_name, io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace"))


_force_utf8("stdout")
_force_utf8("stderr")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# gold 來源：Google 試算表匯出快照（repo 內建，可重現）。
# 2026-07-22 起改以試算表的『Gold 1-10』欄為唯一權威，不再由布林欄推導。
TSV_PATH = Path(__file__).parent / "data" / "gold_sheet.tsv"
OUTPUT_DIR = Path(__file__).parent / "results"
DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_CONDITION = "zeroshot_noname"

# 額度／TPM／token 用量皆為 2026-07-14 實測（讀 rate-limit header ＋ 真實 prompt 量測）。
# Groq 的 RPD 是「每模型」獨立計算 → 多模型可同日平行跑，這是排程的關鍵。
# reasoning: thinking=off 時傳給該模型的 reasoning_effort 值（None=該模型無此參數）。
#   qwen3 系列支援 "none"；gpt-oss 系列最低只到 "low"。不設的話模型預設會 thinking，
#   吃滿 2048 output token 被截斷、答案根本出不來（2026-07-14 事故）。
# delay: 依實測 token 用量算出的 TPM 安全間隔（見 scratchpad/measure_tokens.py）。
MODELS = {
    # 核心規模軸（全 Groq，RPD 各自獨立 → 同日平行）
    "llama-3.1-8b-instant":    {"provider": "groq", "delay": 3.6, "params": "8B",       "rpd": 14400, "reasoning": None},
    "openai/gpt-oss-20b":      {"provider": "groq", "delay": 4.0, "params": "20B-MoE",  "rpd": 1000,  "reasoning": "low"},
    "qwen/qwen3.6-27b":        {"provider": "groq", "delay": 2.5, "params": "27B",      "rpd": 1000,  "reasoning": "none"},
    "qwen/qwen3-32b":          {"provider": "groq", "delay": 2.5, "params": "32B",      "rpd": 1000,  "reasoning": "none"},
    "llama-3.3-70b-versatile": {"provider": "groq", "delay": 2.5, "params": "70B",      "rpd": 1000,  "reasoning": None},
    "openai/gpt-oss-120b":     {"provider": "groq", "delay": 3.0, "params": "120B-MoE", "rpd": 1000,  "reasoning": "low"},
    # === Cerebras 主力（1M tokens/day，是 Groq 70B 的 10 倍）===
    # Groq 的 TPD 綁死大模型（70B 只有 100K/day＝320 語言/天），Cerebras 沒這問題。
    # RPM 較嚴（實測 ~30 RPM，1s 間隔第 6 筆就 429）→ delay 抓 2.5s。
    "gemma-4-31b":             {"provider": "cerebras", "delay": 2.5, "params": "31B",  "rpd": None,  "reasoning": None},
    "gpt-oss-120b":            {"provider": "cerebras", "delay": 3.0, "params": "120B-MoE", "rpd": None, "reasoning": "low"},
    # 閉源錨點
    "gemini-3.1-flash-lite":   {"provider": "gemini", "delay": 5.0, "params": "?",      "rpd": None,  "reasoning": None},
    # 備用
    "meta-llama/llama-4-scout-17b-16e-instruct":
                               {"provider": "groq", "delay": 2.5, "params": "17Bx16E",  "rpd": 1000,  "reasoning": None},
    "zai-glm-4.7":             {"provider": "cerebras", "delay": 2.5, "params": "?",    "rpd": None,  "reasoning": None},
    # === SambaNova（免費 tier，OpenAI 相容）===
    "Meta-Llama-3.3-70B-Instruct":
                               {"provider": "sambanova", "delay": 6.0, "params": "70B",  "rpd": None,  "reasoning": None},
    # 免費額度過低不適合全量（實測 RPD=20），僅供小樣本對照
    "gemini-3.5-flash":        {"provider": "gemini", "delay": 5.0, "params": "?",      "rpd": 20,    "reasoning": None},
    # 已無免費額度（實測 limit=0 或 404），僅供讀舊結果，不可再呼叫
    "gemini-2.0-flash":        {"provider": "gemini", "delay": 5.0, "params": "?",      "rpd": 0,     "reasoning": None},
    "gemini-2.5-flash":        {"provider": "gemini", "delay": 7.0, "params": "?",      "rpd": 0,     "reasoning": None},
    "gemini-2.5-pro":          {"provider": "gemini", "delay": 13.0, "params": "?",     "rpd": 0,     "reasoning": None},
    # 付費備用（未列入核心陣容）
    "claude-sonnet-5":  {"provider": "anthropic", "delay": 1.0, "params": "?", "rpd": None, "reasoning": None},
    "claude-haiku-4-5": {"provider": "anthropic", "delay": 1.0, "params": "?", "rpd": None, "reasoning": None},
    "gpt-4o":           {"provider": "openai", "delay": 0.5, "params": "?", "rpd": None, "reasoning": None},
    "gpt-4o-mini":      {"provider": "openai", "delay": 0.5, "params": "?", "rpd": None, "reasoning": None},
}

PROVIDER_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
}

OPENAI_COMPAT_BASE_URL = {
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "sambanova": "https://api.sambanova.ai/v1",
}

# 只有 Gemini 有原生搜尋（google_search grounding）；其他 provider 的 search 條件直接擋下
SEARCH_PROVIDERS = {"gemini", "anthropic", "openai"}

MAX_RETRIES = 6


class DailyQuotaExhausted(RuntimeError):
    """當日 RPD 額度歸零——重試無意義，應中止整個 run 而非狂等。"""

# thinking 模式（--thinking off|default）
#   off:     claude-sonnet-5 → thinking disabled；gemini-2.5-flash → thinkingBudget 0
#            gemini-2.5-pro 無法全關（API 限制），保留預設並警告
#   default: 各模型 API 預設（sonnet-5 adaptive on、gemini-2.5 dynamic thinking on）
# temperature=0 盡量套用；claude-sonnet-5 不接受非預設 temperature（API 400），只能省略——論文需註明。

# ---------------------------------------------------------------------------
# Prompt construction（與 prompts.md 同步）
# ---------------------------------------------------------------------------

INSTRUCTIONS = """\
Given the following numeral forms (1–10) of a language, analyze the morphological structure of each numeral and determine its formation pattern.

For each numeral, write its decomposition formula:
- If the numeral is a basic/atomic word (not derived from other numerals), write the number itself (e.g., "1", "5")
- If the numeral is derived from other numerals, write the formula using +, -, × (e.g., "5+1", "2×3", "10-1")
"""

# 中性範例版（ablation）：符號示範全部用 1-10 範圍外的數字，不給任何候選答案，
# 排除「範例 priming 導致弱模型套公式」的解釋。
INSTRUCTIONS_NEUTRAL = """\
Given the following numeral forms (1–10) of a language, analyze the morphological structure of each numeral and determine its formation pattern.

For each numeral, write its decomposition formula:
- If the numeral is a basic/atomic word (not derived from other numerals), write the number itself (for instance, a language whose word for eleven is a single unanalyzable word would get "11")
- If the numeral is derived from other numerals, write the formula using +, -, × (for instance, a word for twelve built from 'ten' plus 'two' would be "10+2"; a word for twenty built from 'two' times 'ten' would be "2×10"; a word for nineteen meaning 'twenty minus one' would be "20-1")
"""

PROMPT_VARIANTS = {"standard": INSTRUCTIONS, "neutral": INSTRUCTIONS_NEUTRAL}

ONESHOT_EXAMPLE = """\
Example:
Input: 1: kiŋal, 2: daha, 3: teru, 4: sepac, 5: rima, 6: mumuteru, 7: mupitu, 8: mumusepac, 9: muŋari, 10: maxal
Output: 1, 2, 3, 4, 5, 2×3, 7, 2×4, 9, 10
(6 "mumuteru" contains mu-mu-teru where teru=3, suggesting 2×3; 8 "mumusepac" contains mu-mu-sepac where sepac=4, suggesting 2×4)
"""

SEARCH_PREAMBLE = "You may search the web for information about this language if needed.\n\n"

FOOTER = "Reply with ONLY the 10 formulas, comma-separated, in order from 1 to 10."


def parse_condition(cond: str):
    """'oneshot_name_search' → (shot, give_name, search)。格式錯誤直接 exit。"""
    m = re.fullmatch(r"(zeroshot|oneshot)_(name|noname)(_search)?", cond)
    if not m:
        print(f"ERROR: invalid condition '{cond}' (expected {{zeroshot|oneshot}}_{{name|noname}}[_search])")
        sys.exit(1)
    return m.group(1), m.group(2) == "name", m.group(3) is not None


def make_prompt(lang: dict, condition: str, variant: str = "standard") -> str:
    shot, give_name, search = parse_condition(condition)
    words = lang["words"]
    numeral_lines = "\n".join(f"{i + 1}: {w}" for i, w in enumerate(words))
    name_line = f"Language: {lang['name']}\n" if give_name else ""

    parts = []
    if search:
        parts.append(SEARCH_PREAMBLE)
    parts.append(PROMPT_VARIANTS[variant])
    parts.append("\n")
    if shot == "oneshot":
        parts.append(ONESHOT_EXAMPLE)
        parts.append("\nNow analyze this language:\n")
        parts.append(name_line)
        parts.append(numeral_lines)
    else:
        parts.append(name_line)
        parts.append("Numerals:\n")
        parts.append(numeral_lines)
    parts.append("\n\n")
    parts.append(FOOTER)
    return "".join(parts)


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------

COL_ID = 0
COL_LANG = 1
COL_WORDS_START = 6   # columns 6-15 = words for 1-10
COL_GOLD = 16         # 『Gold 1-10』欄：逗號分隔的 10 個公式，gold 的唯一權威
COL_SPECIAL = 17      # 『特殊』自由文字欄——標註者記錄難以形式化的觀察與存疑
#   ⚠️ 主評分只讀 COL_GOLD。col17 是質性補充（含 '?' 表示標註者自己也不確定），
#      FP 分析務必納入，否則會把標註者已記在自由文字的案例誤計為「人工遺漏」（2026-07-16 事故）。

# 『不確定』標記欄（標註者對該數詞的分析沒把握）。完全排除不計分。
NOT_SURE_COLS = {
    28: 4, 29: 5, 30: 6, 31: 7, 32: 8, 33: 9, 34: 10,
}

# gold 欄裡代表「無語料」的記號。該格完全不計分（不進分子也不進分母）：
# 詞形空白時 prompt 送出的是空行，模型沒有輸入可分析，計分無意義（2026-07-22 決議）。
GOLD_MISSING = "-"

# 非單純詞但運算關係不透明的記號，如 '8&2'（9 的詞形含 8 與 2 的成分，但關係不明）。
# 視同 not_sure：計入 gold 的 complex 類，但標註者自己都無法定調，不因猜錯而苛責。
GOLD_OPAQUE = "&"

# 遞迴展開格的最短子串比對門檻（避免單音節假匹配）。
_MIN_SUBSTR_LEN = 3


def _find_recursive_cells(data):
    """偵測需要遞迴展開才能對齊 gold 的格子。

    邏輯：數詞 N 的詞形包含另一複雜數詞 M 的詞形（子串匹配），
    但 gold[N] 中 M 的編號不存在於操作元（已被展開）。
    回傳 dict: lang_id → set of numeral positions (1-indexed)。
    """
    result = {}
    for lang in data:
        words = lang["words"]
        gold = lang["gold"]
        recursive_pos = set()

        for n_idx in range(10):
            n_gold = gold[n_idx]
            if n_gold == str(n_idx + 1) or n_gold == GOLD_MISSING:
                continue
            n_word = words[n_idx].lower().strip()
            if not n_word or len(n_word) < _MIN_SUBSTR_LEN:
                continue

            # 取 gold[N] 的操作元（數字）
            n_operands = set(re.split(r'[+\-×*]', n_gold))

            for m_idx in range(10):
                if m_idx == n_idx:
                    continue
                m_gold = gold[m_idx]
                if m_gold == str(m_idx + 1) or m_gold == GOLD_MISSING:
                    continue  # M 必須是複雜詞
                m_word = words[m_idx].lower().strip()
                if not m_word or len(m_word) < _MIN_SUBSTR_LEN:
                    continue
                # M 的詞形出現在 N 的詞形中（表層分解用到 M）
                if m_word in n_word:
                    # 但 gold[N] 已把 M 展開（M 的編號不在操作元中）
                    if str(m_idx + 1) not in n_operands:
                        recursive_pos.add(n_idx + 1)
                        break

        if recursive_pos:
            result[lang["id"]] = recursive_pos
    return result


def load_tsv():
    with open(TSV_PATH, encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    data = []
    for row in rows[2:]:
        if not row[COL_ID].strip().isdigit():
            continue
        lang_id = int(row[COL_ID].strip())
        lang_name = row[COL_LANG].strip()
        words = [row[i].strip() if i < len(row) else "" for i in range(COL_WORDS_START, COL_WORDS_START + 10)]
        if not any(words):
            continue

        gold, excluded = build_gold(row)
        not_sure = build_not_sure(row)
        # 運算不透明的格子自動視為 not_sure
        not_sure |= {i + 1 for i, g in enumerate(gold) if GOLD_OPAQUE in g}

        data.append({
            "id": lang_id,
            "name": lang_name,
            "words": words,
            "gold": gold,
            "not_sure": not_sure,
            "excluded": excluded,
        })

    # 遞迴展開格：提示詞未示範此行為，模型缺乏判斷依據，不計分。
    recursive_map = _find_recursive_cells(data)
    for lang in data:
        extra = recursive_map.get(lang["id"], set())
        if extra:
            lang["excluded"] |= extra

    return data


def build_gold(row):
    """讀『Gold 1-10』欄。回傳 (gold, excluded)。

    excluded = 無語料而不計分的數詞位置（1-indexed）。
    """
    raw = row[COL_GOLD].strip() if COL_GOLD < len(row) else ""
    if not raw:
        # 沒填 gold：退回全單純詞，但無語料的位置仍排除
        gold = [str(i) for i in range(1, 11)]
    else:
        gold = [x.strip() for x in raw.split(",")]
        if len(gold) != 10:
            raise ValueError(f"gold 欄需 10 個欄位，ID={row[COL_ID]} 得到 {len(gold)}：{raw!r}")

    excluded = set()
    for i in range(10):
        word = row[COL_WORDS_START + i].strip() if COL_WORDS_START + i < len(row) else ""
        if gold[i] == GOLD_MISSING or not word:
            excluded.add(i + 1)
            gold[i] = GOLD_MISSING
    return gold, excluded


def build_not_sure(row):
    """Return set of numeral positions (1-indexed) flagged as 'not sure'."""
    ns = set()
    for col, numeral in NOT_SURE_COLS.items():
        if col < len(row) and row[col].strip() == "1":
            ns.add(numeral)
    return ns


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def normalize_formula(s: str) -> str:
    """公式樣字串做運算子正規化；詞形（echo）原樣保留，只去空白。"""
    s = s.strip()
    if re.fullmatch(r"[\d\s+\-*xX×()=]+", s):
        s = s.replace("*", "×").replace("x", "×").replace("X", "×")
        s = re.sub(r"\s+", "", s)
    return s


def _commutative_equal(a: str, b: str) -> bool:
    """交換律等價比對：加法和乘法的操作元順序不影響正確性。"""
    if a == b:
        return True
    # 純加法公式：排序操作元後比對
    if "+" in a and "-" not in a and "×" not in a:
        if "+" in b and "-" not in b and "×" not in b:
            return sorted(a.split("+")) == sorted(b.split("+"))
    # 純乘法公式：排序操作元後比對
    if "×" in a and "+" not in a and "-" not in a:
        if "×" in b and "+" not in b and "-" not in b:
            return sorted(a.split("×")) == sorted(b.split("×"))
    return False


def _legacy_norm(s: str) -> str:
    """舊版 normalize（無條件替換）——比對 2026-07-14 前存檔的污染資料用。"""
    s = s.strip().replace("*", "×").replace("x", "×").replace("X", "×")
    return re.sub(r"\s+", "", s)


def is_echo(ai: str, word: str) -> bool:
    """AI 輸出是否只是把詞形抄回來（含 '/' 多寫法拆開比、新舊 normalize 相容）。"""
    if not word:
        return False
    ai_forms = {ai.strip().lower(), _legacy_norm(ai).lower()}
    for alt in re.split(r"[/,;]", word):
        alt = alt.strip()
        if not alt:
            continue
        if {alt.lower(), _legacy_norm(alt).lower()} & ai_forms:
            return True
    return False


def parse_response(text: str) -> list[str] | None:
    """Parse formulas from API response. Returns list of 10 or None.

    容納格式（依序嘗試）：
      1. 單行 10 個逗號分隔（指令要求的格式；可帶 "Output:" 前綴）
      2. 逐行編號 "N: formula" 或 "N. formula"（小模型常見的不服從格式）
      3. 全文逗號切 10 段（fallback）
    """
    # Qwen3 等 reasoning 模型會輸出 <think>...</think>，先剝掉；markdown code fence 也剝
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```\w*\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    # 1. 單行逗號分隔
    for line in lines:
        line = re.sub(r"^(output|answer)\s*[:：]\s*", "", line, flags=re.IGNORECASE)
        parts = [normalize_formula(p) for p in line.split(",")]
        if len(parts) == 10 and all(parts):
            return parts

    # 2. 逐行編號格式
    numbered = {}
    for line in lines:
        m = re.match(r"^(\d{1,2})\s*[:.：]\s*(.+)$", line)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= 10 and idx not in numbered:
                numbered[idx] = normalize_formula(m.group(2))
    if len(numbered) == 10 and all(numbered.values()):
        return [numbered[i] for i in range(1, 11)]

    # 3. 全文逗號切分
    parts = [normalize_formula(p) for p in text.split(",")]
    if len(parts) == 10 and all(parts):
        return parts
    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(ai_output: list[str], gold: list[str], not_sure: set[int], words: list[str] | None = None,
             excluded: set[int] | None = None) -> dict:
    """逐格比對。excluded（無語料）和 not_sure（標註者不確定）完全不計分。"""
    excluded = excluded or set()
    skip = excluded | not_sure
    results = []
    correct = 0
    scored = 0
    for i in range(10):
        numeral = i + 1
        if numeral in skip:
            reason = "Excluded" if numeral in excluded else "NotSure"
            results.append({
                "numeral": numeral,
                "ai": normalize_formula(ai_output[i]),
                "gold": gold[i] if numeral not in excluded else GOLD_MISSING,
                "correct": None,
                "verdict": reason,
                "not_sure": numeral in not_sure,
            })
            continue

        scored += 1
        ai = normalize_formula(ai_output[i])
        expected = normalize_formula(gold[i])
        is_correct = (ai == expected) or _commutative_equal(ai, expected)

        if is_correct:
            verdict = "TN" if expected == str(numeral) else "TP"
            correct += 1
        elif words and is_echo(ai, words[i]):
            verdict = "Echo"
        else:
            if ai == str(numeral) and expected != str(numeral):
                verdict = "FN"
            elif ai != str(numeral) and expected == str(numeral):
                verdict = "FP"
            else:
                verdict = "Wrong"

        results.append({
            "numeral": numeral,
            "ai": ai,
            "gold": expected,
            "correct": is_correct,
            "verdict": verdict,
            "not_sure": False,
        })

    return {"score": correct, "total": scored, "details": results}


# ---------------------------------------------------------------------------
# API calling — 三個 provider
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, model: str, api_key: str, search: bool, thinking: str) -> str:
    """REST 直呼（legacy SDK 已棄用；REST 才有 google_search 工具）。"""
    import requests

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8192},
    }
    if thinking == "off":
        if model == "gemini-2.5-flash":
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
        # gemini-2.5-pro 無法關 thinking（API 最低 budget 128），保留預設
    if search:
        body["tools"] = [{"google_search": {}}]

    for attempt in range(MAX_RETRIES):
        resp = requests.post(url, json=body, timeout=120)
        if resp.status_code == 429:
            err = resp.text
            if "PerDay" in err or "RequestsPerDay" in err:
                m = re.search(r'"quotaValue":\s*"(\d+)"', err)
                cap = m.group(1) if m else "?"
                raise DailyQuotaExhausted(
                    f"{model} 當日免費額度已用完（RPD limit={cap}）。重試無用，需等隔天重置（太平洋時間午夜）或換模型。"
                )
            m = re.search(r"retry.{0,10}?(\d+\.?\d*)\s*s", err, re.IGNORECASE)
            wait = float(m.group(1)) + 5 if m else 65
            print(f"429, wait {wait:.0f}s...", end=" ", flush=True)
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            time.sleep(10 * (attempt + 1))
            continue
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates")
        if not candidates:
            raise RuntimeError(f"no candidates: {json.dumps(data)[:200]}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def call_anthropic(prompt: str, model: str, api_key: str, search: bool, thinking: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    kwargs = {}
    if model.startswith("claude-sonnet-5"):
        # sonnet-5 不接受非預設 temperature（400）；thinking 預設 adaptive on，off 時明確關閉
        if thinking == "off":
            kwargs["thinking"] = {"type": "disabled"}
        kwargs["max_tokens"] = 8192  # 留 thinking 空間（default 模式）
    else:
        kwargs["temperature"] = 0
        kwargs["max_tokens"] = 1024
    if search:
        tool_type = "web_search_20260209" if model.startswith("claude-sonnet-5") else "web_search_20250305"
        kwargs["tools"] = [{"type": tool_type, "name": "web_search", "max_uses": 3}]
        kwargs["max_tokens"] = 8192

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            texts = [b.text for b in resp.content if b.type == "text"]
            return "\n".join(texts)
        except anthropic.RateLimitError as e:
            wait = int(e.response.headers.get("retry-after", "60")) + 5
            print(f"429, wait {wait}s...", end=" ", flush=True)
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(10 * (attempt + 1))
            else:
                raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def call_openai(prompt: str, model: str, api_key: str, search: bool, thinking: str) -> str:
    import openai

    client = openai.OpenAI(api_key=api_key)
    for attempt in range(MAX_RETRIES):
        try:
            if search:
                # web search 走 Responses API；工具名新版 "web_search"，舊版 "web_search_preview"
                try:
                    resp = client.responses.create(
                        model=model, input=prompt, tools=[{"type": "web_search"}],
                    )
                except openai.BadRequestError:
                    resp = client.responses.create(
                        model=model, input=prompt, tools=[{"type": "web_search_preview"}],
                    )
                return resp.output_text
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except openai.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"429, wait {wait}s...", end=" ", flush=True)
            time.sleep(wait)
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(10 * (attempt + 1))
            else:
                raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def call_openai_compatible(prompt: str, model: str, api_key: str, search: bool, thinking: str) -> str:
    """Groq / Cerebras 共用：OpenAI 相容 chat completions。search 條件不支援（main 已擋）。

    reasoning 模型（qwen3 / gpt-oss）預設會 thinking，吃滿 2048 output token 被截斷、
    答案出不來 → thinking=off 時必須傳 reasoning_effort 壓下去。
    """
    import openai

    cfg = MODELS[model]
    provider = cfg["provider"]
    client = openai.OpenAI(api_key=api_key, base_url=OPENAI_COMPAT_BASE_URL[provider])

    kwargs = {}
    if thinking == "off" and cfg.get("reasoning"):
        kwargs["reasoning_effort"] = cfg["reasoning"]
    if cfg.get("reasoning"):
        # 即使 thinking=default 也要留足夠 output 空間，避免 thinking 吃光後截斷
        kwargs["max_tokens"] = 4096

    # Cerebras 是每小時制：飽和後只能等桶子回補，每次只等 30s 所以多試幾次划算
    # （6 次 × 30s 只有 3 分鐘，遇到桶子空的時段會整批 fail；20 次 = 10 分鐘足以撐過）
    max_retries = 20 if provider == "cerebras" else MAX_RETRIES

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            choice = resp.choices[0]
            if choice.finish_reason == "length":
                raise RuntimeError(
                    f"output truncated at {resp.usage.completion_tokens} tokens "
                    f"(finish_reason=length) — thinking 可能沒關成功"
                )
            return choice.message.content or ""
        except openai.RateLimitError as e:
            # 判斷依據用 header（權威）＋訊息中的 "per day" 明示，不要只憑單一 header。
            # 2026-07-16 教訓 A：regex 誤判 TPM 短期限速為「當日額度歸零」，白白中止 4 個批次。
            # 2026-07-19 教訓 B：只查 remaining-requests 會漏掉 TPD（token/day）——TPD 撞牆時
            #   requests 額度還在、只有 tokens 歸零，導致 fail-fast 沒觸發、狂等 90s×6 白燒時間。
            #   → 三個訊號任一成立即判定「當日額度歸零」：訊息含 "per day"/"TPD"、
            #     remaining-requests=0、或 remaining-tokens=0。
            resp = getattr(e, "response", None)
            headers = getattr(resp, "headers", {}) or {}
            rem_req = str(headers.get("x-ratelimit-remaining-requests", "")).strip()
            rem_tok = str(headers.get("x-ratelimit-remaining-tokens", "")).strip()
            reset = (headers.get("x-ratelimit-reset-tokens")
                     or headers.get("x-ratelimit-reset-requests") or "?")
            msg = str(getattr(e, "body", "") or "") + str(e)
            per_day = bool(re.search(r"per\s*day|\bTPD\b|\bRPD\b", msg, re.IGNORECASE))
            if per_day or rem_req in ("0", "0.0") or rem_tok in ("0", "0.0"):
                dim = "tokens/day" if (per_day and "token" in msg.lower()) else "requests/day"
                raise DailyQuotaExhausted(
                    f"{model}（{provider}）當日額度歸零（{dim}，重置需 {reset}）。"
                    f"重試無用，需等額度回補或換模型。\n原始訊息：{msg[:200]}"
                ) from e
            # 其餘 429＝短期限速（TPM/RPM），等 retry-after 或退避後重試
            if attempt == 0 and os.environ.get("DEBUG_429"):
                print(f"\n[DEBUG 429] {model}\n  msg={str(e)[:300]}\n"
                      f"  headers={ {k: v for k, v in headers.items() if 'ratelimit' in k.lower() or 'retry' in k.lower()} }\n",
                      flush=True)
            # Cerebras 是「每小時」限制且無 retry-after header；實測飽和後恢復約 27s，
            # 用平坦 30s 重試比指數退避（30/60/90）省時，且不會把 quota 燒在無效請求上。
            if provider == "cerebras":
                wait = 30.0
            else:
                wait = 30 * (attempt + 1)
                for hk in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
                    hv = headers.get(hk)
                    if hv:
                        m = re.match(r"^([\d.]+)s?$", str(hv).strip())
                        if m:
                            wait = float(m.group(1)) + 2
                            break
                wait = min(wait, 90)
            print(f"429(rate), wait {wait:.0f}s...", end=" ", flush=True)
            time.sleep(wait)
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(10 * (attempt + 1))
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def probe_models(provider: str):
    """列出 provider 實際可用的模型 ID（GET /models），確認 registry 名稱沒寫錯。"""
    api_key = os.environ.get(PROVIDER_KEY_ENV[provider], "")
    if not api_key:
        print(f"ERROR: Set {PROVIDER_KEY_ENV[provider]} first.")
        sys.exit(1)
    if provider in OPENAI_COMPAT_BASE_URL or provider == "openai":
        import openai
        base = OPENAI_COMPAT_BASE_URL.get(provider)
        client = openai.OpenAI(api_key=api_key, base_url=base)
        for m in client.models.list():
            print(f"  {m.id}")
    elif provider == "gemini":
        import requests
        resp = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}", timeout=30)
        resp.raise_for_status()
        for m in resp.json().get("models", []):
            print(f"  {m['name'].removeprefix('models/')}")
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        for m in client.models.list():
            print(f"  {m.id}")


PROVIDER_CALLERS = {
    "gemini": call_gemini,
    "anthropic": call_anthropic,
    "openai": call_openai,
    "groq": call_openai_compatible,
    "cerebras": call_openai_compatible,
    "sambanova": call_openai_compatible,
}


# ---------------------------------------------------------------------------
# Progress management
# ---------------------------------------------------------------------------

def get_output_path(model: str, condition: str, variant: str = "standard") -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if variant == "standard" else f"_{variant}prompt"
    safe_model = model.replace("/", "--")  # e.g. qwen/qwen3-32b → qwen--qwen3-32b
    return OUTPUT_DIR / f"{safe_model}_{condition}{suffix}.jsonl"


def load_progress(path: Path) -> dict[int, dict]:
    done = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if "id" in rec:
                        done[rec["id"]] = rec
    return done


def save_result(path: Path, result: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def reparse_results(output_path: Path, data: list[dict]):
    """用已存的 raw 重新解析既有結果（parser 改版後免重打 API）。API error 記錄會被丟掉以便重跑。"""
    if not output_path.exists():
        print(f"No results file: {output_path}")
        return
    by_id = {d["id"]: d for d in data}
    kept, fixed, still_bad, dropped = 0, 0, 0, 0
    new_lines = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                dropped += 1  # API error → 下次 run 重打
                continue
            lang = by_id.get(rec.get("id"))
            if lang is None:
                kept += 1
            elif rec.get("parse_error") and "raw" in rec:
                parsed = parse_response(rec["raw"])
                if parsed is not None:
                    ev = evaluate(parsed, lang["gold"], lang["not_sure"], lang["words"], lang["excluded"])
                    rec = {
                        "id": lang["id"],
                        "name": lang["name"],
                        "raw": rec["raw"],
                        "ai_output": parsed,
                        "gold": lang["gold"],
                        "eval": ev,
                        "parse_error": False,
                        "format_noncompliant": True,  # 沒照指令格式，靠寬容 parser 救回——論文可報告
                    }
                    fixed += 1
                else:
                    still_bad += 1
            elif "raw" in rec or "ai_output" in rec:
                # 用 raw（優先）或既存 ai_output 依現行 parser/評分規則重評（Echo 分類、normalize 修正）
                parsed = parse_response(rec["raw"]) if "raw" in rec else rec["ai_output"]
                if parsed is None:
                    parsed = rec.get("ai_output")
                rec["ai_output"] = parsed
                rec["gold"] = lang["gold"]
                rec["eval"] = evaluate(parsed, lang["gold"], lang["not_sure"], lang["words"], lang["excluded"])
                rec["parse_error"] = False
                kept += 1
            else:
                kept += 1
            new_lines.append(json.dumps(rec, ensure_ascii=False))
    tmp = output_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
    tmp.replace(output_path)
    print(f"Reparse: {kept} kept, {fixed} fixed, {still_bad} still unparseable, {dropped} API-error dropped")


def save_run_meta(model: str, condition: str, thinking: str, variant: str = "standard"):
    """把本次 run 的有效設定寫進 sidecar，論文方法章節要用。"""
    meta_path = OUTPUT_DIR / f"{get_output_path(model, condition, variant).stem}.meta.json"
    cfg = MODELS[model]
    reasoning_effort = cfg.get("reasoning") if thinking == "off" else None
    meta = {
        "model": model,
        "provider": cfg["provider"],
        "params": cfg.get("params"),
        "rpd": cfg.get("rpd"),
        "condition": condition,
        "prompt_variant": variant,
        "thinking": thinking,
        "reasoning_effort": reasoning_effort,  # 實際送出的值；None=未送此參數
        "temperature": None if model.startswith("claude-sonnet-5") else 0,
        "notes": {
            "claude-sonnet-5": "API 不接受非預設 temperature；thinking off = 明確 disabled",
            "gemini-2.5-pro": "thinking 無法全關（API 最低 budget 128）",
        }.get(model),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results_path: Path):
    done = load_progress(results_path)
    if not done:
        print("No results yet.")
        return

    total_correct = 0
    total_numerals = 0
    errors_by_type = {"FN": 0, "FP": 0, "Wrong": 0, "Echo": 0}
    errors_by_numeral = {i: 0 for i in range(1, 11)}
    parse_failures = 0

    for rec in done.values():
        if rec.get("parse_error"):
            parse_failures += 1
            continue
        ev = rec["eval"]
        total_correct += ev["score"]
        total_numerals += ev["total"]
        for d in ev["details"]:
            if d.get("verdict") in ("Excluded", "NotSure") or d["correct"]:
                continue
            errors_by_type[d["verdict"]] += 1
            errors_by_numeral[d["numeral"]] += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {results_path.name}")
    print(f"Languages processed: {len(done)} ({parse_failures} parse failures)")
    if total_numerals:
        print(f"Overall accuracy: {total_correct}/{total_numerals} ({total_correct / total_numerals * 100:.1f}%)")
    print(f"\nError breakdown:")
    print(f"  FN (missed pattern):     {errors_by_type['FN']}")
    print(f"  FP (hallucinated):       {errors_by_type['FP']}")
    print(f"  Wrong (wrong formula):   {errors_by_type['Wrong']}")
    print(f"  Echo (word repeated):    {errors_by_type['Echo']}")
    print(f"\nErrors by numeral:")
    for num in range(1, 11):
        bar = "█" * min(errors_by_numeral[num], 50)
        print(f"  {num:2d}: {errors_by_numeral[num]:3d} {bar}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ROCLING numeral analysis pipeline (multi-provider)")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    parser.add_argument("--condition", default=DEFAULT_CONDITION)
    parser.add_argument("--thinking", default="off", choices=["off", "default"],
                        help="off=盡量關 thinking（省錢、可比較）；default=各模型 API 預設")
    parser.add_argument("--prompt-variant", default="standard", choices=sorted(PROMPT_VARIANTS),
                        help="standard=原版；neutral=中性範例（priming ablation 用）")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=9999)
    parser.add_argument("--dry-run", action="store_true", help="Parse TSV only, no API calls")
    parser.add_argument("--summary", action="store_true", help="Print summary of existing results")
    parser.add_argument("--reparse", action="store_true",
                        help="Re-parse stored raw responses with current parser (no API calls)")
    parser.add_argument("--list", action="store_true", help="List models and conditions")
    parser.add_argument("--probe", metavar="PROVIDER", choices=sorted(PROVIDER_KEY_ENV),
                        help="List actual available model IDs for a provider")
    args = parser.parse_args()

    if args.list:
        print("Models:")
        for m, cfg in MODELS.items():
            params = cfg.get("params", "?")
            print(f"  {m:24s} provider={cfg['provider']:10s} params={params:4s} delay={cfg['delay']}s")
        print("\nConditions: {zeroshot|oneshot}_{name|noname}[_search]")
        return

    if args.probe:
        probe_models(args.probe)
        return

    _, _, search = parse_condition(args.condition)  # validate early
    if search and MODELS[args.model]["provider"] not in SEARCH_PROVIDERS:
        print(f"ERROR: search condition not supported for {args.model} "
              f"({MODELS[args.model]['provider']} has no native web search). Use a Gemini model.")
        sys.exit(1)
    output_path = get_output_path(args.model, args.condition, args.prompt_variant)

    if args.summary:
        print_summary(output_path)
        return

    if args.reparse:
        reparse_results(output_path, load_tsv())
        print_summary(output_path)
        return

    provider = MODELS[args.model]["provider"]
    delay = MODELS[args.model]["delay"]

    api_key = os.environ.get(PROVIDER_KEY_ENV[provider], "")
    if not api_key and not args.dry_run:
        env = PROVIDER_KEY_ENV[provider]
        print(f"ERROR: Set {env} environment variable first.")
        print(f"  PowerShell:  $env:{env} = 'your-key-here'")
        print(f"  Bash:        export {env}='your-key-here'")
        sys.exit(1)

    print(f"Loading TSV from {TSV_PATH}...")
    data = load_tsv()
    print(f"  {len(data)} languages loaded")

    data = [d for d in data if args.start <= d["id"] <= args.end]
    print(f"  {len(data)} languages in range [{args.start}, {args.end}]")

    if args.dry_run:
        print(f"\n[DRY RUN] condition={args.condition}, variant={args.prompt_variant}, sample prompts:")
        for d in data[:2]:
            print(f"\n--- #{d['id']} {d['name']} ---")
            print(make_prompt(d, args.condition, args.prompt_variant))
            print(f"Gold:  {d['gold']}")
            print(f"Not sure: {d['not_sure']}")
        print(f"\n[DRY RUN] Total languages to process: {len(data)}")
        return

    done = load_progress(output_path)
    remaining = [d for d in data if d["id"] not in done]
    print(f"  {len(done)} already done, {len(remaining)} remaining")
    print(f"  Model: {args.model} ({provider})  Condition: {args.condition}  "
          f"Variant: {args.prompt_variant}  Thinking: {args.thinking}")
    print(f"  Output: {output_path}")
    print(f"  Estimated time: ~{len(remaining) * delay / 60:.0f} min (rate limit only, excl. API latency)")
    print()

    save_run_meta(args.model, args.condition, args.thinking, args.prompt_variant)
    caller = PROVIDER_CALLERS[provider]

    for i, lang in enumerate(remaining):
        prompt = make_prompt(lang, args.condition, args.prompt_variant)
        lang_label = f"#{lang['id']} {lang['name']}"

        try:
            print(f"[{i + 1}/{len(remaining)}] {lang_label}...", end=" ", flush=True)
            raw = caller(prompt, args.model, api_key, search, args.thinking)
            parsed = parse_response(raw)

            if parsed is None:
                result = {
                    "id": lang["id"],
                    "name": lang["name"],
                    "raw": raw,
                    "parse_error": True,
                }
                print(f"PARSE ERROR: {raw[:80]}")
            else:
                ev = evaluate(parsed, lang["gold"], lang["not_sure"], lang["words"], lang["excluded"])
                result = {
                    "id": lang["id"],
                    "name": lang["name"],
                    "raw": raw,
                    "ai_output": parsed,
                    "gold": lang["gold"],
                    "eval": ev,
                    "parse_error": False,
                }
                marks = "".join("✓" if d["correct"] else "✗" for d in ev["details"])
                print(f"{ev['score']}/10 {marks}")

            save_result(output_path, result)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved. Re-run to continue.")
            break
        except DailyQuotaExhausted as e:
            print(f"\n\n{e}\nProgress saved. Stopping run (retrying would just re-hit the same daily cap).")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            save_result(output_path, {
                "id": lang["id"],
                "name": lang["name"],
                "error": str(e),
                "parse_error": True,
            })

        if i < len(remaining) - 1:
            time.sleep(delay)

    print_summary(output_path)


if __name__ == "__main__":
    main()
