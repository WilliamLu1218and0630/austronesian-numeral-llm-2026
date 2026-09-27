"""
從模型偽陽性（FP）中篩出金標準未標記的構詞候選。

過濾三關（對應論文第 5 節）：
  1. 跨模型一致：兩款獨立架構的核心模型同時在同一位置判為構詞複雜
  2. 構詞透明：AI 公式所引用的成分數詞，其詞形須能在目標詞形中找到
  3. 回核金標準：排除 Gold 欄或『特殊』自由文字欄已記錄者

此腳本取代改版前的臨時流程，使候選數字可重現。
用法：python fp_candidates.py [--condition zeroshot_noname]
"""

import argparse
import csv
import json
import re
from pathlib import Path

import pipeline as P

RESULTS = Path(__file__).parent / "results"

# 兩款獨立架構的核心模型（不同家族、不同訓練來源）
MODEL_A = "gemma-4-31b_{cond}_neutralprompt.jsonl"
MODEL_B = "gpt-oss-120b_{cond}_neutralprompt.jsonl"


def strip_marks(w: str) -> str:
    """去掉標註者的 [N] 成分註記與空白，留下可比對的裸詞形。"""
    return re.sub(r"\[\d+\]", "", w).replace(" ", "").lower()


def components_of(formula: str) -> list[int]:
    """取出公式引用的數字操作元。"""
    return [int(x) for x in re.findall(r"\d+", formula)]


MIN_ROOT = 3   # 詞根比對的最短長度；短於此無鑑別力（與字串匹配基線同標準）


def is_transparent(word: str, formula: str, words: list[str],
                   numeral: int, strict: bool = True) -> tuple[bool, str]:
    """成分數詞的詞形是否可見於目標詞形（詞根匹配，$\\geq$ MIN_ROOT 字元）。

    兩種判準都採「至少一個成分可驗證」——要求全部成分可驗會誤殺正字法有出入的
    真案例（如 Tandia 的 *marusibě*，成分 2=*arusi* 可驗但 5=*márási* 因重音
    符號對不上）。

    strict=True（預設）另加兩道防呆，兩者都針對確實會產生假陽性的情形：
      * 自我指涉——成分即該數詞本身，或其詞形與目標同形（如 Teop 的
        ``10=peha savun`` 拿目標詞形匹配自己）；
      * 類別前綴——詞根同時見於半數以上的其他數詞，屬全套數詞共用的構詞前綴
        （如 Madak 的 *lava-*、Nehan 的 *tow-*），不構成成分證據。

    strict=False 不加這兩道，用以重現改版前的判準以便對照。
    """
    target = strip_marks(word)
    if not target:
        return False, ""

    comps = [c for c in components_of(formula) if 1 <= c <= 10]
    if not comps:
        return False, ""

    others = [strip_marks(w) for i, w in enumerate(words)
              if i != numeral - 1 and strip_marks(w)]

    hits = []
    for c in comps:
        if strict and c == numeral:                   # 自我指涉
            continue
        comp = strip_marks(words[c - 1])
        if len(comp) < MIN_ROOT:
            continue
        if strict and comp == target:                 # 與目標同形
            continue
        matched = None
        for ln in range(len(comp), MIN_ROOT - 1, -1):
            if comp[:ln] in target:
                matched = comp[:ln]
                break
        if matched is None:
            continue
        if strict and others and sum(1 for o in others if matched in o) >= len(others) / 2:
            continue                                  # 類別前綴
        hits.append(f"{c}={words[c-1]}")

    return (len(hits) > 0), " ".join(hits)


def arithmetic_ok(formula: str, numeral: int) -> bool:
    """公式算出來是否等於該數字。用於三分類的 A/B 之分。"""
    expr = formula.replace("×", "*")
    if not re.fullmatch(r"[\d\s+\-*()]+", expr):
        return False
    try:
        return eval(expr) == numeral          # noqa: S307 — 已用 regex 限定字元集
    except Exception:
        return False


def fp_map(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("parse_error"):
                continue
            for d in rec["eval"]["details"]:
                if d["verdict"] == "FP":
                    out[(rec["id"], d["numeral"])] = d["ai"]
    return out


def load_special() -> dict:
    """讀『特殊』自由文字欄。"""
    special = {}
    with open(P.TSV_PATH, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row or not row[P.COL_ID].strip().isdigit():
                continue
            txt = row[P.COL_SPECIAL].strip() if len(row) > P.COL_SPECIAL else ""
            if txt:
                special[int(row[P.COL_ID].strip())] = txt
    return special


def run(cond, data, special, a, b, agree, strict):
    """跑一次完整過濾，回傳三分類結果。"""
    A, B, C = [], [], 0
    for (lid, num), (fa, fb) in sorted(agree.items()):
        lang = data.get(lid)
        if lang is None:
            continue
        word = lang["words"][num - 1]
        ok, comps = is_transparent(word, fa, lang["words"], num, strict=strict)
        if not ok:
            C += 1
            continue
        txt = special.get(lid, "")
        recorded = bool(re.search(rf"\b{num}\s*=", txt))
        row = (lid, num, word, fa, fb, comps, txt, recorded)
        (A if arithmetic_ok(fa, num) else B).append(row)
    return A, B, C


def report(label, cond, data, A, B, C, strict):
    tag = "嚴格" if strict else "寬鬆（重現改版前判準）"
    print(f"\n{'='*74}")
    print(f"  {label}｜判準：{tag}")
    print(f"{'='*74}")
    print(f"  [A] 構詞透明 + 算術正確 : {len(A):>3}   ← 最強證據")
    print(f"  [B] 構詞透明 + 算術不符 : {len(B):>3}   ← 詞素真實、分析可議")
    print(f"  [C] 不透明               : {C:>3}")

    fresh = [r for r in A + B if not r[7]]
    print(f"  扣除『特殊』欄已記錄者後的候選：{len(fresh)}")
    if fresh:
        complete = sum(1 for r in fresh if all(data[r[0]]["words"]))
        print(f"    其中資料完整（10/10）：{complete}（{complete/len(fresh)*100:.0f}%）")

    for name, group in (("A", A), ("B", B)):
        if not group:
            continue
        print(f"\n  --- [{name}] ---")
        for lid, num, word, fa, fb, comps, txt, rec in group:
            mark = " (特殊欄已記錄)" if rec else ""
            print(f"  #{lid:<4} {data[lid]['name'][:20]:<20} {num:>2}: "
                  f"{word[:24]:<24} -> {fa:<8} | {comps}{mark}")
    return fresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="zeroshot_noname")
    ap.add_argument("--compare", action="store_true",
                    help="同時輸出寬鬆判準結果，與改版前數字對照")
    args = ap.parse_args()
    cond = args.condition

    data = {d["id"]: d for d in P.load_tsv()}
    special = load_special()

    a = fp_map(RESULTS / MODEL_A.format(cond=cond))
    b = fp_map(RESULTS / MODEL_B.format(cond=cond))
    agree = {k: (a[k], b[k]) for k in a.keys() & b.keys()}

    print(f"=== FP 候選篩選（{cond}，現行 gold）===")
    print(f"  Gemma 31B FP      : {len(a)}")
    print(f"  GPT-OSS 120B FP   : {len(b)}")
    print(f"  兩款一致（關卡 1）: {len(agree)}  涉及 {len({i for i,_ in agree})} 種語言")

    if args.compare:
        A0, B0, C0 = run(cond, data, special, a, b, agree, strict=False)
        report("寬鬆判準", cond, data, A0, B0, C0, strict=False)

    A, B, C = run(cond, data, special, a, b, agree, strict=True)
    fresh = report("嚴格判準", cond, data, A, B, C, strict=True)

    out = RESULTS / f"_fp_candidates_{cond}.tsv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["class", "id", "language", "numeral", "wordform",
                    "ai_gemma", "ai_gptoss", "components", "special_col"])
        for cls, group in (("A", A), ("B", B)):
            for lid, num, word, fa, fb, comps, txt, rec in group:
                if rec:
                    continue
                w.writerow([cls, lid, data[lid]["name"], num, word, fa, fb, comps, txt])
    print(f"\n→ {out}（{len(fresh)} 筆）")


if __name__ == "__main__":
    main()
