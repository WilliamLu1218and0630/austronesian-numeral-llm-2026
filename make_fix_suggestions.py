"""
鐵證級修正建議表（人工核可用，不自動改 gold）
=============================================
輸入：results/_consensus_audit_zeroshot.tsv（共識稽核輸出）
篩出「鐵證級」（詞形含 [N] 成分標記 or col18 有自由文字標註）條目，
逐條給出「原 gold → 建議 → 依據」，並把 col18 深層分析與模型表層分解並列。

分三組（複查難度遞增）：
  A. 建議補標（高信度）：missed 方向，col18 與共識一致 或 只有詞形標記無衝突
  B. 層次選擇：       missed 方向，col18 深層分析 ≠ 模型表層分解（需選標註層次）
  C. 需裁決：         over / diff 方向（over 可能只是模型漏看，未必 gold 錯）

輸出 results/_fix_suggestions.tsv + stdout 人可讀表。
"""

import csv
import io
import re
import sys
from pathlib import Path

import pipeline as P

RESULTS = Path(__file__).parent / "results"
IN_TSV = RESULTS / "_consensus_audit_zeroshot.tsv"
OUT_TSV = RESULTS / "_fix_suggestions.tsv"


def eq(a: str, b: str) -> bool:
    return a == b or P._commutative_equal(a, b)


def main():
    rows = []
    with open(IN_TSV, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)

    hard = [r for r in rows
            if "詞形標記" in r["evidence"] or "col18有標" in r["evidence"]]

    A, B, C = [], [], []
    for r in hard:
        num = int(r["numeral"])
        direction = r["direction"]
        gold = r["gold"]
        cons = r["consensus"]
        col18_f = r.get("col18_this", "")
        not_sure = r["not_sure"] == "True"
        has_wordmark = "詞形標記" in r["evidence"]

        rec = {
            "lang_id": r["lang_id"], "name": r["name"], "numeral": num,
            "word": r["word"], "direction": direction,
            "gold_now": gold, "consensus": cons, "col18_formula": col18_f,
            "n_agree": r["n_agree"], "not_sure": not_sure,
        }

        if direction == "missed":
            if col18_f and not eq(col18_f, cons):
                # 層次差異：col18 深層 vs 模型表層
                rec["suggest"] = col18_f  # 預設尊重標註者深層分析
                rec["layer_conflict"] = True
                rec["basis"] = f"col18深層={col18_f}；模型表層={cons}（{r['n_agree']}家族）"
                rec["action"] = "選層次"
                B.append(rec)
            else:
                rec["suggest"] = cons
                rec["layer_conflict"] = False
                src = []
                if has_wordmark:
                    src.append(f"詞形標記 {r['word']!r}")
                if col18_f:
                    src.append(f"col18={col18_f}(與模型一致)")
                src.append(f"{r['n_agree']}/4 家族共識")
                rec["basis"] = "；".join(src)
                rec["action"] = "補標(高信度)"
                A.append(rec)
        else:
            # over / diff：需裁決
            rec["suggest"] = ""  # 不給單一建議
            rec["layer_conflict"] = bool(col18_f and not eq(col18_f, gold))
            if direction == "over":
                rec["basis"] = (f"gold={gold}，但 {r['n_agree']} 家族說 atomic(模型可能漏看)"
                                + (f"；col18={col18_f}" if col18_f else ""))
            else:
                rec["basis"] = (f"gold={gold} vs 模型={cons}"
                                + (f"；col18={col18_f}" if col18_f else ""))
            rec["action"] = "裁決"
            C.append(rec)

    # 寫 TSV
    cols = ["lang_id", "name", "numeral", "word", "direction",
            "gold_now", "consensus", "col18_formula", "suggest",
            "layer_conflict", "action", "not_sure", "n_agree", "basis"]
    with open(OUT_TSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for rec in A + B + C:
            w.writerow(rec)

    def show(group, title):
        print(f"\n{'='*80}\n{title}（{len(group)} 條）\n{'='*80}")
        for r in group:
            ns = " ⚠not_sure" if r["not_sure"] else ""
            sug = f" → 建議 {r['suggest']}" if r["suggest"] else ""
            print(f"#{r['lang_id']:>4} {r['name'][:20]:<20} 數{r['numeral']}: "
                  f"{r['word'][:24]}")
            print(f"       gold={r['gold_now']}{sug}{ns}")
            print(f"       依據：{r['basis']}")

    show(A, "A. 建議補標（高信度：詞形/col18/模型互相印證）")
    show(B, "B. 層次選擇（col18 深層分析 vs 模型表層分解，需你定標註層次）")
    show(C, "C. 需裁決（over/diff：over 方向模型 atomic 可能只是漏看，未必 gold 錯）")

    print(f"\n{'='*80}")
    print(f"合計 {len(A)+len(B)+len(C)} 條鐵證級：A補標 {len(A)}／B層次 {len(B)}／C裁決 {len(C)}")
    print(f"輸出：{OUT_TSV}")


if __name__ == "__main__":
    main()
