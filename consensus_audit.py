"""
跨模型共識稽核（金標準品質檢查）
================================
邏輯：對每個 (語言, 數詞位置)，收集 N 個「不同底模型家族」的預測公式，
用 pipeline 的交換律比對做分群；若「多家族彼此一致、但與人工 gold 不符」，
就列為人工複查候選（disagreement set）。

方法論定位：模型只做 flag，不當仲裁者。輸出是「該人看的爭議條目」，
不是「人錯了」的判定。三個方向分開報：
  missed  gold=atomic，多家族一致說某公式 → 疑似人工漏標（最高價值）
  over    gold=公式，  多家族一致說 atomic → 疑似人工過度標註
  diff    兩邊都是公式但不同            → 公式分歧

只選「不同底模型家族」各一代表，避免同架構（如 Llama 8B/70B）重複計票。
zero-shot / neutral prompt（純分析力、無範例 priming），是最乾淨的稽核基底。
"""

import csv
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pipeline as P  # 復用 gold 建構、normalize、交換律比對、echo 判定
# 注意：import pipeline 已把 sys.stdout 重包成 UTF-8，這裡不可再包（會關掉底層 buffer）。

RESULTS = Path(__file__).parent / "results"

# 各底模型家族一個代表（皆 892+ 全量覆蓋、不同 pretraining）。
# Gemini 兩檔僅 9/19 筆，覆蓋太少，不納入獨立家族。
FAMILIES = {
    "Llama-70B":    "llama-3.3-70b-versatile_zeroshot_noname_neutralprompt.jsonl",
    "Gemma-31B":    "gemma-4-31b_zeroshot_noname_neutralprompt.jsonl",
    "GPT-OSS-120B": "gpt-oss-120b_zeroshot_noname_neutralprompt.jsonl",
    "Qwen-27B":     "qwen--qwen3.6-27b_zeroshot_noname_neutralprompt.jsonl",
}

# 同家族佐證（不計入獨立票數，只在明細欄供人參考）
WITNESS = {
    "Llama-8B":    "llama-3.1-8b-instant_zeroshot_noname_neutralprompt.jsonl",
    "GPT-OSS-20B": "openai--gpt-oss-20b_zeroshot_noname_neutralprompt.jsonl",
}

MIN_AGREE = 3          # 至少幾個獨立家族一致才 flag（4 家族中的 3/4 或 4/4）
MIN_FAMILIES = 3       # 該位置至少要有幾個家族給出有效預測才判定（否則分母不足）

OUT_TSV = RESULTS / "_consensus_audit_zeroshot.tsv"


def load_model(fname: str) -> dict[int, list[str] | None]:
    """回傳 {lang_id: ai_output(10 公式) 或 None}。parse_error / 無輸出 → None。"""
    path = RESULTS / fname
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("parse_error") or "ai_output" not in rec:
                out[rec["id"]] = None
            else:
                out[rec["id"]] = rec["ai_output"]
    return out


def classify(pred: str, numeral: int, word: str) -> tuple[str, str]:
    """把單一預測歸類。回傳 (kind, value)。
       kind: 'atomic' | 'formula' | 'echo' | 'empty'
       echo（把詞形抄回來）視為無效票，不參與共識。"""
    v = P.normalize_formula(pred)
    if not v:
        return "empty", ""
    if v == str(numeral):
        return "atomic", v
    if P.is_echo(v, word):
        return "echo", v
    return "formula", v


def load_special() -> dict[int, str]:
    """讀原始 TSV 的 col18（特殊自由文字標註），供分級對照——gold 建構未採此欄。"""
    out = {}
    with open(P.TSV_PATH, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row and row[0].strip().isdigit():
                out[int(row[0].strip())] = (
                    row[P.COL_SPECIAL].strip() if len(row) > P.COL_SPECIAL else "")
    return out


def parse_col18(s: str) -> dict[int, str]:
    """從 col18 自由文字抽 {數詞: 公式}。支援 'N=f'、'N＝f'、'N是f'。抓不到的略過。"""
    out = {}
    for seg in re.split(r"[；;,，、]", s):
        m = re.match(r"\s*(\d+)\s*(?:[=＝]|是)\s*([\d\s+\-*×xX²]+)", seg)
        if m:
            num = int(m.group(1))
            f = P.normalize_formula(m.group(2))
            if 1 <= num <= 10 and f:
                out[num] = f
    return out


def evidence_of(word: str, col18_has_this: bool, not_sure_pos: bool) -> str:
    """證據分級。鐵證＝標註者已知『這個數詞』的成分卻沒進布林欄：
         詞形標記 = 詞形內含 [N] 成分註記；
         col18有標 = col18 自由文字有『本數詞』的公式（非同語言其他數詞）。
       not_sure＝標註者自己存疑；三者皆無＝model-only（純模型推論，最需人審慎）。"""
    marks = []
    if re.search(r"\[\d", word):
        marks.append("詞形標記")
    if col18_has_this:
        marks.append("col18有標")
    if not_sure_pos:
        marks.append("not_sure")
    return "+".join(marks) if marks else "model-only"


def cluster(preds: list[tuple[str, str]]) -> list[tuple[str, set[str]]]:
    """把 (family, value) 依交換律等價分群，回傳 [(代表公式, {families})]，最大群在前。"""
    groups: list[tuple[str, set[str]]] = []
    for fam, val in preds:
        for i, (rep, fams) in enumerate(groups):
            if val == rep or P._commutative_equal(val, rep):
                fams.add(fam)
                break
        else:
            groups.append((val, {fam}))
    groups.sort(key=lambda g: len(g[1]), reverse=True)
    return groups


def main():
    print("Loading TSV + gold ...")
    data = P.load_tsv()
    by_id = {d["id"]: d for d in data}
    print(f"  {len(data)} languages")

    fam_preds = {name: load_model(f) for name, f in FAMILIES.items()}
    wit_preds = {name: load_model(f) for name, f in WITNESS.items()}
    special = load_special()
    for name, d in fam_preds.items():
        print(f"  {name:14s} {len(d)} langs")

    flags = []
    for lang in data:
        lid, name, words, gold, not_sure = (
            lang["id"], lang["name"], lang["words"], lang["gold"], lang["not_sure"])

        for i in range(10):
            numeral = i + 1
            word = words[i]
            if not word:
                continue

            # 收集各獨立家族的有效預測（atomic / formula；echo/empty 剔除）
            valid = []
            raw_by_fam = {}
            for fam, preds in fam_preds.items():
                out = preds.get(lid)
                if out is None or i >= len(out):
                    continue
                kind, val = classify(out[i], numeral, word)
                raw_by_fam[fam] = f"{val}" if kind != "echo" else f"⟨echo:{val}⟩"
                if kind in ("atomic", "formula"):
                    valid.append((fam, val))

            if len(valid) < MIN_FAMILIES:
                continue

            groups = cluster(valid)
            rep, fams = groups[0]
            k = len(fams)
            if k < MIN_AGREE:
                continue

            gold_i = P.normalize_formula(gold[i])
            if rep == gold_i or P._commutative_equal(rep, gold_i):
                continue  # 共識與 gold 一致 → 不是爭議

            gold_atomic = (gold_i == str(numeral))
            cons_atomic = (rep == str(numeral))
            if gold_atomic and not cons_atomic:
                direction = "missed"   # 人工可能漏標
            elif not gold_atomic and cons_atomic:
                direction = "over"     # 人工可能過度標註
            else:
                direction = "diff"     # 公式分歧

            # 同家族佐證
            wit = {}
            for wname, preds in wit_preds.items():
                out = preds.get(lid)
                if out is None or i >= len(out):
                    continue
                kind, val = classify(out[i], numeral, word)
                wit[wname] = val if kind != "echo" else f"⟨echo:{val}⟩"

            col18_full = special.get(lid, "")
            col18_this = parse_col18(col18_full).get(numeral, "")
            evidence = evidence_of(word, bool(col18_this), numeral in not_sure)
            flags.append({
                "lang_id": lid, "name": name, "numeral": numeral, "word": word,
                "gold": gold_i, "consensus": rep, "n_agree": k,
                "n_families": len(valid), "direction": direction,
                "evidence": evidence, "not_sure": numeral in not_sure,
                "col18_this": col18_this, "col18": col18_full,
                "families_agree": ",".join(sorted(fams)),
                "all_fam": " | ".join(f"{f}={raw_by_fam[f]}" for f in FAMILIES if f in raw_by_fam),
                "witness": " | ".join(f"{w}={wit[w]}" for w in WITNESS if w in wit),
            })

    # 排序：鐵證級（詞形標記/col18）浮頂 → 一致強度 → missed 方向 → not_sure
    dir_rank = {"missed": 0, "over": 1, "diff": 2}
    def hard(r):  # 有詞形標記或 col18＝布林欄純漏的高信度證據
        return ("詞形標記" in r["evidence"]) or ("col18有標" in r["evidence"])
    flags.sort(key=lambda r: (not hard(r), -r["n_agree"], dir_rank[r["direction"]],
                              not r["not_sure"], r["lang_id"], r["numeral"]))

    # 寫 TSV
    cols = ["lang_id", "name", "numeral", "word", "gold", "consensus",
            "n_agree", "n_families", "direction", "evidence", "not_sure",
            "col18_this", "families_agree", "all_fam", "witness", "col18"]
    with open(OUT_TSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in flags:
            w.writerow(r)

    # Summary
    print(f"\n{'='*70}")
    print(f"共識稽核結果：{len(flags)} 個爭議條目（{MIN_AGREE}+ 家族一致但與 gold 不符）")
    print(f"{'='*70}")
    by_dir = Counter(r["direction"] for r in flags)
    by_k = Counter(r["n_agree"] for r in flags)
    by_num = Counter(r["numeral"] for r in flags)
    ns = sum(1 for r in flags if r["not_sure"])
    n_wordmark = sum(1 for r in flags if "詞形標記" in r["evidence"])
    n_col18 = sum(1 for r in flags if "col18有標" in r["evidence"])
    n_hard = sum(1 for r in flags if hard(r))
    n_modelonly = sum(1 for r in flags if r["evidence"] == "model-only")
    print(f"\n方向分布：")
    print(f"  missed (疑似人工漏標)：      {by_dir['missed']}")
    print(f"  over   (疑似人工過度標註)：  {by_dir['over']}")
    print(f"  diff   (公式分歧)：          {by_dir['diff']}")
    print(f"\n證據分級（決定複查優先序）：")
    print(f"  鐵證級（詞形標記 or col18）：{n_hard}  ← 布林欄純漏，可信度最高")
    print(f"    其中 詞形含 [N] 成分標記：{n_wordmark}")
    print(f"    其中 col18 有自由文字標註：{n_col18}")
    print(f"  純模型推論（model-only）：   {n_modelonly}  ← 無人工佐證，需最審慎")
    print(f"  落在 not_sure 標註列：       {ns}  ← 標註者自己存疑")
    print(f"\n一致強度：")
    for k in sorted(by_k, reverse=True):
        tag = "（unanimous）" if k == len(FAMILIES) else ""
        print(f"  {k}/{len(FAMILIES)} 家族一致：{by_k[k]} {tag}")
    print(f"\n落在數詞位置：")
    for n in range(1, 11):
        print(f"  {n:2d}: {by_num.get(n,0)}")
    print(f"\n輸出：{OUT_TSV}")

    # 抽樣：印 top 20 供人先看
    print(f"\n{'='*78}\nTop 20（鐵證級優先）\n{'='*78}")
    print(f"{'#id':>4} {'數':>2} {'詞形':<18} {'gold':>5} {'共識':>5} {'k':>2} {'方向':<7} {'證據'}")
    for r in flags[:20]:
        print(f"{r['lang_id']:>4} {r['numeral']:>2} {r['word'][:18]:<18} "
              f"{r['gold']:>5} {r['consensus']:>5} {r['n_agree']:>2} "
              f"{r['direction']:<7} {r['evidence']}")


if __name__ == "__main__":
    main()
