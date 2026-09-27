"""
產出論文引用的所有數據，一次到位。

gold 或評分規則改版後跑這支，把輸出對照 paper.tex / figures/*.tex 更新。
輸出寫到 results/_paper_numbers.txt。

用法：python paper_numbers.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from pipeline import load_tsv

RESULTS = Path(__file__).parent / "results"

ZS = [
    ("Llama 8B", "llama-3.1-8b-instant_zeroshot_noname_neutralprompt.jsonl"),
    ("Qwen 27B", "qwen--qwen3.6-27b_zeroshot_noname_neutralprompt.jsonl"),
    ("Gemma 31B", "gemma-4-31b_zeroshot_noname_neutralprompt.jsonl"),
    ("Llama 70B", "llama-3.3-70b-versatile_zeroshot_noname_neutralprompt.jsonl"),
    ("GPT-OSS 120B", "gpt-oss-120b_zeroshot_noname_neutralprompt.jsonl"),
]
OS = [
    ("Llama 8B", "llama-3.1-8b-instant_oneshot_noname_neutralprompt.jsonl"),
    ("Qwen 27B", "qwen--qwen3.6-27b_oneshot_noname_neutralprompt.jsonl"),
    ("Gemma 31B", "gemma-4-31b_oneshot_noname_neutralprompt.jsonl"),
    ("Llama 70B", "llama-3.3-70b-versatile_oneshot_noname_neutralprompt.jsonl"),
    ("GPT-OSS 120B", "gpt-oss-120b_oneshot_noname_neutralprompt.jsonl"),
]


def load(fname):
    p = RESULTS / fname
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def scored_cells(rec):
    """該筆可計分的格（跳過 Excluded / NotSure）。"""
    for d in rec["eval"]["details"]:
        if d.get("verdict") in ("Excluded", "NotSure"):
            continue
        yield d


def model_stats(fname, scored_by_id):
    recs = load(fname)
    if not recs:
        return None
    v = Counter()
    by_numeral = Counter()
    correct = total = 0
    pe = 0
    n_lang = 0
    for rec in recs:
        n_lang += 1
        # API 重試失敗（error）也帶 parse_error=True，與解析失敗一同計入分母，
        # 沿用改版前的處理方式，避免混入未經討論的方法學變動。
        if rec.get("parse_error"):
            pe += 1
            v["ParseErr"] += scored_by_id.get(rec["id"], 10)
            continue
        for d in scored_cells(rec):
            total += 1
            v[d["verdict"]] += 1
            if d["correct"]:
                correct += 1
            else:
                by_numeral[d["numeral"]] += 1
    denom = total + v["ParseErr"]
    return dict(
        n_lang=n_lang, pe=pe,
        acc=(correct / denom * 100) if denom else 0,
        acc_parsed=(correct / total * 100) if total else 0,
        correct=correct, total=total, denom=denom,
        tp=v["TP"], tn=v["TN"], fp=v["FP"], fn=v["FN"],
        wrong=v["Wrong"], echo=v["Echo"], parse_cells=v["ParseErr"],
        by_numeral=by_numeral,
    )


def main():
    out = []

    def p(s=""):
        print(s)
        out.append(s)

    data = load_tsv()
    ref_ids = {r["id"] for r in load(ZS[4][1])}
    data = [d for d in data if d["id"] in ref_ids]
    by_id = {d["id"]: d for d in data}
    scored_by_id = {d["id"]: 10 - len(d["excluded"] | d["not_sure"]) for d in data}

    n_lang = len(data)
    n_cells_raw = n_lang * 10
    n_excluded = sum(len(d["excluded"]) for d in data)
    n_not_sure = sum(len(d["not_sure"] - d["excluded"]) for d in data)
    n_skip = n_excluded + n_not_sure
    n_scored = n_cells_raw - n_skip

    p("=" * 78)
    p("§ 資料規模")
    p("=" * 78)
    p(f"  語言數                     {n_lang}")
    p(f"  格數（10×語言）            {n_cells_raw}")
    p(f"  排除（無語料）             {n_excluded}")
    p(f"  排除（不確定）             {n_not_sure}")
    p(f"  實際計分格數               {n_scored} ({n_scored/n_cells_raw*100:.1f}%)")
    p(f"  含排除格的語言數           {sum(1 for d in data if d['excluded'] | d['not_sure'])}")

    # gold 分布
    atomic = complex_ = 0
    all_atomic_langs = 0
    formula_freq = Counter()
    for d in data:
        has_complex = False
        skip = d["excluded"] | d["not_sure"]
        for i in range(10):
            if (i + 1) in skip:
                continue
            g = d["gold"][i]
            if g == str(i + 1):
                atomic += 1
            else:
                complex_ += 1
                formula_freq[f"{i+1}={g}"] += 1
                has_complex = True
        if not has_complex:
            all_atomic_langs += 1

    p()
    p("=" * 78)
    p("§ Gold 分布（計分格）")
    p("=" * 78)
    p(f"  單純詞 atomic              {atomic} ({atomic/n_scored*100:.1f}%)")
    p(f"  複雜詞 complex             {complex_} ({complex_/n_scored*100:.1f}%)")
    p(f"  1-10 全單純詞的語言        {all_atomic_langs} ({all_atomic_langs/n_lang*100:.0f}%)")
    p(f"  至少一個複雜詞的語言       {n_lang - all_atomic_langs}")
    p()
    p("  各構詞公式出現次數（tab:gold-dist）：")
    for k, c in formula_freq.most_common():
        p(f"    {k:<20} {c}")

    # 模型表
    for label, models in (("zero-shot", ZS), ("one-shot", OS)):
        p()
        p("=" * 78)
        p(f"§ 模型指標（{label}, neutral prompt）")
        p("=" * 78)
        p(f"  {'model':<14}{'n':>5}{'acc%':>7}{'TP':>6}{'TN':>6}{'FP':>6}{'FN':>6}"
          f"{'Wrong':>7}{'Echo':>7}{'PE':>4}")
        p("  " + "-" * 72)
        for name, fname in models:
            m = model_stats(fname, scored_by_id)
            if not m:
                p(f"  {name:<14}  (檔案不存在)")
                continue
            p(f"  {name:<14}{m['n_lang']:>5}{m['acc']:>7.1f}{m['tp']:>6}{m['tn']:>6}"
              f"{m['fp']:>6}{m['fn']:>6}{m['wrong']:>7}{m['echo']:>7}{m['pe']:>4}")

        p()
        p(f"  各數詞錯誤數（fig_pernumeral, {label}）：")
        p(f"  {'model':<14}" + "".join(f"{i:>6}" for i in range(1, 11)))
        for name, fname in models:
            m = model_stats(fname, scored_by_id)
            if not m:
                continue
            p(f"  {name:<14}" + "".join(f"{m['by_numeral'][i]:>6}" for i in range(1, 11)))

    Path(RESULTS / "_paper_numbers.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\n→ 已寫入 {RESULTS / '_paper_numbers.txt'}")


if __name__ == "__main__":
    main()
