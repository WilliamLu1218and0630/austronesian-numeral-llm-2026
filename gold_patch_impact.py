"""
補丁影響分析：補 GOLD_PATCH 後，各模型在這些 cell 的評分怎麼變。
雙向誠實統計：
  上升＝模型原本答了公式被判 FP/Wrong/Echo → 補後 correct（證明 gold 漏、模型對）
  下降＝模型原本答 atomic 被判 TN → 補後變 FN（模型沒偵測到，gold 現在有了）
不改任何檔，只讀 results/*_neutralprompt.jsonl 的既存 verdict。
"""

import csv
import glob
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "results"))
import pipeline as P
from _gold_patch_proposal import GOLD_PATCH

RESULTS = Path(__file__).parent / "results"


def eq(a, b):
    a, b = P.normalize_formula(a), P.normalize_formula(b)
    return a == b or P._commutative_equal(a, b)


def main():
    files = [Path(p) for p in glob.glob(str(RESULTS / "*_neutralprompt.jsonl"))
             if "_contaminated" not in p]
    # patch cells: (id, i, new_formula)
    cells = [(lid, num - 1, f) for lid, d in GOLD_PATCH.items() for num, f in d.items()]

    old_dist = Counter()
    new_correct = 0
    old_correct = 0
    up_by_old = Counter()    # 非correct→correct，按原 verdict
    down = 0                 # correct→非correct（TN→FN）
    pairs = 0                # (model,cell) 總數

    per_model = Counter()    # 每模型淨 correct 變化

    for fp in files:
        model = fp.name.replace("_zeroshot_noname_neutralprompt.jsonl", "") \
                       .replace("_oneshot_noname_neutralprompt.jsonl", "")
        cond = "0shot" if "zeroshot" in fp.name else "1shot"
        by_id = {}
        with open(fp, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if not rec.get("parse_error") and "eval" in rec:
                    by_id[rec["id"]] = rec

        for lid, i, new_f in cells:
            rec = by_id.get(lid)
            if not rec:
                continue
            det = rec["eval"]["details"][i]
            old_v = det["verdict"]
            old_ok = det["correct"]
            ai = det["ai"]
            new_ok = eq(ai, new_f)

            pairs += 1
            old_dist[old_v] += 1
            old_correct += int(old_ok)
            new_correct += int(new_ok)
            if not old_ok and new_ok:
                up_by_old[old_v] += 1
                per_model[f"{model} ({cond})"] += 1
            elif old_ok and not new_ok:
                down += 1
                per_model[f"{model} ({cond})"] -= 1

    print(f"{'='*70}")
    print(f"補丁影響分析（GOLD_PATCH：{len(GOLD_PATCH)} 語言，{len(cells)} cells）")
    print(f"{'='*70}")
    print(f"模型檔（neutralprompt）：{len(files)}")
    print(f"(model,cell) 配對總數：{pairs}")
    print(f"\n這些 cell 上，補丁前後的 correct 數：")
    print(f"  補丁前 correct：{old_correct}")
    print(f"  補丁後 correct：{new_correct}   （淨 {new_correct-old_correct:+d}）")
    print(f"\n上升（原被判錯 → 補後對）：{sum(up_by_old.values())}")
    for v, n in up_by_old.most_common():
        tag = {"FP": "幻覺pattern→其實對(gold漏)", "Wrong": "公式錯→補後對",
               "Echo": "抄詞形→補後對", "FN": "漏pattern→補後對"}.get(v, v)
        print(f"    {v:6s} → correct：{n}   [{tag}]")
    print(f"\n下降（原 atomic 判對 TN → 補後 FN，模型沒偵測到）：{down}")
    print(f"\n原 verdict 分布 on these cells：")
    for v, n in old_dist.most_common():
        print(f"    {v:6s}: {n}")

    print(f"\n各模型淨 correct 變化（正＝評分變好，即原本被低估）：")
    for m, n in sorted(per_model.items(), key=lambda x: -x[1]):
        print(f"    {m:34s} {n:+d}")


if __name__ == "__main__":
    main()
