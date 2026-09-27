"""
比對改版前後的金標準，量化差異來源。

舊 gold（data/gold_legacy.tsv）由改版前的布林欄推導邏輯物化而來，僅供追溯；
新 gold 由 data/gold_sheet.tsv 的『Gold 1-10』欄直接讀取，為現行唯一權威。

用法：python gold_diff.py
"""

import csv
from collections import Counter
from pathlib import Path

import pipeline as P

LEGACY = Path(__file__).parent / "data" / "gold_legacy.tsv"


def load_legacy():
    out = {}
    with open(LEGACY, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[int(r["id"])] = [x.strip() for x in r["gold_1_10"].split(",")]
    return out


def classify(old, new, excluded):
    if new == P.GOLD_MISSING:
        return "排除（無語料）"
    if old == new:
        return None
    old_atomic = old.isdigit()
    new_atomic = new.isdigit()
    if old_atomic and not new_atomic:
        return "補標（單純詞 → 複雜詞）"
    if not old_atomic and new_atomic:
        return "撤標（複雜詞 → 單純詞）"
    # 兩者皆為公式：是否為遞迴展開的加深
    if new.count("+") > old.count("+"):
        return "遞迴展開（分解加深）"
    return "公式改判"


def main():
    legacy = load_legacy()
    new = {d["id"]: d for d in P.load_tsv()}

    kinds = Counter()
    examples = {}
    for lid, old_gold in legacy.items():
        d = new.get(lid)
        if d is None:
            kinds["語言已從新表移除"] += 1
            continue
        for i in range(10):
            k = classify(old_gold[i], d["gold"][i], d["excluded"])
            if k is None:
                continue
            kinds[k] += 1
            examples.setdefault(k, []).append(
                (lid, d["name"], i + 1, old_gold[i], d["gold"][i]))

    total = sum(kinds.values())
    print(f"=== 金標準改版差異（共 {total} 格）===\n")
    for k, c in kinds.most_common():
        print(f"  {k:<22} {c:>4}")

    print("\n=== 各類範例 ===")
    for k, rows in examples.items():
        if k == "排除（無語料）":
            continue
        print(f"\n--- {k}（{len(rows)} 格）---")
        for lid, name, num, o, n in rows[:6]:
            print(f"  #{lid:<4} {name[:22]:<22} 數詞{num:>2}: {o!r} -> {n!r}")
        if len(rows) > 6:
            print(f"  …另 {len(rows)-6} 筆")


if __name__ == "__main__":
    main()
