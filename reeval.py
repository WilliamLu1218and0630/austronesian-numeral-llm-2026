"""
用目前的 gold 標準重新評分既有結果（不呼叫 API）。

gold 改版後跑這支即可，不需重打模型。只覆寫每筆的 gold / eval 兩個欄位，
其餘（raw / ai_output / error / parse_error / format_noncompliant）原樣保留。

用法：
  python reeval.py --dry-run      # 只報告差異，不寫檔
  python reeval.py                # 就地更新 results/*.jsonl
"""

import argparse
import json
import shutil
from pathlib import Path

from pipeline import load_tsv, evaluate, parse_response

RESULTS_DIR = Path(__file__).parent / "results"


def reeval_file(path: Path, by_id: dict, dry_run: bool):
    old_correct = old_total = 0
    new_correct = new_total = 0
    n = n_err = n_pe = n_missing = 0
    out_lines = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1

            # API error 或無 gold 對照的語言：原樣保留
            if rec.get("error"):
                n_err += 1
                out_lines.append(rec)
                continue
            lang = by_id.get(rec.get("id"))
            if lang is None:
                n_missing += 1
                out_lines.append(rec)
                continue

            ev_old = rec.get("eval")
            if ev_old:
                old_correct += ev_old["score"]
                old_total += ev_old["total"]

            # parse_error 的記錄沒有可評分的輸出，只更新 gold 供下游對照
            if rec.get("parse_error") or not rec.get("ai_output"):
                n_pe += 1
                rec["gold"] = lang["gold"]
                out_lines.append(rec)
                continue

            ev = evaluate(rec["ai_output"], lang["gold"], lang["not_sure"],
                          lang["words"], lang["excluded"])
            rec["gold"] = lang["gold"]
            rec["eval"] = ev
            new_correct += ev["score"]
            new_total += ev["total"]
            out_lines.append(rec)

    if not dry_run:
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in out_lines:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(path)

    old_acc = old_correct / old_total * 100 if old_total else 0
    new_acc = new_correct / new_total * 100 if new_total else 0
    return dict(n=n, n_err=n_err, n_pe=n_pe, n_missing=n_missing,
                old_acc=old_acc, new_acc=new_acc,
                old_total=old_total, new_total=new_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只報告，不寫檔")
    ap.add_argument("--pattern", default="*_neutralprompt.jsonl")
    ap.add_argument("--backup", metavar="DIR", help="寫檔前先把 results/ 複製到這裡")
    args = ap.parse_args()

    data = load_tsv()
    by_id = {d["id"]: d for d in data}
    n_ex = sum(len(d["excluded"]) for d in data)
    print(f"Gold loaded: {len(data)} languages, {n_ex} excluded cells (no word form)")

    if args.backup and not args.dry_run:
        dest = Path(args.backup)
        if dest.exists():
            print(f"Backup dir already exists, skipping: {dest}")
        else:
            shutil.copytree(RESULTS_DIR, dest)
            print(f"Backed up results/ -> {dest}")

    paths = sorted(RESULTS_DIR.glob(args.pattern))
    print(f"\n{'file':<58} {'n':>4} {'old acc':>8} {'new acc':>8} {'delta':>7} {'denom':>13}")
    print("-" * 104)
    for p in paths:
        r = reeval_file(p, by_id, args.dry_run)
        delta = r["new_acc"] - r["old_acc"]
        denom = f"{r['old_total']}->{r['new_total']}"
        print(f"{p.name:<58} {r['n']:>4} {r['old_acc']:>7.2f}% {r['new_acc']:>7.2f}% "
              f"{delta:>+6.2f} {denom:>13}")

    if args.dry_run:
        print("\n(dry-run: 沒有寫入任何檔案)")


if __name__ == "__main__":
    main()
