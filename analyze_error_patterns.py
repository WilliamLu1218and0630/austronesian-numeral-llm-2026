"""
Per-numeral error pattern analysis for ROCLING 2026 paper.
Four analyses:
  1. Per-numeral formula confusion (gold × predicted)
  2. Compositional system consistency (quinary/subtractive coherence)
  3. Formula frequency vs error rate correlation
  4. FP hallucination formula distribution
"""

import json
import sys
import io
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RESULTS_DIR = Path(__file__).parent / "results"

MODELS_ZS = [
    ("Qwen 27B",    "qwen--qwen3.6-27b_zeroshot_noname_neutralprompt.jsonl"),
    ("Gemma 31B",   "gemma-4-31b_zeroshot_noname_neutralprompt.jsonl"),
    ("Llama 70B",   "llama-3.3-70b-versatile_zeroshot_noname_neutralprompt.jsonl"),
    ("GPT-OSS 120B","gpt-oss-120b_zeroshot_noname_neutralprompt.jsonl"),
]

MODELS_OS = [
    ("Qwen 27B",    "qwen--qwen3.6-27b_oneshot_noname_neutralprompt.jsonl"),
    ("Gemma 31B",   "gemma-4-31b_oneshot_noname_neutralprompt.jsonl"),
    ("Llama 70B",   "llama-3.3-70b-versatile_oneshot_noname_neutralprompt.jsonl"),
    ("GPT-OSS 120B","gpt-oss-120b_oneshot_noname_neutralprompt.jsonl"),
]


def load_jsonl(fname):
    path = RESULTS_DIR / fname
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def is_scored(d):
    """該格是否納入分析。無語料的格（verdict='Excluded'，gold='-'）一律排除。"""
    return d.get("verdict") != "Excluded" and d.get("gold") != "-"


def classify_operation(formula):
    """Classify a formula string into operation type."""
    if formula.isdigit():
        return "simplex"
    if "×" in formula or "*" in formula:
        return "multiplication"
    if "-" in formula:
        return "subtraction"
    if "+" in formula:
        return "addition"
    return "other"


# =========================================================================
# Analysis 1: Per-numeral formula confusion
# =========================================================================
def analysis1_formula_confusion(models, condition_label):
    print(f"\n{'='*80}")
    print(f"ANALYSIS 1: Per-numeral formula confusion ({condition_label})")
    print(f"{'='*80}")
    print("For numerals 6-9, what does the model predict when gold is X?")
    print("Aggregated across all 4 large models.\n")

    for numeral in [6, 7, 8, 9]:
        idx = numeral - 1
        confusion = defaultdict(Counter)  # confusion[gold_formula][predicted_formula] = count

        for label, fname in models:
            for rec in load_jsonl(fname):
                if rec.get("parse_error"):
                    continue
                d = rec["eval"]["details"][idx]
                if not is_scored(d):
                    continue
                gold = d["gold"]
                ai = d["ai"]
                confusion[gold][ai] += 1

        print(f"\n--- Numeral {numeral} ---")
        for gold_formula in sorted(confusion.keys(), key=lambda x: (x.isdigit(), x)):
            preds = confusion[gold_formula]
            total = sum(preds.values())
            print(f"  Gold={gold_formula:>6} (n={total:>4}): ", end="")
            top_preds = preds.most_common(8)
            parts = []
            for pred, cnt in top_preds:
                pct = 100 * cnt / total
                marker = " ✓" if pred == gold_formula else ""
                parts.append(f"{pred}={cnt}({pct:.0f}%){marker}")
            print(", ".join(parts))


# =========================================================================
# Analysis 1b: Per-model per-numeral confusion (more detailed)
# =========================================================================
def analysis1b_per_model(models, condition_label):
    print(f"\n{'='*80}")
    print(f"ANALYSIS 1b: Per-model prediction bias ({condition_label})")
    print(f"{'='*80}")
    print("For numerals 6-9 with complex gold, model prediction distribution.\n")

    for numeral in [6, 7, 8, 9]:
        idx = numeral - 1
        print(f"\n--- Numeral {numeral} ---")

        for label, fname in models:
            confusion_complex = defaultdict(Counter)
            for rec in load_jsonl(fname):
                if rec.get("parse_error"):
                    continue
                d = rec["eval"]["details"][idx]
                if not is_scored(d):
                    continue
                gold = d["gold"]
                ai = d["ai"]
                if gold == str(numeral):
                    continue  # skip simplex gold
                confusion_complex[gold][ai] += 1

            if not confusion_complex:
                continue
            print(f"  {label}:")
            for gold_f in sorted(confusion_complex.keys()):
                preds = confusion_complex[gold_f]
                total = sum(preds.values())
                correct = preds.get(gold_f, 0)
                print(f"    Gold={gold_f:>6} (n={total:>3}): correct={correct}({100*correct/total:.0f}%) | ", end="")
                wrong_preds = [(p, c) for p, c in preds.most_common(5) if p != gold_f]
                if wrong_preds:
                    print(", ".join(f"{p}={c}" for p, c in wrong_preds))
                else:
                    print("(no errors)")


# =========================================================================
# Analysis 2: Compositional system consistency
# =========================================================================
def analysis2_system_consistency(models, condition_label):
    print(f"\n{'='*80}")
    print(f"ANALYSIS 2: Compositional system consistency ({condition_label})")
    print(f"{'='*80}")

    quinary_pattern = {5: "5+1", 6: "5+2", 7: "5+3", 8: "5+4"}
    sub_pattern = {5: "10-4", 6: "10-3", 7: "10-2", 8: "10-1"}

    for pattern_name, expected_formulas in [("Quinary (5+X)", quinary_pattern),
                                             ("Subtractive (10-X)", sub_pattern)]:
        print(f"\n--- {pattern_name} ---")
        print(f"Languages where ALL of numerals {list(n+1 for n in expected_formulas)} have this pattern.\n")

        for label, fname in models:
            records = load_jsonl(fname)
            if not records:
                continue

            full_pattern_langs = []
            for rec in records:
                if rec.get("parse_error"):
                    continue
                gold = rec["gold"]
                is_full = all(gold[idx] == formula for idx, formula in expected_formulas.items())
                if is_full:
                    full_pattern_langs.append(rec)

            if not full_pattern_langs:
                print(f"  {label}: 0 languages with full pattern")
                continue

            score_dist = Counter()
            for rec in full_pattern_langs:
                correct_count = 0
                for idx in expected_formulas:
                    d = rec["eval"]["details"][idx]
                    if d["verdict"] == "TP":
                        correct_count += 1
                score_dist[correct_count] += 1

            n_langs = len(full_pattern_langs)
            print(f"  {label} (n={n_langs} languages):")
            for score in sorted(score_dist.keys(), reverse=True):
                cnt = score_dist[score]
                bar = "█" * (cnt * 40 // n_langs)
                print(f"    {score}/{len(expected_formulas)} correct: {cnt:>3} ({100*cnt/n_langs:>5.1f}%) {bar}")

            all_correct = score_dist.get(len(expected_formulas), 0)
            partial = sum(cnt for s, cnt in score_dist.items() if 0 < s < len(expected_formulas))
            none_correct = score_dist.get(0, 0)
            print(f"    → all-or-nothing ratio: {100*(all_correct+none_correct)/n_langs:.0f}% "
                  f"(all={all_correct}, none={none_correct}, partial={partial})")


# =========================================================================
# Analysis 3: Formula frequency vs error rate
# =========================================================================
def analysis3_frequency_vs_error(models, condition_label):
    print(f"\n{'='*80}")
    print(f"ANALYSIS 3: Formula frequency vs error rate ({condition_label})")
    print(f"{'='*80}")
    print("For each gold formula type: frequency across 892 languages, and average error rate.\n")

    formula_freq = Counter()
    formula_errors = defaultdict(lambda: {"correct": 0, "total": 0})

    first_model_records = load_jsonl(models[0][1])
    for rec in first_model_records:
        if rec.get("parse_error"):
            continue
        for d in rec["eval"]["details"]:
            if not is_scored(d):
                continue
            gold = d["gold"]
            if gold != str(d["numeral"]):
                formula_freq[gold] += 1

    for label, fname in models:
        for rec in load_jsonl(fname):
            if rec.get("parse_error"):
                continue
            for d in rec["eval"]["details"]:
                if not is_scored(d):
                    continue
                gold = d["gold"]
                if gold == str(d["numeral"]):
                    continue
                formula_errors[gold]["total"] += 1
                if d["verdict"] == "TP":
                    formula_errors[gold]["correct"] += 1

    print(f"{'Formula':>10} {'Freq':>5} {'Op Type':>14} {'Total':>6} {'Correct':>8} {'ErrRate':>8}")
    print("-" * 65)

    formulas_sorted = sorted(formula_freq.keys(), key=lambda f: formula_freq[f], reverse=True)
    for formula in formulas_sorted:
        freq = formula_freq[formula]
        stats = formula_errors[formula]
        total = stats["total"]
        correct = stats["correct"]
        err_rate = 100 * (1 - correct / total) if total > 0 else 0
        op_type = classify_operation(formula)
        print(f"{formula:>10} {freq:>5} {op_type:>14} {total:>6} {correct:>8} {err_rate:>7.1f}%")

    print(f"\nCorrelation check (Spearman):")
    freqs = [formula_freq[f] for f in formulas_sorted]
    errs = [100 * (1 - formula_errors[f]["correct"] / formula_errors[f]["total"])
            if formula_errors[f]["total"] > 0 else 0 for f in formulas_sorted]
    n = len(freqs)
    if n > 2:
        rank_f = _rank(freqs)
        rank_e = _rank(errs)
        d_sq = sum((rf - re)**2 for rf, re in zip(rank_f, rank_e))
        rho = 1 - (6 * d_sq) / (n * (n**2 - 1))
        print(f"  ρ = {rho:.3f} (negative = rarer formulas have higher error rates)")


def _rank(values):
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2 + 1
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


# =========================================================================
# Analysis 4: FP hallucination formula distribution
# =========================================================================
def analysis4_fp_distribution(models, condition_label):
    print(f"\n{'='*80}")
    print(f"ANALYSIS 4: FP hallucination formula distribution ({condition_label})")
    print(f"{'='*80}")
    print("When model predicts complex for simplex gold, which formulas are hallucinated?\n")

    fp_by_numeral = defaultdict(Counter)
    fp_by_model = defaultdict(Counter)
    fp_total = Counter()

    for label, fname in models:
        for rec in load_jsonl(fname):
            if rec.get("parse_error"):
                continue
            for d in rec["eval"]["details"]:
                if d["verdict"] == "FP":
                    numeral = d["numeral"]
                    ai = d["ai"]
                    fp_by_numeral[numeral][ai] += 1
                    fp_by_model[label][ai] += 1
                    fp_total[ai] += 1

    print("--- By numeral position ---")
    for numeral in range(1, 11):
        if numeral not in fp_by_numeral:
            continue
        preds = fp_by_numeral[numeral]
        total = sum(preds.values())
        print(f"  Numeral {numeral:>2} ({total:>4} FPs): ", end="")
        top = preds.most_common(6)
        print(", ".join(f"{p}={c}({100*c/total:.0f}%)" for p, c in top))

    print(f"\n--- Overall FP formula distribution ---")
    grand_total = sum(fp_total.values())
    print(f"  Total FPs: {grand_total}")
    for formula, count in fp_total.most_common(15):
        op = classify_operation(formula)
        print(f"  {formula:>10} ({op:>14}): {count:>4} ({100*count/grand_total:.1f}%)")

    print(f"\n--- By model ---")
    for label, _ in models:
        preds = fp_by_model[label]
        total = sum(preds.values())
        if total == 0:
            continue
        add = sum(c for f, c in preds.items() if classify_operation(f) == "addition")
        sub = sum(c for f, c in preds.items() if classify_operation(f) == "subtraction")
        mul = sum(c for f, c in preds.items() if classify_operation(f) == "multiplication")
        print(f"  {label:>15}: {total:>4} FPs → add={add}({100*add/total:.0f}%), "
              f"sub={sub}({100*sub/total:.0f}%), mul={mul}({100*mul/total:.0f}%)")


# =========================================================================
# Main
# =========================================================================
def main():
    print("ROCLING 2026 — Per-numeral Error Pattern Analysis")
    print("=" * 80)

    print("\n\n" + "▓" * 80)
    print("  ZERO-SHOT CONDITION (C1)")
    print("▓" * 80)
    analysis1_formula_confusion(MODELS_ZS, "zero-shot")
    analysis1b_per_model(MODELS_ZS, "zero-shot")
    analysis2_system_consistency(MODELS_ZS, "zero-shot")
    analysis3_frequency_vs_error(MODELS_ZS, "zero-shot")
    analysis4_fp_distribution(MODELS_ZS, "zero-shot")

    print("\n\n" + "▓" * 80)
    print("  ONE-SHOT CONDITION (C2)")
    print("▓" * 80)
    analysis1_formula_confusion(MODELS_OS, "one-shot")
    analysis1b_per_model(MODELS_OS, "one-shot")
    analysis2_system_consistency(MODELS_OS, "one-shot")
    analysis3_frequency_vs_error(MODELS_OS, "one-shot")
    analysis4_fp_distribution(MODELS_OS, "one-shot")


if __name__ == "__main__":
    main()
