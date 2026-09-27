"""
Compute baselines and recall/F1 for the complex class.
1. Majority-class baseline (always predict atomic)
2. String-match heuristic baseline (substring detection)
3. Precision/Recall/F1 for "complex" class from model JSONL results
"""

import csv
import json
import sys
import io
import re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RESULTS_DIR = Path(__file__).parent / "results"

# gold 建構邏輯統一由 pipeline 提供，避免兩份程式各自漂移（2026-07-22 起）。
# load_tsv() 回傳的每筆含 words / gold / not_sure / excluded。
from pipeline import load_tsv, GOLD_MISSING  # noqa: E402


def is_complex(gold_val, numeral_idx):
    """Gold value is complex if it's not just the numeral number itself."""
    return gold_val != str(numeral_idx + 1)


def string_match_heuristic(words):
    """
    Simple substring heuristic:
    For numeral N (index 2-9, i.e. numerals 3-10), check if the word contains
    substrings from smaller numerals (length >= 3 chars).
    Returns list of 10 predictions: "atomic" or "complex".
    """
    preds = ["atomic"] * 10  # numerals 1-2 always atomic

    # Normalize words for comparison (lowercase, strip)
    norm_words = [w.lower().strip() for w in words]

    for target_idx in range(2, 10):  # numerals 3-10
        target = norm_words[target_idx]
        if not target or len(target) < 3:
            continue

        # Check if target contains any substring from numerals 1-5
        # (the most common components in formulas)
        found_component = False
        for comp_idx in range(5):  # check numerals 1-5
            comp = norm_words[comp_idx]
            if not comp or len(comp) < 3:
                continue
            # Check if component (or its root, first 3+ chars) appears in target
            # Use minimum 3 chars to avoid spurious matches
            min_len = max(3, len(comp) // 2)
            for substr_len in range(len(comp), min_len - 1, -1):
                substr = comp[:substr_len]
                if substr in target and target != comp:
                    found_component = True
                    break
            if found_component:
                break

        # Also check if numeral 10's word appears as component (for subtraction)
        if not found_component and target_idx < 9:  # for 3-9, check if 10 is component
            comp10 = norm_words[9]
            if comp10 and len(comp10) >= 3:
                min_len = max(3, len(comp10) // 2)
                for substr_len in range(len(comp10), min_len - 1, -1):
                    substr = comp10[:substr_len]
                    if substr in target and target != comp10:
                        found_component = True
                        break

        if found_component:
            preds[target_idx] = "complex"

    return preds


def compute_baselines(data):
    """Compute majority-class and string-match baselines."""
    total = 0
    gold_complex = 0
    gold_atomic = 0
    majority_correct = 0
    heuristic_correct = 0
    # For complex class metrics
    heur_tp = 0  # heuristic says complex, gold is complex
    heur_fp = 0  # heuristic says complex, gold is atomic
    heur_fn = 0  # heuristic says atomic, gold is complex

    for lang in data:
        words = lang["words"]
        gold = lang["gold"]
        excluded = lang["excluded"]
        heur_preds = string_match_heuristic(words)

        not_sure = lang["not_sure"]
        skip = excluded | not_sure
        for i in range(10):
            if (i + 1) in skip:
                continue
            total += 1
            g_complex = is_complex(gold[i], i)

            if g_complex:
                gold_complex += 1
            else:
                gold_atomic += 1
                majority_correct += 1  # majority always says atomic

            # Heuristic evaluation
            h_complex = (heur_preds[i] == "complex")
            if not g_complex and not h_complex:
                heuristic_correct += 1  # TN
            elif g_complex and h_complex:
                heuristic_correct += 1  # TP
                heur_tp += 1
            elif not g_complex and h_complex:
                heur_fp += 1  # FP
            elif g_complex and not h_complex:
                heur_fn += 1  # FN

    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS (n={total} numeral tokens from {len(data)} languages)")
    print(f"{'='*60}")
    print(f"\nGold distribution:")
    print(f"  Atomic (simplex): {gold_atomic} ({100*gold_atomic/total:.1f}%)")
    print(f"  Complex:          {gold_complex} ({100*gold_complex/total:.1f}%)")
    print(f"\n--- Majority-class baseline (always predict atomic) ---")
    print(f"  Accuracy: {100*majority_correct/total:.1f}%")
    print(f"  Complex recall: 0.0%  (never predicts complex)")
    print(f"\n--- String-match heuristic baseline ---")
    print(f"  Accuracy: {100*heuristic_correct/total:.1f}%")
    heur_prec = heur_tp / (heur_tp + heur_fp) * 100 if (heur_tp + heur_fp) > 0 else 0
    heur_rec = heur_tp / (heur_tp + heur_fn) * 100 if (heur_tp + heur_fn) > 0 else 0
    heur_f1 = 2 * heur_prec * heur_rec / (heur_prec + heur_rec) if (heur_prec + heur_rec) > 0 else 0
    print(f"  Complex class — P: {heur_prec:.1f}%  R: {heur_rec:.1f}%  F1: {heur_f1:.1f}%")
    print(f"  (TP={heur_tp}, FP={heur_fp}, FN={heur_fn})")

    return total, gold_complex, gold_atomic


def compute_model_metrics(jsonl_path, total_expected=None, scored_by_id=None):
    """Compute precision/recall/F1 for complex class from a model's JSONL.

    parse_error 的語言以該語言「實際可計分的格數」計入分母（非固定 10），
    與 evaluate() 排除無語料格的規則一致。verdict='Excluded' 不落任何桶，自然被略過。
    """
    tp = fp = fn = tn = echo = wrong = parse_err = 0
    scored_by_id = scored_by_id or {}

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("parse_error"):
                parse_err += scored_by_id.get(rec["id"], 10)
                continue
            for d in rec["eval"]["details"]:
                v = d["verdict"]
                if v == "TP":
                    tp += 1
                elif v == "TN":
                    tn += 1
                elif v == "FP":
                    fp += 1
                elif v == "FN":
                    fn += 1
                elif v == "Wrong":
                    wrong += 1
                elif v == "Echo":
                    echo += 1

    total = tp + tn + fp + fn + wrong + echo + parse_err
    acc = (tp + tn) / total * 100 if total else 0

    # Complex class: gold says complex → TP + FN + Wrong
    # Model predicted complex → TP + FP + Wrong
    gold_complex_total = tp + fn + wrong
    pred_complex_total = tp + fp + wrong  # model said some formula (right or wrong)

    # For recall of complex: correctly identified as complex (TP only)
    # For precision: of all model "complex" predictions, how many match gold exactly (TP)
    # Note: "Wrong" means model predicted a formula but wrong one — it detected complexity but got formula wrong
    # We can report two versions:
    #   Strict: only TP counts as correct complex detection
    #   Lenient: TP + Wrong both count as "detected complexity" (just wrong formula)

    # Strict
    prec_strict = tp / (tp + fp + wrong) * 100 if (tp + fp + wrong) > 0 else 0
    rec_strict = tp / (tp + fn + wrong) * 100 if (tp + fn + wrong) > 0 else 0
    f1_strict = 2 * prec_strict * rec_strict / (prec_strict + rec_strict) if (prec_strict + rec_strict) > 0 else 0

    # Lenient (detection): did the model correctly identify that it's complex?
    # TP + Wrong = model said formula AND gold says complex
    detected = tp + wrong  # model said complex AND gold is complex
    pred_any_formula = tp + fp + wrong  # model said any formula
    gold_has_formula = tp + fn + wrong  # gold says complex

    prec_detect = detected / pred_any_formula * 100 if pred_any_formula > 0 else 0
    rec_detect = detected / gold_has_formula * 100 if gold_has_formula > 0 else 0
    f1_detect = 2 * prec_detect * rec_detect / (prec_detect + rec_detect) if (prec_detect + rec_detect) > 0 else 0

    return {
        "total": total,
        "acc": acc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "wrong": wrong, "echo": echo, "pe": parse_err,
        "prec_strict": prec_strict,
        "rec_strict": rec_strict,
        "f1_strict": f1_strict,
        "prec_detect": prec_detect,
        "rec_detect": rec_detect,
        "f1_detect": f1_detect,
        "gold_complex": tp + fn + wrong,
    }


def main():
    print("Loading TSV...")
    data = load_tsv()
    print(f"  Loaded {len(data)} languages")

    # Filter to the language set used in the paper (matching the JSONL files)
    # Load one JSONL to get the ID set
    ref_jsonl = RESULTS_DIR / "gpt-oss-120b_zeroshot_noname_neutralprompt.jsonl"
    paper_ids = set()
    with open(ref_jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            paper_ids.add(rec["id"])
    data_filtered = [d for d in data if d["id"] in paper_ids]
    print(f"  Filtered to {len(data_filtered)} languages (matching the paper)")

    # 每語言實際可計分格數（供 parse_error 的分母使用）：扣掉 excluded + not_sure
    scored_by_id = {d["id"]: 10 - len(d["excluded"] | d["not_sure"]) for d in data_filtered}
    n_excluded = sum(len(d["excluded"]) for d in data_filtered)
    n_not_sure = sum(len(d["not_sure"] - d["excluded"]) for d in data_filtered)
    n_total = len(data_filtered) * 10
    print(f"  Excluded cells (no word form): {n_excluded} / {n_total}")
    print(f"  Excluded cells (not sure):     {n_not_sure} / {n_total}")
    print(f"  Scored cells:                  {n_total - n_excluded - n_not_sure}")

    compute_baselines(data_filtered)

    # Model metrics
    print(f"\n{'='*60}")
    print("MODEL COMPLEX-CLASS METRICS (zero-shot, neutral prompt)")
    print(f"{'='*60}")
    print(f"\n{'Model':<20} {'Acc%':>5} {'P_strict':>8} {'R_strict':>8} {'F1_strict':>9} {'P_detect':>8} {'R_detect':>8} {'F1_detect':>9}")
    print("-" * 95)

    models = [
        ("Llama 8B", "llama-3.1-8b-instant_zeroshot_noname_neutralprompt.jsonl"),
        ("Qwen 27B", "qwen--qwen3.6-27b_zeroshot_noname_neutralprompt.jsonl"),
        ("Gemma 31B", "gemma-4-31b_zeroshot_noname_neutralprompt.jsonl"),
        ("Llama 70B", "llama-3.3-70b-versatile_zeroshot_noname_neutralprompt.jsonl"),
        ("GPT-OSS 120B", "gpt-oss-120b_zeroshot_noname_neutralprompt.jsonl"),
    ]

    for label, fname in models:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"  {label}: FILE NOT FOUND")
            continue
        m = compute_model_metrics(path, scored_by_id=scored_by_id)
        print(f"  {label:<18} {m['acc']:>5.1f} {m['prec_strict']:>8.1f} {m['rec_strict']:>8.1f} {m['f1_strict']:>9.1f} {m['prec_detect']:>8.1f} {m['rec_detect']:>8.1f} {m['f1_detect']:>9.1f}")
        print(f"    (TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, Wrong={m['wrong']}, Echo={m['echo']}, GoldComplex={m['gold_complex']})")

    # Also one-shot
    print(f"\n{'='*60}")
    print("MODEL COMPLEX-CLASS METRICS (one-shot, neutral prompt)")
    print(f"{'='*60}")
    print(f"\n{'Model':<20} {'Acc%':>5} {'P_strict':>8} {'R_strict':>8} {'F1_strict':>9} {'P_detect':>8} {'R_detect':>8} {'F1_detect':>9}")
    print("-" * 95)

    models_os = [
        ("Llama 8B", "llama-3.1-8b-instant_oneshot_noname_neutralprompt.jsonl"),
        ("Qwen 27B", "qwen--qwen3.6-27b_oneshot_noname_neutralprompt.jsonl"),
        ("Gemma 31B", "gemma-4-31b_oneshot_noname_neutralprompt.jsonl"),
        ("Llama 70B", "llama-3.3-70b-versatile_oneshot_noname_neutralprompt.jsonl"),
        ("GPT-OSS 120B", "gpt-oss-120b_oneshot_noname_neutralprompt.jsonl"),
    ]

    for label, fname in models_os:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"  {label}: FILE NOT FOUND")
            continue
        m = compute_model_metrics(path, scored_by_id=scored_by_id)
        print(f"  {label:<18} {m['acc']:>5.1f} {m['prec_strict']:>8.1f} {m['rec_strict']:>8.1f} {m['f1_strict']:>9.1f} {m['prec_detect']:>8.1f} {m['rec_detect']:>8.1f} {m['f1_detect']:>9.1f}")
        print(f"    (TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, Wrong={m['wrong']}, Echo={m['echo']}, GoldComplex={m['gold_complex']})")


if __name__ == "__main__":
    main()
