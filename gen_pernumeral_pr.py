"""
Generate per-numeral Precision/Recall figure (fig_pernumeral.tex/.pdf).

Replaces the old per-numeral accuracy figure.
Also outputs the 5+1 error rates for §4.4 inline example.

Usage: python gen_pernumeral_pr.py
"""

import json
import io
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pipeline import load_tsv, GOLD_MISSING, evaluate

RESULTS = Path(__file__).parent / "results"
FIGURES = Path(__file__).parent / "figures"

MODELS_ZS = [
    ("Qwen 27B",     "qwen--qwen3.6-27b_zeroshot_noname_neutralprompt.jsonl"),
    ("Llama 70B",    "llama-3.3-70b-versatile_zeroshot_noname_neutralprompt.jsonl"),
    ("Gemma 31B",    "gemma-4-31b_zeroshot_noname_neutralprompt.jsonl"),
    ("GPT-OSS 120B", "gpt-oss-120b_zeroshot_noname_neutralprompt.jsonl"),
]


def load_results(fname):
    p = RESULTS / fname
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def compute_per_numeral_pr(data_by_id, fname):
    recs = load_results(fname)
    if not recs:
        return None

    counts = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "Wrong": 0})

    for rec in recs:
        if rec.get("parse_error"):
            continue
        for d in rec["eval"]["details"]:
            if d.get("verdict") in ("Excluded", "NotSure"):
                continue
            n = d["numeral"]
            v = d["verdict"]
            if v in ("TP", "FP", "FN", "TN", "Wrong"):
                counts[n][v] += 1
            elif v == "Echo":
                gold = d["gold"]
                if gold != str(n):
                    counts[n]["FN"] += 1
                else:
                    counts[n]["TN"] += 1

    result = {}
    for n in range(1, 11):
        c = counts[n]
        tp, fp, fn, w = c["TP"], c["FP"], c["FN"], c["Wrong"]
        p = (tp / (tp + fp + w) * 100) if (tp + fp + w) > 0 else None
        r = (tp / (tp + fn + w) * 100) if (tp + fn + w) > 0 else None
        result[n] = {"TP": tp, "FP": fp, "FN": fn, "TN": c["TN"],
                     "Wrong": w, "P": p, "R": r}
    return result


def compute_formula_error_rate(data_by_id, fname, target_numeral, target_gold):
    """Count error rate for a specific gold formula."""
    recs = load_results(fname)
    if not recs:
        return None
    total = 0
    errors = 0
    for rec in recs:
        if rec.get("parse_error"):
            continue
        lang = data_by_id.get(rec["id"])
        if not lang:
            continue
        idx = target_numeral - 1
        if target_numeral in lang["excluded"] or target_numeral in lang["not_sure"]:
            continue
        if lang["gold"][idx] != target_gold:
            continue
        total += 1
        for d in rec["eval"]["details"]:
            if d["numeral"] == target_numeral:
                if not d.get("correct", False):
                    errors += 1
                break
    return errors, total


def gen_tex(all_pr):
    """Generate PGFplots .tex for two-panel P/R figure."""
    colors = {
        "Qwen 27B":     "okorange",
        "Llama 70B":    "okblue",
        "Gemma 31B":    "okvermil",
        "GPT-OSS 120B": "okgreen",
    }
    marks = {
        "Qwen 27B":     "diamond*",
        "Llama 70B":    "*",
        "Gemma 31B":    "square*",
        "GPT-OSS 120B": "triangle*",
    }
    mark_sizes = {
        "Qwen 27B":     "2pt",
        "Llama 70B":    "1.6pt",
        "Gemma 31B":    "1.5pt",
        "GPT-OSS 120B": "2.1pt",
    }

    lines = []
    lines.append(r"""\documentclass[border=2pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usepackage{fontspec}
\usepackage{xeCJK}
\IfFontExistsTF{AR PL UKai TW}{%
  \setCJKmainfont[AutoFakeBold=2.5]{AR PL UKai TW}%
}{%
  \setCJKmainfont[AutoFakeBold=2.5]{DFKai-SB}%
}
\definecolor{okblue}{HTML}{0072B2}
\definecolor{okorange}{HTML}{E69F00}
\definecolor{okvermil}{HTML}{D55E00}
\definecolor{okgreen}{HTML}{009E73}
\setlength{\columnwidth}{7.7cm}
\pgfplotsset{
  prfig/.style={
    width=\columnwidth,
    height=4.0cm,
    tick label style={font=\scriptsize},
    label style={font=\small},
    legend style={font=\tiny, draw=none, fill=white, fill opacity=0.85,
                  text opacity=1, legend columns=2, column sep=3pt},
    ymajorgrids,
    grid style={gray!25},
    axis line style={gray!50},
    tick align=outside,
    tick pos=left,
    xmin=5.5, xmax=10.5,
    ymin=20, ymax=100,
    xtick={6,7,8,9,10},
    ytick={20,40,60,80,100},
  }
}
\begin{document}
\begin{tikzpicture}""")

    # Panel (a): Precision
    lines.append(r"""\begin{axis}[
  prfig,
  ylabel={精確率 $P$（\%）},
  xticklabels={},
  at={(0,4.2cm)},
  legend pos=south west,
]""")
    for name, _ in MODELS_ZS:
        pr = all_pr[name]
        coords = " ".join(f"({n},{pr[n]['P']:.1f})" for n in range(6, 11) if pr[n]["P"] is not None)
        lines.append(
            f"\\addplot[{colors[name]}, semithick, mark={marks[name]}, mark size={mark_sizes[name]}]"
        )
        lines.append(f"  coordinates {{{coords}}};")
    lines.append(r"\legend{Qwen 27B, Llama 70B, Gemma 31B, GPT-OSS 120B}")
    lines.append(r"\end{axis}")

    # Panel (b): Recall
    lines.append(r"""\begin{axis}[
  prfig,
  ylabel={召回率 $R$（\%）},
  xlabel={數詞},
  at={(0,0)},
]""")
    for name, _ in MODELS_ZS:
        pr = all_pr[name]
        coords = " ".join(f"({n},{pr[n]['R']:.1f})" for n in range(6, 11) if pr[n]["R"] is not None)
        lines.append(
            f"\\addplot[{colors[name]}, semithick, mark={marks[name]}, mark size={mark_sizes[name]}]"
        )
        lines.append(f"  coordinates {{{coords}}};")
    lines.append(r"\end{axis}")

    lines.append(r"""\end{tikzpicture}
\end{document}""")

    return "\n".join(lines)


def main():
    data = load_tsv()
    ref_recs = load_results(MODELS_ZS[-1][1])
    ref_ids = {r["id"] for r in ref_recs}
    data = [d for d in data if d["id"] in ref_ids]
    data_by_id = {d["id"]: d for d in data}

    print(f"Languages: {len(data)}")

    # Per-numeral P/R
    all_pr = {}
    for name, fname in MODELS_ZS:
        pr = compute_per_numeral_pr(data_by_id, fname)
        all_pr[name] = pr
        print(f"\n{name}:")
        print(f"  {'N':>3} {'TP':>5} {'FP':>5} {'FN':>5} {'W':>5} {'TN':>5} {'P':>7} {'R':>7}")
        for n in range(1, 11):
            c = pr[n]
            p_str = f"{c['P']:.1f}" if c['P'] is not None else "n/a"
            r_str = f"{c['R']:.1f}" if c['R'] is not None else "n/a"
            print(f"  {n:>3} {c['TP']:>5} {c['FP']:>5} {c['FN']:>5} {c['Wrong']:>5} {c['TN']:>5} {p_str:>7} {r_str:>7}")

    # 5+1 error rates
    print("\n" + "=" * 60)
    print("5+1 error rates (numeral 6, gold='5+1'):")
    for name, fname in MODELS_ZS:
        result = compute_formula_error_rate(data_by_id, fname, 6, "5+1")
        if result:
            errors, total = result
            print(f"  {name}: {errors}/{total} = {errors/total*100:.1f}%")

    # Generate .tex
    tex = gen_tex(all_pr)
    tex_path = FIGURES / "fig_pernumeral.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"\n-> {tex_path}")


if __name__ == "__main__":
    main()
