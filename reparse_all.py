"""一鍵重新 reparse 所有條件（不打 API，只用新 gold 重評）。"""
import subprocess, sys, os

os.environ.setdefault("GROQ_API_KEY", "dummy")
os.environ.setdefault("CEREBRAS_API_KEY", "dummy")

MODELS = [
    "llama-3.1-8b-instant",
    "qwen/qwen3.6-27b",
    "gemma-4-31b",
    "gpt-oss-120b",
    "llama-3.3-70b-versatile",
]
CONDITIONS = ["zeroshot_noname", "oneshot_noname"]

for m in MODELS:
    for c in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"  {m}  {c}")
        print(f"{'='*60}")
        subprocess.run(
            [sys.executable, "pipeline.py",
             "--model", m, "--condition", c,
             "--prompt-variant", "neutral", "--reparse"],
        )

print("\n\n=== 全部 reparse 完成，接著跑 paper_numbers.py ===\n")
subprocess.run([sys.executable, "paper_numbers.py"])
print("\n=== 接著跑 compute_baselines.py ===\n")
subprocess.run([sys.executable, "compute_baselines.py"])
