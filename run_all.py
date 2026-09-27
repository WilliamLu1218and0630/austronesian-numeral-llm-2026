"""
ROCLING 2026 — 平行批次執行器

Groq 的 RPD 是「每模型」獨立計算，所以核心規模軸的六個模型可以同一天平行跑完，
不必一天一個模型排隊。每個模型一天最多跑一個條件（892 語言 < 1000 RPD），
llama-3.1-8b 例外（14400 RPD，可多條件）。

用法：
  python run_all.py --condition zeroshot_noname                # 跑核心六模型（背景平行）
  python run_all.py --condition oneshot_noname --variant neutral
  python run_all.py --condition zeroshot_noname --end 20       # 小樣本試跑
  python run_all.py --plan                                     # 只印排程estimate，不執行
  python run_all.py --status                                   # 看各結果檔進度
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# 注意：不要在這裡包 sys.stdout。匯入 pipeline 時它會自行換成 UTF-8 wrapper，
# 重複包裝會讓前一層被 GC 時關掉底層 buffer（ValueError: I/O operation on closed file）。

HERE = Path(__file__).parent
PIPELINE = HERE / "pipeline.py"
RESULTS = HERE / "results"

sys.path.insert(0, str(HERE))
from pipeline import MODELS, load_tsv  # noqa: E402  (匯入時會設好 UTF-8 stdout)

# 核心規模軸：8B → 31B → 120B
# 2026-07-16 改版：Groq 的隱藏 TPD 綁死大模型（70B=100K/day＝320 語言/天，892 語言要 3 天），
# 改用 Cerebras（1M tokens/day）跑中大型模型。8B 留在 Groq（500K TPD 夠用且已跑完）。
CORE_MODELS = [
    "llama-3.1-8b-instant",   # 8B   — Groq，zeroshot 已完成 892
    "gemma-4-31b",            # 31B  — Cerebras，實測 98%、297 tok/筆 → 3367 語言/天
    "gpt-oss-120b",           # 120B — Cerebras，實測 78%、801 tok/筆 → 1248 語言/天
]
# 已收集的 Groq 資料（受 TPD 限制未跑完，論文可當補充或小樣本對照）
GROQ_SUPPLEMENTARY = [
    "qwen/qwen3-32b",           # 892 完成（但 reasoning off 造成 echo，需註明）
    "qwen/qwen3.6-27b",         # 606
    "openai/gpt-oss-20b",       # 233
    "llama-3.3-70b-versatile",  # 349
]
# 閉源錨點單獨跑（不同 provider、額度未知，避免拖累主批次）
ANCHOR_MODELS = ["gemini-3.1-flash-lite"]


def out_path(model: str, condition: str, variant: str) -> Path:
    suffix = "" if variant == "standard" else f"_{variant}prompt"
    return RESULTS / f"{model.replace('/', '--')}_{condition}{suffix}.jsonl"


def live_quota(model: str, cfg: dict):
    """打一次最小請求讀 header，回傳 (剩餘請求數, 重置時間)。非 Groq/Cerebras 回 (None, None)。"""
    import os

    import requests

    base = {"groq": "https://api.groq.com/openai/v1",
            "cerebras": "https://api.cerebras.ai/v1"}.get(cfg["provider"])
    if not base:
        return None, None
    env = {"groq": "GROQ_API_KEY", "cerebras": "CEREBRAS_API_KEY"}[cfg["provider"]]
    key = os.environ.get(env, "")
    if not key:
        return None, None
    try:
        r = requests.post(f"{base}/chat/completions",
                          json={"model": model, "messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 1},
                          headers={"Authorization": f"Bearer {key}"}, timeout=30)
        h = {k.lower(): v for k, v in r.headers.items()}
        rem = h.get("x-ratelimit-remaining-requests")
        return (int(rem) if rem is not None else None,
                h.get("x-ratelimit-reset-requests", "?"))
    except Exception:
        return None, None


def count_done(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    p = argparse.ArgumentParser(description="平行跑多模型批次")
    p.add_argument("--condition", default="zeroshot_noname")
    p.add_argument("--variant", default="neutral", choices=["standard", "neutral"])
    p.add_argument("--end", type=int, default=9999)
    p.add_argument("--models", nargs="*", help="指定模型（預設＝核心六模型）")
    p.add_argument("--anchor", action="store_true", help="同時跑閉源錨點 gemini")
    p.add_argument("--plan", action="store_true", help="只印排程，不執行")
    p.add_argument("--status", action="store_true", help="印各結果檔進度")
    p.add_argument("--detach", action="store_true",
                   help="子行程脫離父行程，session 結束也不會被砍（Windows）")
    args = p.parse_args()

    models = args.models or list(CORE_MODELS)
    if args.anchor:
        models += ANCHOR_MODELS

    if args.status:
        print(f"{'model':45s} {'condition':22s} {'variant':9s} {'done':>5s}")
        print("-" * 88)
        for f in sorted(RESULTS.glob("*.jsonl")):
            n = count_done(f)
            stem = f.stem
            variant = "neutral" if stem.endswith("_neutralprompt") else "standard"
            stem = stem.removesuffix("_neutralprompt")
            model, _, cond = stem.partition("_")
            print(f"{model.replace('--', '/'):45s} {cond:22s} {variant:9s} {n:5d}")
        return

    n_langs = len([d for d in load_tsv() if d["id"] <= args.end])
    print(f"Condition: {args.condition}  Variant: {args.variant}  Languages: {n_langs}\n")
    print(f"{'model':45s} {'params':10s} {'done':>5s} {'todo':>5s} {'quota now':>11s} {'ETA':>7s}  fits?")
    print("-" * 104)

    plan = []
    for m in models:
        cfg = MODELS[m]
        done = count_done(out_path(m, args.condition, args.variant))
        todo = max(n_langs - done, 0)
        eta_min = todo * cfg["delay"] / 60
        # 讀真實剩餘額度，不要拿 RPD 上限當可用量（已用掉的要扣掉）
        remaining, reset = live_quota(m, cfg)
        if remaining is None:
            quota_s, fits = "?", "?"
        else:
            quota_s = f"{remaining}"
            fits = "OK" if todo <= remaining else f"NO(need {todo}, {reset} to reset)"
        print(f"{m:45s} {str(cfg.get('params')):10s} {done:5d} {todo:5d} {quota_s:>11s} {eta_min:6.0f}m  {fits}")
        if todo:
            plan.append(m)

    if args.plan:
        print("\n[PLAN ONLY] 未執行。拿掉 --plan 就開跑。")
        return
    if not plan:
        print("\n全部已完成，無需執行。")
        return

    print(f"\n平行啟動 {len(plan)} 個 process{'（detached，可獨立存活）' if args.detach else ''}...\n")
    procs = {}
    # Windows: DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP 讓子行程脫離父行程，
    # 這樣 session 結束／監看中止都不會把批次砍掉（2026-07-16 教訓：跑到一半被 teardown 砍）。
    creationflags = 0
    if args.detach and sys.platform == "win32":
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    for m in plan:
        cmd = [sys.executable, str(PIPELINE), "--model", m,
               "--condition", args.condition, "--prompt-variant", args.variant,
               "--end", str(args.end)]
        log = RESULTS / f"_log_{m.replace('/', '--')}_{args.condition}_{args.variant}.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "a", encoding="utf-8")  # append：續跑時保留先前 log
        procs[m] = (subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE),
                                     creationflags=creationflags), fh, log)
        print(f"  started {m}  (pid={procs[m][0].pid})  → {log.name}")

    if args.detach:
        print("\n已 detached，批次會在背景獨立跑完。用 `python run_all.py --status` 查進度。")
        for _, fh, _ in procs.values():
            fh.close()
        return

    print("\n進度（每 60 秒更新，Ctrl-C 只停監看、不停背景任務）：")
    try:
        while True:
            alive = []
            for m, (proc, _, _) in procs.items():
                done = count_done(out_path(m, args.condition, args.variant))
                state = "running" if proc.poll() is None else f"exit={proc.returncode}"
                alive.append(f"{m.split('/')[-1]}:{done}/{n_langs}({state})")
            print(f"  [{time.strftime('%H:%M:%S')}] " + "  ".join(alive))
            if all(proc.poll() is not None for proc, _, _ in procs.values()):
                break
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n監看中止；背景 process 仍在跑。用 --status 查進度。")
        return

    for _, fh, _ in procs.values():
        fh.close()
    print("\n全部完成。各模型統計：")
    for m in plan:
        subprocess.run([sys.executable, str(PIPELINE), "--model", m, "--condition", args.condition,
                        "--prompt-variant", args.variant, "--summary"], cwd=str(HERE))


if __name__ == "__main__":
    main()
