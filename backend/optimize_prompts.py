#!/usr/bin/env python3
"""
optimize_prompts.py — Automated Prompt Optimization Loop
=========================================================

Runs the staged pipeline (Stage 1 + Stage 2), judges outputs with Gemini,
identifies the weakest prompt, rewrites it with Gemini 2.5-pro, saves it to
prompt_config.json, and repeats for N cycles.

Because _load_prompt() re-reads prompt_config.json on every API call, the
running server picks up prompt changes automatically — no restart needed.

Scores tracked per cycle:
  geometry_accuracy       → drives: stage1_placement_instruction
  background_preservation → drives: stage1_placement_instruction
  style_match             → drives: stage2_analyzer_instruction
  overall                 → drives: stage2_materialize_instruction

Usage
-----
  python optimize_prompts.py --folder "Tests\\replace in a render" --cycles 5
  python optimize_prompts.py --folder "Tests\\replace in a render" --cycles 3 \\
      --stage2-only --stage1-summary "opt_runs/.../stage1_summary_xxx.json"
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx", file=sys.stderr)
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

PROMPT_CONFIG_PATH = Path(__file__).parent / "prompt_config.json"
HISTORY_DIR = Path(__file__).parent / "optimizer_history"
OPTIMIZER_MODEL = "gemini-2.5-pro"
SCORE_THRESHOLD = 7.5
MAX_IMAGES_FOR_CONTEXT = 2

DIMENSION_TO_KEYS = {
    "geometry_accuracy":        ["stage1_placement_instruction"],
    "background_preservation":  ["stage1_placement_instruction"],
    "style_match":              ["stage2_analyzer_instruction", "stage2_materialize_instruction"],
    "overall":                  ["stage2_materialize_instruction"],
}

KEY_DESCRIPTIONS = {
    "stage1_placement_instruction": (
        "Stage 1 system prompt. Controls how Gemini places the white/grey mass model "
        "into the real scene. Affects: correct position, scale, perspective, "
        "geometry accuracy, and background preservation."
    ),
    "stage2_analyzer_instruction": (
        "Stage 2 analyzer prompt. Instructs the text model to write a detailed "
        "materialization brief from the placed-mass image + style reference. "
        "Affects: how well the final building matches the reference style."
    ),
    "stage2_materialize_instruction": (
        "Stage 2 image-gen prompt. The direct instruction sent to the image model "
        "that paints photorealistic materials onto the white mass. "
        "Affects: style match, realism, background preservation, geometry lock."
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if PROMPT_CONFIG_PATH.exists():
        return json.loads(PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def save_config(cfg: dict) -> None:
    PROMPT_CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [config] prompt_config.json updated ({len(cfg)} keys)")


def backup_config(cycle: int) -> None:
    HISTORY_DIR.mkdir(exist_ok=True)
    dst = HISTORY_DIR / f"prompt_config_cycle{cycle:02d}_{datetime.now():%Y%m%dT%H%M%S}.json"
    shutil.copy(PROMPT_CONFIG_PATH, dst)
    print(f"  [config] backup → {dst.name}")


def run_test_script(extra_args: list[str]) -> int:
    cmd = [sys.executable, str(Path(__file__).parent / "run_paired_tests.py")] + extra_args
    print(f"\n  $ {' '.join(cmd)}\n")
    return subprocess.call(cmd)


def avg_scores(entries: list[dict]) -> dict[str, float]:
    keys = ["geometry_accuracy", "style_match", "background_preservation", "overall"]
    totals: dict[str, list[float]] = {k: [] for k in keys}
    for e in entries:
        j = e.get("judge")
        if not j:
            continue
        for k in keys:
            if k in j:
                totals[k].append(float(j[k]))
    return {k: (sum(v) / len(v)) if v else 0.0 for k, v in totals.items()}


def worst_entries(entries: list[dict], dimension: str, n: int) -> list[dict]:
    judged = [e for e in entries if e.get("judge") and dimension in e["judge"]]
    judged.sort(key=lambda e: e["judge"][dimension])
    return judged[:n]


def load_image_b64(path: Path) -> str | None:
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except Exception:
        return None


def rewrite_prompt(
    current_prompt: str,
    key: str,
    weak_dimension: str,
    avg_score: float,
    worst: list[dict],
    stage2_dir: Path | None,
    gemini_key: str,
) -> str | None:
    """Ask Gemini 2.5-pro to rewrite the prompt based on judge feedback."""
    examples_text = ""
    for i, e in enumerate(worst[:MAX_IMAGES_FOR_CONTEXT], 1):
        j = e.get("judge", {})
        reasoning = j.get("reasoning", "no reasoning available")
        score = j.get(weak_dimension, "?")
        examples_text += (
            f"\nExample {i}: pair={e['pair']} model={e.get('model', '?')}\n"
            f"  {weak_dimension} score: {score}/10\n"
            f"  Judge reasoning: {reasoning}\n"
        )

    rewriter_prompt = (
        "You are an expert prompt engineer specializing in architectural AI visualization.\n"
        f"Your goal: rewrite the system prompt below to improve '{weak_dimension}', "
        f"which currently averages {avg_score:.1f}/10 across test cases.\n\n"
        f"WHAT THIS PROMPT CONTROLS:\n{KEY_DESCRIPTIONS.get(key, key)}\n\n"
        f"JUDGE FEEDBACK from the worst-scoring outputs:\n{examples_text}\n"
        "CURRENT PROMPT:\n"
        "```\n"
        f"{current_prompt}\n"
        "```\n\n"
        "REWRITING RULES:\n"
        "1. Fix exactly what the judge flagged — add or strengthen those specific rules.\n"
        "2. Keep all existing rules that are working.\n"
        "3. Keep the prompt under 800 words.\n"
        "4. Output ONLY the new prompt text. No explanation, no markdown fences, "
        "no preamble. Start directly with the first word of the prompt."
    )

    parts: list[dict] = [{"text": rewriter_prompt}]

    # Add worst output images as visual context for the rewriter
    if stage2_dir and stage2_dir.exists():
        imgs_added = 0
        for e in worst[:MAX_IMAGES_FOR_CONTEXT]:
            if imgs_added >= MAX_IMAGES_FOR_CONTEXT:
                break
            local_file = e.get("local_file")
            if not local_file:
                continue
            p = stage2_dir / local_file
            if p.exists():
                b64 = load_image_b64(p)
                if b64:
                    parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
                    imgs_added += 1

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.4},
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{OPTIMIZER_MODEL}:generateContent?key={gemini_key}"
    )
    try:
        with httpx.Client(timeout=120) as http:
            r = http.post(url, json=payload)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as exc:
        print(f"  [optimizer] Gemini rewrite failed: {exc}")
        return None


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated prompt optimization loop for the staged render pipeline"
    )
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3,
                        help="Number of optimization cycles (default 3)")
    parser.add_argument("--repeat", type=int, default=2,
                        help="Stage-2 repeats per job for score averaging (default 2)")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--models", default="gemini-staged-pro")
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD,
                        help=f"Rewrite prompts scoring below this value (default {SCORE_THRESHOLD})")
    parser.add_argument("--stage2-only", action="store_true",
                        help="Skip Stage 1 in every cycle — reuse --stage1-summary")
    parser.add_argument("--stage1-summary", type=Path, default=None,
                        help="Required with --stage2-only")
    args = parser.parse_args()

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    if args.stage2_only and not args.stage1_summary:
        print("ERROR: --stage2-only requires --stage1-summary", file=sys.stderr)
        sys.exit(1)

    HISTORY_DIR.mkdir(exist_ok=True)
    history: list[dict] = []
    stage1_summary_path: Path | None = args.stage1_summary

    print(f"\n{'═'*72}")
    print(f"  PROMPT OPTIMIZER  cycles={args.cycles}  repeat={args.repeat}  "
          f"threshold={args.score_threshold}")
    print(f"{'═'*72}")

    for cycle in range(1, args.cycles + 1):
        print(f"\n{'─'*72}")
        print(f"  CYCLE {cycle} / {args.cycles}")
        print(f"{'─'*72}")

        backup_config(cycle)

        # ── Stage 1 ───────────────────────────────────────────────────────────
        if not args.stage2_only or stage1_summary_path is None:
            print(f"\n[Cycle {cycle}] Stage 1 — placing mass in scene…")
            run_test_script([
                "--folder", str(args.folder),
                "--models", args.models,
                "--result", "1",
                "--server", args.server,
            ])
            opt_dir = Path("opt_runs") / args.folder.name
            s1_dirs = sorted(opt_dir.glob("stage1_*"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
            if not s1_dirs:
                print("  ERROR: no stage1 directory found — aborting cycle")
                continue
            jsons = list(s1_dirs[0].glob("stage1_summary_*.json"))
            if not jsons:
                print("  ERROR: no stage1_summary JSON found — aborting cycle")
                continue
            stage1_summary_path = max(jsons, key=lambda p: p.stat().st_mtime)
            print(f"  Stage-1 summary: {stage1_summary_path}")

        # ── Stage 2 + judge ───────────────────────────────────────────────────
        print(f"\n[Cycle {cycle}] Stage 2 — materializing + judging…")
        run_test_script([
            "--folder", str(args.folder),
            "--models", args.models,
            "--result", "2",
            "--stage1-summary", str(stage1_summary_path),
            "--repeat", str(args.repeat),
            "--judge",
            "--server", args.server,
        ])

        opt_dir = Path("opt_runs") / args.folder.name
        s2_dirs = sorted(opt_dir.glob("stage2_*"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if not s2_dirs:
            print("  ERROR: no stage2 directory found — skipping optimization")
            continue
        jsons = list(s2_dirs[0].glob("stage2_summary_*.json"))
        if not jsons:
            print("  ERROR: no stage2_summary JSON found — skipping optimization")
            continue
        s2_json = max(jsons, key=lambda p: p.stat().st_mtime)
        entries = json.loads(s2_json.read_text())
        scores = avg_scores(entries)
        history.append({"cycle": cycle, "scores": scores})

        # ── Print scores ──────────────────────────────────────────────────────
        print(f"\n  Cycle {cycle} avg scores:")
        for dim, val in scores.items():
            flag = "  ← WEAK" if val < args.score_threshold and dim != "overall" else ""
            print(f"    {dim:<30} {val:.1f}/10{flag}")

        # ── Find weak dimensions ──────────────────────────────────────────────
        weak_dims = sorted(
            [(d, v) for d, v in scores.items()
             if v < args.score_threshold and d != "overall"],
            key=lambda x: x[1]
        )

        if not weak_dims:
            print(f"\n  All criteria ≥ {args.score_threshold} — nothing to rewrite.")
            continue

        # ── Rewrite weak prompts ──────────────────────────────────────────────
        cfg = load_config()
        any_updated = False
        already_rewritten: set[str] = set()

        for dim, dim_score in weak_dims:
            for pkey in DIMENSION_TO_KEYS.get(dim, []):
                if pkey in already_rewritten:
                    continue
                current = cfg.get(pkey, "")
                if not current:
                    continue

                print(f"\n  Rewriting '{pkey}'  (drives {dim} = {dim_score:.1f}/10)…")
                bad = worst_entries(entries, dim, MAX_IMAGES_FOR_CONTEXT)
                new_prompt = rewrite_prompt(
                    current_prompt=current,
                    key=pkey,
                    weak_dimension=dim,
                    avg_score=dim_score,
                    worst=bad,
                    stage2_dir=s2_dirs[0],
                    gemini_key=gemini_key,
                )
                if new_prompt and new_prompt != current:
                    cfg[pkey] = new_prompt
                    any_updated = True
                    already_rewritten.add(pkey)
                    print(f"  ✓ {pkey} rewritten ({len(new_prompt)} chars)")
                    # Save diff to history
                    (HISTORY_DIR / f"cycle{cycle:02d}_{pkey}.txt").write_text(
                        f"=== BEFORE  ({dim}={dim_score:.1f}) ===\n{current}\n\n"
                        f"=== AFTER ===\n{new_prompt}\n",
                        encoding="utf-8",
                    )
                else:
                    print(f"  [skip] no change for {pkey}")

        if any_updated:
            save_config(cfg)
            print("  Server will use new prompts on the next job (no restart needed).")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("  OPTIMIZATION COMPLETE")
    print(f"{'═'*72}")
    if history:
        dims = ["geometry_accuracy", "style_match", "background_preservation", "overall"]
        print(f"  {'Cycle':<8}" + "".join(f"{d[:10]:>12}" for d in dims))
        for h in history:
            s = h["scores"]
            print(f"  {h['cycle']:<8}" + "".join(f"{s.get(d, 0):>12.1f}" for d in dims))

    (HISTORY_DIR / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"\n  History: {HISTORY_DIR}/history.json")
    print(f"  Prompts: {PROMPT_CONFIG_PATH}")


if __name__ == "__main__":
    main()
