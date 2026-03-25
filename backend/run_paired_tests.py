#!/usr/bin/env python3
"""
run_paired_tests.py — Paired Mass+Render Batch Tester
======================================================

Scans a parent folder whose subdirectories each contain one mass image
and one render image (named *--mass.* and *--render.*). For every subfolder
pair it calls:
  - POST /api/style-transfer  × N Gemini models
  - POST /api/update-render   × N Gemini models

All jobs run concurrently. Progress is polled until every job is settled.

Usage
-----
  python run_paired_tests.py --folder "C:\\path\\to\\replace in a render"
  python run_paired_tests.py --folder ./tests/replace_in_a_render --server http://localhost:8000
  python run_paired_tests.py --folder ./tests --tabs style-transfer
  python run_paired_tests.py --folder ./tests --models gemini-25-pro,gemini-25-flash

Folder layout expected
----------------------
  parent_folder/
    200A/
      200--mass.png
      200--render.jpg
    200B/
      200--mass.png
      200--render.jpg
    201A/
      201--mass.png
      201--render.jpg
    ...

Each subdirectory is one test case (one building). The file whose name
contains "--mass" is the wireframe/massing model; the one containing
"--render" is the reference render used for materiality matching.

Output
------
  Results printed to terminal and saved to opt_runs/<folder_name>/paired_test_summary_<id>.json.
"""

import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx", file=sys.stderr)
    sys.exit(1)

GEMINI_MODELS = ["gemini-25-pro", "gemini-25-flash", "gemini-direct", "gemini-staged-pro", "gemini-staged-flash"]
TABS = ["style-transfer", "update-render"]
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
POLL_INTERVAL = 5  # seconds between status polls


def find_pairs(folder: Path) -> list[dict]:
    """Return a list of {name, mass_path, render_path} for each valid subfolder."""
    pairs = []
    for subdir in sorted(folder.iterdir()):
        if not subdir.is_dir():
            continue
        images = [p for p in subdir.iterdir() if p.suffix.lower() in ALLOWED_SUFFIXES]
        mass = next((p for p in images if "--mass" in p.stem.lower()), None)
        render = next((p for p in images if "--render" in p.stem.lower()), None)
        if not mass:
            print(f"  [warn] Skipping {subdir.name}: no file with '--mass' in its name")
            continue
        if not render:
            print(f"  [warn] Skipping {subdir.name}: no file with '--render' in its name")
            continue
        pairs.append({"name": subdir.name, "mass": mass, "render": render})
    return pairs


def submit_job(server: str, tab: str, model: str, pair: dict, client: httpx.Client) -> str:
    """Upload the pair and submit a single job; returns the job_id."""
    mass_path: Path = pair["mass"]
    render_path: Path = pair["render"]

    with mass_path.open("rb") as mass_fh, render_path.open("rb") as render_fh:
        if tab == "style-transfer":
            r = client.post(
                f"{server}/api/style-transfer",
                files={
                    "reference": (render_path.name, render_fh, "image/jpeg"),
                    "mass": (mass_path.name, mass_fh, "image/png"),
                },
                data={"model": model, "prompt": ""},
                timeout=60,
            )
        else:  # update-render
            r = client.post(
                f"{server}/api/update-render",
                files={
                    "render": (render_path.name, render_fh, "image/jpeg"),
                    "mass": (mass_path.name, mass_fh, "image/png"),
                },
                data={"model": model, "prompt": ""},
                timeout=60,
            )
    r.raise_for_status()
    return r.json()["jobId"]


def poll_all(server: str, job_entries: list[dict], client: httpx.Client) -> None:
    """Poll /api/job/{id} for every entry until all are settled."""
    while True:
        done = errors = pending = 0
        for entry in job_entries:
            if entry["status"] in ("done", "error"):
                done += entry["status"] == "done"
                errors += entry["status"] == "error"
                continue
            try:
                r = client.get(f"{server}/api/job/{entry['job_id']}", timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    entry["status"] = data["status"]
                    entry["output_url"] = data.get("output_url")
                    entry["error_msg"] = data.get("error") or (
                        "job failed (no details from server)" if data["status"] == "error" else None
                    )
                    if entry["status"] == "done":
                        done += 1
                    elif entry["status"] == "error":
                        errors += 1
                    else:
                        pending += 1
                else:
                    pending += 1
            except Exception:
                pending += 1

        total = len(job_entries)
        print(
            f"\r  Progress: {done} done / {errors} errors / {pending} pending (total {total})   ",
            end="",
            flush=True,
        )
        if pending == 0:
            print()
            break
        time.sleep(POLL_INTERVAL)


def print_summary(job_entries: list[dict], server: str) -> None:
    done = [e for e in job_entries if e["status"] == "done"]
    errors = [e for e in job_entries if e["status"] == "error"]

    print(f"\n{'─' * 72}")
    print(f"Total    : {len(job_entries)}")
    print(f"Done     : {len(done)}")
    print(f"Errors   : {len(errors)}")
    print(f"{'─' * 72}")

    if done:
        print("\nSuccessful results:")
        for e in done:
            out = e.get("output_url", "")
            full_url = f"{server}{out}" if out and out.startswith("/") else out
            print(f"  [{e['pair']:6s}] [{e['tab']:16s}] [{e['model']:16s}]")
            print(f"     mass  : {e['mass']}")
            print(f"     render: {e['render']}")
            print(f"     output: {full_url}")

    if errors:
        print("\nFailed jobs:")
        for e in errors:
            msg = e.get("error_msg") or e.get("submit_error") or "?"
            print(f"  [{e['pair']:6s}] [{e['tab']:16s}] [{e['model']:16s}]  {msg}")


def _download_outputs(
    job_entries: list[dict],
    out_dir: Path,
    server: str,
    client: "httpx.Client",
    model_short_map: dict,
) -> None:
    """Download output images for all done entries into out_dir."""
    for entry in job_entries:
        if entry["status"] != "done" or not entry.get("output_url"):
            continue
        url = entry["output_url"]
        full_url = f"{server}{url}" if url.startswith("/") else url
        model_short = model_short_map.get(entry["model"], entry["model"])
        img_name = f"{entry['pair']}_{entry['tab']}_{model_short}.png"
        img_path = out_dir / img_name
        try:
            r = client.get(full_url, timeout=30)
            r.raise_for_status()
            img_path.write_bytes(r.content)
            entry["local_file"] = img_name
            print(f"  Saved {img_name}")
        except Exception as exc:
            print(f"  WARN: could not download {img_name}: {exc}")


def judge_entry(server: str, entry: dict, client: "httpx.Client") -> dict | None:
    """Call /api/judge for a completed entry. Returns scores dict or None on failure."""
    if entry["status"] != "done" or not entry.get("output_url"):
        return None
    mass_path: Path = entry["_mass_path"]
    render_path: Path = entry["_render_path"]
    output_url: str = entry["output_url"]
    try:
        with mass_path.open("rb") as mf, render_path.open("rb") as rf:
            r = client.post(
                f"{server}/api/judge",
                files={
                    "mass":   (mass_path.name, mf, "image/png"),
                    "render": (render_path.name, rf, "image/jpeg"),
                },
                data={"output_url": output_url},
                timeout=90,
            )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"  [judge warn] {entry['pair']} / {entry['model']}: {exc}")
        return None


def _print_judge_summary(job_entries: list[dict]) -> None:
    """Print a ranked leaderboard of judge scores grouped by model."""
    judged = [e for e in job_entries if e.get("judge")]
    if not judged:
        return

    # Group by model
    from collections import defaultdict
    by_model: dict[str, list[dict]] = defaultdict(list)
    for e in judged:
        by_model[e["model"]].append(e["judge"])

    score_keys = ["geometry_accuracy", "style_match", "background_preservation", "overall"]

    print(f"\n{'─' * 72}")
    print("JUDGE SCORES (avg per model)")
    print(f"{'─' * 72}")
    print(f"  {'Model':<22} {'Geometry':>9} {'Style':>7} {'BG':>5} {'Overall':>9}")
    print(f"  {'─'*22} {'─'*9} {'─'*7} {'─'*5} {'─'*9}")

    rows = []
    for model, score_list in by_model.items():
        avgs = {k: statistics.mean(s[k] for s in score_list if k in s) for k in score_keys}
        rows.append((model, avgs))

    # Sort by overall descending
    rows.sort(key=lambda x: x[1].get("overall", 0), reverse=True)
    for model, avgs in rows:
        print(
            f"  {model:<22} "
            f"{avgs.get('geometry_accuracy', 0):>9.1f} "
            f"{avgs.get('style_match', 0):>7.1f} "
            f"{avgs.get('background_preservation', 0):>5.1f} "
            f"{avgs.get('overall', 0):>9.1f}"
        )
    print(f"{'─' * 72}")


def _run_retry(args: "argparse.Namespace", server: str, client: "httpx.Client", model_short_map: dict) -> None:
    """Re-submit all errored entries from an existing summary and merge results back."""
    summary_path: Path = args.retry_failed
    if not summary_path.exists():
        print(f"ERROR: --retry-failed file not found: {summary_path}", file=sys.stderr)
        sys.exit(1)

    all_entries: list[dict] = json.loads(summary_path.read_text())
    failed = [e for e in all_entries if e.get("status") == "error"]
    if not failed:
        print("No failed entries found in the summary — nothing to retry.")
        return

    out_dir = summary_path.parent
    print(f"\nRetrying {len(failed)} failed job(s) from: {summary_path}")

    # Build a lookup of pair name → {mass: Path, render: Path}
    pair_lookup: dict = {}
    for e in failed:
        pair_name = e["pair"]
        if pair_name not in pair_lookup:
            subdir = args.folder / pair_name
            images = [p for p in subdir.iterdir() if p.suffix.lower() in ALLOWED_SUFFIXES]
            mass = next((p for p in images if "--mass" in p.stem.lower()), None)
            render = next((p for p in images if "--render" in p.stem.lower()), None)
            if not mass or not render:
                print(f"  [warn] Cannot find images for pair {pair_name}, skipping")
                continue
            pair_lookup[pair_name] = {"name": pair_name, "mass": mass, "render": render}

    # Re-submit
    retry_entries: list[dict] = []
    for e in failed:
        pair_name = e["pair"]
        pair = pair_lookup.get(pair_name)
        if not pair:
            continue
        tab = e["tab"]
        model = e["model"]
        # Reset entry fields
        e["job_id"] = None
        e["status"] = "error"
        e["output_url"] = None
        e["error_msg"] = None
        e["submit_error"] = None
        e.pop("local_file", None)
        try:
            job_id = submit_job(server, tab, model, pair, client)
            e["job_id"] = job_id
            e["status"] = "pending"
            print(f"  Submitted [{pair_name:6s}] [{tab:16s}] [{model:16s}] → {job_id[:8]}…")
        except Exception as exc:
            e["submit_error"] = str(exc)
            print(f"  FAILED    [{pair_name:6s}] [{tab:16s}] [{model:16s}]  {exc}")
        retry_entries.append(e)

    # Poll retried jobs
    pending = [e for e in retry_entries if e.get("job_id")]
    if pending:
        print(f"\nWaiting for {len(pending)} retried job(s)…")
        poll_all(server, pending, client)

    print_summary(all_entries, server)

    print("\nDownloading new output images…")
    _download_outputs(retry_entries, out_dir, server, client, model_short_map)

    # Save back to the same file
    summary_path.write_text(json.dumps(all_entries, indent=2, default=str))
    print(f"\nUpdated results saved to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ArchRender AI — Paired Mass+Render Batch Test Runner"
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Parent folder whose subdirs each contain *--mass.* + *--render.* files",
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Base URL of the running server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--models",
        default=",".join(GEMINI_MODELS),
        help=f"Comma-separated model IDs (default: {','.join(GEMINI_MODELS)})",
    )
    parser.add_argument(
        "--tabs",
        default=",".join(TABS),
        help=f"Comma-separated tabs to test (default: {','.join(TABS)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save JSON results to this file (default: paired_test_summary_<id>.json)",
    )
    parser.add_argument(
        "--retry-failed",
        type=Path,
        default=None,
        metavar="SUMMARY_JSON",
        help="Re-submit all errored jobs from an existing summary JSON and merge results back",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Run each job N times (default 1). Useful for variance analysis.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        default=False,
        help="After all jobs finish, ask Gemini to score each output and print a leaderboard",
    )
    args = parser.parse_args()

    server = args.server.rstrip("/")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tabs = [t.strip() for t in args.tabs.split(",") if t.strip()]

    if not args.folder.is_dir():
        print(f"ERROR: --folder is not a directory: {args.folder}", file=sys.stderr)
        sys.exit(1)

    MODEL_SHORT = {
        "gemini-25-pro":      "pro",
        "gemini-25-flash":    "flash",
        "gemini-direct":      "direct",
        "gemini-staged-pro":  "staged-pro",
        "gemini-staged-flash":"staged-flash",
    }

    with httpx.Client() as client:
        # ── Health check ──────────────────────────────────────────────────────
        try:
            r = client.get(f"{server}/api/health", timeout=5)
            r.raise_for_status()
        except Exception as exc:
            print(f"ERROR: Cannot reach server at {server}: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Server OK: {server}")

        # ── Retry-failed mode ─────────────────────────────────────────────────
        if args.retry_failed:
            _run_retry(args, server, client, MODEL_SHORT)
            return

        # ── Discover pairs ────────────────────────────────────────────────────
        pairs = find_pairs(args.folder)
        if not pairs:
            print(
                f"ERROR: No valid mass+render pairs found in {args.folder}", file=sys.stderr
            )
            sys.exit(1)

        repeat = max(1, args.repeat)
        print(f"\nFound {len(pairs)} pair(s): {[p['name'] for p in pairs]}")
        total_jobs = len(pairs) * len(tabs) * len(models) * repeat
        repeat_note = f" × {repeat} repeat(s)" if repeat > 1 else ""
        print(
            f"Submitting {total_jobs} job(s)  "
            f"({len(pairs)} pair(s) × {len(tabs)} tab(s) × {len(models)} model(s){repeat_note})"
        )

        # ── Submit all jobs ───────────────────────────────────────────────────
        job_entries: list[dict] = []
        for pair in pairs:
            for tab in tabs:
                for model in models:
                    for run_idx in range(1, repeat + 1):
                        run_label = f"run{run_idx}" if repeat > 1 else ""
                        entry: dict = {
                            "job_id": None,
                            "pair": pair["name"],
                            "tab": tab,
                            "model": model,
                            "run": run_idx,
                            "mass": pair["mass"].name,
                            "render": pair["render"].name,
                            "_mass_path": pair["mass"],    # used by judge, not serialised
                            "_render_path": pair["render"],
                            "status": "error",
                            "output_url": None,
                            "error_msg": None,
                            "submit_error": None,
                        }
                        try:
                            job_id = submit_job(server, tab, model, pair, client)
                            entry["job_id"] = job_id
                            entry["status"] = "pending"
                            run_tag = f" r{run_idx}" if repeat > 1 else ""
                            print(
                                f"  Submitted [{pair['name']:6s}] "
                                f"[{tab:16s}] [{model:16s}]{run_tag} → {job_id[:8]}…"
                            )
                        except Exception as exc:
                            entry["submit_error"] = str(exc)
                            print(
                                f"  FAILED    [{pair['name']:6s}] "
                                f"[{tab:16s}] [{model:16s}]  {exc}"
                            )
                        job_entries.append(entry)

        # ── Poll ──────────────────────────────────────────────────────────────
        pending_entries = [e for e in job_entries if e["job_id"]]
        if pending_entries:
            print(
                f"\nWaiting for {len(pending_entries)} job(s) "
                "(real Gemini API calls — this may take a few minutes)…"
            )
            poll_all(server, pending_entries, client)

        # ── Summary ───────────────────────────────────────────────────────────
        print_summary(job_entries, server)

        # ── Resolve output folder ──────────────────────────────────────────────
        run_id = str(uuid.uuid4())[:8]
        if args.output:
            out_dir = args.output.parent
            json_path = args.output
        else:
            out_dir = Path("opt_runs") / args.folder.name
            out_dir.mkdir(parents=True, exist_ok=True)
            json_path = out_dir / f"paired_test_summary_{run_id}.json"

        # ── Download output images into the same folder ────────────────────────
        print("\nDownloading output images…")
        _download_outputs(job_entries, out_dir, server, client, MODEL_SHORT)

        # ── Judge (optional) ──────────────────────────────────────────────────
        if args.judge:
            done_count = sum(1 for e in job_entries if e["status"] == "done")
            print(f"\nJudging {done_count} successful output(s) with Gemini…")
            for i, entry in enumerate(job_entries, 1):
                if entry["status"] != "done":
                    continue
                run_tag = f" r{entry['run']}" if repeat > 1 else ""
                print(
                    f"  [{i}/{done_count}] {entry['pair']} / {entry['model']}{run_tag}…",
                    end=" ", flush=True,
                )
                scores = judge_entry(server, entry, client)
                if scores:
                    entry["judge"] = scores
                    print(f"overall={scores.get('overall', '?')}")
                else:
                    print("failed")
            _print_judge_summary(job_entries)

        # ── Strip internal keys before saving ─────────────────────────────────
        serialisable = [
            {k: v for k, v in e.items() if not k.startswith("_")}
            for e in job_entries
        ]

        # ── Save JSON ─────────────────────────────────────────────────────────
        json_path.write_text(json.dumps(serialisable, indent=2, default=str))
        print(f"\nFull results saved to: {json_path}")


if __name__ == "__main__":
    main()
