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
import sys
import time
import uuid
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx", file=sys.stderr)
    sys.exit(1)

GEMINI_MODELS = ["gemini-25-pro", "gemini-25-flash", "gemini-direct"]
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
    args = parser.parse_args()

    server = args.server.rstrip("/")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tabs = [t.strip() for t in args.tabs.split(",") if t.strip()]

    if not args.folder.is_dir():
        print(f"ERROR: --folder is not a directory: {args.folder}", file=sys.stderr)
        sys.exit(1)

    with httpx.Client() as client:
        # ── Health check ──────────────────────────────────────────────────────
        try:
            r = client.get(f"{server}/api/health", timeout=5)
            r.raise_for_status()
        except Exception as exc:
            print(f"ERROR: Cannot reach server at {server}: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Server OK: {server}")

        # ── Discover pairs ────────────────────────────────────────────────────
        pairs = find_pairs(args.folder)
        if not pairs:
            print(
                f"ERROR: No valid mass+render pairs found in {args.folder}", file=sys.stderr
            )
            sys.exit(1)

        print(f"\nFound {len(pairs)} pair(s): {[p['name'] for p in pairs]}")
        total_jobs = len(pairs) * len(tabs) * len(models)
        print(
            f"Submitting {total_jobs} job(s)  "
            f"({len(pairs)} pair(s) × {len(tabs)} tab(s) × {len(models)} model(s))"
        )

        # ── Submit all jobs ───────────────────────────────────────────────────
        job_entries: list[dict] = []
        for pair in pairs:
            for tab in tabs:
                for model in models:
                    entry: dict = {
                        "job_id": None,
                        "pair": pair["name"],
                        "tab": tab,
                        "model": model,
                        "mass": pair["mass"].name,
                        "render": pair["render"].name,
                        "status": "error",
                        "output_url": None,
                        "error_msg": None,
                        "submit_error": None,
                    }
                    try:
                        job_id = submit_job(server, tab, model, pair, client)
                        entry["job_id"] = job_id
                        entry["status"] = "pending"
                        print(
                            f"  Submitted [{pair['name']:6s}] "
                            f"[{tab:16s}] [{model:16s}] → {job_id[:8]}…"
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

        # ── Save JSON ─────────────────────────────────────────────────────────
        run_id = str(uuid.uuid4())[:8]
        if args.output:
            out_path = args.output
        else:
            opt_runs_dir = Path("opt_runs") / args.folder.name
            opt_runs_dir.mkdir(parents=True, exist_ok=True)
            out_path = opt_runs_dir / f"paired_test_summary_{run_id}.json"
        out_path.write_text(json.dumps(job_entries, indent=2, default=str))
        print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
