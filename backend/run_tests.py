#!/usr/bin/env python3
"""
run_tests.py — Photo Library Test Runner
=========================================

Uploads all images from a local folder to the library, triggers a full test run
(Style Transfer + Update Render × all 3 Gemini models), polls until complete,
and prints a results summary to the terminal.

Usage
-----
  # Upload from separate folders:
  python run_tests.py --mass ./my_masses --render ./my_renders

  # Upload from a single mixed folder (files whose name starts with "mass_"
  # are treated as mass images; everything else is treated as render images):
  python run_tests.py --mixed ./my_photos

  # Point at a running server (default: http://localhost:8000):
  python run_tests.py --mass ./masses --render ./renders --server http://localhost:8000

  # Skip upload if images are already in the library:
  python run_tests.py --skip-upload

Outputs
-------
  Results are saved by the server into test_results/{run_id}.json.
  This script also writes a local summary to run_summary_{run_id}.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx", file=sys.stderr)
    sys.exit(1)

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
POLL_INTERVAL = 5  # seconds between status polls


def upload_folder(server: str, folder: Path, role: str, client: httpx.Client) -> list[dict]:
    """Upload every image in *folder* to the library with the given role."""
    images = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in ALLOWED_SUFFIXES]
    if not images:
        print(f"  [warn] No images found in {folder}")
        return []
    uploaded = []
    for img in images:
        print(f"  Uploading {role}: {img.name} ...", end=" ", flush=True)
        with img.open("rb") as fh:
            r = client.post(
                f"{server}/api/library/upload",
                files={"file": (img.name, fh, "image/png")},
                data={"role": role},
                timeout=30,
            )
        r.raise_for_status()
        info = r.json()
        print(f"ok ({info['id']})")
        uploaded.append(info)
    return uploaded


def trigger_run(server: str, client: httpx.Client) -> dict:
    """POST /api/library/run-tests and return the response JSON."""
    r = client.post(f"{server}/api/library/run-tests", timeout=15)
    r.raise_for_status()
    return r.json()


def poll_run(server: str, run_id: str, client: httpx.Client) -> dict:
    """Poll /api/library/test-results/{run_id} until all jobs are settled."""
    url = f"{server}/api/library/test-results/{run_id}"
    while True:
        r = client.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        entries = data.get("entries", [])
        total = len(entries)
        done = sum(1 for e in entries if e.get("status") == "done")
        errors = sum(1 for e in entries if e.get("status") == "error")
        pending = total - done - errors
        print(f"\r  Progress: {done} done / {errors} errors / {pending} pending (total {total})    ",
              end="", flush=True)
        if pending == 0:
            print()
            return data
        time.sleep(POLL_INTERVAL)


def print_summary(data: dict, server: str) -> None:
    entries = data.get("entries", [])
    print(f"\n{'─'*70}")
    print(f"Run ID   : {data.get('run_id')}")
    print(f"Timestamp: {data.get('timestamp')}")
    print(f"Total    : {len(entries)}")
    done = [e for e in entries if e.get("status") == "done"]
    errors = [e for e in entries if e.get("status") == "error"]
    print(f"Done     : {len(done)}")
    print(f"Errors   : {len(errors)}")
    print(f"{'─'*70}")

    if done:
        print("\nSuccessful results:")
        for e in done:
            out = e.get("output_url", "")
            full_url = f"{server}{out}" if out and out.startswith("/") else out
            print(f"  [{e['tab']:15s}] [{e['model']:15s}]  {e['mass']} + {e['render']}")
            print(f"     → {full_url}")

    if errors:
        print("\nFailed:")
        for e in errors:
            print(f"  [{e['tab']:15s}] [{e['model']:15s}]  {e.get('error','?')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ArchRender AI — Library Test Runner")
    parser.add_argument("--server", default="http://localhost:8000",
                        help="Base URL of the running server (default: http://localhost:8000)")
    parser.add_argument("--mass", type=Path, default=None,
                        help="Folder of mass / wireframe images to upload")
    parser.add_argument("--render", type=Path, default=None,
                        help="Folder of reference render images to upload")
    parser.add_argument("--mixed", type=Path, default=None,
                        help="Single folder: files named mass_* → mass role, others → render role")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip upload step (use images already in the library)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save full JSON results to this file (default: run_summary_<id>.json)")
    args = parser.parse_args()

    server = args.server.rstrip("/")

    with httpx.Client() as client:
        # ── Health check ──────────────────────────────────────────────────────
        try:
            r = client.get(f"{server}/api/health", timeout=5)
            r.raise_for_status()
        except Exception as exc:
            print(f"ERROR: Cannot reach server at {server}: {exc}", file=sys.stderr)
            sys.exit(1)

        # ── Upload ───────────────────────────────────────────────────────────
        if not args.skip_upload:
            if args.mixed:
                folder = args.mixed
                if not folder.is_dir():
                    print(f"ERROR: --mixed path is not a directory: {folder}", file=sys.stderr)
                    sys.exit(1)
                mass_imgs = [p for p in sorted(folder.iterdir())
                             if p.suffix.lower() in ALLOWED_SUFFIXES and p.stem.lower().startswith("mass")]
                render_imgs = [p for p in sorted(folder.iterdir())
                               if p.suffix.lower() in ALLOWED_SUFFIXES and not p.stem.lower().startswith("mass")]
                print(f"Mixed folder: {len(mass_imgs)} mass images, {len(render_imgs)} render images")
                for img_path in mass_imgs:
                    with img_path.open("rb") as fh:
                        r = client.post(f"{server}/api/library/upload",
                                        files={"file": (img_path.name, fh, "image/png")},
                                        data={"role": "mass"}, timeout=30)
                        r.raise_for_status()
                    print(f"  Uploaded mass: {img_path.name}")
                for img_path in render_imgs:
                    with img_path.open("rb") as fh:
                        r = client.post(f"{server}/api/library/upload",
                                        files={"file": (img_path.name, fh, "image/png")},
                                        data={"role": "render"}, timeout=30)
                        r.raise_for_status()
                    print(f"  Uploaded render: {img_path.name}")
            else:
                if args.mass:
                    if not args.mass.is_dir():
                        print(f"ERROR: --mass path is not a directory: {args.mass}", file=sys.stderr)
                        sys.exit(1)
                    print(f"Uploading mass images from {args.mass}:")
                    upload_folder(server, args.mass, "mass", client)
                if args.render:
                    if not args.render.is_dir():
                        print(f"ERROR: --render path is not a directory: {args.render}", file=sys.stderr)
                        sys.exit(1)
                    print(f"Uploading render images from {args.render}:")
                    upload_folder(server, args.render, "render", client)

        # ── Verify library is non-empty ───────────────────────────────────────
        lib = client.get(f"{server}/api/library", timeout=10).json()
        print(f"\nLibrary: {len(lib['mass'])} mass image(s), {len(lib['render'])} render image(s)")
        if not lib["mass"] or not lib["render"]:
            print("ERROR: Library needs at least one mass and one render image.", file=sys.stderr)
            sys.exit(1)

        # ── Trigger test run ──────────────────────────────────────────────────
        print("\nTriggering test run...")
        run_info = trigger_run(server, client)
        run_id = run_info["run_id"]
        print(f"  Run ID    : {run_id}")
        print(f"  Job count : {run_info['job_count']} "
              f"({run_info['pairs']} pair(s) × 2 tabs × 3 models)")
        print(f"  Results   : {server}{run_info['results_url']}")

        # ── Poll ─────────────────────────────────────────────────────────────
        print("\nWaiting for results (this may take a while — real Gemini API calls)...")
        results = poll_run(server, run_id, client)

        # ── Summary ──────────────────────────────────────────────────────────
        print_summary(results, server)

        # ── Save JSON ────────────────────────────────────────────────────────
        out_path = args.output or Path(f"run_summary_{run_id[:8]}.json")
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
