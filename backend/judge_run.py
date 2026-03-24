#!/usr/bin/env python3
"""
judge_run.py — AI Judge for Paired Test Results
================================================

Reads a paired_test_summary_*.json from opt_runs/ and judges each output
image against the original mass + render inputs using two independent AI judges:

  1. Gemini 2.5 Flash  — native multimodal, sees all 3 images separately
  2. Replicate (Llama 3.2 90B Vision) — sees a side-by-side composite image

Scoring rubric (each 1–5):
  geometry_transfer   — does the output match the mass model's geometry/silhouette?
  style_preservation  — does the output preserve the reference render's materials/lighting?
  overall             — overall quality as an architectural visualization

Usage
-----
  python judge_run.py \\
      --summary "opt_runs/replace in a render/paired_test_summary_abc12345.json" \\
      --source-folder "Tests/replace in a render"

Output
------
  opt_runs/<folder>/paired_test_summary_<id>_judged.json  — enriched with judge scores
  Comparison table printed to terminal
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image as PILImage
except ImportError:
    print("Pillow not installed. Run: pip install Pillow", file=sys.stderr)
    sys.exit(1)


# ── Replicate vision model ─────────────────────────────────────────────────────
REPLICATE_VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"

# ── Shared rubric ──────────────────────────────────────────────────────────────
GEMINI_PROMPT = """\
You are an expert architectural visualization judge.
You are given THREE images:
  Image 1 (MASS):   The mass/wireframe model — the exact geometry blueprint.
  Image 2 (RENDER): The reference render — the style target (materials, lighting, atmosphere).
  Image 3 (OUTPUT): The AI-generated result to evaluate.

Study Image 1 (MASS) carefully before scoring. Pay close attention to:
  - Number of towers and their EXACT relative heights (e.g. tower A is 100%, tower B is 65%)
  - Overall outer silhouette traced left-to-right across the top
  - Crown/top termination of each tower (flat, pointed, curved, forked, etc.)
  - Podium or base structure (shape, size, presence/absence)
  - Connecting elements: sky bridges, structural links between towers
  - Distinctive features: setbacks, chamfers, banding, openings, organic curves
  - Spatial arrangement: which tower is left/right/front/back

Then score the OUTPUT (Image 3) on these criteria.
Respond ONLY with valid JSON — no markdown, no preamble, no explanation outside the JSON:

{
  "geometry_transfer": <integer 1-5>,
  "proportions_accuracy": <integer 1-5>,
  "architectural_elements": <integer 1-5>,
  "style_preservation": <integer 1-5>,
  "overall": <integer 1-5>,
  "explanation": "<3-4 sentences covering geometry fidelity, proportion accuracy, element preservation, and style>"
}

Scoring guide:
  geometry_transfer:      5 = outer silhouette and tower count exactly match the mass
                          3 = roughly correct shape but notable deviations
                          1 = completely wrong geometry or wrong number of towers
  proportions_accuracy:   5 = relative tower heights and widths match the mass exactly
                          3 = proportions roughly similar but some towers too tall/short/wide
                          1 = proportions bear no resemblance to the mass
  architectural_elements: 5 = all distinctive elements present (crown type, podium, sky bridges,
                              setbacks, banding) exactly as shown in the mass
                          3 = some elements carried over, others missing or distorted
                          1 = none of the mass's distinctive architectural elements appear
  style_preservation:     5 = materials, lighting, atmosphere perfectly match the reference render
                          3 = some style elements carried over but significant mismatches
                          1 = no recognizable style match
  overall:                5 = excellent, production-ready architectural visualization
                          3 = usable but needs improvement
                          1 = unusable
"""

REPLICATE_PROMPT = """\
You are an expert architectural visualization judge.
You are given a single composite image with THREE panels side by side:
  LEFT panel:   The mass/wireframe model (geometry blueprint)
  MIDDLE panel: The reference render (style target — materials, lighting, atmosphere)
  RIGHT panel:  The AI-generated output to evaluate

Before scoring, carefully study the LEFT panel (mass model) for:
  - Number of towers and their exact relative heights
  - Overall outer silhouette shape
  - Crown/top of each tower (flat, pointed, curved, etc.)
  - Podium or base structure
  - Sky bridges or connecting links between towers
  - Any distinctive features: setbacks, banding, chamfers, openings

Then score the RIGHT panel. Respond ONLY with valid JSON, no other text:

{
  "geometry_transfer": <integer 1-5>,
  "proportions_accuracy": <integer 1-5>,
  "architectural_elements": <integer 1-5>,
  "style_preservation": <integer 1-5>,
  "overall": <integer 1-5>,
  "explanation": "<3-4 sentences covering geometry, proportions, elements, and style>"
}

geometry_transfer:      5=output silhouette/tower count exactly matches mass; 1=completely wrong
proportions_accuracy:   5=relative tower heights and widths match exactly; 1=no resemblance
architectural_elements: 5=all crowns/podium/sky bridges/distinctive features present; 1=none preserved
style_preservation:     5=materials/lighting/atmosphere perfectly match reference; 1=no match
overall:                5=excellent architectural visualization; 1=unusable
"""


# ── Image helpers ──────────────────────────────────────────────────────────────

def b64_image(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return base64.b64encode(raw).decode(), mime


def make_composite(mass_path: Path, render_path: Path, output_path: Path, target_height: int = 512) -> bytes:
    """Combine 3 images side by side at the same height for Replicate judge."""
    imgs = []
    for p in (mass_path, render_path, output_path):
        img = PILImage.open(p).convert("RGB")
        ratio = target_height / img.height
        new_w = max(1, int(img.width * ratio))
        imgs.append(img.resize((new_w, target_height), PILImage.LANCZOS))

    gap = 8
    total_w = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    composite = PILImage.new("RGB", (total_w, target_height), (180, 180, 180))
    x = 0
    for img in imgs:
        composite.paste(img, (x, 0))
        x += img.width + gap

    buf = io.BytesIO()
    composite.save(buf, format="PNG")
    return buf.getvalue()


def parse_judge_json(text: str) -> dict:
    """Extract and parse JSON from model output, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    if "{" in text and "}" in text:
        text = text[text.index("{") : text.rindex("}") + 1]
    return json.loads(text)


# ── Judge implementations ──────────────────────────────────────────────────────

async def judge_with_gemini(mass_path: Path, render_path: Path, output_path: Path) -> dict:
    """Send mass + render + output to Gemini 2.5 Flash as separate images."""
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        return {"error": "GEMINI_API_KEY not set"}

    mass_b64, mass_mime = b64_image(mass_path)
    render_b64, render_mime = b64_image(render_path)
    output_b64, output_mime = b64_image(output_path)

    payload = {
        "contents": [{
            "parts": [
                {"text": GEMINI_PROMPT},
                {"inline_data": {"mime_type": mass_mime,   "data": mass_b64}},
                {"inline_data": {"mime_type": render_mime, "data": render_b64}},
                {"inline_data": {"mime_type": output_mime, "data": output_b64}},
            ]
        }],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512},
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={key}"
    )
    async with httpx.AsyncClient(timeout=90) as http:
        r = await http.post(url, json=payload)
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return parse_judge_json(text)


async def judge_with_replicate(mass_path: Path, render_path: Path, output_path: Path) -> dict:
    """Send a side-by-side composite to Llama 3.2 90B Vision on Replicate."""
    token = os.getenv("REPLICATE_API_TOKEN", "")
    if not token:
        return {"error": "REPLICATE_API_TOKEN not set"}

    composite_bytes = make_composite(mass_path, render_path, output_path)
    composite_b64 = f"data:image/png;base64,{base64.b64encode(composite_bytes).decode()}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "image": composite_b64,
            "prompt": REPLICATE_PROMPT,
            "max_tokens": 512,
            "temperature": 0.1,
        }
    }

    url = f"https://api.replicate.com/v1/models/{REPLICATE_VISION_MODEL}/predictions"
    async with httpx.AsyncClient(timeout=300) as http:
        r = await http.post(url, json=payload, headers=headers)
        r.raise_for_status()
        prediction = r.json()

        get_url = prediction["urls"]["get"]
        while prediction["status"] not in ("succeeded", "failed", "canceled"):
            await asyncio.sleep(3)
            r = await http.get(get_url, headers=headers)
            r.raise_for_status()
            prediction = r.json()

        if prediction["status"] != "succeeded":
            return {"error": f"Replicate prediction {prediction['status']}"}

        output = prediction.get("output", "")
        text = "".join(output) if isinstance(output, list) else str(output)
        return parse_judge_json(text.strip())


# ── Per-entry judging ──────────────────────────────────────────────────────────

async def judge_entry(entry: dict, summary_dir: Path, source_folder: Path) -> None:
    local_file = entry.get("local_file")
    if not local_file:
        entry["judge_error"] = "no local_file in entry (image not downloaded)"
        return

    output_path = summary_dir / local_file
    mass_path   = source_folder / entry["pair"] / entry["mass"]
    render_path = source_folder / entry["pair"] / entry["render"]

    for p, label in [(output_path, "output"), (mass_path, "mass"), (render_path, "render")]:
        if not p.exists():
            entry["judge_error"] = f"{label} file not found: {p}"
            print(f"  SKIP  [{entry['pair']:6s}] [{entry['tab']:16s}] [{entry['model']:16s}]  {label} missing")
            return

    label = f"[{entry['pair']:6s}] [{entry['tab']:16s}] [{entry['model']:16s}]"
    print(f"  Judging {label}...", end=" ", flush=True)

    gemini_result, replicate_result = await asyncio.gather(
        judge_with_gemini(mass_path, render_path, output_path),
        judge_with_replicate(mass_path, render_path, output_path),
        return_exceptions=True,
    )

    entry["judge_gemini"]    = gemini_result    if not isinstance(gemini_result,    Exception) else {"error": str(gemini_result)}
    entry["judge_replicate"] = replicate_result if not isinstance(replicate_result, Exception) else {"error": str(replicate_result)}

    g = entry["judge_gemini"].get("overall",    "ERR")
    r = entry["judge_replicate"].get("overall", "ERR")
    print(f"Gemini={g}/5  Replicate={r}/5")


# ── Output table ───────────────────────────────────────────────────────────────

def print_comparison_table(entries: list[dict]) -> None:
    judged = [e for e in entries if "judge_gemini" in e or "judge_replicate" in e]
    if not judged:
        print("\nNo judged entries to display.")
        return

    W = 126
    print(f"\n{'─' * W}")
    print(f"{'Pair':8} {'Tab':16} {'Model':20} │ {'──────── Gemini judge ────────':30} │ {'──────── Replicate judge ────────':30}")
    print(f"{'':8} {'':16} {'':20} │ {'geo  pro  elm  sty  ovr':30} │ {'geo  pro  elm  sty  ovr':30}")
    print(f"{'─' * W}")

    def fmt(j: dict) -> str:
        if "error" in j:
            return f"{'ERR':30}"
        g  = j.get("geometry_transfer",    "?")
        p  = j.get("proportions_accuracy", "?")
        el = j.get("architectural_elements","?")
        s  = j.get("style_preservation",   "?")
        o  = j.get("overall",              "?")
        return f"{g!s:4} {p!s:4} {el!s:4} {s!s:4} {o!s:4} "

    for e in judged:
        g_str = fmt(e.get("judge_gemini",    {"error": "missing"}))
        r_str = fmt(e.get("judge_replicate", {"error": "missing"}))
        print(f"{e['pair']:8} {e['tab']:16} {e['model']:20} │ {g_str} │ {r_str}")

    print(f"{'─' * W}")

    # Per-model averages
    model_scores: dict = defaultdict(lambda: {"gemini": [], "replicate": []})
    for e in judged:
        m = e["model"]
        for judge_key, label in [("judge_gemini", "gemini"), ("judge_replicate", "replicate")]:
            j = e.get(judge_key, {})
            if isinstance(j.get("overall"), (int, float)):
                model_scores[m][label].append(j["overall"])

    print("\nAverage 'overall' score per model:")
    for model, scores in sorted(model_scores.items()):
        g_avg = f"{sum(scores['gemini'])   / len(scores['gemini']):.1f}"   if scores["gemini"]    else "N/A"
        r_avg = f"{sum(scores['replicate'])/ len(scores['replicate']):.1f}" if scores["replicate"] else "N/A"
        print(f"  {model:25}  Gemini judge: {g_avg:>4}   Replicate judge: {r_avg:>4}")

    # Explanations
    print("\nJudge explanations:")
    for e in judged:
        print(f"\n  [{e['pair']}] [{e['tab']}] [{e['model']}]")
        for label, key in [("Gemini   ", "judge_gemini"), ("Replicate", "judge_replicate")]:
            j = e.get(key, {})
            expl = j.get("explanation") or j.get("error") or "—"
            print(f"    {label}: {expl}")


# ── Entry point ────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> None:
    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"ERROR: summary file not found: {summary_path}", file=sys.stderr)
        sys.exit(1)

    entries: list[dict] = json.loads(summary_path.read_text())
    summary_dir   = summary_path.parent
    source_folder = Path(args.source_folder)

    if not source_folder.is_dir():
        print(f"ERROR: --source-folder is not a directory: {source_folder}", file=sys.stderr)
        sys.exit(1)

    done_entries = [e for e in entries if e.get("status") == "done" and e.get("local_file")]
    skip_entries = [e for e in entries if e.get("status") != "done" or not e.get("local_file")]

    print(f"Summary:  {summary_path}")
    print(f"Source:   {source_folder}")
    print(f"Entries:  {len(done_entries)} to judge, {len(skip_entries)} skipped (error/no image)\n")

    if not done_entries:
        print("Nothing to judge.")
        return

    # Judge all entries concurrently (both judges run in parallel per entry)
    await asyncio.gather(*[judge_entry(e, summary_dir, source_folder) for e in done_entries])

    print_comparison_table(done_entries)

    out_path = summary_path.with_name(summary_path.stem + "_judged.json")
    out_path.write_text(json.dumps(entries, indent=2, default=str))
    print(f"\nJudged results saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ArchRender AI Judge — scores test outputs with Gemini + Replicate"
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Path to paired_test_summary_*.json produced by run_paired_tests.py",
    )
    parser.add_argument(
        "--source-folder",
        required=True,
        help="Original test folder passed to run_paired_tests.py (contains pair subdirectories)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
