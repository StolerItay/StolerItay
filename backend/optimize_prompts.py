#!/usr/bin/env python3
"""
optimize_prompts.py — Prompt Optimization Agent for ArchRender AI
==================================================================

Uses a textual gradient descent loop to iteratively improve the Gemini
instruction prompts by testing against expected results.

Workflow (per iteration):
  1. Generate outputs using current prompts (calls Gemini API directly)
  2. Gemini judges each output vs the expected --result image
     → scores geometry fidelity, height preservation, style quality
  3. Aggregate failures into a feedback summary ("textual gradient")
  4. Gemini rewrites the prompts to address the failures
  5. Save the best-scoring prompts to prompt_config.json
  6. main.py re-reads prompt_config.json on every request (hot-reload)

Folder layout expected:
  parent_folder/
    200A/
      200--mass.png
      200--render.jpg
      200--result.jpg   ← expected / desired photorealistic output
    200B/
      ...

Usage:
  python optimize_prompts.py --folder ./Tests/replace_in_a_render
  python optimize_prompts.py --folder ./tests --iterations 5
  python optimize_prompts.py --folder ./tests --iterations 3 --output-dir ./runs
  python optimize_prompts.py --folder ./tests --prompt direct_render_instruction
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

# Google Drive uploader (optional — only loaded if --drive-folder is given)
_DriveUploader = None

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx", file=sys.stderr)
    sys.exit(1)

# Load .env from the same directory as this script (same as main.py)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv optional; user can set env vars manually

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROMPT_CONFIG_PATH = BASE_DIR / "prompt_config.json"
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# ── Prompt keys and defaults ──────────────────────────────────────────────────
# These mirror the defaults embedded in main.py so the optimizer starts from
# the current best-known baseline.

DEFAULTS: dict[str, str] = {
    "update_render_direct_instruction": (
        "You are an expert architectural visualization artist. "
        "Image 1 is a new architectural mass/volume model — the EXACT geometric blueprint for a new building. "
        "Image 2 is an existing photorealistic render of a site whose building will be replaced.\n\n"
        "STEP 1 — STUDY Image 1 carefully before generating anything:\n"
        "- Count every distinct volume, tower, and mass block.\n"
        "- Note the exact overall silhouette from base to crown.\n"
        "- Identify every architectural element: setbacks, fins, sky bridges, balcony lines, "
        "floor plates, podium shape, canopy, lattice, crown geometry.\n"
        "- Memorize the width-to-height ratio of every volume.\n\n"
        "STEP 2 — REPLACE the building in Image 2 with EXACTLY what you studied in Step 1.\n\n"
        "SHAPE RULES — ABSOLUTE, NON-NEGOTIABLE:\n"
        "- THE OVERALL 3-D MASSING ENVELOPE MUST BE IDENTICAL TO IMAGE 1. "
        "Do not simplify, round off, or generalize the shape.\n"
        "- EVERY VOLUME must be present: same count, same relative position, same proportions.\n"
        "- HEIGHT IS FIXED. Do NOT compress, elongate, or rescale any volume vertically.\n"
        "- THE OUTER SILHOUETTE (top profile, crown, sides) must match Image 1 exactly — "
        "pixel for pixel in composition.\n\n"
        "ARCHITECTURAL ELEMENTS — REPRODUCE ALL OF THEM:\n"
        "- Crown / top termination: reproduce the exact shape, cutouts, fins, or taper.\n"
        "- Setbacks: every step-back in the massing must appear at the correct floor level.\n"
        "- Podium / base: exact footprint, curved or straight edges, canopy or lattice if present.\n"
        "- Connecting elements: sky bridges, structural links — keep them where Image 1 shows them.\n"
        "- Facade grid: floor lines, balcony rails, structural bays — maintain their rhythm and density.\n"
        "- DO NOT omit, merge, smooth, or 'improve' any element. If it is in Image 1, it must appear.\n\n"
        "SCENE INTEGRATION (from Image 2 — already good, keep it):\n"
        "- Preserve camera angle, perspective, and all surroundings exactly.\n"
        "- Match lighting, shadows, and atmosphere of Image 2.\n"
        "- Apply photorealistic facade materials appropriate to the scene.\n"
        "Output only the rendered image, no text."
    ),
    "direct_render_instruction": (
        "You are an expert architectural visualization artist. "
        "The first image is an architectural mass/volume model — treat it as a strict geometric blueprint. "
        "The second image is a reference architectural render showing the desired materials and style. "
        "\n\nCRITICAL — GEOMETRY RULES (from Image 1, absolutely non-negotiable):\n"
        "- SILHOUETTE: The outer silhouette of the building must be PIXEL-IDENTICAL to Image 1. "
        "Do not alter the boundary, outline, or overall form in any way.\n"
        "- HEIGHT: The height of each tower is a fixed constraint. Do NOT compress, elongate, or "
        "rescale any building vertically. Tower heights and their ratios must be preserved exactly.\n"
        "- PROPORTIONS: Each tower's width-to-height ratio must match Image 1 exactly. "
        "Do not make towers wider, narrower, taller, or shorter than shown.\n"
        "- Reproduce the EXACT number of towers and their relative positions.\n"
        "- Preserve the precise crown/top profile of every tower (shape, slant, cutouts, fins).\n"
        "- Keep every connecting element: sky bridges, structural links, transitions between towers.\n"
        "- Maintain the base/podium form: its footprint, curved elements, canopy, or lattice structure.\n"
        "- Do NOT simplify, merge, add, smooth, or omit any architectural feature shown in the mass.\n"
        "- This is a STYLE TRANSFER only — you are changing materials and lighting, NOT redesigning the building.\n"
        "\nSTYLE (from Image 2 only): apply the facade materials, glass type and color, structural "
        "finish, lighting, sky, vegetation, and overall atmosphere.\n"
        "Output only the rendered image, no text."
    ),
    "analyzer_instruction": (
        "You are a senior architectural visualization director. "
        "Analyze both images carefully:\n"
        "- Image 1: an architectural mass/volume model — the EXACT geometric blueprint.\n"
        "- Image 2: a reference architectural render showing the target style only.\n\n"
        "Write a detailed image generation prompt (300-500 words) that will guide an AI image model "
        "to render Image 1's geometry in the exact style of Image 2.\n\n"
        "SECTION 1 — GEOMETRY (from Image 1, must be described with full precision):\n"
        "- Exact number of towers and their relative heights. Express each tower's height as a "
        "fraction of the tallest tower (e.g. 'main tower full height, secondary tower 60% as tall').\n"
        "- CRITICAL: describe the exact outer silhouette of the entire composition — this is the "
        "single most important constraint; the generated image must match it exactly.\n"
        "- Crown/top profile of each tower: describe the exact shape, any cutouts, fins, tapers, or distinctive terminations.\n"
        "- All connecting elements: sky bridges, structural links, podium transitions between towers — describe location and form.\n"
        "- Base/podium structure: footprint, curved or lattice elements, canopy, entrance volumes.\n"
        "- Any other distinctive geometric features (setbacks, chamfers, openings).\n"
        "IMPORTANT: the prompt you write must explicitly instruct the image model that:\n"
        "1. The outer silhouette and every tower's height must be IDENTICAL to Image 1 — "
        "do not alter height, width, or outline under any circumstances.\n"
        "2. This is a STYLE TRANSFER only — change materials and lighting, NOT the building geometry.\n"
        "3. It must NOT compress, elongate, widen, narrow, simplify, merge, add, or omit any feature.\n\n"
        "SECTION 2 — STYLE (from Image 2 only):\n"
        "- Facade materials, textures, colors, glass type.\n"
        "- Structural and architectural surface details.\n"
        "- Lighting conditions, time of day, shadows.\n"
        "- Sky, weather, atmosphere.\n"
        "- Surrounding context, ground, vegetation.\n"
        "- Camera angle and framing.\n\n"
        "Output only the prompt text, no preamble."
    ),
}

OPTIMIZABLE_KEYS = list(DEFAULTS.keys())

# Maps --tab argument to the primary prompt key for that tab
TAB_DEFAULT_KEY = {
    "style-transfer": "direct_render_instruction",
    "update-render": "update_render_direct_instruction",
}

# ── Prompt config I/O ─────────────────────────────────────────────────────────

def load_prompt_config() -> dict:
    if PROMPT_CONFIG_PATH.exists():
        return json.loads(PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def save_prompt_config(config: dict) -> None:
    PROMPT_CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → Saved prompts to {PROMPT_CONFIG_PATH}")


def get_instruction(config: dict, key: str) -> str:
    return config.get(key) or DEFAULTS.get(key, "")


# ── Image helpers ─────────────────────────────────────────────────────────────

def _b64(p: Path) -> tuple[str, str]:
    suffix = p.suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else "image/png"
    return base64.b64encode(p.read_bytes()).decode(), mime


# ── Server: generate via local backend ────────────────────────────────────────

async def server_generate(
    mass: Path,
    render: Path,
    tab: str,
    server: str,
) -> Optional[bytes]:
    """Submit a job to the local backend server and return the output image bytes.

    The server reads prompt_config.json on every request, so prompt changes
    are picked up automatically without restarting.
    """
    endpoint = "update-render" if tab == "update-render" else "style-transfer"

    async with httpx.AsyncClient(timeout=60) as http:
        # Submit job
        if tab == "update-render":
            files = {
                "render": (render.name, render.read_bytes(), "image/jpeg"),
                "mass":   (mass.name,   mass.read_bytes(),   "image/png"),
            }
        else:
            files = {
                "reference": (render.name, render.read_bytes(), "image/jpeg"),
                "mass":      (mass.name,   mass.read_bytes(),   "image/png"),
            }
        r = await http.post(
            f"{server}/api/{endpoint}",
            files=files,
            data={"model": "gemini-direct", "prompt": ""},
        )
        r.raise_for_status()
        job_id = r.json()["jobId"]
        print(f"    job {job_id[:8]}… submitted", flush=True)

    # Poll until done (max 5 min)
    for attempt in range(60):
        await asyncio.sleep(5)
        async with httpx.AsyncClient(timeout=10) as http:
            try:
                r = await http.get(f"{server}/api/job/{job_id}")
                data = r.json()
            except Exception:
                continue
        status = data.get("status")
        if status == "done":
            out_url = server + data["output_url"]
            async with httpx.AsyncClient(timeout=60) as http:
                img_r = await http.get(out_url)
            print(f"    ✓ done after {(attempt+1)*5}s", flush=True)
            return img_r.content
        elif status == "error":
            print(f"    ✗ job error: {data.get('error', '?')}", flush=True)
            return None

    print("    ✗ job timed out (5 min)", flush=True)
    return None


# ── Gemini: judge ─────────────────────────────────────────────────────────────

async def gemini_judge(
    mass: Path,
    expected: Path,
    generated: bytes,
    key: str,
) -> dict:
    """Score generated output vs expected result."""
    mass_b64, mass_mime = _b64(mass)
    exp_b64, exp_mime = _b64(expected)
    gen_b64 = base64.b64encode(generated).decode()

    judge_prompt = (
        "You are evaluating an AI-generated architectural visualization.\n"
        "Three images are provided:\n"
        "- Image 1: architectural mass/wireframe model (exact geometric blueprint)\n"
        "- Image 2: expected photorealistic result (ground truth target)\n"
        "- Image 3: AI-generated output to evaluate\n\n"
        "Score Image 3 on EACH criterion (integer 0-10):\n"
        "  overall_shape        — MOST CRITICAL. Does the complete 3-D silhouette of the building "
        "in Image 3 match Image 1 EXACTLY? Same number of volumes, same massing envelope, same "
        "top profile, same width-to-height ratios. A completely wrong shape scores 0.\n"
        "  height_preservation  — Are every tower/volume height preserved exactly from Image 1? "
        "Compressed, stretched, or rescaled volumes score 0.\n"
        "  architectural_elements — Are ALL specific architectural features from Image 1 present and "
        "correctly reproduced: crown details, podium, setbacks, fins, sky bridges, balcony lines, "
        "floor plates, structural grid, canopy, lattice? Missing or simplified elements score 0.\n"
        "  style_quality        — Does Image 3 match Image 2's materials, glass type, facade texture, "
        "and color palette?\n"
        "  scene_integration    — Does the building blend naturally into the scene: correct scale, "
        "seamless edges, consistent perspective?\n"
        "  context_preservation — Are ALL surrounding elements from Image 2 unchanged: sky, roads, "
        "vegetation, adjacent buildings, ground?\n"
        "  lighting_match       — Does lighting direction, shadow, and color temperature match Image 2?\n\n"
        "Also provide:\n"
        "  issues         — top 3 specific problems; always start with shape/element failures if any\n"
        "  geometry_delta — describe EXACTLY how Image 3's overall shape differs from Image 1 "
        "(mention every volume that is wrong, missing, or distorted)\n\n"
        "Return ONLY valid JSON (no markdown, no extra text):\n"
        '{"overall_shape":0,"height_preservation":0,"architectural_elements":0,'
        '"style_quality":0,"scene_integration":0,"context_preservation":0,'
        '"lighting_match":0,"issues":["","",""],"geometry_delta":""}'
    )

    payload = {
        "contents": [{"parts": [
            {"text": judge_prompt},
            {"inline_data": {"mime_type": mass_mime, "data": mass_b64}},
            {"inline_data": {"mime_type": exp_mime, "data": exp_b64}},
            {"inline_data": {"mime_type": "image/png", "data": gen_b64}},
        ]}],
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={key}"
    )
    async with httpx.AsyncClient(timeout=120) as http:
        r = await http.post(url, json=payload)
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1:
            raise ValueError(f"No JSON in judge response: {text[:300]}")
        return json.loads(text[start:end])


# ── Gemini: optimize ──────────────────────────────────────────────────────────

async def gemini_optimize(
    key_name: str,
    current_instruction: str,
    feedback: str,
    key: str,
) -> str:
    """Use Gemini to write an improved instruction based on aggregated feedback."""
    context = {
        "direct_render_instruction": (
            "This instruction is given directly to an IMAGE GENERATION model. "
            "It receives: Image 1 = architectural mass/wireframe, Image 2 = style reference render. "
            "It must output a photorealistic rendered image."
        ),
        "analyzer_instruction": (
            "This instruction is given to a TEXT model that acts as an analyst. "
            "It receives: Image 1 = architectural mass/wireframe, Image 2 = style reference render. "
            "It must output a 300-500 word text prompt that will then be given to an IMAGE GENERATION model. "
            "The analyst's job is to describe the geometry and style so precisely that the image generator "
            "cannot deviate from them."
        ),
    }.get(key_name, "This is an instruction for an architectural visualization AI.")

    opt_prompt = (
        f"You are a prompt engineer specializing in architectural visualization AI.\n\n"
        f"CONTEXT: {context}\n\n"
        "TASK: Rewrite the instruction below to fix the failures listed in the test results.\n\n"
        "CURRENT INSTRUCTION:\n"
        "---\n" + current_instruction + "\n---\n\n"
        "AGGREGATED TEST FAILURES:\n"
        "---\n" + feedback + "\n---\n\n"
        "THE TWO KNOWN FAILURE MODES (fix these first):\n"
        "1. OVERALL SHAPE — the model generates a generic or simplified building instead of "
        "reproducing the exact 3-D massing from Image 1. Every volume, every setback, every "
        "protrusion must match.\n"
        "2. ARCHITECTURAL ELEMENTS — specific features (crown details, fins, sky bridges, podium, "
        "balcony lines, floor plates, lattice, canopy) are omitted or smoothed away. "
        "These must be reproduced one-by-one.\n\n"
        "Write an IMPROVED instruction that forces the model to:\n"
        "- Study the mass model volume-by-volume before generating anything.\n"
        "- Reproduce every architectural element explicitly — no simplification allowed.\n"
        "- Treat the mass model as a hard constraint, not a suggestion.\n\n"
        "Note: atmosphere, camera angle, and landscape are already good — do NOT over-specify them.\n\n"
        "Be commanding. Use ALL CAPS for shape and element constraints.\n"
        "Return ONLY the new instruction text — no explanation, no markdown, no preamble."
    )

    payload = {"contents": [{"parts": [{"text": opt_prompt}]}]}
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-pro:generateContent?key={key}"
    )
    async with httpx.AsyncClient(timeout=180) as http:
        r = await http.post(url, json=payload)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


# ── Triplet finder ─────────────────────────────────────────────────────────────

def find_triplets(folder: Path) -> list[dict]:
    triplets = []
    for subdir in sorted(folder.iterdir()):
        if not subdir.is_dir():
            continue
        images = [p for p in subdir.iterdir() if p.suffix.lower() in ALLOWED_SUFFIXES]
        mass   = next((p for p in images if "--mass"   in p.stem.lower()), None)
        render = next((p for p in images if "--render" in p.stem.lower()), None)
        result = next((p for p in images if "--result" in p.stem.lower()), None)
        if not mass or not render or not result:
            missing = [n for n, p in [("mass", mass), ("render", render), ("result", result)] if not p]
            print(f"  [skip] {subdir.name}: missing {', '.join(missing)}")
            continue
        triplets.append({"name": subdir.name, "mass": mass, "render": render, "result": result})
    return triplets


# ── Score helpers ─────────────────────────────────────────────────────────────

def _avg(scores: list[dict], key: str) -> float:
    vals = [s[key] for s in scores if isinstance(s.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def composite_score(scores: list[dict]) -> float:
    """Weighted composite (shape-first — reflects known failure mode):
    overall_shape 35% + height_preservation 20% + architectural_elements 25%
    + style_quality 10% + scene_integration 5% + context_preservation 3%
    + lighting_match 2%
    """
    return (
        0.35 * _avg(scores, "overall_shape")
        + 0.25 * _avg(scores, "architectural_elements")
        + 0.20 * _avg(scores, "height_preservation")
        + 0.10 * _avg(scores, "style_quality")
        + 0.05 * _avg(scores, "scene_integration")
        + 0.03 * _avg(scores, "context_preservation")
        + 0.02 * _avg(scores, "lighting_match")
    )


def build_feedback_summary(scores: list[dict], names: list[str]) -> str:
    lines = []
    for name, score in zip(names, scores):
        if not score:
            continue
        lines.append(
            f"[{name}] "
            f"shape={score.get('overall_shape','?')}/10 "
            f"arch_elements={score.get('architectural_elements','?')}/10 "
            f"proportion={score.get('height_preservation','?')}/10 "
            f"style={score.get('style_quality','?')}/10 "
            f"scene={score.get('scene_integration','?')}/10 "
            f"context={score.get('context_preservation','?')}/10 "
            f"lighting={score.get('lighting_match','?')}/10"
        )
        for issue in (score.get("issues") or [])[:2]:
            lines.append(f"  issue: {issue}")
        delta = (score.get("geometry_delta") or "").strip()
        if delta:
            lines.append(f"  geometry_delta: {delta[:250]}")
    lines.append(
        f"\nAverages across {len(scores)} case(s): "
        f"shape={_avg(scores,'overall_shape'):.1f} "
        f"arch_elements={_avg(scores,'architectural_elements'):.1f} "
        f"proportion={_avg(scores,'height_preservation'):.1f} "
        f"style={_avg(scores,'style_quality'):.1f} "
        f"scene={_avg(scores,'scene_integration'):.1f} "
        f"context={_avg(scores,'context_preservation'):.1f} "
        f"lighting={_avg(scores,'lighting_match'):.1f}"
    )
    return "\n".join(lines)


# ── One iteration ─────────────────────────────────────────────────────────────

async def run_iteration(
    triplets: list[dict],
    tab: str,
    server: str,
    key: str,
    output_dir: Path,
    iteration: int,
    drive=None,
    drive_folder_id: Optional[str] = None,
) -> tuple[list[dict], list[Optional[bytes]]]:
    scores: list[dict] = []
    images: list[Optional[bytes]] = []

    for triplet in triplets:
        name = triplet["name"]
        print(f"  Generating [{name}] ...", flush=True)
        img = await server_generate(triplet["mass"], triplet["render"], tab, server)
        images.append(img)

        if img is None:
            print(f"  [{name}] generation failed, skipping judge")
            scores.append({})
            continue

        out_path = output_dir / f"iter{iteration:02d}_{name}.png"
        out_path.write_bytes(img)

        if drive and drive_folder_id:
            try:
                link = drive.upload_file(out_path, drive_folder_id)
                print(f"  [Drive] {out_path.name} → {link}", flush=True)
            except Exception as exc:
                print(f"  [Drive] upload failed for {out_path.name}: {exc}", flush=True)

        print(f"  Judging  [{name}] ...", flush=True)
        try:
            score = await gemini_judge(triplet["mass"], triplet["result"], img, key)
            scores.append(score)
            print(
                f"  [{name}] shape={score.get('overall_shape','?')} "
                f"arch={score.get('architectural_elements','?')} "
                f"proportion={score.get('height_preservation','?')} "
                f"style={score.get('style_quality','?')} "
                f"scene={score.get('scene_integration','?')} "
                f"context={score.get('context_preservation','?')} "
                f"lighting={score.get('lighting_match','?')}",
                flush=True,
            )
        except Exception as exc:
            print(f"  [{name}] judge error: {exc}", flush=True)
            scores.append({})

    return scores, images


# ── Main ───────────────────────────────────────────────────────────────────────

async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="ArchRender AI — Prompt Optimization Agent"
    )
    parser.add_argument(
        "--folder", type=Path, required=True,
        help="Parent folder whose subdirs contain --mass, --render, and --result files",
    )
    parser.add_argument(
        "--iterations", type=int, default=4,
        help="Optimization iterations (default: 4)",
    )
    parser.add_argument(
        "--tab", choices=list(TAB_DEFAULT_KEY.keys()), default=None,
        help="Tab to optimize: style-transfer or update-render. "
             "Sets --prompt automatically if --prompt is not given.",
    )
    parser.add_argument(
        "--prompt", choices=OPTIMIZABLE_KEYS, default=None,
        help=f"Which prompt key to optimize. "
             f"Options: {', '.join(OPTIMIZABLE_KEYS)}. "
             "Defaults to the primary key for --tab (or direct_render_instruction).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to save intermediate outputs and scores (default: ./opt_runs/<id>)",
    )
    parser.add_argument(
        "--server", default="http://localhost:8000",
        help="Base URL of the running ArchRender backend (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--key", default=None,
        help="Gemini API key (default: GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--drive-folder", default=None,
        help="Google Drive folder ID to upload results into "
             "(default: GOOGLE_DRIVE_FOLDER_ID env var)",
    )
    parser.add_argument(
        "--drive-credentials", default=None,
        help="Path to Google service account JSON credentials file "
             "(default: GOOGLE_APPLICATION_CREDENTIALS env var)",
    )
    args = parser.parse_args()

    server = args.server.rstrip("/")

    key = args.key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        print("ERROR: GEMINI_API_KEY not set. Use --key or set the environment variable.", file=sys.stderr)
        sys.exit(1)

    if not args.folder.is_dir():
        print(f"ERROR: --folder is not a directory: {args.folder}", file=sys.stderr)
        sys.exit(1)

    run_id = str(uuid.uuid4())[:8]
    output_dir = args.output_dir or (BASE_DIR / "opt_runs" / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # ── Google Drive setup (optional) ─────────────────────────────────────────
    drive: Optional[object] = None
    drive_run_folder_id: Optional[str] = None
    drive_folder_id = args.drive_folder or os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
    if drive_folder_id:
        creds_path = args.drive_credentials or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        if not creds_path:
            print(
                "WARNING: --drive-folder given but no credentials found.\n"
                "Pass --drive-credentials or set GOOGLE_APPLICATION_CREDENTIALS.",
                file=sys.stderr,
            )
        else:
            try:
                from drive_upload import DriveUploader
                drive = DriveUploader(drive_folder_id, creds_path)
                drive_run_folder_id = drive.create_run_folder(run_id)
                print(f"Drive folder   : {drive_folder_id} / {run_id}")
            except Exception as exc:
                print(f"WARNING: Drive setup failed: {exc}", file=sys.stderr)
                drive = None
    print()

    # Find triplets
    print("Scanning for triplets (mass + render + result)...")
    triplets = find_triplets(args.folder)
    if not triplets:
        print(
            "ERROR: No valid triplets found.\n"
            "Each subfolder needs a --mass file, a --render file, AND a --result file.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Found {len(triplets)} triplet(s): {[t['name'] for t in triplets]}\n")

    # Resolve which tab and prompt key to optimize
    tab = args.tab or "update-render"
    if args.prompt:
        prompt_key = args.prompt
    elif args.tab:
        prompt_key = TAB_DEFAULT_KEY[args.tab]
    else:
        prompt_key = TAB_DEFAULT_KEY[tab]

    # Load current prompts
    config = load_prompt_config()
    instruction = get_instruction(config, prompt_key)
    print(f"Optimizing: {prompt_key}")
    print(f"Starting instruction ({len(instruction)} chars):\n{instruction[:200]}...\n")

    best_score = -1.0
    best_instruction = instruction
    best_iteration = 0
    history = []

    print(f"{'─' * 64}")
    for iteration in range(1, args.iterations + 1):
        print(f"\n── Iteration {iteration}/{args.iterations} {'─' * 44}")

        scores, _ = await run_iteration(
            triplets, tab, server, key, output_dir, iteration,
            drive=drive, drive_folder_id=drive_run_folder_id,
        )
        valid = [s for s in scores if s]

        if not valid:
            print("  No valid scores this iteration.")
            continue

        cscore = composite_score(valid)
        valid_names = [t["name"] for t, s in zip(triplets, scores) if s]
        feedback = build_feedback_summary(valid, valid_names)

        print(f"\n  Composite score: {cscore:.2f}/10  "
              f"(shape={_avg(valid,'overall_shape'):.1f} "
              f"arch={_avg(valid,'architectural_elements'):.1f} "
              f"height={_avg(valid,'height_preservation'):.1f} "
              f"style={_avg(valid,'style_quality'):.1f} "
              f"scene={_avg(valid,'scene_integration'):.1f} "
              f"context={_avg(valid,'context_preservation'):.1f} "
              f"lighting={_avg(valid,'lighting_match'):.1f})")

        history.append({
            "iteration": iteration,
            "composite": cscore,
            "overall_shape": _avg(valid, "overall_shape"),
            "architectural_elements": _avg(valid, "architectural_elements"),
            "height": _avg(valid, "height_preservation"),
            "style": _avg(valid, "style_quality"),
            "scene_integration": _avg(valid, "scene_integration"),
            "context_preservation": _avg(valid, "context_preservation"),
            "lighting_match": _avg(valid, "lighting_match"),
            "instruction": instruction,
        })

        if cscore > best_score:
            best_score = cscore
            best_instruction = instruction
            best_iteration = iteration
            print("  ✓ New best — saving to prompt_config.json")
            config[prompt_key] = best_instruction
            save_prompt_config(config)

        # Save per-iteration scores
        scores_path = output_dir / f"iter{iteration:02d}_scores.json"
        scores_path.write_text(
            json.dumps({
                "iteration": iteration, "composite": cscore,
                "cases": [{"name": t["name"], "scores": s} for t, s in zip(triplets, scores)],
                "instruction": instruction,
            }, indent=2),
            encoding="utf-8",
        )
        if drive and drive_run_folder_id:
            try:
                drive.upload_file(scores_path, drive_run_folder_id)
            except Exception as exc:
                print(f"  [Drive] scores upload failed: {exc}", flush=True)

        # Generate improved instruction for next iteration
        if iteration < args.iterations:
            print(f"\n  Generating improved instruction...")
            try:
                instruction = await gemini_optimize(prompt_key, instruction, feedback, key)
                print(f"  New instruction ({len(instruction)} chars): {instruction[:120]}...")
            except Exception as exc:
                print(f"  Optimizer error: {exc} — keeping current instruction")

    # Final summary
    print(f"\n{'─' * 64}")
    print(f"Best iteration : {best_iteration}")
    print(f"Best score     : {best_score:.2f}/10")

    # Save history
    history_path = output_dir / "optimization_history.json"
    history_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    print(f"History saved  : {history_path}")

    # Upload final files to Drive
    if drive and drive_run_folder_id:
        for final_path in [history_path, PROMPT_CONFIG_PATH]:
            if final_path.exists():
                try:
                    link = drive.upload_file(final_path, drive_run_folder_id)
                    print(f"[Drive] {final_path.name} → {link}")
                except Exception as exc:
                    print(f"[Drive] upload failed for {final_path.name}: {exc}", file=sys.stderr)

    print("\nDone! The optimized prompts are already in prompt_config.json.")
    print("The running server picks them up on the next request (no restart needed).")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
