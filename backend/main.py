import sys

# Ensure Python 3.11+
if sys.version_info < (3, 11):
    print(f"ERROR: Python 3.11+ required, running {sys.version}")
    print("Run with: py -3.11 main.py")
    sys.exit(1)

import os
import uuid
import json
import asyncio
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import io
import httpx
from PIL import Image as _PILImage
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Path helpers (works both in dev and when frozen by PyInstaller) ──────────

def _base_dir() -> Path:
    """Directory that holds the exe (frozen) or the script (dev)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def _bundle_dir() -> Path:
    """Root of the PyInstaller bundle (_MEIPASS) or the script dir in dev."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).parent


BASE_DIR = _base_dir()
BUNDLE_DIR = _bundle_dir()

PROMPT_CONFIG_PATH = BASE_DIR / "prompt_config.json"


def _load_prompt(key: str, default: str) -> str:
    """Load a prompt override from prompt_config.json; falls back to default.
    File is re-read on every call so the optimizer can update prompts while the
    server is running without requiring a restart."""
    try:
        if PROMPT_CONFIG_PATH.exists():
            cfg = json.loads(PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
            if key in cfg and isinstance(cfg[key], str) and cfg[key].strip():
                return cfg[key]
    except Exception:
        pass
    return default


app = FastAPI(title="ArchRender AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (use Redis/DB for production)
jobs: dict[str, dict] = {}

# Runtime dirs live next to the exe so they persist between runs
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
LIBRARY_MASS_DIR = BASE_DIR / "library" / "mass"
LIBRARY_RENDER_DIR = BASE_DIR / "library" / "render"
TEST_RESULTS_DIR = BASE_DIR / "test_results"

for _d in (UPLOADS_DIR, OUTPUTS_DIR, LIBRARY_MASS_DIR, LIBRARY_RENDER_DIR, TEST_RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/library/mass", StaticFiles(directory=str(LIBRARY_MASS_DIR)), name="library_mass")
app.mount("/library/render", StaticFiles(directory=str(LIBRARY_RENDER_DIR)), name="library_render")
app.mount("/test_results", StaticFiles(directory=str(TEST_RESULTS_DIR)), name="test_results")


def save_upload(file: UploadFile) -> Path:
    ext = Path(file.filename or "file.png").suffix or ".png"
    dest = UPLOADS_DIR / f"{uuid.uuid4()}{ext}"
    content = file.file.read()
    dest.write_bytes(content)
    return dest


def image_to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix, "image/png")
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def get_api_token(api_token: Optional[str] = None) -> str:
    return api_token or os.getenv("REPLICATE_API_TOKEN", "")


def get_gemini_key() -> str:
    return os.getenv("GEMINI_API_KEY", "")


# Maximum pixel dimension sent to Gemini. Images larger than this are resized
# proportionally before base64-encoding. Larger images cause Gemini API timeouts.
_GEMINI_MAX_PX = 1536


def _resize_for_gemini(source: "Path | bytes", is_render: bool = False) -> tuple[bytes, str]:
    """Return (image_bytes, mime_type) resized so the longest side ≤ _GEMINI_MAX_PX.

    Renders (photos) are re-encoded as JPEG (quality=90) for smaller payloads.
    Mass models (diagrams) are kept as PNG to preserve clean edges.
    If the image is already within the limit it is returned as-is (original bytes).
    """
    raw = source if isinstance(source, bytes) else source.read_bytes()
    try:
        img = _PILImage.open(io.BytesIO(raw))
        w, h = img.size
        if max(w, h) > _GEMINI_MAX_PX:
            ratio = _GEMINI_MAX_PX / max(w, h)
            new_w, new_h = max(1, int(w * ratio)), max(1, int(h * ratio))
            img = img.resize((new_w, new_h), _PILImage.LANCZOS)
            print(f"  [resize] {w}×{h} → {new_w}×{new_h}", flush=True)
        buf = io.BytesIO()
        if is_render:
            img.convert("RGB").save(buf, format="JPEG", quality=90)
            return buf.getvalue(), "image/jpeg"
        else:
            img.save(buf, format="PNG")
            return buf.getvalue(), "image/png"
    except Exception:
        # Fall back to raw bytes if PIL fails
        suffix = (source.suffix.lower() if isinstance(source, Path) else "")
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        return raw, mime


def _b64_for_gemini(source: "Path | bytes", is_render: bool = False) -> tuple[str, str]:
    """Convenience wrapper: resize → base64 encode → return (b64_str, mime_type)."""
    img_bytes, mime = _resize_for_gemini(source, is_render=is_render)
    return base64.b64encode(img_bytes).decode(), mime


def _crop_mass_to_content(mass_path: Path) -> bytes:
    """Crop white/transparent padding from a mass model image so the geometry fills the frame.

    Returns PNG bytes of the cropped+padded image. If cropping fails for any reason
    the original file bytes are returned unchanged.
    """
    try:
        img = _PILImage.open(mass_path).convert("RGBA")
        r, g, b, a = img.split()
        # Build a mask of non-white, non-transparent pixels
        rgb = _PILImage.merge("RGB", (r, g, b))
        diff = _PILImage.new("RGB", img.size, (255, 255, 255))
        from PIL import ImageChops, ImageFilter
        delta = ImageChops.difference(rgb, diff)
        # Threshold: any pixel that differs from white by > 15 in any channel
        mask = delta.convert("L").point(lambda x: 255 if x > 15 else 0)
        # Also consider alpha — non-transparent pixels count
        alpha_mask = a.point(lambda x: 255 if x > 30 else 0)
        from PIL import ImageOps
        combined = _PILImage.new("L", img.size, 0)
        combined.paste(mask, mask=mask)
        combined.paste(alpha_mask, mask=alpha_mask)
        bbox = combined.getbbox()
        if bbox is None:
            return mass_path.read_bytes()
        # Add 5% padding around the content
        w, h = img.size
        pad_x = max(int((bbox[2] - bbox[0]) * 0.05), 10)
        pad_y = max(int((bbox[3] - bbox[1]) * 0.05), 10)
        x0 = max(bbox[0] - pad_x, 0)
        y0 = max(bbox[1] - pad_y, 0)
        x1 = min(bbox[2] + pad_x, w)
        y1 = min(bbox[3] + pad_y, h)
        cropped = img.crop((x0, y0, x1, y1)).convert("RGB")
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return mass_path.read_bytes()


async def _extract_mass_geometry(mass_path: Path, key: str, cropped_bytes: bytes) -> str:
    """Call Gemini text model to extract a structured geometry description from the mass model.

    Returns a plain-text geometry contract that can be injected into generation prompts.
    """
    mass_b64 = base64.b64encode(cropped_bytes).decode()
    prompt = (
        "You are analyzing an architectural mass/wireframe model image.\n"
        "Extract a precise geometry description covering ALL of the following — be specific and quantitative:\n\n"
        "1. TOWER COUNT: exact number of distinct vertical volumes/towers visible\n"
        "2. TOWER HEIGHTS: express each tower's height as a percentage of the tallest tower "
        "(e.g. 'Tower A: 100%, Tower B: 65%, Tower C: 58%')\n"
        "3. TOWER SHAPES: describe the cross-section and profile of each tower "
        "(e.g. 'cylindrical', 'rectangular with chamfered corners', 'curved blade', "
        "'stacked segmented volumes', 'organic tapering form')\n"
        "4. SILHOUETTE: describe the overall outer silhouette as seen from this camera angle — "
        "trace the top profile left to right\n"
        "5. CROWN/TOP: exact top termination of each tower "
        "(e.g. 'flat circular crown with gold rim', 'pointed blade', 'rounded top', 'forked peaks')\n"
        "6. PODIUM/BASE: describe the base structure "
        "(e.g. 'rectangular podium', 'organic curved lattice', 'woven rib structure', 'no podium')\n"
        "7. DISTINCTIVE FEATURES: any unique elements "
        "(e.g. 'horizontal banding every 4 floors', 'golden parametric ribs connecting towers', "
        "'curved inward taper at mid-height', 'sky bridge at 60% height')\n"
        "8. RELATIVE POSITIONS: spatial arrangement of towers "
        "(e.g. 'left tower flanks right, central tower set back', 'two towers side by side')\n\n"
        "Output ONLY the numbered list above, no preamble. Be concise but precise."
    )
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/png", "data": mass_b64}},
        ]}],
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={key}"
    )
    try:
        async with httpx.AsyncClient(timeout=60) as http:
            r = await http.post(url, json=payload)
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[mass geometry extraction failed: {e}]", flush=True)
        return ""


async def gemini_generate_render(mass_path: Path, reference_path: Path, prompt: str = "") -> Path:
    """Send mass + reference to Gemini 2.0 Flash image generation. Returns saved output path."""
    key = get_gemini_key()
    if not key:
        raise ValueError("GEMINI_API_KEY not set in .env")

    # Use original mass (uncropped) for generation — the mass is already positioned at the exact
    # scale and location the building occupies in the render scene; cropping would destroy that.
    mass_b64, mass_mime = _b64_for_gemini(mass_path, is_render=False)
    ref_b64, ref_mime   = _b64_for_gemini(reference_path, is_render=True)

    # For geometry extraction, crop to content so the text model can read details clearly,
    # but the original (uncropped) mass is what gets sent to the image generator.
    cropped_for_extraction = _crop_mass_to_content(mass_path)
    # Extract explicit geometry description from the mass model
    geometry_contract = await _extract_mass_geometry(mass_path, key, cropped_for_extraction)
    geometry_section = ""
    if geometry_contract:
        geometry_section = (
            "\n\nGEOMETRY CONTRACT — extracted directly from Image 1. "
            "Every detail below is a HARD CONSTRAINT. Do NOT deviate:\n"
            f"{geometry_contract}\n"
            "The output MUST match every point above exactly. "
            "If any tower count, height ratio, silhouette feature, crown shape, or podium form "
            "differs from the contract above, the output is WRONG.\n"
        )

    base_instruction = (
        "You are an expert architectural visualization artist. "
        "Image 1 is an architectural mass/volume model — the EXACT geometric blueprint. "
        "Image 2 is a reference render showing the target materials and style ONLY.\n"
        "\n⚠ GEOMETRY IS LOCKED TO IMAGE 1 — THIS IS NON-NEGOTIABLE:\n"
        "- The complete 3-D silhouette, tower count, height ratios, crown shapes, podium form, "
        "and every architectural element in Image 1 must appear UNCHANGED in the output.\n"
        "- Do NOT normalize, simplify, smooth, redesign, or 'improve' any geometry.\n"
        "- Do NOT compress or elongate any tower vertically.\n"
        "- Do NOT merge separate volumes or add volumes that are not in Image 1.\n"
        "- This is a STYLE TRANSFER only: change materials and lighting, NOT the building geometry.\n"
        "\nSTYLE (from Image 2 only): facade materials, glass type, structural finish, "
        "lighting, sky, vegetation, and atmosphere.\n"
        "Output only the rendered image, no text."
    )
    instruction = _load_prompt("direct_render_instruction", base_instruction)
    instruction += geometry_section
    if prompt.strip():
        instruction += f"\nAdditional direction: {prompt.strip()}"

    payload = {
        "contents": [{
            "parts": [
                {"text": instruction},
                {"inline_data": {"mime_type": mass_mime, "data": mass_b64}},
                {"inline_data": {"mime_type": ref_mime, "data": ref_b64}},
            ]
        }],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }

    # Try image-capable models in order; skip on 404 (removed) or unsupported 400.
    _image_gen_models = [
        "gemini-2.0-flash-exp",
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image-preview",
    ]
    data = None
    _errors: list[str] = []
    for _mid in _image_gen_models:
        _url = f"https://generativelanguage.googleapis.com/v1beta/models/{_mid}:generateContent?key={key}"
        async with httpx.AsyncClient(timeout=120) as http:
            r = await http.post(_url, json=payload)
            print(f"[Gemini image-gen] {_mid} → {r.status_code}: {r.text[:300]}", flush=True)
            if r.status_code in (404, 400, 403):
                _errors.append(f"{_mid}: {r.status_code} {r.text[:120]}")
                continue
            r.raise_for_status()
            data = r.json()
            break

    if data is None:
        raise ValueError(f"No Gemini image-gen model succeeded. Errors: {'; '.join(_errors)}")

    for part in data["candidates"][0]["content"]["parts"]:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline:
            img_bytes = base64.b64decode(inline["data"])
            out_path = OUTPUTS_DIR / f"{uuid.uuid4()}.png"
            out_path.write_bytes(img_bytes)
            return out_path

    raise ValueError(f"Gemini returned no image. Response: {data}")


async def gemini_25_analyze(mass_path: Path, reference_path: Path, extra_prompt: str = "", analyzer_model: str = "gemini-2.5-pro") -> str:
    """Use a Gemini text/vision model to deeply analyze mass + reference and produce a detailed render prompt."""
    key = get_gemini_key()
    if not key:
        raise ValueError("GEMINI_API_KEY not set in .env")

    mass_b64, mass_mime = _b64_for_gemini(mass_path, is_render=False)
    ref_b64, ref_mime   = _b64_for_gemini(reference_path, is_render=True)

    instruction = _load_prompt("analyzer_instruction", (
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
    ))
    if extra_prompt.strip():
        instruction += f"\n\nAdditional direction from the user: {extra_prompt.strip()}"

    payload = {
        "contents": [{
            "parts": [
                {"text": instruction},
                {"inline_data": {"mime_type": mass_mime, "data": mass_b64}},
                {"inline_data": {"mime_type": ref_mime, "data": ref_b64}},
            ]
        }],
    }

    _url = f"https://generativelanguage.googleapis.com/v1beta/models/{analyzer_model}:generateContent?key={key}"
    async with httpx.AsyncClient(timeout=180) as http:
        r = await http.post(_url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


async def gemini_25_render(mass_path: Path, reference_path: Path, prompt: str = "", analyzer_model: str = "gemini-2.5-pro") -> Path:
    """Two-step: Gemini analyzer writes detailed prompt → image-gen model renders."""
    rich_prompt = await gemini_25_analyze(mass_path, reference_path, prompt, analyzer_model)
    return await gemini_generate_render(mass_path, reference_path, rich_prompt)


async def _analyze_for_materialization(
    placed_mass_path: Path,
    style_ref_path: Path,
    geometry_contract: str,
    prompt: str,
    analyzer_model: str,
    key: str,
) -> str:
    """Stage-2 analyzer: given a placed white-mass scene + style reference,
    produce a detailed materialization prompt that instructs the image-gen model
    to paint materials/lighting onto the white volume — without touching geometry
    or background.
    """
    placed_b64, placed_mime = _b64_for_gemini(placed_mass_path, is_render=True)
    style_b64,  style_mime  = _b64_for_gemini(style_ref_path,   is_render=True)

    geometry_block = ""
    if geometry_contract:
        geometry_block = (
            "\n\nGEOMETRY CONTRACT (already correctly placed in Image 1 — "
            "must be reproduced exactly):\n"
            f"{geometry_contract}\n"
        )

    _s2a_base = (
        "You are a senior architectural visualization director.\n"
        "You have two images:\n"
        "Image 1: the SCENE — a white/grey mass model of a new building already composited into "
        "a real urban environment at the correct scale, position, and camera angle. The background "
        "(sky, roads, trees, neighboring buildings) is real photography.\n"
        "Image 2: the STYLE REFERENCE — a photorealistic architectural render of the TARGET building. "
        "This shows EXACTLY how the finished building must look: its facade materials, colors, "
        "glass type, cladding texture, window patterns, balconies, structural details, and how "
        "light hits the surface at this time of day.\n\n"
        "YOUR TASK: Write a precise image generation brief (300-450 words) so that an AI image "
        "model can paint Image 2's exact appearance onto the white mass in Image 1.\n\n"
        "The brief MUST:\n"
        "1. Start with a direct statement: 'Paint the building in Image 2's style onto the white "
        "mass in Image 1. The finished building must be visually indistinguishable in style from "
        "Image 2.'\n"
        "2. Describe Image 2's facade in exhaustive detail:\n"
        "   - Primary facade material: exact color, finish (glossy/matte/reflective), texture\n"
        "   - Glass: tint color, reflectivity, pattern (curtain wall / punched windows / full-floor bands)\n"
        "   - Secondary materials: cladding panels, metal fins, concrete, stone — colors and placement\n"
        "   - Window grid: mullion spacing, horizontal vs vertical emphasis, depth of reveals\n"
        "   - Special features: balconies, louvers, crown treatment, base podium\n"
        "   - Lighting: sun angle, shadow direction, warm/cool tones, reflections on glass\n"
        "3. State what must NOT change: background, camera angle, building silhouette/proportions\n\n"
        "Output only the brief text. No preamble, no section headers."
    )
    instruction = _load_prompt("stage2_analyzer_instruction", _s2a_base) + geometry_block
    if prompt.strip():
        instruction += f"\n\nAdditional user direction: {prompt.strip()}"

    payload = {
        "contents": [{"parts": [
            {"text": instruction},
            {"inline_data": {"mime_type": placed_mime, "data": placed_b64}},
            {"inline_data": {"mime_type": style_mime,  "data": style_b64}},
        ]}],
    }
    _url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{analyzer_model}:generateContent?key={key}"
    )
    async with httpx.AsyncClient(timeout=180) as http:
        r = await http.post(_url, json=payload)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


async def gemini_materialize_placed_mass(
    placed_mass_path: Path,
    style_ref_path: Path,
    geometry_contract: str = "",
    prompt: str = "",
    analyzer_model: str = "gemini-2.5-pro",
) -> Path:
    """Stage 2 of the staged pipeline — materialization only.

    The placed_mass_path (stage-1 output) already shows the white/grey mass
    correctly seated in the scene at the right scale, angle, and position.
    This function applies photorealistic materials and lighting to ONLY the
    white volume, keeping the background and proportions pixel-perfect.

    Pipeline:
      analyzer  → detailed materialization prompt (geometry + bg locked)
      image-gen → paint style onto white mass
    """
    key = get_gemini_key()
    if not key:
        raise ValueError("GEMINI_API_KEY not set in .env")

    rich_prompt = await _analyze_for_materialization(
        placed_mass_path, style_ref_path, geometry_contract, prompt, analyzer_model, key
    )
    print(f"[staged s2] materialization prompt ({len(rich_prompt)} chars)", flush=True)

    placed_b64, placed_mime = _b64_for_gemini(placed_mass_path, is_render=True)
    style_b64,  style_mime  = _b64_for_gemini(style_ref_path,   is_render=True)

    geometry_section = ""
    if geometry_contract:
        geometry_section = (
            "\n\nGEOMETRY CONTRACT — the white mass in Image 1 already matches these specs exactly. "
            "Reproduce them faithfully:\n"
            f"{geometry_contract}\n"
        )

    _s2m_base = (
        "You are a photorealistic architectural visualization artist.\n\n"
        "You have two images:\n"
        "Image 1 (SCENE): A real urban photograph with a white/grey building mass composited in. "
        "The mass shows the EXACT position, scale, silhouette, and camera angle of the new building. "
        "Everything outside the white mass is real photography that must not change.\n"
        "Image 2 (STYLE TARGET): A photorealistic render showing exactly how the finished building "
        "must look — its facade, materials, colors, glass, cladding, windows, and lighting.\n\n"
        "YOUR GOAL: Produce an output image that looks like Image 1 BUT with the white/grey mass "
        "replaced by the building from Image 2, rendered with full photorealistic quality. "
        "The finished building in the output must be VISUALLY IDENTICAL IN STYLE to Image 2 — "
        "same materials, same colors, same facade treatment, same glass type.\n\n"
        "STRICT RULES:\n"
        "1. STYLE PRIORITY — The building's appearance must match Image 2 as closely as possible. "
        "Same facade color, same glass tint, same cladding pattern, same window grid.\n"
        "2. SILHOUETTE LOCKED — Use the exact outline of the white mass from Image 1. "
        "Do not change height, width, or shape.\n"
        "3. BACKGROUND UNTOUCHED — Every pixel outside the building mass must be identical to Image 1.\n"
        "4. CAMERA FIXED — Same viewpoint, perspective, and framing as Image 1.\n"
        "5. LIGHTING CONSISTENT — The building's shadows and reflections must match the "
        "real-world lighting visible in Image 1's background.\n\n"
        "Think of it as: 'Replace the white mass with the building from Image 2, keeping everything "
        "else from Image 1 exactly as-is.'"
    )
    base_instruction = (
        _load_prompt("stage2_materialize_instruction", _s2m_base)
        + geometry_section
        + f"\n\nDetailed materialization guide:\n{rich_prompt}"
    )

    payload = {
        "contents": [{"parts": [
            {"text": base_instruction},
            {"inline_data": {"mime_type": placed_mime, "data": placed_b64}},
            {"inline_data": {"mime_type": style_mime,  "data": style_b64}},
        ]}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }

    _image_gen_models = [
        "gemini-2.0-flash-exp",
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image-preview",
    ]
    data = None
    _errors: list[str] = []
    for _mid in _image_gen_models:
        _url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{_mid}:generateContent?key={key}"
        )
        async with httpx.AsyncClient(timeout=180) as http:
            r = await http.post(_url, json=payload)
        print(f"[staged s2 materialize] {_mid} → {r.status_code}: {r.text[:200]}", flush=True)
        if r.status_code in (404, 400, 403):
            _errors.append(f"{_mid}: {r.status_code} {r.text[:120]}")
            continue
        if r.status_code != 200:
            _errors.append(f"{_mid}: {r.status_code} {r.text[:120]}")
            continue
        data = r.json()
        break

    if data:
        for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            if "inlineData" in part:
                img_bytes = base64.b64decode(part["inlineData"]["data"])
                out_path = OUTPUTS_DIR / f"{uuid.uuid4()}.png"
                out_path.write_bytes(img_bytes)
                return out_path

    raise ValueError(f"Stage-2 materialization: Gemini returned no image. Errors: {_errors}")


def _erase_building_from_render(mass_path: Path, render_path: Path) -> bytes:
    """Erase the existing building from the render using the mass model silhouette as a mask.

    Replaces the building region with a blurred fill so Gemini cannot copy the old building's
    geometry or style — only the mass model remains as the building reference.
    """
    try:
        from PIL import ImageFilter, ImageChops

        render = _PILImage.open(render_path).convert("RGB")
        mass = _PILImage.open(mass_path).convert("RGBA")
        mass_resized = mass.resize(render.size, _PILImage.LANCZOS)

        # Build mask: non-white / non-transparent pixels in mass = building footprint
        r, g, b, a = mass_resized.split()
        rgb = _PILImage.merge("RGB", (r, g, b))
        white = _PILImage.new("RGB", mass_resized.size, (255, 255, 255))
        delta = ImageChops.difference(rgb, white)
        color_mask = delta.convert("L").point(lambda x: 255 if x > 20 else 0)
        alpha_mask = a.point(lambda x: 255 if x > 30 else 0)
        building_mask = _PILImage.new("L", render.size, 0)
        building_mask.paste(color_mask, mask=color_mask)
        building_mask.paste(alpha_mask, mask=alpha_mask)
        # Dilate slightly to ensure full coverage
        building_mask = building_mask.filter(ImageFilter.MaxFilter(size=21))

        # Fill the building region with a heavy blur of the render (plausible background)
        blurred = render.filter(ImageFilter.GaussianBlur(radius=40))
        result = _PILImage.composite(blurred, render, building_mask)

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return render_path.read_bytes()


async def gemini_generate_render_update(mass_path: Path, render_path: Path, prompt: str = "") -> Path:
    """Like gemini_generate_render but uses the update-render instruction (scene integration)."""
    key = get_gemini_key()
    if not key:
        raise ValueError("GEMINI_API_KEY not set in .env")

    # Erase the old building from the render so Gemini cannot copy it
    clean_render_bytes = _erase_building_from_render(mass_path, render_path)

    # Original mass is sent to the generator (preserves scale/position in the scene)
    mass_b64, mass_mime = _b64_for_gemini(mass_path, is_render=False)
    ref_b64, ref_mime   = _b64_for_gemini(clean_render_bytes, is_render=True)

    # Crop for geometry extraction only (text model reads fine-detail better without white padding)
    cropped_for_extraction = _crop_mass_to_content(mass_path)
    geometry_contract = await _extract_mass_geometry(mass_path, key, cropped_for_extraction)
    geometry_section = ""
    if geometry_contract:
        geometry_section = (
            "\n\nGEOMETRY CONTRACT — extracted directly from Image 1. "
            "Every detail below is a HARD CONSTRAINT. Do NOT deviate:\n"
            f"{geometry_contract}\n"
            "The output MUST match every point above exactly. "
            "If any tower count, height ratio, silhouette feature, crown shape, or podium form "
            "differs from the contract above, the output is WRONG.\n"
        )

    base_instruction = (
        "You are an expert architectural visualization artist. "
        "Image 1 is a new architectural mass/volume model — the EXACT geometric blueprint for the new building. "
        "Image 2 is an existing photorealistic render of the site. The building in Image 2 must be replaced.\n\n"
        "⚠ TASK: Replace the existing building in Image 2 with the new building defined by Image 1. "
        "The new building occupies EXACTLY the same position and footprint in the scene.\n\n"
        "GEOMETRY RULES (from Image 1, absolutely non-negotiable):\n"
        "- Reproduce the EXACT outer silhouette of Image 1 — do not alter height, width, or outline.\n"
        "- Tower height is a fixed constraint. Do NOT compress, elongate, or rescale vertically.\n"
        "- Width-to-height ratio must match Image 1 exactly.\n"
        "- Preserve every feature: crown shape, podium, setbacks, connecting elements, lattice structures.\n"
        "- Do NOT simplify, normalize, redesign, or 'improve' any aspect of the geometry from Image 1.\n\n"
        "SCENE INTEGRATION (from Image 2):\n"
        "- Match the camera angle, perspective, and focal length of Image 2 EXACTLY.\n"
        "- Preserve ALL surrounding elements unchanged: sky, roads, vegetation, other buildings, infrastructure.\n"
        "- Match the lighting direction, quality, and color temperature of Image 2.\n"
        "- The new building must cast shadows consistent with Image 2's sun angle and atmosphere.\n"
        "- Adapt facade materials to look photorealistic within Image 2's environmental context.\n"
        "Output only the rendered image, no text."
    )
    instruction = _load_prompt("update_render_direct_instruction", base_instruction)
    instruction += geometry_section
    if prompt.strip():
        instruction += f"\nAdditional direction: {prompt.strip()}"

    payload = {
        "contents": [{"parts": [
            {"text": instruction},
            {"inline_data": {"mime_type": mass_mime, "data": mass_b64}},
            {"inline_data": {"mime_type": ref_mime, "data": ref_b64}},
        ]}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }

    _image_gen_models = [
        "gemini-2.0-flash-exp",
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image-preview",
    ]
    data = None
    _errors: list[str] = []
    for _mid in _image_gen_models:
        _url = f"https://generativelanguage.googleapis.com/v1beta/models/{_mid}:generateContent?key={key}"
        async with httpx.AsyncClient(timeout=120) as http:
            r = await http.post(_url, json=payload)
            print(f"[Gemini update-render] {_mid} → {r.status_code}: {r.text[:300]}", flush=True)
            if r.status_code in (404, 400, 403):
                _errors.append(f"{_mid}: {r.status_code} {r.text[:120]}")
                continue
            r.raise_for_status()
            data = r.json()
            break

    if data is None:
        raise ValueError(f"No Gemini image-gen model succeeded. Errors: {'; '.join(_errors)}")

    for part in data["candidates"][0]["content"]["parts"]:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline:
            img_bytes = base64.b64decode(inline["data"])
            out_path = OUTPUTS_DIR / f"{uuid.uuid4()}.png"
            out_path.write_bytes(img_bytes)
            return out_path

    raise ValueError(f"Gemini returned no image. Response: {data}")


async def gemini_place_mass_in_scene(mass_path: Path, render_path: Path) -> tuple[Path, str]:
    """Stage 1 of the staged pipeline.

    Places the mass/wireframe model cleanly into the background scene at the
    correct position, scale, and perspective — outputting a clean white/grey
    massing diagram in the scene rather than a fully rendered image.

    Returns (placed_path, geometry_contract) so Stage 2 can lock the proportions.
    """
    key = get_gemini_key()
    if not key:
        raise ValueError("GEMINI_API_KEY not set in .env")

    mass_b64, mass_mime = _b64_for_gemini(mass_path, is_render=False)
    ref_b64, ref_mime   = _b64_for_gemini(render_path, is_render=True)

    # Extract geometry contract from the mass so the placement prompt can enforce exact proportions
    cropped_for_extraction = _crop_mass_to_content(mass_path)
    geometry_contract = await _extract_mass_geometry(mass_path, key, cropped_for_extraction)
    geometry_section = ""
    if geometry_contract:
        geometry_section = (
            "\n\nGEOMETRY CONTRACT — these proportions are ABSOLUTE CONSTRAINTS for the placed mass:\n"
            f"{geometry_contract}\n"
            "Every tower height ratio, silhouette feature, crown shape, and podium form listed above "
            "MUST appear correctly in your output. Do NOT rescale, simplify, or omit any element.\n"
        )

    _s1_base = (
        "You are an architectural visualization compositor.\n\n"
        "You have two images with completely different roles:\n\n"
        "Image 1 = GEOMETRY SOURCE ONLY. This is a massing/wireframe diagram of a new building. "
        "Extract ONLY the building's shape: number of towers, relative heights, silhouette, crown forms, "
        "podium, connecting elements. IGNORE Image 1's background entirely — it does not exist for your "
        "purposes. The sky, ground, and surroundings in Image 1 are irrelevant and must never appear in "
        "the output.\n\n"
        "Image 2 = THE SCENE. This is the real-world photograph you must work with. The output background "
        "must be 100% identical to Image 2 — same sky, same roads, same trees, same city, same lighting, "
        "same camera angle. You are only allowed to modify the area where the existing building sits.\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "Produce Image 2 with the existing building on site replaced by a clean flat WHITE/LIGHT-GREY "
        "solid volume that has the exact silhouette and proportions of the building shape from Image 1.\n\n"
        "STRICT RULES:\n"
        "1. OUTPUT BACKGROUND = Image 2 exactly. Every pixel outside the building area must be "
        "pixel-perfect identical to Image 2. Never use Image 1's background.\n"
        "2. WHITE MASS ONLY — the new building must be a flat, featureless white or light-grey solid. "
        "No materials, no glass, no windows, no textures, no facade details, no shadows on the mass.\n"
        "3. GEOMETRY FROM IMAGE 1 — reproduce the exact silhouette: tower count, relative heights, "
        "crown shape, podium, all connecting elements. Do not simplify.\n"
        "4. SCALE CORRECT — the building base must sit on the correct ground plane of Image 2 with "
        "proper perspective foreshortening, sized consistently with surrounding buildings.\n"
        "5. NO DRIFT — the output must look like a photograph of Image 2's city with a white cardboard "
        "model placed in it. Not a render. Not Image 1's scene. Image 2's scene only.\n"
    )
    instruction = _load_prompt("stage1_placement_instruction", _s1_base) + geometry_section + (
        "\nOutput: the scene from Image 2 with the clean mass from Image 1 placed on site. "
        "No text, no annotations."
    )

    payload = {
        "contents": [{"parts": [
            {"text": instruction},
            {"inline_data": {"mime_type": mass_mime, "data": mass_b64}},
            {"inline_data": {"mime_type": ref_mime,  "data": ref_b64}},
        ]}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }

    _image_gen_models = [
        "gemini-2.0-flash-exp",
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image-preview",
    ]
    data = None
    _errors: list[str] = []
    for _mid in _image_gen_models:
        _url = f"https://generativelanguage.googleapis.com/v1beta/models/{_mid}:generateContent?key={key}"
        async with httpx.AsyncClient(timeout=120) as http:
            r = await http.post(_url, json=payload)
            print(f"[Gemini place-mass stage-1] {_mid} → {r.status_code}: {r.text[:300]}", flush=True)
            if r.status_code in (404, 400, 403):
                _errors.append(f"{_mid}: {r.status_code} {r.text[:120]}")
                continue
            r.raise_for_status()
            data = r.json()
            break

    if data is None:
        raise ValueError(f"Stage-1 placement: no Gemini image-gen model succeeded. Errors: {'; '.join(_errors)}")

    for part in data["candidates"][0]["content"]["parts"]:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline:
            img_bytes = base64.b64decode(inline["data"])
            placed_path = OUTPUTS_DIR / f"placed_{uuid.uuid4().hex}.png"
            placed_path.write_bytes(img_bytes)
            return placed_path, geometry_contract

    raise ValueError(f"Stage-1 placement: Gemini returned no image. Response: {data}")


async def gemini_staged_update_render(
    mass_path: Path,
    render_path: Path,
    prompt: str = "",
    analyzer_model: str = "gemini-2.5-pro",
) -> Path:
    """Two-stage update-render pipeline.

    Stage 1 — Placement: gemini_place_mass_in_scene
        Positions the mass model cleanly in the scene at the correct perspective
        and scale. Background is untouched; building is a clean volume diagram.

    Stage 2 — Materialize: gemini_materialize_placed_mass
        The placed mass already has correct position/scale/angle in the scene.
        Stage 2 only paints photorealistic materials and lighting onto the white
        volume — geometry, proportions, background, and camera angle are all locked.
        Style reference is the original render (not erased) so Gemini can match
        materials/lighting directly from the real scene photograph.
    """
    print("[staged] Stage 1: placing mass in scene…", flush=True)
    placed_mass_path, geometry_contract = await gemini_place_mass_in_scene(mass_path, render_path)
    try:
        print("[staged] Stage 2: materializing placed mass…", flush=True)
        out_path = await gemini_materialize_placed_mass(
            placed_mass_path,
            render_path,          # original render — best style reference
            geometry_contract,
            prompt,
            analyzer_model,
        )
        return out_path
    finally:
        placed_mass_path.unlink(missing_ok=True)


async def gemini_25_analyze_three_image(
    original_mass_path: Path,
    modified_mass_path: Path,
    reference_path: Path,
    extra_prompt: str = "",
    analyzer_model: str = "gemini-2.5-pro",
) -> str:
    """
    Delta-aware analyst for the three-image workflow.

    Learns from the existing two-step chain pattern but extends it:
    - Image 1 = original geometry (baseline — before edits)
    - Image 2 = modified geometry (after edits — ALL changes are intentional)
    - Image 3 = photorealistic style reference

    The analyst identifies the geometry delta (what changed between 1 and 2),
    then writes a rich 300-500 word render prompt that locks in the new form
    while applying Image 3's materials, lighting, and atmosphere.
    """
    key = get_gemini_key()
    if not key:
        raise ValueError("GEMINI_API_KEY not set in .env")

    orig_b64, orig_mime = _b64_for_gemini(original_mass_path, is_render=False)
    mod_b64,  mod_mime  = _b64_for_gemini(modified_mass_path, is_render=False)
    ref_b64,  ref_mime  = _b64_for_gemini(reference_path, is_render=True)

    instruction = _load_prompt("analyzer_three_image_instruction", (
        "You are a senior architectural visualization director. "
        "Analyze all three images carefully:\n"
        "- Image 1: the ORIGINAL architectural mass model (baseline geometry — before any design edits).\n"
        "- Image 2: the MODIFIED architectural mass model (updated design — after edits). "
        "Every geometric difference from Image 1 is an intentional design decision and must be preserved exactly.\n"
        "- Image 3: a photorealistic reference render showing the target visual style.\n\n"
        "First, identify every geometric difference between Image 1 and Image 2 "
        "(new volumes, changed profiles, added bulges, shifted silhouettes, new elements).\n\n"
        "Then write a detailed image generation prompt (300-500 words) that will guide an AI image model "
        "to render Image 2's exact geometry in the style of Image 3. Cover:\n"
        "1. Building geometry from Image 2 — describe the modified form with full precision:\n"
        "   - Explicitly state each tower's height as a fraction of the tallest tower.\n"
        "   - Describe the exact outer silhouette — instruct the generator that it must be IDENTICAL.\n"
        "   - Call out every intentional deviation from Image 1 so the generator does not 'correct' them.\n"
        "   - Instruct the generator: this is a STYLE TRANSFER only — do NOT alter height, width, "
        "silhouette, or proportions; do NOT compress, elongate, widen, or narrow any element.\n"
        "2. Facade materials, textures, colors, glass type (from Image 3)\n"
        "3. Structural and architectural details (from Image 3)\n"
        "4. Lighting conditions, time of day, shadows (from Image 3)\n"
        "5. Sky, weather, atmosphere (from Image 3)\n"
        "6. Surrounding context, ground, vegetation (from Image 3)\n"
        "7. Camera angle and framing (match Image 2's perspective exactly)\n"
        "Output only the prompt text, no preamble."
    ))
    if extra_prompt.strip():
        instruction += f"\n\nAdditional direction from the user: {extra_prompt.strip()}"

    payload = {
        "contents": [{
            "parts": [
                {"text": instruction},
                {"inline_data": {"mime_type": orig_mime, "data": orig_b64}},
                {"inline_data": {"mime_type": mod_mime, "data": mod_b64}},
                {"inline_data": {"mime_type": ref_mime, "data": ref_b64}},
            ]
        }],
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{analyzer_model}:generateContent?key={key}"
    async with httpx.AsyncClient(timeout=180) as http:
        r = await http.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


async def gemini_generate_render_three_image(
    original_mass_path: Path,
    modified_mass_path: Path,
    reference_path: Path,
    prompt: str = "",
) -> Path:
    """
    Direct three-image render (no separate analyst step).

    Mirrors gemini_generate_render but sends all three images so the model can
    see the geometry delta directly and won't auto-correct intentional design changes.
    Follows the same model-fallback chain pattern as the two-image generator.
    """
    key = get_gemini_key()
    if not key:
        raise ValueError("GEMINI_API_KEY not set in .env")

    orig_b64, orig_mime = _b64_for_gemini(original_mass_path, is_render=False)
    mod_b64,  mod_mime  = _b64_for_gemini(modified_mass_path, is_render=False)
    ref_b64,  ref_mime  = _b64_for_gemini(reference_path, is_render=True)

    instruction = _load_prompt("direct_render_three_image_instruction", (
        "You are an expert architectural visualization artist. "
        "You are given three images:\n"
        "- Image 1: the ORIGINAL architectural mass model (baseline — before design edits).\n"
        "- Image 2: the MODIFIED architectural mass model (new design — after edits). "
        "Every geometric difference from Image 1 is intentional. Preserve all changes exactly — "
        "do not smooth, correct, or revert any deviation.\n"
        "- Image 3: a photorealistic reference render showing the target visual style.\n\n"
        "CRITICAL — GEOMETRY RULES (from Image 2, absolutely non-negotiable):\n"
        "- SILHOUETTE: The outer silhouette of the building must match Image 2 EXACTLY. "
        "Do not alter the height, width, outline, or proportions in any way.\n"
        "- HEIGHT: The height of each tower is a fixed constraint. Do NOT compress, elongate, or "
        "rescale any building vertically. Tower heights and their ratios must be preserved exactly.\n"
        "- PROPORTIONS: Width-to-height ratios must match Image 2 exactly for every element.\n"
        "- Do NOT simplify, merge, add, or omit any architectural feature.\n"
        "- This is a STYLE TRANSFER only — change materials and lighting, NOT the building's shape.\n\n"
        "Generate a photorealistic architectural visualization of Image 2's exact geometry rendered in "
        "the exact style of Image 3: match its facade materials, glass type and color, structural elements, "
        "lighting, sky, vegetation, and overall atmosphere. "
        "Output only the rendered image, no text."
    ))
    if prompt.strip():
        instruction += f" Additional direction: {prompt.strip()}"

    payload = {
        "contents": [{
            "parts": [
                {"text": instruction},
                {"inline_data": {"mime_type": orig_mime, "data": orig_b64}},
                {"inline_data": {"mime_type": mod_mime, "data": mod_b64}},
                {"inline_data": {"mime_type": ref_mime, "data": ref_b64}},
            ]
        }],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }

    _image_gen_models = [
        "gemini-2.0-flash-exp",
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image-preview",
    ]
    data = None
    _errors: list[str] = []
    for _mid in _image_gen_models:
        _url = f"https://generativelanguage.googleapis.com/v1beta/models/{_mid}:generateContent?key={key}"
        async with httpx.AsyncClient(timeout=120) as http:
            r = await http.post(_url, json=payload)
            print(f"[Gemini 3-img direct] {_mid} → {r.status_code}: {r.text[:300]}", flush=True)
            if r.status_code in (404, 400, 403):
                _errors.append(f"{_mid}: {r.status_code} {r.text[:120]}")
                continue
            r.raise_for_status()
            data = r.json()
            break

    if data is None:
        raise ValueError(f"No Gemini image-gen model succeeded. Errors: {'; '.join(_errors)}")

    for part in data["candidates"][0]["content"]["parts"]:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline:
            img_bytes = base64.b64decode(inline["data"])
            out_path = OUTPUTS_DIR / f"{uuid.uuid4()}.png"
            out_path.write_bytes(img_bytes)
            return out_path

    raise ValueError(f"Gemini returned no image. Response: {data}")


async def gemini_25_render_three_image(
    original_mass_path: Path,
    modified_mass_path: Path,
    reference_path: Path,
    prompt: str = "",
    analyzer_model: str = "gemini-2.5-pro",
) -> Path:
    """
    Three-image two-step chain:
      gemini_25_analyze_three_image (delta-aware analyst) →
      gemini_generate_render (image-gen on modified mass + reference)

    Mirrors the two-image gemini_25_render pattern but the analyst now sees
    the original geometry baseline so it can explicitly describe design deltas
    and prevent the generator from 'correcting' intentional geometry changes.
    """
    rich_prompt = await gemini_25_analyze_three_image(
        original_mass_path, modified_mass_path, reference_path, prompt, analyzer_model
    )
    return await gemini_generate_render(modified_mass_path, reference_path, rich_prompt)


async def gemini_new_angle(
    render_path: Path,
    angle_prompt: str,
    style_prompt: str,
    reference_path: Optional[Path] = None,
) -> Path:
    """
    Generate a new camera angle of the same building using Gemini image generation.

    Follows the same model-fallback chain and inline_data pattern as gemini_generate_render.
    When a reference image is provided it is included as a second image so the model can
    match the desired composition angle directly from pixels rather than from text alone.
    """
    key = get_gemini_key()
    if not key:
        raise ValueError("GEMINI_API_KEY not set in .env")

    render_b64, render_mime = _b64_for_gemini(render_path, is_render=True)

    angle_part = angle_prompt.strip() if angle_prompt.strip() else "a compelling new viewpoint"
    style_part = (
        style_prompt.strip() if style_prompt.strip()
        else "matching the exact facade materials, lighting and atmosphere of the source render"
    )

    if reference_path:
        ref_b64, ref_mime = _b64(reference_path)
        instruction = (
            "You are an expert architectural visualization artist. "
            "Image 1 is a photorealistic architectural render of a building. "
            "Image 2 shows a reference for the desired camera angle and composition. "
            f"Generate a photorealistic render of the exact same building from: {angle_part}. "
            "Match the camera composition shown in Image 2. "
            "Preserve every facade material, glass type, structural element, and architectural detail "
            f"from Image 1 exactly. {style_part}. "
            "Output only the rendered image, no text."
        )
        parts = [
            {"text": instruction},
            {"inline_data": {"mime_type": render_mime, "data": render_b64}},
            {"inline_data": {"mime_type": ref_mime, "data": ref_b64}},
        ]
    else:
        instruction = (
            "You are an expert architectural visualization artist. "
            "The image shows a photorealistic architectural render of a building. "
            f"Generate a photorealistic render of the exact same building from: {angle_part}. "
            "Preserve every facade material, glass type, structural element, and architectural detail exactly. "
            f"{style_part}. "
            "Output only the rendered image, no text."
        )
        parts = [
            {"text": instruction},
            {"inline_data": {"mime_type": render_mime, "data": render_b64}},
        ]

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }

    _image_gen_models = [
        "gemini-2.0-flash-exp",
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image-preview",
    ]
    data = None
    _errors: list[str] = []
    for _mid in _image_gen_models:
        _url = f"https://generativelanguage.googleapis.com/v1beta/models/{_mid}:generateContent?key={key}"
        async with httpx.AsyncClient(timeout=120) as http:
            r = await http.post(_url, json=payload)
            print(f"[Gemini new-angle] {_mid} → {r.status_code}: {r.text[:300]}", flush=True)
            if r.status_code in (404, 400, 403):
                _errors.append(f"{_mid}: {r.status_code} {r.text[:120]}")
                continue
            r.raise_for_status()
            data = r.json()
            break

    if data is None:
        raise ValueError(f"No Gemini image-gen model succeeded. Errors: {'; '.join(_errors)}")

    for part in data["candidates"][0]["content"]["parts"]:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline:
            img_bytes = base64.b64decode(inline["data"])
            out_path = OUTPUTS_DIR / f"{uuid.uuid4()}.png"
            out_path.write_bytes(img_bytes)
            return out_path

    raise ValueError(f"Gemini returned no image. Response: {data}")


async def gemini_edit_region(
    base_image_path: Optional[Path],
    base_image_url: Optional[str],
    mask_data_url: str,
    prompt: str,
) -> Path:
    """
    Gemini-based region editing (inpainting alternative).

    Sends the base render + the painted mask as two images. The model sees
    exactly which pixels to change (white = edit here) and blends the result
    into the surrounding context. Less pixel-precise than Flux Fill on hard
    mask edges, but understands architectural context holistically.
    """
    key = get_gemini_key()
    if not key:
        raise ValueError("GEMINI_API_KEY not set in .env")

    # Resolve base image to b64
    if base_image_path and base_image_path.exists():
        suffix = base_image_path.suffix.lower().lstrip(".")
        base_mime = "image/jpeg" if suffix in ("jpg", "jpeg") else "image/png"
        base_b64 = base64.b64encode(base_image_path.read_bytes()).decode()
    elif base_image_url:
        async with httpx.AsyncClient(timeout=30) as http:
            r = await http.get(base_image_url)
            r.raise_for_status()
            raw = r.content
        base_mime = "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg"
        base_b64 = base64.b64encode(raw).decode()
    else:
        raise ValueError("No base image provided for Gemini edit")

    # Mask is a data URI
    mask_bytes = base64.b64decode(mask_data_url.split(",", 1)[1])
    mask_b64 = base64.b64encode(mask_bytes).decode()

    instruction = (
        "You are an expert architectural visualization artist. "
        "Image 1 is a photorealistic architectural render. "
        "Image 2 is an edit mask where white pixels mark the region to change. "
        f"Edit only the white-masked region: {prompt.strip()}. "
        "The result must blend seamlessly into the untouched surroundings — "
        "match perspective, lighting, shadow, scale, and material quality exactly. "
        "Output only the edited image, no text."
    )

    payload = {
        "contents": [{
            "parts": [
                {"text": instruction},
                {"inline_data": {"mime_type": base_mime, "data": base_b64}},
                {"inline_data": {"mime_type": "image/png", "data": mask_b64}},
            ]
        }],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }

    _image_gen_models = [
        "gemini-2.0-flash-exp",
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image-preview",
    ]
    data = None
    _errors: list[str] = []
    for _mid in _image_gen_models:
        _url = f"https://generativelanguage.googleapis.com/v1beta/models/{_mid}:generateContent?key={key}"
        async with httpx.AsyncClient(timeout=120) as http:
            r = await http.post(_url, json=payload)
            print(f"[Gemini edit-region] {_mid} → {r.status_code}: {r.text[:300]}", flush=True)
            if r.status_code in (404, 400, 403):
                _errors.append(f"{_mid}: {r.status_code} {r.text[:120]}")
                continue
            r.raise_for_status()
            data = r.json()
            break

    if data is None:
        raise ValueError(f"No Gemini image-gen model succeeded. Errors: {'; '.join(_errors)}")

    for part in data["candidates"][0]["content"]["parts"]:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline:
            img_bytes = base64.b64decode(inline["data"])
            out_path = OUTPUTS_DIR / f"{uuid.uuid4()}.png"
            out_path.write_bytes(img_bytes)
            return out_path

    raise ValueError(f"Gemini returned no image. Response: {data}")


async def gemini_describe_style(image_path: Path, extra_prompt: str = "") -> str:
    """Use Gemini Vision (REST API) to extract a precise architectural style prompt from an image."""
    key = get_gemini_key()
    if not key:
        return extra_prompt  # fallback: just use the user prompt

    suffix = image_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else "image/png"
    b64 = base64.b64encode(image_path.read_bytes()).decode()

    system = (
        "You are an expert architectural visualization prompter. "
        "Analyze the image and output ONLY a concise technical prompt (max 130 words) for an AI image generator. "
        "Describe precisely:\n"
        "- Facade materials (glass color/finish, metal type/color, stone, wood)\n"
        "- Structural elements (ribs, fins, lattice, frames — material, color, density)\n"
        "- Lighting (time of day, warm/cool, accent light colors and placement)\n"
        "- Sky and atmosphere (color gradient, clouds, haze)\n"
        "- Vegetation (species, density, terrace/facade integration)\n"
        "- Overall render quality (photorealistic, CGI, photography style)\n"
        "Output ONLY the prompt text, no explanations or labels."
    )
    if extra_prompt.strip():
        system += f"\n\nAlso incorporate this user direction: {extra_prompt.strip()}"

    payload = {
        "contents": [{
            "parts": [
                {"text": system},
                {"inline_data": {"mime_type": mime, "data": b64}},
            ]
        }]
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent?key={key}"
    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


async def replicate_run(model_id: str, input_data: dict, api_token: str) -> str:
    """Call the Replicate REST API directly (no SDK). Returns the output URL."""

    # Convert any open file handles to base64 data URIs
    processed: dict = {}
    for key, value in input_data.items():
        if hasattr(value, "read"):
            raw = value.read()
            mime = "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg"
            processed[key] = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
        else:
            processed[key] = value

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    # Official models have no ":" (e.g. "black-forest-labs/flux-canny-pro").
    # Versioned community models include a hash after ":" (e.g. "owner/model:abc123").
    if ":" in model_id:
        url = "https://api.replicate.com/v1/predictions"
        _, version = model_id.rsplit(":", 1)
        payload: dict = {"version": version, "input": processed}
    else:
        url = f"https://api.replicate.com/v1/models/{model_id}/predictions"
        payload = {"input": processed}

    async with httpx.AsyncClient(timeout=300) as http:
        r = await http.post(url, json=payload, headers=headers)
        r.raise_for_status()
        prediction = r.json()

        # Poll until the prediction finishes
        get_url = prediction["urls"]["get"]
        while prediction["status"] not in ("succeeded", "failed", "canceled"):
            await asyncio.sleep(2)
            r = await http.get(get_url, headers=headers)
            r.raise_for_status()
            prediction = r.json()

        if prediction["status"] != "succeeded":
            raise RuntimeError(f"Replicate prediction failed: {prediction.get('error')}")

        output = prediction["output"]
        return output[0] if isinstance(output, list) else str(output)


# Official BFL serverless models — no version hash needed.
CONTROLNET_MODELS = {
    "flux-controlnet-canny": {
        "id": "black-forest-labs/flux-canny-pro",
        "input_key": "control_image",
        "extra": {"guidance": 30, "steps": 28, "safety_tolerance": 5, "output_format": "png"},
    },
    "flux-controlnet-depth": {
        "id": "black-forest-labs/flux-depth-pro",
        "input_key": "control_image",
        "extra": {"guidance": 15, "steps": 28, "safety_tolerance": 5, "output_format": "png"},
    },
    "sdxl-controlnet": {
        "id": "diffusers/controlnet-canny-sdxl-1.0:a398a399f1238d5651c7bb7b5417823f1d559fc2ab1b7fa3f06a45d57c971db4",
        "input_key": "image",
        "extra": {"num_inference_steps": 50, "guidance_scale": 9.0,
                   "controlnet_conditioning_scale": 0.85,
                   "negative_prompt": "blurry, low quality, distorted, deformed, cartoon, illustration, painting, sketch, amateur, watermark, text"},
    },
}


async def run_controlnet_render(
    render_path: Path,
    mass_path: Path,
    prompt: str,
    model: str,
    job_id: str,
    api_token: Optional[str] = None,
) -> None:
    """Structure-guided render using official BFL canny/depth-pro models or Gemini.

    For Gemini models the existing render acts as the style reference and the new
    mass provides the geometry — exactly the same roles as in the style-transfer tab
    but surfaced here for users who already have a render they want to update.
    """
    try:
        jobs[job_id]["status"] = "processing"

        # ── Gemini routes ────────────────────────────────────────────────────
        if model in ("gemini-25-pro", "gemini-25-flash"):
            # Erase the old building so Gemini cannot copy its geometry/style.
            clean_render_path = UPLOADS_DIR / f"clean_{uuid.uuid4().hex}.png"
            clean_render_path.write_bytes(_erase_building_from_render(mass_path, render_path))
            try:
                analyzer = "gemini-2.5-pro" if model == "gemini-25-pro" else "gemini-2.5-flash"
                out_path = await gemini_25_render(mass_path, clean_render_path, prompt, analyzer_model=analyzer)
            finally:
                clean_render_path.unlink(missing_ok=True)
            jobs[job_id].update({"status": "done", "output_url": f"/outputs/{out_path.name}"})
            return
        elif model == "gemini-direct":
            # gemini_generate_render_update does its own erasing internally
            out_path = await gemini_generate_render_update(mass_path, render_path, prompt)
            jobs[job_id].update({"status": "done", "output_url": f"/outputs/{out_path.name}"})
            return
        elif model in ("gemini-staged-pro", "gemini-staged-flash"):
            # Two-stage pipeline: place mass in scene first, then render
            analyzer = "gemini-2.5-pro" if model == "gemini-staged-pro" else "gemini-2.5-flash"
            out_path = await gemini_staged_update_render(mass_path, render_path, prompt, analyzer_model=analyzer)
            jobs[job_id].update({"status": "done", "output_url": f"/outputs/{out_path.name}"})
            return

        # ── Flux / SDXL routes ───────────────────────────────────────────────
        cfg = CONTROLNET_MODELS.get(model, CONTROLNET_MODELS["flux-controlnet-canny"])

        # Gemini Flash extracts scene lighting, atmosphere, materials from the existing render.
        full_prompt = await gemini_describe_style(render_path, prompt)

        token = get_api_token(api_token)

        if model == "sdxl-controlnet":
            # SDXL: use existing render as img2img base + mass as control
            model_input = {
                "image": open(render_path, "rb"),
                cfg["input_key"]: open(mass_path, "rb"),
                "prompt": full_prompt,
                **cfg["extra"],
            }
        else:
            # BFL Canny Pro / Depth Pro: mass is the structural guide image
            model_input = {
                cfg["input_key"]: open(mass_path, "rb"),
                "prompt": full_prompt,
                **cfg["extra"],
            }
        output_url = await replicate_run(cfg["id"], model_input, token)

        jobs[job_id].update({"status": "done", "output_url": output_url})
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e) or repr(e)})


async def run_style_transfer(
    reference_path: Path,
    mass_path: Path,
    prompt: str,
    model: str,
    job_id: str,
    api_token: Optional[str] = None,
    original_mass_path: Optional[Path] = None,
) -> None:
    """Style transfer: apply reference image style onto new mass.

    When original_mass_path is provided the pipeline uses the three-image workflow:
    the analyst sees the geometry delta (original vs modified) so intentional design
    changes are described explicitly and the generator won't auto-correct them.
    """
    try:
        jobs[job_id]["status"] = "processing"

        token = get_api_token(api_token)

        if model == "gemini-25-pro":
            if original_mass_path:
                out_path = await gemini_25_render_three_image(
                    original_mass_path, mass_path, reference_path, prompt, analyzer_model="gemini-2.5-pro"
                )
            else:
                out_path = await gemini_25_render(mass_path, reference_path, prompt, analyzer_model="gemini-2.5-pro")
            output_url = f"/outputs/{out_path.name}"

        elif model == "gemini-25-flash":
            if original_mass_path:
                out_path = await gemini_25_render_three_image(
                    original_mass_path, mass_path, reference_path, prompt, analyzer_model="gemini-2.5-flash"
                )
            else:
                out_path = await gemini_25_render(mass_path, reference_path, prompt, analyzer_model="gemini-2.5-flash")
            output_url = f"/outputs/{out_path.name}"

        elif model == "gemini-direct":
            if original_mass_path:
                out_path = await gemini_generate_render_three_image(
                    original_mass_path, mass_path, reference_path, prompt
                )
            else:
                out_path = await gemini_generate_render(mass_path, reference_path, prompt)
            output_url = f"/outputs/{out_path.name}"

        elif model == "flux-redux-controlnet":
            # When original mass is present, use the delta-aware analyst (flash) for a richer
            # structural prompt instead of the single-image style extractor.
            if original_mass_path:
                style_prompt = await gemini_25_analyze_three_image(
                    original_mass_path, mass_path, reference_path, prompt, analyzer_model="gemini-2.5-flash"
                )
            else:
                style_prompt = await gemini_describe_style(reference_path, prompt)
            output_url = await replicate_run(
                "black-forest-labs/flux-canny-pro",
                {"control_image": open(mass_path, "rb"),
                 "prompt": style_prompt,
                 "guidance": 28,
                 "steps": 28,
                 "safety_tolerance": 5,
                 "output_format": "png"},
                token,
            )
        elif model == "flux-redux-only":
            output_url = await replicate_run(
                "black-forest-labs/flux-redux-dev",
                {"redux_image": open(reference_path, "rb"),
                 "num_inference_steps": 50, "guidance": 3.5},
                token,
            )
        else:  # sdxl-img2img
            if original_mass_path:
                style_prompt = await gemini_25_analyze_three_image(
                    original_mass_path, mass_path, reference_path, prompt, analyzer_model="gemini-2.5-flash"
                )
            else:
                style_prompt = await gemini_describe_style(reference_path, prompt)
            output_url = await replicate_run(
                "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                {"image": open(mass_path, "rb"), "prompt": style_prompt,
                 "prompt_strength": 0.80, "num_inference_steps": 50, "guidance_scale": 9.0,
                 "negative_prompt": "blurry, low quality, distorted, deformed, cartoon, illustration, painting, sketch, amateur"},
                token,
            )

        jobs[job_id].update({"status": "done", "output_url": output_url})
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e) or repr(e)})


async def run_new_angle(
    render_path: Path,
    reference_path: Optional[Path],
    angle_prompt: str,
    style_prompt: str,
    model: str,
    job_id: str,
    api_token: Optional[str] = None,
) -> None:
    """Generate a new camera angle of a building."""
    try:
        jobs[job_id]["status"] = "processing"

        # ── Gemini route ─────────────────────────────────────────────────────
        if model == "gemini":
            out_path = await gemini_new_angle(render_path, angle_prompt, style_prompt, reference_path)
            jobs[job_id].update({"status": "done", "output_url": f"/outputs/{out_path.name}"})
            return

        # ── Flux routes ───────────────────────────────────────────────────────
        style_part = style_prompt.strip() if style_prompt.strip() else "matching the exact materials, lighting and atmosphere of the original render"
        combined_prompt = (
            f"award-winning architectural visualization of the exact same building, {angle_prompt}, "
            f"{style_part}, photorealistic CGI render, dramatic cinematic lighting, "
            "volumetric atmosphere, ultra-detailed facade materials, "
            "professional architectural photography, hyperrealistic, 8K ultra resolution"
        )

        token = get_api_token(api_token)

        if model == "zero123plus":
            output_url = await replicate_run(
                "black-forest-labs/flux-canny-pro",
                {"control_image": open(render_path, "rb"), "prompt": combined_prompt,
                 "guidance": 25, "steps": 28, "safety_tolerance": 5, "output_format": "png"},
                token,
            )
        elif model == "flux-redux":
            output_url = await replicate_run(
                "black-forest-labs/flux-redux-dev",
                {"redux_image": open(render_path, "rb"),
                 "num_inference_steps": 50, "guidance": 3.5},
                token,
            )
        else:  # flux-img2img
            output_url = await replicate_run(
                "black-forest-labs/flux-dev",
                {"image": open(render_path, "rb"), "prompt": combined_prompt,
                 "strength": 0.75, "num_inference_steps": 28, "guidance": 3.5},
                token,
            )

        jobs[job_id].update({"status": "done", "output_url": output_url})
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e) or repr(e)})


async def run_inpaint(
    base_image_url: Optional[str],
    base_image_path: Optional[Path],
    mask_data_url: str,
    prompt: str,
    inpaint_model: str,
    job_id: str,
    api_token: Optional[str] = None,
) -> None:
    """Inpaint a masked region of an image."""
    try:
        jobs[job_id]["status"] = "processing"

        # ── Gemini route ─────────────────────────────────────────────────────
        if inpaint_model == "gemini-edit":
            out_path = await gemini_edit_region(base_image_path, base_image_url, mask_data_url, prompt)
            jobs[job_id].update({"status": "done", "output_url": f"/outputs/{out_path.name}"})
            return

        # ── Flux / SD routes ──────────────────────────────────────────────────
        full_prompt = (
            f"award-winning architectural detail, {prompt}, "
            "seamlessly integrated, perfectly matching surrounding materials and lighting, "
            "photorealistic CGI render, ultra-detailed, professional architectural visualization, "
            "8K resolution, hyperrealistic"
        )

        # Decode mask from data URL
        mask_data = mask_data_url.split(",", 1)[1]
        mask_bytes = base64.b64decode(mask_data)
        mask_path = UPLOADS_DIR / f"{uuid.uuid4()}_mask.png"
        mask_path.write_bytes(mask_bytes)

        image_input: any
        if base_image_path and base_image_path.exists():
            image_input = open(base_image_path, "rb")
        else:
            image_input = base_image_url

        inpaint_model_id = {
            "flux-fill-pro": "black-forest-labs/flux-fill-pro",
            "flux-fill-dev": "black-forest-labs/flux-fill-dev",
            "sd-inpainting": "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
        }.get(inpaint_model, "black-forest-labs/flux-fill-pro")

        is_flux_fill = "flux-fill" in inpaint_model_id
        mask_fh = open(mask_path, "rb")
        try:
            model_input = {
                "image": image_input,
                "mask": mask_fh,
                "prompt": full_prompt,
                **({"num_inference_steps": 50, "guidance": 30, "output_format": "png"}
                   if is_flux_fill
                   else {"num_inference_steps": 50, "guidance_scale": 9.0,
                         "negative_prompt": "blurry, low quality, distorted, deformed, cartoon, illustration, painting, sketch, amateur, watermark"}),
            }

            token = get_api_token(api_token)
            output_url = await replicate_run(inpaint_model_id, model_input, token)
        finally:
            mask_fh.close()
            if hasattr(image_input, "close"):
                image_input.close()

        jobs[job_id].update({"status": "done", "output_url": output_url})
        mask_path.unlink(missing_ok=True)
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e) or repr(e)})


async def run_place_mass(mass_path: Path, render_path: Path, job_id: str) -> None:
    """Background task for /api/place-mass — runs Stage 1 only."""
    try:
        jobs[job_id]["status"] = "processing"
        placed_path, geometry_contract = await gemini_place_mass_in_scene(mass_path, render_path)
        jobs[job_id].update({
            "status": "done",
            "output_url": f"/outputs/{placed_path.name}",
            "geometry_contract": geometry_contract,
        })
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e) or repr(e)})
    finally:
        mass_path.unlink(missing_ok=True)
        render_path.unlink(missing_ok=True)


async def run_materialize_mass(
    placed_mass_path: Path,
    render_path: Path,
    geometry_contract: str,
    prompt: str,
    analyzer_model: str,
    job_id: str,
) -> None:
    """Background task for /api/materialize-mass — runs Stage 2 only."""
    try:
        jobs[job_id]["status"] = "processing"
        out_path = await gemini_materialize_placed_mass(
            placed_mass_path, render_path, geometry_contract, prompt, analyzer_model
        )
        jobs[job_id].update({"status": "done", "output_url": f"/outputs/{out_path.name}"})
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e) or repr(e)})
    finally:
        placed_mass_path.unlink(missing_ok=True)
        render_path.unlink(missing_ok=True)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.post("/api/update-render")
async def update_render(
    render: UploadFile = File(...),
    mass: UploadFile = File(...),
    prompt: str = Form(""),
    model: str = Form("flux-controlnet-canny"),
    replicate_api_token: Optional[str] = Form(None),
):
    render_path = save_upload(render)
    mass_path = save_upload(mass)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    asyncio.create_task(run_controlnet_render(render_path, mass_path, prompt, model, job_id, replicate_api_token))
    return {"jobId": job_id}


@app.post("/api/place-mass")
async def place_mass(
    render: UploadFile = File(...),
    mass: UploadFile = File(...),
):
    """Stage 1 only — place the mass model into the scene. Returns job whose output is
    the placed white/grey mass composite. The job result also carries geometry_contract."""
    render_path = save_upload(render)
    mass_path = save_upload(mass)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    asyncio.create_task(run_place_mass(mass_path, render_path, job_id))
    return {"jobId": job_id}


@app.post("/api/materialize-mass")
async def materialize_mass(
    placed_mass: UploadFile = File(...),
    render: UploadFile = File(...),
    geometry_contract: str = Form(""),
    prompt: str = Form(""),
    model: str = Form("gemini-staged-pro"),
):
    """Stage 2 only — materialize a placed white-mass composite into a photorealistic render.
    placed_mass is the Stage-1 output (white/grey mass already in scene).
    render is the original reference render (for style/materials).
    geometry_contract is the JSON/text from Stage-1 (optional but recommended)."""
    placed_path = save_upload(placed_mass)
    render_path = save_upload(render)
    analyzer = "gemini-2.5-pro" if model == "gemini-staged-pro" else "gemini-2.5-flash"
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    asyncio.create_task(run_materialize_mass(placed_path, render_path, geometry_contract, prompt, analyzer, job_id))
    return {"jobId": job_id}


@app.post("/api/style-transfer")
async def style_transfer(
    reference: UploadFile = File(...),
    mass: UploadFile = File(...),
    prompt: str = Form(""),
    model: str = Form("flux-redux-controlnet"),
    replicate_api_token: Optional[str] = Form(None),
    original_mass: Optional[UploadFile] = File(None),
):
    reference_path = save_upload(reference)
    mass_path = save_upload(mass)
    original_mass_path = (
        save_upload(original_mass)
        if original_mass and original_mass.filename
        else None
    )
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    asyncio.create_task(
        run_style_transfer(reference_path, mass_path, prompt, model, job_id, replicate_api_token, original_mass_path)
    )
    return {"jobId": job_id}


@app.post("/api/new-angle")
async def new_angle(
    render: UploadFile = File(...),
    reference: Optional[UploadFile] = File(None),
    angle_prompt: str = Form(""),
    style_prompt: str = Form(""),
    model: str = Form("zero123plus"),
    replicate_api_token: Optional[str] = Form(None),
):
    render_path = save_upload(render)
    reference_path = save_upload(reference) if reference else None
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    asyncio.create_task(run_new_angle(render_path, reference_path, angle_prompt, style_prompt, model, job_id, replicate_api_token))
    return {"jobId": job_id}


@app.post("/api/inpaint")
async def inpaint(
    base_image: Optional[UploadFile] = File(None),
    base_image_url: Optional[str] = Form(None),
    mask_data_url: str = Form(...),
    prompt: str = Form(...),
    model: str = Form("flux-fill-pro"),
    replicate_api_token: Optional[str] = Form(None),
):
    base_path: Optional[Path] = None
    if base_image and base_image.filename:
        base_path = save_upload(base_image)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    asyncio.create_task(run_inpaint(base_image_url, base_path, mask_data_url, prompt, model, job_id, replicate_api_token))
    return {"jobId": job_id}


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status": job["status"],
        "output_url": job.get("output_url"),
        "error": job.get("error"),
        "geometry_contract": job.get("geometry_contract"),
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "replicate_configured": bool(os.getenv("REPLICATE_API_TOKEN"))}


@app.post("/api/judge")
async def judge_output(
    mass: UploadFile = File(...),
    render: UploadFile = File(...),
    output_url: str = Form(...),
):
    """Score a generated output against the mass model and reference render.

    Returns JSON:
      { "scores": {"geometry": 1-10, "style": 1-10, "background": 1-10, "overall": 1-10},
        "reasoning": "<text>" }
    """
    key = get_gemini_key()
    if not key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    mass_bytes   = await mass.read()
    render_bytes = await render.read()

    # Resolve output image from local filesystem (URL is /outputs/<filename>)
    out_filename = output_url.lstrip("/").split("/")[-1]
    out_path = OUTPUTS_DIR / out_filename
    if not out_path.exists():
        raise HTTPException(status_code=404, detail=f"Output file not found: {out_filename}")
    output_bytes = out_path.read_bytes()

    # Resize all three for the judge (text model — just needs to see detail, not huge)
    mass_b64,   mass_mime   = _b64_for_gemini(mass_bytes,   is_render=False)
    render_b64, render_mime = _b64_for_gemini(render_bytes, is_render=True)
    output_b64, output_mime = _b64_for_gemini(output_bytes, is_render=True)

    judge_prompt = (
        "You are a senior architectural visualization quality judge. "
        "You will evaluate an AI-generated architectural rendering against two reference images.\n\n"
        "Image 1: the MASS MODEL — the exact geometric blueprint (towers, heights, silhouette, crown shapes, podium).\n"
        "Image 2: the REFERENCE RENDER — the target photorealistic style, materials, lighting, and scene context.\n"
        "Image 3: the AI OUTPUT — the image to evaluate.\n\n"
        "Score Image 3 on these four criteria (1–10 each, where 10 is perfect):\n"
        "1. geometry_accuracy: Does Image 3 faithfully reproduce Image 1's exact shape? "
        "Check tower count, relative heights, silhouette, crown forms, podium.\n"
        "2. style_match: Does Image 3 match Image 2's materials, lighting, atmosphere, and photorealism?\n"
        "3. background_preservation: Is the surrounding scene in Image 3 unchanged from Image 2? "
        "(sky, roads, trees, neighboring buildings, ground plane)\n"
        "4. overall: Overall quality as a convincing architectural visualization.\n\n"
        "Respond ONLY with valid JSON in this exact format, no other text:\n"
        '{"geometry_accuracy": <1-10>, "style_match": <1-10>, '
        '"background_preservation": <1-10>, "overall": <1-10>, '
        '"reasoning": "<one concise sentence per criterion, separated by |>"}'
    )

    payload = {
        "contents": [{"parts": [
            {"text": judge_prompt},
            {"inline_data": {"mime_type": mass_mime,    "data": mass_b64}},
            {"inline_data": {"mime_type": render_mime,  "data": render_b64}},
            {"inline_data": {"mime_type": output_mime,  "data": output_b64}},
        ]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    judge_model = "gemini-2.5-flash"
    judge_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{judge_model}:generateContent?key={key}"
    )
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(judge_url, json=payload)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Gemini judge error {r.status_code}: {r.text[:300]}")

    try:
        raw_text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        scores = json.loads(raw_text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not parse judge response: {exc}")

    return scores





@app.post("/api/validate-token")
async def validate_token(token: str = Form(...)):
    try:
        async with httpx.AsyncClient() as http:
            for auth_scheme in ("Bearer", "Token"):
                r = await http.get(
                    "https://api.replicate.com/v1/account",
                    headers={"Authorization": f"{auth_scheme} {token}"},
                    timeout=8,
                )
                if r.status_code == 200:
                    username = r.json().get("username", "")
                    return {"valid": True, "username": username}
        return {"valid": False, "error": "Invalid token"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


# ── Photo library ────────────────────────────────────────────────────────────

LIBRARY_GEMINI_MODELS = ["gemini-25-pro", "gemini-25-flash", "gemini-direct"]
LIBRARY_TABS = ["style-transfer", "update-render"]

_ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _library_image_info(path: Path, role: str) -> dict:
    return {
        "id": path.name,
        "name": path.stem,
        "role": role,
        "url": f"/library/{role}/{path.name}",
    }


def _load_run_result(run_file: Path) -> dict:
    try:
        return json.loads(run_file.read_text())
    except Exception:
        return {}


def _save_run_result(run_id: str, entries: list[dict]) -> Path:
    data = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    out = TEST_RESULTS_DIR / f"{run_id}.json"
    out.write_text(json.dumps(data, indent=2))
    return out


@app.post("/api/library/upload")
async def library_upload(
    file: UploadFile = File(...),
    role: str = Form(...),
):
    """Upload an image to the photo library. role must be 'mass' or 'render'."""
    if role not in ("mass", "render"):
        raise HTTPException(status_code=422, detail="role must be 'mass' or 'render'")
    suffix = Path(file.filename or "file.png").suffix.lower()
    if suffix not in _ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=422, detail=f"Unsupported image type: {suffix}")
    dest_dir = LIBRARY_MASS_DIR if role == "mass" else LIBRARY_RENDER_DIR
    dest = dest_dir / f"{uuid.uuid4()}{suffix}"
    dest.write_bytes(await file.read())
    return _library_image_info(dest, role)


@app.get("/api/library")
async def library_list():
    """List all images in the library, grouped by role."""
    masses = [_library_image_info(p, "mass") for p in sorted(LIBRARY_MASS_DIR.iterdir())
              if p.suffix.lower() in _ALLOWED_IMAGE_SUFFIXES]
    renders = [_library_image_info(p, "render") for p in sorted(LIBRARY_RENDER_DIR.iterdir())
               if p.suffix.lower() in _ALLOWED_IMAGE_SUFFIXES]
    return {"mass": masses, "render": renders, "total": len(masses) + len(renders)}


async def run_library_test_suite(
    mass_paths: list[Path],
    render_paths: list[Path],
    run_id: str,
) -> None:
    """
    For every mass+render pair: run Style Transfer and Update Render with every
    Gemini model. Results are stored in test_results/{run_id}.json.
    """
    entries: list[dict] = []
    tasks: list[asyncio.Task] = []

    for mass_path in mass_paths:
        for render_path in render_paths:
            for tab in LIBRARY_TABS:
                for model in LIBRARY_GEMINI_MODELS:
                    job_id = str(uuid.uuid4())
                    jobs[job_id] = {"status": "pending"}
                    entry: dict = {
                        "run_id": run_id,
                        "job_id": job_id,
                        "tab": tab,
                        "model": model,
                        "mass": mass_path.name,
                        "render": render_path.name,
                        "status": "pending",
                        "output_url": None,
                        "error": None,
                    }
                    entries.append(entry)

                    if tab == "style-transfer":
                        coro = run_style_transfer(mass_path, render_path, "", model, job_id)
                    else:  # update-render
                        coro = run_controlnet_render(render_path, mass_path, "", model, job_id)

                    tasks.append(asyncio.create_task(coro))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Reflect final job state back into entries
    for i, entry in enumerate(entries):
        job = jobs.get(entry["job_id"], {})
        entry["status"] = job.get("status", "error")
        entry["output_url"] = job.get("output_url")
        exc = results[i]
        if isinstance(exc, Exception):
            entry["status"] = "error"
            entry["error"] = str(exc)

    _save_run_result(run_id, entries)


@app.post("/api/library/run-tests")
async def library_run_tests(background_tasks=None):
    """
    Kick off a full test run against every mass+render pair in the library.
    All 3 Gemini models × Style Transfer + Update Render = 6 jobs per pair.
    Returns the run_id immediately; results appear in GET /api/library/test-results/{run_id}.
    """
    mass_paths = sorted(p for p in LIBRARY_MASS_DIR.iterdir()
                        if p.suffix.lower() in _ALLOWED_IMAGE_SUFFIXES)
    render_paths = sorted(p for p in LIBRARY_RENDER_DIR.iterdir()
                          if p.suffix.lower() in _ALLOWED_IMAGE_SUFFIXES)
    if not mass_paths:
        raise HTTPException(status_code=422, detail="No mass images in library. Upload at least one.")
    if not render_paths:
        raise HTTPException(status_code=422, detail="No render images in library. Upload at least one.")

    run_id = str(uuid.uuid4())
    job_count = len(mass_paths) * len(render_paths) * len(LIBRARY_TABS) * len(LIBRARY_GEMINI_MODELS)

    asyncio.create_task(run_library_test_suite(mass_paths, render_paths, run_id))

    return {
        "run_id": run_id,
        "job_count": job_count,
        "pairs": len(mass_paths) * len(render_paths),
        "results_url": f"/api/library/test-results/{run_id}",
    }


@app.get("/api/library/test-results")
async def library_test_results_list():
    """List all saved test run summaries (newest first)."""
    runs = []
    for f in sorted(TEST_RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = _load_run_result(f)
        if data:
            entries = data.get("entries", [])
            done = sum(1 for e in entries if e.get("status") == "done")
            runs.append({
                "run_id": data.get("run_id"),
                "timestamp": data.get("timestamp"),
                "total": len(entries),
                "done": done,
                "errors": sum(1 for e in entries if e.get("status") == "error"),
                "pending": len(entries) - done - sum(1 for e in entries if e.get("status") == "error"),
            })
    return {"runs": runs}


@app.get("/api/library/test-results/{run_id}")
async def library_test_results_detail(run_id: str):
    """Full detail for a single test run including all input/output URLs."""
    result_file = TEST_RESULTS_DIR / f"{run_id}.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    data = _load_run_result(result_file)
    # Refresh status for still-running jobs
    for entry in data.get("entries", []):
        if entry.get("status") in ("pending", "processing"):
            job = jobs.get(entry["job_id"], {})
            entry["status"] = job.get("status", entry["status"])
            entry["output_url"] = job.get("output_url", entry.get("output_url"))
            if entry["status"] in ("done", "error"):
                # Persist the update
                _save_run_result(run_id, data["entries"])
    return data


# ── Serve React frontend (must be last) ─────────────────────────────────────

FRONTEND_DIST = BUNDLE_DIR / "frontend_dist"

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file = FRONTEND_DIST / full_path
        if file.exists() and file.is_file():
            return FileResponse(str(file))
        return FileResponse(str(FRONTEND_DIST / "index.html"))


# ── Entry point for PyInstaller ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    port = 8000

    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port)
