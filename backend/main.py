import sys

# Ensure Python 3.11+
if sys.version_info < (3, 11):
    print(f"ERROR: Python 3.11+ required, running {sys.version}")
    print("Run with: py -3.11 main.py")
    sys.exit(1)

import os
import uuid
import asyncio
import base64
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()

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
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


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


async def gemini_generate_render(mass_path: Path, reference_path: Path, prompt: str = "") -> Path:
    """Send mass + reference to Gemini 2.0 Flash image generation. Returns saved output path."""
    key = get_gemini_key()
    if not key:
        raise ValueError("GEMINI_API_KEY not set in .env")

    def _b64(p: Path) -> tuple[str, str]:
        suffix = p.suffix.lower().lstrip(".")
        mime = "image/jpeg" if suffix in ("jpg", "jpeg") else "image/png"
        return base64.b64encode(p.read_bytes()).decode(), mime

    mass_b64, mass_mime = _b64(mass_path)
    ref_b64, ref_mime = _b64(reference_path)

    instruction = (
        "You are an expert architectural visualization artist. "
        "The first image is an architectural mass/volume model (a simplified 3D building shape). "
        "The second image is a reference architectural render showing the desired style. "
        "Generate a photorealistic architectural visualization of the building mass that exactly matches "
        "the reference's facade materials, glass type and color, structural elements, lighting, sky, "
        "vegetation, and overall atmosphere. Preserve the building geometry from the first image. "
        "Output only the rendered image, no text."
    )
    if prompt.strip():
        instruction += f" Additional direction: {prompt.strip()}"

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
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash-preview-image-generation",
    ]
    data = None
    _errors: list[str] = []
    for _mid in _image_gen_models:
        _url = f"https://generativelanguage.googleapis.com/v1beta/models/{_mid}:generateContent?key={key}"
        async with httpx.AsyncClient(timeout=120) as http:
            r = await http.post(_url, json=payload)
            if r.status_code in (404, 400):
                _errors.append(f"{_mid}: {r.status_code} {r.text[:120]}")
                continue
            r.raise_for_status()
            data = r.json()
            break

    if data is None:
        raise ValueError(f"No Gemini image-gen model succeeded. Errors: {'; '.join(_errors)}")

    for part in data["candidates"][0]["content"]["parts"]:
        if "inline_data" in part:
            img_bytes = base64.b64decode(part["inline_data"]["data"])
            out_path = OUTPUTS_DIR / f"{uuid.uuid4()}.png"
            out_path.write_bytes(img_bytes)
            return out_path

    raise ValueError(f"Gemini returned no image. Response: {data}")


async def gemini_25_analyze(mass_path: Path, reference_path: Path, extra_prompt: str = "", analyzer_model: str = "gemini-2.5-pro") -> str:
    """Use a Gemini text/vision model to deeply analyze mass + reference and produce a detailed render prompt."""
    key = get_gemini_key()
    if not key:
        raise ValueError("GEMINI_API_KEY not set in .env")

    def _b64(p: Path) -> tuple[str, str]:
        suffix = p.suffix.lower().lstrip(".")
        mime = "image/jpeg" if suffix in ("jpg", "jpeg") else "image/png"
        return base64.b64encode(p.read_bytes()).decode(), mime

    mass_b64, mass_mime = _b64(mass_path)
    ref_b64, ref_mime = _b64(reference_path)

    instruction = (
        "You are a senior architectural visualization director. "
        "Analyze both images carefully:\n"
        "- Image 1: an architectural mass/volume model (simplified 3D building geometry).\n"
        "- Image 2: a reference architectural render showing the target style.\n\n"
        "Write a detailed image generation prompt (300-500 words) that will guide an AI image model "
        "to render Image 1's geometry in the exact style of Image 2. Cover:\n"
        "1. Building geometry and form (from Image 1)\n"
        "2. Facade materials, textures, colors, glass type (from Image 2)\n"
        "3. Structural and architectural details (from Image 2)\n"
        "4. Lighting conditions, time of day, shadows (from Image 2)\n"
        "5. Sky, weather, atmosphere (from Image 2)\n"
        "6. Surrounding context, ground, vegetation (from Image 2)\n"
        "7. Camera angle and framing\n"
        "Output only the prompt text, no preamble."
    )
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
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(_url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


async def gemini_25_render(mass_path: Path, reference_path: Path, prompt: str = "", analyzer_model: str = "gemini-2.5-pro") -> Path:
    """Two-step: Gemini analyzer writes detailed prompt → image-gen model renders."""
    rich_prompt = await gemini_25_analyze(mass_path, reference_path, prompt, analyzer_model)
    return await gemini_generate_render(mass_path, reference_path, rich_prompt)


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
    """Structure-guided render using official BFL canny/depth-pro models."""
    try:
        jobs[job_id]["status"] = "processing"

        cfg = CONTROLNET_MODELS.get(model, CONTROLNET_MODELS["flux-controlnet-canny"])

        # Gemini analyzes the existing render → extracts scene lighting, atmosphere, materials.
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
        jobs[job_id].update({"status": "error", "error": str(e)})


async def run_style_transfer(
    reference_path: Path,
    mass_path: Path,
    prompt: str,
    model: str,
    job_id: str,
    api_token: Optional[str] = None,
) -> None:
    """Style transfer: apply reference image style onto new mass."""
    try:
        jobs[job_id]["status"] = "processing"

        token = get_api_token(api_token)

        if model == "gemini-25-pro":
            out_path = await gemini_25_render(mass_path, reference_path, prompt, analyzer_model="gemini-2.5-pro")
            output_url = f"/outputs/{out_path.name}"

        elif model == "gemini-25-flash":
            out_path = await gemini_25_render(mass_path, reference_path, prompt, analyzer_model="gemini-2.5-flash")
            output_url = f"/outputs/{out_path.name}"

        elif model == "gemini-direct":
            out_path = await gemini_generate_render(mass_path, reference_path, prompt)
            output_url = f"/outputs/{out_path.name}"

        elif model == "flux-redux-controlnet":
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
        jobs[job_id].update({"status": "error", "error": str(e)})


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
        jobs[job_id].update({"status": "error", "error": str(e)})


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
        model_input = {
            "image": image_input,
            "mask": open(mask_path, "rb"),
            "prompt": full_prompt,
            **({"num_inference_steps": 50, "guidance": 30, "output_format": "png"}
               if is_flux_fill
               else {"num_inference_steps": 50, "guidance_scale": 9.0,
                     "negative_prompt": "blurry, low quality, distorted, deformed, cartoon, illustration, painting, sketch, amateur, watermark"}),
        }

        token = get_api_token(api_token)
        output_url = await replicate_run(inpaint_model_id, model_input, token)
        jobs[job_id].update({"status": "done", "output_url": output_url})
        mask_path.unlink(missing_ok=True)
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e)})


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


@app.post("/api/style-transfer")
async def style_transfer(
    reference: UploadFile = File(...),
    mass: UploadFile = File(...),
    prompt: str = Form(""),
    model: str = Form("flux-redux-controlnet"),
    replicate_api_token: Optional[str] = Form(None),
):
    reference_path = save_upload(reference)
    mass_path = save_upload(mass)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    asyncio.create_task(run_style_transfer(reference_path, mass_path, prompt, model, job_id, replicate_api_token))
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
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "replicate_configured": bool(os.getenv("REPLICATE_API_TOKEN"))}


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
