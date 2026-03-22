import os
import sys
import uuid
import asyncio
import base64
from pathlib import Path
from typing import Optional

import replicate
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


CONTROLNET_MODELS = {
    "flux-controlnet-canny": {
        "id": "xlabs-ai/flux-dev-controlnet:9a8db105db745f8b11ad3afe5c8bd892428b2a43ade0b67edc4e0ccd52ff2fda",
        "input_key": "control_image",
        "extra": {"control_type": "canny", "controlnet_conditioning_scale": 0.7,
                   "num_inference_steps": 50, "guidance_scale": 4.5},
    },
    "flux-controlnet-depth": {
        "id": "xlabs-ai/flux-dev-controlnet:9a8db105db745f8b11ad3afe5c8bd892428b2a43ade0b67edc4e0ccd52ff2fda",
        "input_key": "control_image",
        "extra": {"control_type": "depth", "controlnet_conditioning_scale": 0.7,
                   "num_inference_steps": 50, "guidance_scale": 4.5},
    },
    "sdxl-controlnet": {
        "id": "diffusers/controlnet-canny-sdxl-1.0:a398a399f1238d5651c7bb7b5417823f1d559fc2ab1b7fa3f06a45d57c971db4",
        "input_key": "image",
        "extra": {"num_inference_steps": 50, "guidance_scale": 9.0,
                   "controlnet_conditioning_scale": 0.85,
                   "negative_prompt": "blurry, low quality, distorted, deformed, cartoon, illustration, painting, sketch, amateur, watermark, text"},
    },
}


def get_replicate_client(api_token: Optional[str] = None) -> replicate.Client:
    token = api_token or os.getenv("REPLICATE_API_TOKEN", "")
    return replicate.Client(api_token=token)


async def run_controlnet_render(
    render_path: Path,
    mass_path: Path,
    prompt: str,
    model: str,
    job_id: str,
    api_token: Optional[str] = None,
) -> None:
    """ControlNet-guided render: replace building with new mass."""
    try:
        jobs[job_id]["status"] = "processing"

        client = get_replicate_client(api_token)
        cfg = CONTROLNET_MODELS.get(model, CONTROLNET_MODELS["flux-controlnet-canny"])

        if not prompt.strip():
            # No prompt: extract style/materials/atmosphere from the original render using Redux,
            # then constrain the result to the new mass shape with ControlNet.
            # This preserves the exact materiality, lighting, vegetation and site atmosphere
            # of the reference render without requiring the user to describe it in text.
            redux_output = await asyncio.to_thread(
                client.run,
                "black-forest-labs/flux-redux-dev",
                input={
                    "redux_image": open(render_path, "rb"),
                    "num_inference_steps": 50,
                    "guidance": 3.5,
                },
            )
            redux_url = str(redux_output[0] if isinstance(redux_output, list) else redux_output)

            output = await asyncio.to_thread(
                client.run,
                "xlabs-ai/flux-dev-controlnet:9a8db105db745f8b11ad3afe5c8bd892428b2a43ade0b67edc4e0ccd52ff2fda",
                input={
                    "control_image": open(mass_path, "rb"),
                    "image": redux_url,
                    "prompt": "award-winning architectural visualization, photorealistic CGI render, ultra-detailed, 8K ultra resolution",
                    "prompt_strength": 0.80,
                    "controlnet_conditioning_scale": 0.7,
                    "control_type": "canny" if model != "flux-controlnet-depth" else "depth",
                    "num_inference_steps": 50,
                    "guidance_scale": 4.5,
                },
            )
        else:
            full_prompt = (
                f"award-winning architectural visualization, {prompt}, "
                "photorealistic CGI render, dramatic cinematic lighting, golden hour atmosphere, "
                "volumetric light rays, ultra-detailed facade materials, glass curtain wall reflections, "
                "ambient occlusion, ray-traced global illumination, professional architectural photography, "
                "hyperrealistic, 8K ultra resolution, sharp focus, "
                "Zaha Hadid Architects quality render, architectural digest cover shot"
            )

            model_input = {cfg["input_key"]: open(mass_path, "rb"), "prompt": full_prompt, **cfg["extra"]}

            if model in ("flux-controlnet-canny", "flux-controlnet-depth"):
                model_input["image"] = open(render_path, "rb")
                model_input["prompt_strength"] = 0.80

            output = await asyncio.to_thread(
                client.run,
                cfg["id"],
                input=model_input,
            )

        output_url = output[0] if isinstance(output, list) else output
        jobs[job_id].update({"status": "done", "output_url": str(output_url)})
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

        # If no prompt given, let the reference image speak for itself - Redux will extract
        # all style/material/atmosphere information visually, no text description needed.
        style_prompt = (
            "award-winning architectural visualization, "
            "faithfully matching the style, materials and atmosphere of the reference image, "
            "photorealistic CGI render, cinematic lighting, ultra-detailed facade, "
            "professional architectural photography, hyperrealistic, 8K ultra resolution"
            if not prompt.strip()
            else (
                f"award-winning architectural visualization, {prompt}, "
                "faithfully matching the style, materials and atmosphere of the reference image, "
                "photorealistic CGI render, cinematic lighting, ultra-detailed facade, "
                "professional architectural photography, hyperrealistic, 8K ultra resolution"
            )
        )

        client = get_replicate_client(api_token)
        if model == "flux-redux-controlnet":
            # Step 1: Redux extracts style/look from the reference image
            redux_output = await asyncio.to_thread(
                client.run,
                "black-forest-labs/flux-redux-dev",
                input={"redux_image": open(reference_path, "rb"),
                       "num_inference_steps": 50, "guidance": 3.5},
            )
            redux_url = str(redux_output[0] if isinstance(redux_output, list) else redux_output)

            # Step 2: ControlNet constrains the redux-styled image to the mass shape
            output = await asyncio.to_thread(
                client.run,
                "xlabs-ai/flux-dev-controlnet:9a8db105db745f8b11ad3afe5c8bd892428b2a43ade0b67edc4e0ccd52ff2fda",
                input={"control_image": open(mass_path, "rb"), "image": redux_url,
                       "prompt": style_prompt, "prompt_strength": 0.80,
                       "controlnet_conditioning_scale": 0.7, "num_inference_steps": 50,
                       "guidance_scale": 4.5, "control_type": "canny"},
            )
        elif model == "flux-redux-only":
            output = await asyncio.to_thread(
                client.run,
                "black-forest-labs/flux-redux-dev",
                input={"redux_image": open(reference_path, "rb"),
                       "num_inference_steps": 50, "guidance": 3.5},
            )
        else:  # sdxl-img2img
            output = await asyncio.to_thread(
                client.run,
                "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input={"image": open(mass_path, "rb"), "prompt": style_prompt,
                       "prompt_strength": 0.80, "num_inference_steps": 50, "guidance_scale": 9.0,
                       "negative_prompt": "blurry, low quality, distorted, deformed, cartoon, illustration, painting, sketch, amateur"},
            )

        output_url = output[0] if isinstance(output, list) else output
        jobs[job_id].update({"status": "done", "output_url": str(output_url)})
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

        # If no style prompt, the render itself is the style reference - Redux will
        # extract materiality, atmosphere and vegetation directly from the image.
        style_part = style_prompt.strip() if style_prompt.strip() else "matching the exact materials, lighting and atmosphere of the original render"
        combined_prompt = (
            f"award-winning architectural visualization of the exact same building, {angle_prompt}, "
            f"{style_part}, photorealistic CGI render, dramatic cinematic lighting, "
            "volumetric atmosphere, ultra-detailed facade materials, "
            "professional architectural photography, hyperrealistic, 8K ultra resolution"
        )

        client = get_replicate_client(api_token)
        if model == "zero123plus":
            output = await asyncio.to_thread(
                client.run,
                "sudo-ai/zero123plus:0e3a8a2cc1f5b88a0c24a40a5fd10d84be4b76c0e83a55a7b1f7c7ac67df5432",
                input={"image": open(render_path, "rb"), "scale": 4.0, "num_inference_steps": 50},
            )
        elif model == "flux-redux":
            # Redux always uses the render image as visual style reference regardless of text prompt
            output = await asyncio.to_thread(
                client.run,
                "black-forest-labs/flux-redux-dev",
                input={"redux_image": open(render_path, "rb"),
                       "num_inference_steps": 50, "guidance": 3.5},
            )
        else:  # flux-img2img
            output = await asyncio.to_thread(
                client.run,
                "black-forest-labs/flux-dev:a60b88a054a2c9e7f6d5e5c8a42b2d6e7c8f9a1b2c3d4e5f6a7b8c9d0e1f2a3b",
                input={"image": open(render_path, "rb"), "prompt": combined_prompt,
                       "prompt_strength": 0.75, "num_inference_steps": 50, "guidance_scale": 4.5},
            )

        output_url = output[0] if isinstance(output, list) else output
        jobs[job_id].update({"status": "done", "output_url": str(output_url)})
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

        client = get_replicate_client(api_token)
        output = await asyncio.to_thread(
            client.run,
            inpaint_model_id,
            input=model_input,
        )

        output_url = output[0] if isinstance(output, list) else output
        jobs[job_id].update({"status": "done", "output_url": str(output_url)})
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
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://api.replicate.com/v1/account",
                headers={"Authorization": f"Token {token}"},
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
