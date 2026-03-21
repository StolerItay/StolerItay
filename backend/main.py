import os
import uuid
import asyncio
import base64
from pathlib import Path
from typing import Optional

import replicate
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ArchRender AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (use Redis/DB for production)
jobs: dict[str, dict] = {}

UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


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


async def run_controlnet_render(
    render_path: Path,
    mass_path: Path,
    prompt: str,
    job_id: str,
) -> None:
    """Flux Dev ControlNet (canny) — replace building with new mass at high quality."""
    try:
        jobs[job_id]["status"] = "processing"

        full_prompt = (
            f"Photorealistic architectural render, {prompt}, "
            "8K resolution, professional architectural photography, "
            "detailed materials, realistic lighting, ultra sharp"
        )

        output = await asyncio.to_thread(
            replicate.run,
            "xlabs-ai/flux-dev-controlnet:f2c31c31d81278a91b2447a304dae654c96f5f37f72305b4651489e3dad14f27",
            input={
                "control_image": open(mass_path, "rb"),
                "prompt": full_prompt,
                "controlnet_conditioning_scale": 0.75,
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
                "control_type": "canny",
            },
        )

        output_url = output[0] if isinstance(output, list) else output
        jobs[job_id].update({"status": "done", "output_url": str(output_url)})
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e)})


async def run_style_transfer(
    reference_path: Path,
    mass_path: Path,
    prompt: str,
    job_id: str,
) -> None:
    """Flux Redux (style transfer) + Flux ControlNet: apply reference style to new mass."""
    try:
        jobs[job_id]["status"] = "processing"

        # Step 1: extract style embedding from reference via Flux Redux
        redux_output = await asyncio.to_thread(
            replicate.run,
            "black-forest-labs/flux-redux-dev:2a6b1ca2f8ab1f5e9704f62edc88b22afbab43cb2b2bc98e6b0c6e27e87a27c4",
            input={
                "redux_image": open(reference_path, "rb"),
                "prompt": (
                    f"Photorealistic architectural render of this building mass, "
                    f"{prompt}, matching the style and materials of the reference"
                ),
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
            },
        )
        style_url = redux_output[0] if isinstance(redux_output, list) else redux_output

        # Step 2: apply that styled result through ControlNet guided by the mass
        output = await asyncio.to_thread(
            replicate.run,
            "xlabs-ai/flux-dev-controlnet:f2c31c31d81278a91b2447a304dae654c96f5f37f72305b4651489e3dad14f27",
            input={
                "control_image": open(mass_path, "rb"),
                "prompt": (
                    f"Photorealistic architectural render, {prompt}, "
                    "matching the style of the reference image, high detail, 8K"
                ),
                "controlnet_conditioning_scale": 0.7,
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
                "control_type": "canny",
            },
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
    job_id: str,
) -> None:
    """Zero123++ for novel view synthesis; falls back to Flux img2img if angle_prompt only."""
    try:
        jobs[job_id]["status"] = "processing"

        if not angle_prompt and reference_path:
            # Pure novel-view: use Zero123++ to synthesise new camera angles
            output = await asyncio.to_thread(
                replicate.run,
                "sudo-ai/zero123plus:0e3a8a2cc1f5b88a0c24a40a5fd10d84be4b76c0e83a55a7b1f7c7ac67df5432",
                input={
                    "image": open(render_path, "rb"),
                    "scale": 4.0,
                    "num_inference_steps": 36,
                },
            )
        else:
            # Prompt-guided angle: Flux Redux keeps identity, prompt steers viewpoint
            combined_prompt = (
                f"Architectural render of the exact same building, {angle_prompt}, "
                f"{style_prompt}, photorealistic, professional CGI, 8K, detailed facade"
            )
            output = await asyncio.to_thread(
                replicate.run,
                "black-forest-labs/flux-redux-dev:2a6b1ca2f8ab1f5e9704f62edc88b22afbab43cb2b2bc98e6b0c6e27e87a27c4",
                input={
                    "redux_image": open(render_path, "rb"),
                    "prompt": combined_prompt,
                    "num_inference_steps": 28,
                    "guidance_scale": 3.5,
                },
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
    job_id: str,
) -> None:
    """Inpaint a masked region of an image."""
    try:
        jobs[job_id]["status"] = "processing"

        full_prompt = (
            f"Photorealistic architectural detail, {prompt}, "
            "seamlessly integrated, matching surrounding context, "
            "professional render quality"
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

        output = await asyncio.to_thread(
            replicate.run,
            "black-forest-labs/flux-fill-pro",
            input={
                "image": image_input,
                "mask": open(mask_path, "rb"),
                "prompt": full_prompt,
                "num_inference_steps": 28,
                "guidance": 30,
                "output_format": "png",
            },
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
):
    render_path = save_upload(render)
    mass_path = save_upload(mass)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    asyncio.create_task(run_controlnet_render(render_path, mass_path, prompt, job_id))
    return {"jobId": job_id}


@app.post("/api/style-transfer")
async def style_transfer(
    reference: UploadFile = File(...),
    mass: UploadFile = File(...),
    prompt: str = Form(""),
):
    reference_path = save_upload(reference)
    mass_path = save_upload(mass)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    asyncio.create_task(run_style_transfer(reference_path, mass_path, prompt, job_id))
    return {"jobId": job_id}


@app.post("/api/new-angle")
async def new_angle(
    render: UploadFile = File(...),
    reference: Optional[UploadFile] = File(None),
    angle_prompt: str = Form(""),
    style_prompt: str = Form(""),
):
    render_path = save_upload(render)
    reference_path = save_upload(reference) if reference else None
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    asyncio.create_task(run_new_angle(render_path, reference_path, angle_prompt, style_prompt, job_id))
    return {"jobId": job_id}


@app.post("/api/inpaint")
async def inpaint(
    base_image: Optional[UploadFile] = File(None),
    base_image_url: Optional[str] = Form(None),
    mask_data_url: str = Form(...),
    prompt: str = Form(...),
):
    base_path: Optional[Path] = None
    if base_image and base_image.filename:
        base_path = save_upload(base_image)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    asyncio.create_task(run_inpaint(base_image_url, base_path, mask_data_url, prompt, job_id))
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
