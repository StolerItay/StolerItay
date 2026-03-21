# ArchRender AI

Architectural render editor powered by Stable Diffusion + ControlNet.

## Features

1. **Update Render** — Upload an existing render + new building mass; AI replaces the building preserving the scene context
2. **Style Transfer** — Transfer the visual style of a reference render onto a new building mass
3. **New Angle** — Generate a new camera angle of a building from an existing render, guided by angle presets or a custom prompt
4. **Inpaint Tool** — Paint over any area in the result and describe what to put there; only the painted region is changed

All modes accept a text prompt for materiality, texture, and render style.

## Setup

### 1. Get a Replicate API token

Sign up at [replicate.com](https://replicate.com) and copy your API token.

### 2. Backend

```bash
cd backend
cp .env.example .env
# edit .env and set REPLICATE_API_TOKEN=r8_...
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Architecture

```
frontend/   React + Vite + TypeScript + Tailwind
backend/    FastAPI + Replicate API
```

### AI Models Used

| Feature | Model |
|---|---|
| Update Render / Style Transfer | ControlNet Canny (jagilley/controlnet-canny) |
| New Angle | Stable Diffusion img2img |
| Inpaint | Stable Diffusion Inpainting |

## How It Works

1. User uploads images and writes a style prompt
2. Frontend POSTs to FastAPI → job ID returned immediately
3. Backend runs the Replicate model asynchronously
4. Frontend polls `/api/job/{id}` every 2 seconds
5. When done, the result image URL is displayed
6. User can click **Inpaint**, paint a yellow mask over any area, add a prompt, and regenerate only that region
