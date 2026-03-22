"""
pytest test suite for ArchRender AI backend.
Tests run fully offline — all Replicate API calls are mocked.

Run:
    cd backend
    source venv/bin/activate
    pip install pytest pytest-asyncio httpx
    pytest test_main.py -v
"""

import base64
import io
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Patch replicate before importing main so no real API calls ever happen
import replicate as _replicate_mod

FAKE_OUTPUT_URL = "https://example.com/result.png"


@pytest.fixture(autouse=True)
def mock_replicate(monkeypatch):
    """Replace replicate.Client.run with a fake that returns a URL immediately.

    Also patches asyncio.to_thread in main.py so background tasks complete
    synchronously during TestClient polling, including multi-step pipelines
    (e.g. Redux → ControlNet) that make two sequential thread calls.
    """
    fake_client = MagicMock()
    fake_client.run.return_value = [FAKE_OUTPUT_URL]
    monkeypatch.setattr("replicate.Client", lambda **kw: fake_client)

    # Make asyncio.to_thread call the function directly (no real thread)
    # so coroutines complete in one event-loop tick.
    async def _instant_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr("main.asyncio.to_thread", _instant_to_thread)
    return fake_client


# Import app after patching
from main import app, jobs  # noqa: E402

client = TestClient(app)


# ── helpers ─────────────────────────────────────────────────────────────────

def _png_bytes() -> bytes:
    """Minimal valid 1x1 white PNG."""
    import struct, zlib
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(c[4:]) & 0xFFFFFFFF)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def _png_file(name="test.png"):
    return (name, io.BytesIO(_png_bytes()), "image/png")


def _data_uri_mask() -> str:
    """1x1 PNG encoded as data URI (used as inpaint mask)."""
    b64 = base64.b64encode(_png_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def _wait_for_job(job_id: str, timeout_steps: int = 30) -> dict:
    """Poll /api/job/{id} until done or error.

    Each GET request gives the async event loop a chance to advance pending
    background tasks.  Two-step pipelines (e.g. Redux → ControlNet) need a
    couple of extra ticks, hence the higher default.
    """
    import time
    data = {"status": "pending"}
    for _ in range(timeout_steps):
        r = client.get(f"/api/job/{job_id}")
        assert r.status_code == 200
        data = r.json()
        if data["status"] in ("done", "error"):
            return data
        time.sleep(0.01)  # yield to event loop between polls
    return data  # type: ignore[return-value]


# ── /api/update-render ───────────────────────────────────────────────────────

class TestUpdateRender:
    def test_returns_job_id(self):
        r = client.post(
            "/api/update-render",
            files={"render": _png_file("render.png"), "mass": _png_file("mass.png")},
            data={"prompt": "glass and steel facade", "model": "flux-controlnet-canny"},
        )
        assert r.status_code == 200
        assert "jobId" in r.json()

    def test_empty_prompt_accepted(self):
        """Empty prompt should not cause a 422 validation error."""
        r = client.post(
            "/api/update-render",
            files={"render": _png_file(), "mass": _png_file()},
            data={"prompt": "", "model": "flux-controlnet-canny"},
        )
        assert r.status_code == 200

    def test_missing_mass_returns_422(self):
        r = client.post(
            "/api/update-render",
            files={"render": _png_file()},
            data={"prompt": "test"},
        )
        assert r.status_code == 422

    def test_job_reaches_done(self):
        r = client.post(
            "/api/update-render",
            files={"render": _png_file(), "mass": _png_file()},
            data={"prompt": "concrete panels"},
        )
        job_id = r.json()["jobId"]
        # TestClient runs background tasks synchronously
        result = _wait_for_job(job_id)
        assert result["status"] == "done"
        assert FAKE_OUTPUT_URL in result.get("output_url", "")

    def test_all_models_accepted(self):
        for model in ("flux-controlnet-canny", "flux-controlnet-depth", "sdxl-controlnet"):
            r = client.post(
                "/api/update-render",
                files={"render": _png_file(), "mass": _png_file()},
                data={"model": model},
            )
            assert r.status_code == 200, f"model {model} rejected"


# ── /api/style-transfer ──────────────────────────────────────────────────────

class TestStyleTransfer:
    def test_returns_job_id(self):
        r = client.post(
            "/api/style-transfer",
            files={"reference": _png_file("ref.png"), "mass": _png_file("mass.png")},
            data={"prompt": "wood and stone", "model": "flux-redux-controlnet"},
        )
        assert r.status_code == 200
        assert "jobId" in r.json()

    def test_empty_prompt_accepted(self):
        r = client.post(
            "/api/style-transfer",
            files={"reference": _png_file(), "mass": _png_file()},
            data={"prompt": ""},
        )
        assert r.status_code == 200

    def test_job_reaches_done(self):
        r = client.post(
            "/api/style-transfer",
            files={"reference": _png_file(), "mass": _png_file()},
            data={"model": "flux-redux-controlnet"},
        )
        job_id = r.json()["jobId"]
        result = _wait_for_job(job_id)
        assert result["status"] == "done"


# ── /api/new-angle ───────────────────────────────────────────────────────────

class TestNewAngle:
    def test_returns_job_id(self):
        r = client.post(
            "/api/new-angle",
            files={"render": _png_file()},
            data={"angle_prompt": "aerial view", "style_prompt": "golden hour", "model": "flux-redux"},
        )
        assert r.status_code == 200
        assert "jobId" in r.json()

    def test_empty_style_prompt_accepted(self):
        r = client.post(
            "/api/new-angle",
            files={"render": _png_file()},
            data={"angle_prompt": "street level", "style_prompt": ""},
        )
        assert r.status_code == 200

    def test_optional_reference_image(self):
        r = client.post(
            "/api/new-angle",
            files={"render": _png_file(), "reference": _png_file("ref.png")},
            data={"angle_prompt": "aerial"},
        )
        assert r.status_code == 200

    def test_job_reaches_done(self):
        r = client.post(
            "/api/new-angle",
            files={"render": _png_file()},
            data={"model": "flux-redux"},
        )
        job_id = r.json()["jobId"]
        result = _wait_for_job(job_id)
        assert result["status"] == "done"


# ── /api/inpaint ─────────────────────────────────────────────────────────────

class TestInpaint:
    def test_with_uploaded_image(self):
        r = client.post(
            "/api/inpaint",
            files={"base_image": _png_file()},
            data={"mask_data_url": _data_uri_mask(), "prompt": "add trees", "model": "flux-fill-pro"},
        )
        assert r.status_code == 200
        assert "jobId" in r.json()

    def test_with_url_image(self):
        r = client.post(
            "/api/inpaint",
            data={
                "base_image_url": FAKE_OUTPUT_URL,
                "mask_data_url": _data_uri_mask(),
                "prompt": "glass panels",
                "model": "flux-fill-dev",
            },
        )
        assert r.status_code == 200

    def test_missing_prompt_returns_422(self):
        r = client.post(
            "/api/inpaint",
            data={"mask_data_url": _data_uri_mask()},
        )
        assert r.status_code == 422

    def test_job_reaches_done(self):
        r = client.post(
            "/api/inpaint",
            files={"base_image": _png_file()},
            data={"mask_data_url": _data_uri_mask(), "prompt": "stone cladding"},
        )
        job_id = r.json()["jobId"]
        result = _wait_for_job(job_id)
        assert result["status"] == "done"


# ── /api/job/{id} ────────────────────────────────────────────────────────────

class TestJobStatus:
    def test_unknown_job_returns_404(self):
        r = client.get(f"/api/job/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_pending_job_returns_pending(self):
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending"}
        r = client.get(f"/api/job/{job_id}")
        assert r.status_code == 200
        assert r.json()["status"] == "pending"

    def test_done_job_returns_url(self):
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "done", "output_url": FAKE_OUTPUT_URL}
        r = client.get(f"/api/job/{job_id}")
        assert r.json()["output_url"] == FAKE_OUTPUT_URL


# ── helpers unit tests ────────────────────────────────────────────────────────

class TestHelpers:
    def test_image_to_data_uri_png(self, tmp_path):
        from main import image_to_data_uri
        f = tmp_path / "img.png"
        f.write_bytes(_png_bytes())
        uri = image_to_data_uri(f)
        assert uri.startswith("data:image/png;base64,")

    def test_image_to_data_uri_jpg(self, tmp_path):
        from main import image_to_data_uri
        f = tmp_path / "img.jpg"
        f.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)  # minimal JPG-like bytes
        uri = image_to_data_uri(f)
        assert uri.startswith("data:image/jpeg;base64,")

    def test_get_replicate_client_uses_env(self, monkeypatch):
        from main import get_replicate_client
        monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_test_token")
        c = get_replicate_client()
        assert c is not None

    def test_get_replicate_client_uses_param(self):
        from main import get_replicate_client
        c = get_replicate_client("r8_custom_token")
        assert c is not None
