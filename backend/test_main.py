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
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

FAKE_OUTPUT_URL = "https://example.com/result.png"


@pytest.fixture(autouse=True)
def mock_replicate(monkeypatch):
    """Replace replicate_run with a fake that returns a URL immediately."""
    async def _fake_run(model_id, input_data, api_token):
        return FAKE_OUTPUT_URL

    monkeypatch.setattr("main.replicate_run", _fake_run)


@pytest.fixture(autouse=True)
def mock_gemini(monkeypatch, tmp_path):
    """Mock all Gemini API functions so tests run offline."""
    import struct, zlib

    def _png_bytes_local():
        def chunk(tag, data):
            c = struct.pack(">I", len(data)) + tag + data
            return c + struct.pack(">I", zlib.crc32(c[4:]) & 0xFFFFFFFF)
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        idat = chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        iend = chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    async def _fake_gemini_render(mass_path, reference_path, prompt=""):
        out = tmp_path / f"fake_gemini_{uuid.uuid4()}.png"
        out.write_bytes(_png_bytes_local())
        return out

    async def _fake_gemini_25_render(mass_path, reference_path, prompt="", analyzer_model="gemini-2.5-pro"):
        out = tmp_path / f"fake_gemini25_{uuid.uuid4()}.png"
        out.write_bytes(_png_bytes_local())
        return out

    async def _fake_gemini_describe(image_path, extra_prompt=""):
        return extra_prompt or "modern glass and steel facade"

    # Three-image workflow mocks
    async def _fake_gemini_25_analyze_three(original_mass_path, modified_mass_path, reference_path,
                                             extra_prompt="", analyzer_model="gemini-2.5-pro"):
        return extra_prompt or "delta-aware render prompt: bulging tower preserved, twilight lighting"

    async def _fake_gemini_generate_render_three(original_mass_path, modified_mass_path, reference_path, prompt=""):
        out = tmp_path / f"fake_gemini3img_{uuid.uuid4()}.png"
        out.write_bytes(_png_bytes_local())
        return out

    async def _fake_gemini_25_render_three(original_mass_path, modified_mass_path, reference_path,
                                            prompt="", analyzer_model="gemini-2.5-pro"):
        out = tmp_path / f"fake_gemini25_3img_{uuid.uuid4()}.png"
        out.write_bytes(_png_bytes_local())
        return out

    monkeypatch.setattr("main.gemini_generate_render", _fake_gemini_render)
    monkeypatch.setattr("main.gemini_25_render", _fake_gemini_25_render)
    monkeypatch.setattr("main.gemini_describe_style", _fake_gemini_describe)
    monkeypatch.setattr("main.gemini_25_analyze_three_image", _fake_gemini_25_analyze_three)
    monkeypatch.setattr("main.gemini_generate_render_three_image", _fake_gemini_generate_render_three)
    monkeypatch.setattr("main.gemini_25_render_three_image", _fake_gemini_25_render_three)


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
    """Poll /api/job/{id} until done or error."""
    import time
    data = {"status": "pending"}
    for _ in range(timeout_steps):
        r = client.get(f"/api/job/{job_id}")
        assert r.status_code == 200
        data = r.json()
        if data["status"] in ("done", "error"):
            return data
        time.sleep(0.01)
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

    def test_controlnet_models_reach_done(self):
        """Flux Canny/Depth Pro must complete successfully (not fall back to flux-dev)."""
        for model in ("flux-controlnet-canny", "flux-controlnet-depth"):
            r = client.post(
                "/api/update-render",
                files={"render": _png_file(), "mass": _png_file()},
                data={"model": model},
            )
            result = _wait_for_job(r.json()["jobId"])
            assert result["status"] == "done", f"{model} failed: {result}"


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

    @pytest.mark.parametrize("model", [
        "gemini-25-pro", "gemini-25-flash", "gemini-direct",
        "flux-redux-controlnet", "flux-redux-only", "sdxl-img2img",
    ])
    def test_all_models_reach_done(self, model):
        r = client.post(
            "/api/style-transfer",
            files={"reference": _png_file(), "mass": _png_file()},
            data={"model": model},
        )
        assert r.status_code == 200, f"model {model} rejected at submission"
        result = _wait_for_job(r.json()["jobId"])
        assert result["status"] == "done", f"model {model} did not reach done: {result}"

    def test_gemini_models_return_local_url(self):
        """Gemini models save locally and return /outputs/… path, not an external URL."""
        for model in ("gemini-25-pro", "gemini-25-flash", "gemini-direct"):
            r = client.post(
                "/api/style-transfer",
                files={"reference": _png_file(), "mass": _png_file()},
                data={"model": model},
            )
            result = _wait_for_job(r.json()["jobId"])
            assert result["output_url"].startswith("/outputs/"), \
                f"{model} returned unexpected url: {result['output_url']}"


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


# ── /api/style-transfer (three-image delta workflow) ─────────────────────────

class TestStyleTransferThreeImage:
    """Three-image workflow: reference + modified mass + original mass (delta mode)."""

    def test_three_image_returns_job_id(self):
        r = client.post(
            "/api/style-transfer",
            files={
                "reference": _png_file("ref.png"),
                "mass": _png_file("modified_mass.png"),
                "original_mass": _png_file("original_mass.png"),
            },
            data={"prompt": "twilight lighting, warm glow", "model": "gemini-25-pro"},
        )
        assert r.status_code == 200
        assert "jobId" in r.json()

    def test_three_image_job_reaches_done(self):
        r = client.post(
            "/api/style-transfer",
            files={
                "reference": _png_file("ref.png"),
                "mass": _png_file("modified_mass.png"),
                "original_mass": _png_file("original_mass.png"),
            },
            data={"model": "gemini-25-pro"},
        )
        result = _wait_for_job(r.json()["jobId"])
        assert result["status"] == "done"
        assert result["output_url"].startswith("/outputs/")

    @pytest.mark.parametrize("model", [
        "gemini-25-pro", "gemini-25-flash", "gemini-direct",
        "flux-redux-controlnet", "sdxl-img2img",
    ])
    def test_three_image_all_gemini_models_reach_done(self, model):
        r = client.post(
            "/api/style-transfer",
            files={
                "reference": _png_file("ref.png"),
                "mass": _png_file("mass.png"),
                "original_mass": _png_file("orig.png"),
            },
            data={"model": model},
        )
        assert r.status_code == 200, f"model {model} rejected at submission"
        result = _wait_for_job(r.json()["jobId"])
        assert result["status"] == "done", f"model {model} did not reach done: {result}"

    def test_two_image_still_works_without_original_mass(self):
        """Backward compatibility: omitting original_mass uses the existing two-image path."""
        r = client.post(
            "/api/style-transfer",
            files={"reference": _png_file(), "mass": _png_file()},
            data={"model": "gemini-25-pro"},
        )
        result = _wait_for_job(r.json()["jobId"])
        assert result["status"] == "done"

    def test_three_image_gemini_models_return_local_url(self):
        for model in ("gemini-25-pro", "gemini-25-flash", "gemini-direct"):
            r = client.post(
                "/api/style-transfer",
                files={
                    "reference": _png_file(),
                    "mass": _png_file(),
                    "original_mass": _png_file(),
                },
                data={"model": model},
            )
            result = _wait_for_job(r.json()["jobId"])
            assert result["output_url"].startswith("/outputs/"), \
                f"{model} three-image returned unexpected url: {result['output_url']}"


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
        f.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
        uri = image_to_data_uri(f)
        assert uri.startswith("data:image/jpeg;base64,")

    def test_get_api_token_uses_env(self, monkeypatch):
        from main import get_api_token
        monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_test_token")
        assert get_api_token() == "r8_test_token"

    def test_get_api_token_uses_param(self):
        from main import get_api_token
        assert get_api_token("r8_custom_token") == "r8_custom_token"
