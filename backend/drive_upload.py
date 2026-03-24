"""
drive_upload.py — Google Drive upload helper for optimize_prompts.py
=====================================================================

Supports two auth methods:
  1. Service Account JSON  (--drive-credentials path/to/service_account.json)
  2. OAuth2 token          (run setup_drive_auth.py once → drive_token.json saved automatically)

Usage:
    from drive_upload import DriveUploader

    uploader = DriveUploader(root_folder_id="1AbCdEf...")
    run_folder_id = uploader.create_run_folder("dcd38488")
    uploader.upload_file(local_path, run_folder_id)

Requirements:
    pip install google-api-python-client google-auth google-auth-oauthlib
"""

from __future__ import annotations

import mimetypes
import sys
from pathlib import Path

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
except ImportError:
    print(
        "Google Drive libraries not installed.\n"
        "Run: pip install google-api-python-client google-auth google-auth-oauthlib",
        file=sys.stderr,
    )
    sys.exit(1)

SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_PATH = Path(__file__).parent / "drive_token.json"


def _load_credentials(credentials_path: str | Path | None):
    """Load credentials from service account JSON, OAuth2 token, or fail clearly."""
    # Option 1: explicit service account JSON
    if credentials_path:
        creds_path = Path(credentials_path)
        if not creds_path.exists():
            raise FileNotFoundError(f"Credentials file not found: {creds_path}")
        try:
            from google.oauth2 import service_account
            return service_account.Credentials.from_service_account_file(
                str(creds_path), scopes=SCOPES
            )
        except Exception:
            pass
        # Maybe it's an OAuth2 client secret — try loading as saved token
        try:
            from google.oauth2.credentials import Credentials
            creds = Credentials.from_authorized_user_file(str(creds_path), SCOPES)
            return creds
        except Exception as exc:
            raise ValueError(
                f"Could not load credentials from {creds_path}. "
                "Pass a service account JSON or a saved OAuth2 token. "
                f"Original error: {exc}"
            )

    # Option 2: saved OAuth2 token from setup_drive_auth.py
    if TOKEN_PATH.exists():
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
        return creds

    raise RuntimeError(
        "No Google Drive credentials found.\n"
        "Either:\n"
        "  1. Run setup_drive_auth.py once to authorize via browser, OR\n"
        "  2. Pass --drive-credentials path/to/service_account.json"
    )


class DriveUploader:
    def __init__(
        self,
        root_folder_id: str,
        credentials_path: str | Path | None = None,
    ) -> None:
        creds = _load_credentials(credentials_path)
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        self._root_folder_id = root_folder_id
        self._folder_cache: dict[str, str] = {}

    def create_run_folder(self, run_id: str) -> str:
        """Create a subfolder named run_id inside the root folder. Returns folder ID."""
        if run_id in self._folder_cache:
            return self._folder_cache[run_id]

        meta = {
            "name": run_id,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [self._root_folder_id],
        }
        folder = self._service.files().create(body=meta, fields="id").execute()
        folder_id: str = folder["id"]
        self._folder_cache[run_id] = folder_id
        return folder_id

    def upload_file(self, local_path: str | Path, folder_id: str) -> str:
        """Upload a local file to the given Drive folder. Returns the view link."""
        local_path = Path(local_path)
        mime_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
        meta = {"name": local_path.name, "parents": [folder_id]}
        media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=False)
        result = (
            self._service.files()
            .create(body=meta, media_body=media, fields="id,webViewLink")
            .execute()
        )
        return result.get("webViewLink", result["id"])

    def upload_bytes(self, data: bytes, filename: str, folder_id: str) -> str:
        """Upload raw bytes as a file to the given Drive folder. Returns the view link."""
        import io
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        meta = {"name": filename, "parents": [folder_id]}
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=False)
        result = (
            self._service.files()
            .create(body=meta, media_body=media, fields="id,webViewLink")
            .execute()
        )
        return result.get("webViewLink", result["id"])
