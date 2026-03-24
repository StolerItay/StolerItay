"""
drive_upload.py — Google Drive upload helper for optimize_prompts.py
=====================================================================

Usage:
    from drive_upload import DriveUploader

    uploader = DriveUploader(
        credentials_path="path/to/service_account.json",
        root_folder_id="1AbCdEf...",   # shared Drive folder ID
    )
    run_folder_id = uploader.create_run_folder("dcd38488")
    uploader.upload_file(local_path, run_folder_id)

Requirements:
    pip install google-api-python-client google-auth
"""

from __future__ import annotations

import mimetypes
import sys
from pathlib import Path

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except ImportError:
    print(
        "Google Drive libraries not installed.\n"
        "Run: pip install google-api-python-client google-auth",
        file=sys.stderr,
    )
    sys.exit(1)

SCOPES = ["https://www.googleapis.com/auth/drive"]


class DriveUploader:
    def __init__(self, credentials_path: str | Path, root_folder_id: str) -> None:
        creds = service_account.Credentials.from_service_account_file(
            str(credentials_path), scopes=SCOPES
        )
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
        """Upload a local file to the given Drive folder. Returns the file ID."""
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
        """Upload raw bytes as a file to the given Drive folder. Returns the file ID."""
        import io
        from googleapiclient.http import MediaIoBaseUpload

        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        meta = {"name": filename, "parents": [folder_id]}
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=False)
        result = (
            self._service.files()
            .create(body=meta, media_body=media, fields="id,webViewLink")
            .execute()
        )
        return result.get("webViewLink", result["id"])
