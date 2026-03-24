"""
setup_drive_auth.py — One-time Google Drive OAuth2 authorization
================================================================

Run this ONCE to authorize Google Drive access.
After running, a token file is saved and reused automatically.

Usage:
    python setup_drive_auth.py --credentials client_secret_xxx.json

Requirements:
    pip install google-api-python-client google-auth google-auth-oauthlib
"""

import argparse
import sys
from pathlib import Path

TOKEN_PATH = Path(__file__).parent / "drive_token.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:
    print(
        "Missing dependency. Run:\n"
        "  pip install google-api-python-client google-auth google-auth-oauthlib",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Authorize Google Drive access (run once)")
    parser.add_argument(
        "--credentials", required=True,
        help="Path to the OAuth2 client_secret JSON downloaded from GCP Console",
    )
    args = parser.parse_args()

    creds_path = Path(args.credentials)
    if not creds_path.exists():
        print(f"ERROR: file not found: {creds_path}", file=sys.stderr)
        sys.exit(1)

    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
    creds = flow.run_local_server(port=0)

    TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    print(f"\n✓ Authorization complete. Token saved to: {TOKEN_PATH}")
    print("You can now use --drive-folder without --drive-credentials.")


if __name__ == "__main__":
    main()
