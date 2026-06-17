"""One-time HuggingFace login helper for sam-audio-lite.

Run this once before using the app for the first time:

    uv run python setup_hf_token.py
"""

import getpass
import sys


INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════╗
║           sam-audio-lite — HuggingFace Setup                 ║
╚══════════════════════════════════════════════════════════════╝

The SAM-Audio models are hosted on HuggingFace and require a
free account + access token to download.

Step 1 — Create a HuggingFace account (if you don't have one):
  https://huggingface.co/join

Step 2 — Request access to the SAM-Audio models.
  You need to accept the licence on the model you plan to use:
  • ex: https://huggingface.co/facebook/sam-audio-base
  Click "Agree and access repository" on each page.
  Access is granted instantly.

Step 3 — Create a read token:
  https://huggingface.co/settings/tokens
  Click "New token", choose "Read" (the minimum required), copy it.
  It looks like: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

print(INSTRUCTIONS)

try:
    from huggingface_hub import login, whoami  # type: ignore
except ImportError:
    print("ERROR: huggingface_hub is not installed.")
    print("Run:  uv sync")
    sys.exit(1)

token = getpass.getpass("Paste your HuggingFace token (input hidden): ").strip()

if not token:
    print("\nNo token entered. Exiting.")
    sys.exit(1)

if not token.startswith("hf_"):
    print("\nWarning: token doesn't look like a HuggingFace token (expected hf_…).")
    confirm = input("Continue anyway? [y/N]: ").strip().lower()
    if confirm != "y":
        sys.exit(1)

try:
    login(token=token, add_to_git_credential=False)
    user = whoami()["name"]
    print(f"\nLogged in as: {user}")
    print("Token saved to ~/.cache/huggingface/token")
    print("You can now run the app:  uv run python -m sam_audio_lite.app")
except Exception as exc:
    print(f"\nLogin failed: {exc}")
    print("Double-check the token and that you have accepted the model licences.")
    sys.exit(1)
