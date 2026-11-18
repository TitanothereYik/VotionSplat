"""One question to the local vision model. Run with the hyworld venv python."""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / ".hf_cache"
os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
os.environ["HF_XET_CACHE"] = str(CACHE / "xet")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"


def main() -> int:
    image = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("shim", REPO / "tools" / "hyworld" / "openai_shim.py")
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    client = shim.LocalOpenAI()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Is this scene indoor or outdoor? Answer with one word."},
        {"type": "image_url", "image_url": {"url": str(image)}},
    ]}]
    try:
        reply = client.chat.completions.create(messages=messages, max_tokens=16)
    except Exception:
        traceback.print_exc()
        return 1
    print("VLM answer:", reply.choices[0].message.content.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
