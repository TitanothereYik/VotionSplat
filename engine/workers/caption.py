"""engine.workers.caption — Florence-2 fills the surroundings box only. Does not Generate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from engine.job import REPO_ROOT, apply_huggingface_cache_env, huggingface_cache_root, job_progress
from engine.pano.defaults import FLORENCE_HUB_ID, FLORENCE_REVISION, FLORENCE_TASK

MANIFEST_PATH = REPO_ROOT / "tools" / "model_manifest.json"


def _pin_florence(hub_id: str, revision: str) -> None:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for entry in data.get("models", []):
        if entry.get("id") == "florence":
            entry["repo"] = hub_id
            entry["revision"] = revision
            break
    MANIFEST_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"STATUS // florence pin written {hub_id} rev={revision}", flush=True)


def _try_load(hub_id: str, dtype, device):
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    print(f"STATUS // caption snapshot {hub_id} rev={FLORENCE_REVISION}", flush=True)
    job_progress(2, 4, "download Florence-2")
    local = snapshot_download(hub_id, revision=FLORENCE_REVISION)
    print(f"STATUS // caption load {hub_id} from {local}", flush=True)
    job_progress(3, 4, "load Florence-2")
    try:
        model = Florence2ForConditionalGeneration.from_pretrained(local, dtype=dtype)
    except TypeError:
        model = Florence2ForConditionalGeneration.from_pretrained(local, torch_dtype=dtype)
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(local)
    rev = getattr(getattr(model, "config", None), "_commit_hash", None) or ""
    if not rev:
        try:
            from huggingface_hub import model_info

            rev = str(getattr(model_info(hub_id), "sha", "") or "")
        except Exception:
            rev = ""
    return model, processor, str(rev)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Florence-2 caption worker")
    p.add_argument("--image", required=True)
    args = p.parse_args(argv)
    path = Path(args.image)
    if not path.is_file():
        print(f"STATUS // caption missing image: {path}", flush=True)
        return 2
    apply_huggingface_cache_env(os.environ)
    os.environ.setdefault("HF_HOME", str(huggingface_cache_root()))
    print(f"STATUS // caption hub={FLORENCE_HUB_ID} rev={FLORENCE_REVISION} task={FLORENCE_TASK}", flush=True)
    job_progress(1, 4, "caption start")
    try:
        import torch
        from PIL import Image
    except Exception as exc:
        print(f"STATUS // caption import failed: {exc}", flush=True)
        return 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        model, processor, rev = _try_load(FLORENCE_HUB_ID, dtype, device)
    except Exception as exc:
        print(f"STATUS // caption failed: {type(exc).__name__}: {exc}", flush=True)
        return 1
    try:
        print(f"FLORENCE_HUB\t{FLORENCE_HUB_ID}", flush=True)
        print(f"FLORENCE_REV\t{rev or FLORENCE_REVISION}", flush=True)
        image = Image.open(path).convert("RGB")
        inputs = processor(text=FLORENCE_TASK, images=image, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        generated = model.generate(**inputs, max_new_tokens=64, num_beams=3, do_sample=False)
        text = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        caption = str(text).strip()
        post = getattr(processor, "post_process_generation", None)
        if callable(post):
            try:
                parsed = post(text, task=FLORENCE_TASK, image_size=(image.width, image.height))
                if isinstance(parsed, dict):
                    caption = str(parsed.get(FLORENCE_TASK, caption)).strip()
            except Exception:
                pass
        print(f"CAPTION\t{caption}", flush=True)
        _pin_florence(FLORENCE_HUB_ID, rev or FLORENCE_REVISION)
        print("STATUS // caption done", flush=True)
        job_progress(4, 4, "caption done")
        return 0
    except Exception as exc:
        print(f"STATUS // caption failed: {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
