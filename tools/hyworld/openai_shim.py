"""Local Qwen3-VL standing in for the OpenAI client HY-World expects."""

from __future__ import annotations

import base64
import gc
import io
import os
import threading

from PIL import Image

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MAX_PIXELS = 1792 * 896
_LOCK = threading.Lock()
_STATE: dict = {"model": None, "processor": None}


def _keep_loaded() -> bool:
    return os.environ.get("HY_VLM_KEEP", "1") != "0"


def unload() -> None:
    _STATE["model"] = None
    _STATE["processor"] = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load():
    if _STATE["model"] is None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        _STATE["processor"] = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=False)
        _STATE["model"] = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        ).eval()
    return _STATE["model"], _STATE["processor"]


def _image(url: str) -> Image.Image:
    if url.startswith("data:"):
        raw = base64.b64decode(url.split(",", 1)[1])
        image = Image.open(io.BytesIO(raw))
    else:
        image = Image.open(url.removeprefix("file://"))
    image = image.convert("RGB")
    width, height = image.size
    if width * height > MAX_PIXELS:
        scale = (MAX_PIXELS / (width * height)) ** 0.5
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BICUBIC)
    return image


def _messages(messages: list) -> list:
    converted = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            items = [{"type": "text", "text": content}]
        else:
            items = []
            for item in content or []:
                if item.get("type") == "text":
                    items.append({"type": "text", "text": item.get("text") or ""})
                else:
                    url = ""
                    if item.get("type") == "image_url":
                        url = (item.get("image_url") or {}).get("url") or ""
                    elif item.get("type") == "image":
                        url = item.get("image") or ""
                    if url:
                        items.append({"type": "image", "image": _image(url)})
        converted.append({"role": message.get("role") or "user", "content": items})
    return converted


class _Obj:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class Completions:
    def create(self, *args, **kwargs):
        try:
            return self._create(*args, **kwargs)
        except Exception as exc:
            print(f"HY\twarn\tvision model failed: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            raise

    def _create(self, model=None, messages=None, max_tokens=1024, temperature=0.0, seed=None, **kwargs):
        del model, kwargs
        with _LOCK:
            mdl, processor = _load()
            import torch

            batch = processor.apply_chat_template(
                _messages(messages or []),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(mdl.device)
            generate = {"max_new_tokens": int(max_tokens), "do_sample": False}
            if temperature and float(temperature) > 0.1:
                if seed is not None:
                    torch.manual_seed(int(seed))
                generate = {
                    "max_new_tokens": int(max_tokens),
                    "do_sample": True,
                    "temperature": float(temperature),
                }
            with torch.inference_mode():
                out = mdl.generate(**batch, **generate)
            text = processor.batch_decode(out[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)[0]
            del batch, out
            if not _keep_loaded():
                unload()
            else:
                torch.cuda.empty_cache()
        return _Obj(choices=[_Obj(index=0, finish_reason="stop", message=_Obj(role="assistant", content=text))])


class LocalOpenAI:
    unload = staticmethod(unload)

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.chat = _Obj(completions=Completions())
