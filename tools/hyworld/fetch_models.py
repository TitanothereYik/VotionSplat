"""Download the weights HY-World stage 1, 2 and 5 need. Run with the hyworld venv python."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / ".hf_cache"
os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
os.environ["HF_XET_CACHE"] = str(CACHE / "xet")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["TORCH_HOME"] = str(CACHE / "torch")


def main() -> int:
    from huggingface_hub import snapshot_download

    jobs = [
        ("Qwen/Qwen3-VL-8B-Instruct", None),
        ("naver-iv/zim-anything-vitl", ["zim_vit_l_2092/*"]),
        ("IDEA-Research/grounding-dino-tiny", ["*.json", "*.txt", "model.safetensors"]),
    ]
    for repo, patterns in jobs:
        print(f"fetch {repo}", flush=True)
        path = snapshot_download(repo_id=repo, allow_patterns=patterns, cache_dir=str(CACHE / "hub"))
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        print(f"ok {repo} {size / 1e9:.2f} GB {path}", flush=True)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
    print("ok lpips vgg", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
