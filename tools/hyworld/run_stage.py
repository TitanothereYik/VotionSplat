"""Launch one HY-World stage inside the hyworld venv.

Patches glob paths, forces a one-process gloo group, and turns tqdm into the
app's PROGRESS lines. The OpenAI client is the local Qwen shim.
"""

from __future__ import annotations

import glob
import os
import runpy
import socket
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORLD = Path(sys.prefix) / "worldgen"
STAGES = {
    "traj_generate": WORLD / "traj_generate.py",
    "traj_render": WORLD / "traj_render.py",
    "align_bank": REPO / "tools" / "hyworld" / "align_bank.py",
    "gen_gs_data": WORLD / "gen_gs_data.py",
    "world_gs_trainer": WORLD / "world_gs_trainer.py",
    "view_ckpt": REPO / "tools" / "hyworld" / "view_ckpt.py",
}


def _free_port() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = str(sock.getsockname()[1])
    sock.close()
    return port


def _patch_glob() -> None:
    original = glob.glob
    original_iglob = glob.iglob

    def glob_fixed(pathname, *args, **kwargs):
        return [path.replace("\\", "/") for path in original(pathname, *args, **kwargs)]

    def iglob_fixed(pathname, *args, **kwargs):
        for path in original_iglob(pathname, *args, **kwargs):
            yield path.replace("\\", "/")

    glob.glob = glob_fixed
    glob.iglob = iglob_fixed


def _patch_distributed() -> None:
    import torch.distributed as dist

    original_init = dist.init_process_group
    original_gather = dist.all_gather

    def init_process_group(*args, **kwargs):
        args = list(args)
        kwargs["backend"] = "gloo"
        if args:
            args[0] = "gloo"
            kwargs.pop("backend")
        return original_init(*args, **kwargs)

    def all_gather(tensor_list, tensor, *args, **kwargs):
        if dist.is_initialized() and dist.get_world_size() == 1:
            tensor_list[0].copy_(tensor)
            return None
        return original_gather(tensor_list, tensor, *args, **kwargs)

    dist.init_process_group = init_process_group
    dist.all_gather = all_gather


_NULL = open(os.devnull, "w", encoding="utf-8")


def _patch_tqdm() -> None:
    import tqdm as tqdm_mod

    class JobTqdm(tqdm_mod.tqdm):
        def __init__(self, *args, **kwargs):
            if len(args) < 5:
                kwargs["file"] = _NULL
            super().__init__(*args, **kwargs)

        def _job_line(self, force: bool = False) -> None:
            if getattr(self, "disable", False):
                return
            now = time.time()
            total = int(getattr(self, "total", 0) or 0)
            current = int(getattr(self, "n", 0) or 0)
            if not force and now - getattr(self, "_job_last", 0.0) < 0.5 and not (total and current >= total):
                return
            self._job_last = now
            pct = int(100 * current / total) if total else 0
            desc = str(getattr(self, "desc", "") or "").replace("\t", " ").replace("\n", " ")
            try:
                print(f"PROGRESS\tjob\t{current}\t{total}\t{pct}\t{desc}", flush=True)
            except (OSError, ValueError):
                pass

        def update(self, n=1):
            displayed = super().update(n)
            self._job_line()
            return displayed

        def close(self):
            if not getattr(self, "_job_closed", False):
                self._job_closed = True
                self._job_line(force=True)
            super().close()

    tqdm_mod.tqdm = JobTqdm
    tqdm_mod.std.tqdm = JobTqdm
    if hasattr(tqdm_mod, "auto"):
        tqdm_mod.auto.tqdm = JobTqdm


def _inject_openai() -> None:
    shim_path = REPO / "tools" / "hyworld" / "openai_shim.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("votion_openai_shim", shim_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {shim_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake = types.ModuleType("openai")
    fake.OpenAI = module.LocalOpenAI
    sys.modules["openai"] = fake


def _prepare_env(stage: str) -> None:
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["USE_LIBUV"] = "0"
    device = os.environ.get("CUDA_DEVICE_INDEX", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = _free_port()
    os.environ.setdefault("SAM3_ROOT", r"D:\sam3\sam3")
    os.environ.setdefault("SAM3_CHECKPOINT", r"D:\sam3\checkpoints\sam3.pt")
    os.environ["HY_VLM_KEEP"] = "0" if stage == "traj_generate" else "1"
    cache = REPO / ".hf_cache"
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ["TORCH_HOME"] = str(cache / "torch")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        split = argv.index("--")
        head, child = argv[:split], argv[split + 1 :]
    else:
        head, child = argv, []
    stage = ""
    if "--stage" in head:
        stage = head[head.index("--stage") + 1]
    if stage not in STAGES:
        print("STATUS // unknown hyworld stage", stage, flush=True)
        return 2
    script = STAGES[stage]
    if not script.is_file() and stage != "align_bank":
        print(f"STATUS // missing {script}", flush=True)
        return 2
    _prepare_env(stage)
    here = str(Path(__file__).resolve().parent)
    for extra in (here, str(REPO)):
        if extra not in sys.path:
            sys.path.append(extra)
    if stage == "traj_generate" and "--scene" in head:
        from engine.hyworld.prepare import prepare_scene

        def _value(flag: str) -> str:
            at = head.index(flag) + 1 if flag in head else len(head)
            return head[at] if at < len(head) else ""

        try:
            prepare_scene(_value("--scene"), _value("--scene-type"), "--fresh" in head)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"STATUS // {exc}", flush=True)
            return 2
    _patch_glob()
    _patch_distributed()
    _patch_tqdm()
    _inject_openai()
    sys.path.insert(0, str(WORLD))
    os.chdir(WORLD)
    for index, arg in enumerate(child):
        if arg in {"--target_path", "--root_path", "--data_dir", "--result_dir", "--data-dir", "--result-dir"} and index + 1 < len(child):
            child[index + 1] = child[index + 1].replace("\\", "/")
    sys.argv = [str(script), *child]
    print(f"STATUS // hyworld {stage} start", flush=True)
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code not in (None, 0):
            print(f"STATUS // hyworld {stage} exit {code}", flush=True)
            return int(code) if isinstance(code, int) else 1
    print(f"STATUS // hyworld {stage} exit 0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
