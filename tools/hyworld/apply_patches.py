"""Apply the Windows and one-GPU edits to a copy of vendored worldgen.

The copy under ``.venvs/hyworld/worldgen`` is patched. ``vendor/hyworld`` stays
as the pinned download.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _swap(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"patch anchor missing in {path.name}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _ensure(path: Path, old: str, new: str) -> None:
    """Like _swap, but a no-op when the edit is already there."""
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"patch anchor missing in {path.name}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _ensure_any(path: Path, olds: list[str], new: str) -> None:
    """Like _ensure, for an edit that replaces an earlier version of itself."""
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    for old in olds:
        if old in text:
            path.write_text(text.replace(old, new, 1), encoding="utf-8")
            return
    raise SystemExit(f"patch anchor missing in {path.name}: {olds[0][:80]!r}")


def patch(root: Path) -> None:
    traj = root / "traj_generate.py"
    _swap(traj, "SAM_BATCH_SIZE = 4", "SAM_BATCH_SIZE = 2")
    _swap(
        traj,
        'HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")',
        'HF_CACHE_DIR = os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub")',
    )
    _swap(
        traj,
        """    print("Models Initializing...")
    zim_predictor = build_zim_model("vit_l", resolve_zim_checkpoint(), device=device)
    gd_processor, gd_model = build_gd_model(resolve_gd_checkpoint(), device=device)

    depth_model = MoGeModel.from_pretrained(MOGE_ID).to(device).eval()

    # VLM & SAM3
    client = OpenAI(api_key="EMPTY", base_url=f"http://{LLM_ADDR}:{LLM_PORT}/v1")
    sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
    print("Models Initializing over.")
""",
        """    print("Models stay off the GPU until a step needs them.")
    client = OpenAI(api_key="EMPTY", base_url=f"http://{LLM_ADDR}:{LLM_PORT}/v1")
    zim_predictor = None
    gd_processor = None
    gd_model = None
    depth_model = None
    sam3_processor = None

    def _free_gpu() -> None:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
""",
    )
    _swap(
        traj,
        """                if meta_info["scene_type"] == "outdoor":
                    sky_mask = torch.tensor(~get_zim_mask(full_img, "sky.", 0.3, 0.3, zim_predictor, gd_processor, gd_model, DEVICE=device))
""",
        """                if meta_info["scene_type"] == "outdoor":
                    if zim_predictor is None:
                        zim_predictor = build_zim_model("vit_l", resolve_zim_checkpoint(), device=device)
                        gd_processor, gd_model = build_gd_model(resolve_gd_checkpoint(), device=device)
                    sky_mask = torch.tensor(~get_zim_mask(full_img, "sky.", 0.3, 0.3, zim_predictor, gd_processor, gd_model, DEVICE=device))
""",
    )
    _swap(
        traj,
        """            transforms.ToPILImage()(((~sky_mask).float() * 255).to(torch.uint8)).save(f"{scene_path}/render_results/sky_mask.png")
""",
        """            transforms.ToPILImage()(((~sky_mask).float() * 255).to(torch.uint8)).save(f"{scene_path}/render_results/sky_mask.png")
            zim_predictor = None
            gd_processor = None
            gd_model = None
            _free_gpu()
""",
    )
    _swap(
        traj,
        """            with timer.track("Predict panorama depth"):
                full_depth = pred_pano_depth(depth_model, full_img)
""",
        """            with timer.track("Predict panorama depth"):
                if depth_model is None:
                    depth_model = MoGeModel.from_pretrained(MOGE_ID).to(device).eval()
                full_depth = pred_pano_depth(depth_model, full_img)
""",
    )
    _swap(
        traj,
        """            with timer.track("[IO] Save panorama depth"):
                torch.save(full_depth, f"{scene_path}/render_results/full_depth_prediction.pt")
""",
        """            with timer.track("[IO] Save panorama depth"):
                torch.save(full_depth, f"{scene_path}/render_results/full_depth_prediction.pt")
            depth_model = None
            _free_gpu()
""",
    )
    _swap(
        traj,
        """                print(f"SAM3 process for {scene_path}...")
                for i in range(0, len(unique_objects), SAM_BATCH_SIZE):
                    batch_objects = unique_objects[i: i + SAM_BATCH_SIZE]
                    batch_objects = [obj for obj in batch_objects if len(obj.split()) < 8 and obj.lower() not in ("sun")]
                    batch_images = [full_img] * len(batch_objects)
                    if len(batch_objects) == 0:
                        continue
                    with timer.track("SAM3 segmentation"):
                        inputs = sam3_processor(images=batch_images, text=batch_objects, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = sam3_model(**inputs)
                        results = sam3_processor.post_process_instance_segmentation(outputs, threshold=0.4, mask_threshold=0.5, target_sizes=[full_img.size[::-1]] * len(batch_objects))
""",
        """                print(f"SAM3 process for {scene_path}...")
                if hasattr(OpenAI, "unload"):
                    OpenAI.unload()
                _free_gpu()
                if sam3_processor is None:
                    sam3_processor = _load_local_sam3(device)
                for i in range(0, len(unique_objects), SAM_BATCH_SIZE):
                    batch_objects = unique_objects[i: i + SAM_BATCH_SIZE]
                    batch_objects = [obj for obj in batch_objects if len(obj.split()) < 8 and obj.lower() not in ("sun")]
                    if len(batch_objects) == 0:
                        continue
                    with timer.track("SAM3 segmentation"):
                        state = sam3_processor.set_image(full_img)
                        results = []
                        for name in batch_objects:
                            state = sam3_processor.set_text_prompt(name, state)
                            masks = state["masks"]
                            if isinstance(masks, torch.Tensor) and masks.ndim == 4:
                                masks = masks.squeeze(1)
                            results.append({"masks": masks, "scores": state["scores"]})
                            sam3_processor.reset_all_prompts(state)
""",
    )
    _swap(
        traj,
        """                        create_and_save_combined_pcd(global_pcd, segmentation_data, os.path.join(scene_path, "render_results", "combined_with_markers.ply"))
""",
        """                        create_and_save_combined_pcd(global_pcd, segmentation_data, os.path.join(scene_path, "render_results", "combined_with_markers.ply"))
                sam3_processor = None
                _free_gpu()
""",
    )
    _swap(
        traj,
        'obj_label = seg_labels.get(i, "unknown").strip().replace(" ", "_")',
        'obj_label = re.sub(r"[^A-Za-z0-9_]+", "_", seg_labels.get(i, "unknown")).strip("_") or "obj"',
    )
    if "import re\n" not in traj.read_text(encoding="utf-8"):
        _swap(traj, "import os\n", "import os\nimport re\n")
    _swap(
        traj,
        "def save_image(splitted_image, path):\n",
        '''def _load_local_sam3(device):
    """Meta sam3.pt, not the transformers Hub build."""
    root = os.environ.get("SAM3_ROOT", r"D:\\sam3\\sam3")
    ckpt = os.environ.get("SAM3_CHECKPOINT", r"D:\\sam3\\checkpoints\\sam3.pt")
    if root not in sys.path:
        sys.path.insert(0, root)
    from sam3.model.sam3_image_processor import Sam3Processor as LocalProcessor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False, device=str(device))
    return LocalProcessor(model, device=str(device), confidence_threshold=0.4)


def save_image(splitted_image, path):
''',
    )
    if "import sys\n" not in traj.read_text(encoding="utf-8"):
        _swap(traj, "import shutil\n", "import shutil\nimport sys\n")

    render = root / "traj_render.py"
    _swap(
        render,
        '''    except Exception as e:
        return render_path, False, str(e)
''',
        '''    except Exception as e:
        prompt = "a walk through the scene"
        folder = os.path.dirname(render_path)
        for _ in range(6):
            folder = os.path.dirname(folder)
            candidate = os.path.join(folder, "prompt.txt")
            if os.path.isfile(candidate):
                prompt = open(candidate, encoding="utf-8").read().strip() or prompt
                break
        with open(output_path, "w", encoding="utf-8") as write:
            json.dump({"prompt": prompt}, write)
        return render_path, False, str(e)
''',
    )
    _swap(
        render,
        '''    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
    )
''',
        '''    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )
''',
    )

    navi = root / "src" / "navi_utils.py"
    _swap(
        navi,
        "pairs = np.array(list(tree.query_pairs(r=search_radius)))",
        'pairs = tree.query_pairs(r=search_radius, output_type="ndarray")',
    )
    _swap(
        navi,
        "nav_node_centers=nav_graph.centers,",
        "nav_node_centers=None if nav_graph is None else nav_graph.centers,",
    )
    _swap(
        navi,
        '''    locations, index_ray, _ = tm_scene.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False
    )
''',
        '''    try:
        locations, index_ray, _ = tm_scene.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False
        )
    except Exception as exc:
        print(f"navmesh ray cast skipped: {exc}")
        locations = np.zeros((0, 3))
        index_ray = np.zeros((0,), dtype=np.int64)
''',
    )

    cpp = root / "third_party" / "navmesh" / "navmesh_builder.cpp"
    _swap(cpp, "#include <vector>\n", "#include <vector>\n#include <algorithm>\n")
    _swap(
        cpp,
        "float y = pmesh->bmin[1] + pv[vi*3 + 1] * cfg.cs;",
        "float y = pmesh->bmin[1] + pv[vi*3 + 1] * cfg.ch;",
    )
    setup = root / "third_party" / "navmesh" / "setup.py"
    _swap(
        setup,
        'extra_compile_args=["-O3"] if not sys.platform.startswith("win") else ["/O2"],',
        'extra_compile_args=["-O3"] if not sys.platform.startswith("win") else ["/O2", "/utf-8"],',
    )

    trainer = root / "world_gs_trainer.py"
    _swap(trainer, "                    dist.barrier()\n", "                    if self.world_size > 1:\n                        dist.barrier()\n")
    text = trainer.read_text(encoding="utf-8")
    old_loader = """        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )"""
    new_loader = """        _workers = int(os.environ.get("HY_NUM_WORKERS", "0"))
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=_workers,
            persistent_workers=_workers > 0,
            pin_memory=True,
        )"""
    count = text.count(old_loader)
    if count == 0:
        raise SystemExit("trainer dataloader anchor missing")
    trainer.write_text(text.replace(old_loader, new_loader), encoding="utf-8")
    print(f"patched {root} ({count} dataloaders)")
    patch_live(root)


def patch_live(root: Path) -> None:
    """Edits added after the first build. Safe to run again."""
    traj = root / "traj_generate.py"
    _ensure(
        traj,
        '            meta_info["scene_type"] = clean_text\n',
        '            meta_info["scene_type"] = "indoor" if "indoor" in clean_text.lower() else "outdoor"\n',
    )
    _ensure(
        traj,
        '    return LocalProcessor(model, device=str(device), confidence_threshold=0.4)\n',
        '    class Bf16Processor(LocalProcessor):\n'
        '        """sam3.pt needs bfloat16 autocast, and numpy cannot read bfloat16 scores."""\n'
        '\n'
        '        @staticmethod\n'
        '        def _floats(state):\n'
        '            for key in ("scores", "boxes"):\n'
        '                value = state.get(key)\n'
        '                if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:\n'
        '                    state[key] = value.float()\n'
        '            return state\n'
        '\n'
        '        def set_image(self, *args, **kwargs):\n'
        '            with torch.autocast("cuda", dtype=torch.bfloat16):\n'
        '                return self._floats(super().set_image(*args, **kwargs))\n'
        '\n'
        '        def set_text_prompt(self, *args, **kwargs):\n'
        '            with torch.autocast("cuda", dtype=torch.bfloat16):\n'
        '                return self._floats(super().set_text_prompt(*args, **kwargs))\n'
        '\n'
        '    return Bf16Processor(model, device=str(device), confidence_threshold=0.4)\n',
    )
    render = root / "traj_render.py"
    _ensure(
        render,
        '        rank0_log(f"Get {len(traj_list)} trajectories at all for {scene_path}")\n',
        '        _keep = set()\n'
        '        _keep_file = f"{scene_path}/_votion/selection.json"\n'
        '        if os.path.exists(_keep_file):\n'
        '            with open(_keep_file, encoding="utf-8") as _f:\n'
        '                _keep = set(json.load(_f).get("keep") or [])\n'
        '        if _keep:\n'
        '            traj_list = [p for p in traj_list if "/".join(p.split("/")[-2:]) in _keep]\n'
        '        rank0_log(f"Get {len(traj_list)} trajectories at all for {scene_path}")\n',
    )
    _ensure(
        render,
        '            if total_render_list:\n',
        '            if _keep:\n'
        '                total_render_list = [p for p in total_render_list if "/".join(p.split("/")[-3:-1]) in _keep]\n'
        '            if total_render_list:\n',
    )
    trainer = root / "world_gs_trainer.py"
    _ensure(
        trainer,
        '        pbar = tqdm.tqdm(range(init_step, max_steps))\n'
        '        for step in pbar:\n'
        '            loss_decay = (max_steps - step) / max_steps\n',
        '        import votion_live\n'
        '        self._votion = votion_live.TrainerHook(self)\n'
        '        init_step = self._votion.resume(schedulers)\n'
        '        pbar = tqdm.tqdm(range(init_step, max_steps))\n'
        '        for step in pbar:\n'
        '            self._votion.tick(step)\n'
        '            loss_decay = (max_steps - step) / max_steps\n',
    )
    _ensure(
        trainer,
        '            pbar.set_description(desc)\n',
        '            pbar.set_description(desc)\n'
        '            self._votion.report(step, max_steps, loss.item(), colors, pixels, len(self.splats["means"]))\n',
    )
    _ensure(
        trainer,
        '    else:\n        runner.train()\n',
        '    else:\n        runner.train()\n        runner._votion.finish()\n',
    )
    _ensure(
        trainer,
        'valloader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=1)',
        'valloader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=0)',
    )
    _ensure(
        trainer,
        '        self._votion = votion_live.TrainerHook(self)\n'
        '        pbar = tqdm.tqdm(range(init_step, max_steps))\n',
        '        self._votion = votion_live.TrainerHook(self)\n'
        '        init_step = self._votion.resume(schedulers)\n'
        '        pbar = tqdm.tqdm(range(init_step, max_steps))\n',
    )
    _ensure(
        trainer,
        '                loss = loss + lambda_mask * mask_loss_val\n'
        '\n'
        '            loss.backward()\n',
        '                loss = loss + lambda_mask * mask_loss_val\n'
        '\n'
        '            loss = loss * self._votion.loss_weight(data)\n'
        '            loss.backward()\n',
    )
    opencv = root / "gs" / "opencv.py"
    _ensure(
        opencv,
        '        if normalize:\n'
        '            T1, scale = similarity_from_cameras(camtoworlds)\n',
        '        _votion_frame = os.environ.get("VOTION_FRAME", "")\n'
        '        if normalize and _votion_frame:\n'
        '            with open(_votion_frame, encoding="utf-8") as _vf:\n'
        '                _fr = json.load(_vf)\n'
        '            transform = np.asarray(_fr["transform"], dtype=np.float64)\n'
        '            self.rescale = float(_fr["rescale"])\n'
        '            camtoworlds = transform_cameras(transform, camtoworlds)\n'
        '            points = transform_points(transform, points)\n'
        '            self.center_point = np.asarray(_fr["center_point"], dtype=np.float32).reshape(3)\n'
        '            self.up_direction = np.asarray(_fr["up_direction"], dtype=np.float32)\n'
        '            self.facing_direction = np.asarray(_fr["facing_direction"], dtype=np.float32)\n'
        '        elif normalize:\n'
        '            T1, scale = similarity_from_cameras(camtoworlds)\n',
    )
    _ensure(
        opencv,
        '        self.scene_scale = np.max(dists)\n',
        '        self.scene_scale = np.max(dists)\n'
        '        if os.environ.get("VOTION_FRAME", ""):\n'
        '            with open(os.environ["VOTION_FRAME"], encoding="utf-8") as _vf:\n'
        '                self.scene_scale = float(json.load(_vf)["scene_scale"])\n',
    )
    # HiRes frames: depth and normal maps stay at the Expand size, so scale them to the
    # full frame, then cut the same random crop as the image.
    _ensure(
        opencv,
        '        if self.patch_size is not None:\n'
        '            # Random crop.\n'
        '            h, w = image.shape[:2]\n'
        '            x = np.random.randint(0, max(w - self.patch_size, 1))\n'
        '            y = np.random.randint(0, max(h - self.patch_size, 1))\n'
        '            image = image[y: y + self.patch_size, x: x + self.patch_size]\n'
        '            K[0, 2] -= x\n'
        '            K[1, 2] -= y\n',
        '        _votion_full = image.shape[:2]\n'
        '        _votion_crop = None\n'
        '        if self.patch_size is not None:\n'
        '            # Random crop.\n'
        '            h, w = image.shape[:2]\n'
        '            x = np.random.randint(0, max(w - self.patch_size, 1))\n'
        '            y = np.random.randint(0, max(h - self.patch_size, 1))\n'
        '            image = image[y: y + self.patch_size, x: x + self.patch_size]\n'
        '            K[0, 2] -= x\n'
        '            K[1, 2] -= y\n'
        '            _votion_crop = (y, x)\n',
    )
    _ensure(
        opencv,
        '            "image_name": image_name,\n'
        '        }\n',
        '            "image_name": image_name,\n'
        '        }\n'
        '        if _votion_crop is not None:\n'
        '            data["votion_crop"] = torch.tensor(_votion_crop, dtype=torch.long)\n',
    )
    _ensure_any(
        opencv,
        [
            '                depth = load_16bit_png_depth(depth_path)\n'
            '                if depth.shape[:2] != image.shape[:2]:\n'
            '                    depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)\n',
            '                depth = load_16bit_png_depth(depth_path)\n',
        ],
        '                depth = load_16bit_png_depth(depth_path)\n'
        '                if depth.shape[:2] != _votion_full:\n'
        '                    depth = cv2.resize(depth, (_votion_full[1], _votion_full[0]), interpolation=cv2.INTER_NEAREST)\n'
        '                if _votion_crop is not None:\n'
        '                    depth = depth[_votion_crop[0]: _votion_crop[0] + image.shape[0], _votion_crop[1]: _votion_crop[1] + image.shape[1]]\n',
    )
    _ensure_any(
        opencv,
        [
            '                assert normal.ndim == 3 and normal.shape[2] == 3\n'
            '                if normal.shape[:2] != image.shape[:2]:\n'
            '                    normal = cv2.resize(normal, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)\n',
            '                assert normal.ndim == 3 and normal.shape[2] == 3\n',
        ],
        '                assert normal.ndim == 3 and normal.shape[2] == 3\n'
        '                if normal.shape[:2] != _votion_full:\n'
        '                    normal = cv2.resize(normal, (_votion_full[1], _votion_full[0]), interpolation=cv2.INTER_LINEAR)\n'
        '                if _votion_crop is not None:\n'
        '                    normal = normal[_votion_crop[0]: _votion_crop[0] + image.shape[0], _votion_crop[1]: _votion_crop[1] + image.shape[1]]\n',
    )
    _ensure(
        trainer,
        '                        depths_gt = self.sky_depth_maps[image_id_val].unsqueeze(0).to(device)\n',
        '                        _sky = self.sky_depth_maps[image_id_val]\n'
        '                        if "votion_crop" in data:\n'
        '                            _cy, _cx = (int(v) for v in data["votion_crop"].reshape(-1)[:2])\n'
        '                            _sky = _sky[_cy: _cy + depths_gt.shape[-2], _cx: _cx + depths_gt.shape[-1]]\n'
        '                        depths_gt = _sky.unsqueeze(0).to(device)\n',
    )
    sky = root / "gs" / "sky_depth.py"
    _ensure(
        sky,
        '            gt_depth = torch.from_numpy(load_16bit_png_depth(depth_path)).float()\n',
        '            _depth = load_16bit_png_depth(depth_path)\n'
        '            _w, _h = parser.imsize_dict[camera_id]\n'
        '            if _depth.shape[:2] != (_h, _w):\n'
        '                import cv2\n'
        '                _depth = cv2.resize(_depth, (_w, _h), interpolation=cv2.INTER_NEAREST)\n'
        '            gt_depth = torch.from_numpy(_depth).float()\n',
    )
    _ensure(
        sky,
        '            normal_img = np.array(PILImage.open(normal_path)) / 255.0\n',
        '            normal_img = np.array(PILImage.open(normal_path)) / 255.0\n'
        '            _w, _h = parser.imsize_dict[camera_id]\n'
        '            if normal_img.shape[:2] != (_h, _w):\n'
        '                import cv2\n'
        '                normal_img = cv2.resize(normal_img, (_w, _h), interpolation=cv2.INTER_LINEAR)\n',
    )
    print(f"live patches ok {root}")


if __name__ == "__main__":
    if "--live-only" in sys.argv:
        patch_live(Path(sys.argv[1]))
    else:
        patch(Path(sys.argv[1]))
