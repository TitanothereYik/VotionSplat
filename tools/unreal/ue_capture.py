"""Import one tagged PLY and save a HighResShot for every camera in a job JSON.

Run inside the editor:
    UnrealEditor.exe <uproject> -ExecutePythonScript=<this file> -log
The job path is the VOTION_JOB environment variable, or unreal/validation/jobs/current.json.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import unreal

JOB = Path(os.environ.get("VOTION_JOB") or str(Path(__file__).resolve().parents[2] / "unreal" / "validation" / "jobs" / "current.json"))
STATE = {"phase": "start", "tick": 0, "camera": 0, "wait": 0, "task": None, "cameras": [], "job": {}}


def _log(text: str) -> None:
    unreal.log(f"VotionCapture: {text}")


def _tick(_delta: float) -> None:
    try:
        _tick_body()
    except Exception as exc:
        _log(f"failed: {exc}")
        STATE["phase"] = "failed"
        unreal.EditorPythonScripting.set_keep_python_script_alive(False)
        unreal.SystemLibrary.quit_editor()


def _tick_body() -> None:
    STATE["tick"] += 1
    phase = STATE["phase"]
    if phase == "start":
        _start()
    elif phase == "settle" and time.perf_counter() >= STATE["ready_at"]:
        if STATE["job"].get("perf_seconds"):
            _begin_perf()
        else:
            _begin_camera()
    elif phase == "perf":
        now = time.perf_counter()
        if now >= STATE["perf_begin"]:
            if STATE["perf_frames"] == 0:
                STATE["perf_start"] = now
            STATE["perf_frames"] += 1
            elapsed = now - STATE["perf_start"]
            if elapsed >= float(STATE["job"]["perf_seconds"]):
                name = STATE["cameras"][STATE["camera"]]["name"]
                _log(
                    f"VotionPerf: name={name} fps={STATE['perf_frames'] / elapsed:.1f} "
                    f"ms={1000.0 * elapsed / STATE['perf_frames']:.2f}"
                )
                unreal.SystemLibrary.execute_console_command(None, "ProfileGPU")
                STATE["phase"] = "perf_cool"
                STATE["wait"] = STATE["tick"] + 60
    elif phase == "perf_cool" and STATE["tick"] > STATE["wait"]:
        STATE["camera"] += 1
        if STATE["camera"] >= len(STATE["cameras"]):
            _advance()
        else:
            _begin_perf()
    elif phase == "shot":
        task = STATE["task"]
        if task is not None and task.is_task_done():
            STATE["phase"] = "cooldown"
            STATE["wait"] = STATE["tick"] + 10
    elif phase == "cooldown" and STATE["tick"] > STATE["wait"]:
        _collect()
        STATE["camera"] += 1
        if STATE["camera"] >= len(STATE["cameras"]):
            _advance()
        else:
            if STATE["job"].get("perf_seconds"):
                _begin_perf()
            else:
                _begin_camera()


def _strip_level(actors) -> None:
    # The default builder brush is a cube at the origin. It writes depth and hides the middle of a scene.
    keep = {"VotionSplatActor", "CameraActor", "WorldSettings", "DefaultPhysicsVolume"}
    removed = 0
    for actor in list(actors.get_all_level_actors()):
        if actor.get_class().get_name() in keep:
            continue
        actors.destroy_actor(actor)
        removed += 1
    _log(f"removed {removed} background actors")


def _load_job() -> dict:
    if "queue" not in STATE:
        raw = json.loads(JOB.read_text(encoding="utf-8"))
        STATE["queue"] = raw["queue"] if "queue" in raw else [raw]
        STATE["queue_base"] = raw
        STATE["queue_index"] = 0
    item = dict(STATE["queue"][STATE["queue_index"]])
    base = STATE["queue_base"]
    if "out_dir" not in item and "out_root" in base:
        item["out_dir"] = str(Path(base["out_root"]) / item["name"])
    for key in ("cameras", "asset_dir", "settle_seconds", "console", "perf_seconds"):
        item.setdefault(key, base.get(key))
    item["quit"] = False
    _log(f"job {item.get('name', item['ply'])} ({STATE['queue_index'] + 1}/{len(STATE['queue'])})")
    return item


def _start() -> None:
    STATE["phase"] = "importing"
    STATE["camera"] = 0
    job = _load_job()
    STATE["job"] = job
    STATE["cameras"] = json.loads(Path(job["cameras"]).read_text(encoding="utf-8"))
    task = unreal.AssetImportTask()
    task.filename = job["ply"]
    task.destination_path = job.get("asset_dir", "/Game/Validation")
    task.automated = True
    task.replace_existing = True
    task.save = True
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in actors.get_all_level_actors():
        if actor.get_class().get_name() == "VotionSplatActor":
            actors.destroy_actor(actor)
    asset_path = task.destination_path + "/" + Path(job["ply"]).stem
    asset = unreal.EditorAssetLibrary.load_asset(asset_path)
    if asset is None:
        _log(f"import failed for {job['ply']}")
        STATE["phase"] = "failed"
        unreal.EditorPythonScripting.set_keep_python_script_alive(False)
        unreal.SystemLibrary.quit_editor()
        return
    _log(
        "asset "
        f"splats={asset.get_editor_property('num_splats')} "
        f"degree={asset.get_editor_property('sh_degree')} "
        f"raster={asset.get_editor_property('raster_profile')} "
        f"near={asset.get_editor_property('near_plane_ply')}"
    )
    spawned = actors.spawn_actor_from_object(asset, unreal.Vector(0, 0, 0))
    spawned.set_actor_rotation(unreal.Rotator(roll=0, pitch=0, yaw=0), False)
    _strip_level(actors)
    for command in job.get("console", []) + [
        "r.Tonemapper.GrainQuantization 0",
        "r.DefaultFeature.Bloom 0",
        "r.BloomQuality 0",
        "r.EyeAdaptationQuality 0",
        "r.DefaultFeature.AutoExposure 0",
        "showflag.atmosphere 0",
        "showflag.fog 0",
        "showflag.clouds 0",
        "showflag.landscape 0",
        "showflag.bsp 0",
        "showflag.builderbrush 0",
    ]:
        unreal.SystemLibrary.execute_console_command(None, command)
    if job.get("perf_seconds"):
        settings_class = unreal.load_class(None, "/Script/UnrealEd.EditorPerformanceSettings")
        if settings_class is not None:
            settings = unreal.get_default_object(settings_class)
            for property_name in ("b_throttle_cpu_when_not_foreground", "throttle_cpu_when_not_foreground"):
                try:
                    settings.set_editor_property(property_name, False)
                    _log(f"disabled {property_name}")
                    break
                except Exception as exc:
                    _log(f"{property_name} unavailable: {exc}")
        unreal.SystemLibrary.execute_console_command(None, "Slate.bAllowThrottling 0")
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_set_viewport_realtime(True)
    STATE["phase"] = "settle"
    STATE["ready_at"] = time.perf_counter() + float(job.get("settle_seconds", 20))
    STATE["spawned"] = spawned
    _log(f"imported {job['ply']}")


def _begin_perf() -> None:
    camera = STATE["cameras"][STATE["camera"]]
    loc = camera["loc_cm"]
    unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(
        unreal.Vector(loc[0], loc[1], loc[2]),
        unreal.Rotator(roll=camera["roll"], pitch=camera["pitch"], yaw=camera["yaw"]))
    STATE["perf_begin"] = time.perf_counter() + 3.0
    STATE["perf_frames"] = 0
    STATE["phase"] = "perf"
    _log(f"perf {camera['name']}")


def _begin_camera() -> None:
    STATE["phase"] = "camera"
    camera = STATE["cameras"][STATE["camera"]]
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    cam = STATE.get("cam")
    if cam is None:
        cam = actors.spawn_actor_from_class(unreal.CameraActor, unreal.Vector(0, 0, 0))
        STATE["cam"] = cam
    loc = camera["loc_cm"]
    cam.set_actor_location(unreal.Vector(loc[0], loc[1], loc[2]), False, False)
    cam.set_actor_rotation(unreal.Rotator(roll=camera["roll"], pitch=camera["pitch"], yaw=camera["yaw"]), False)
    component = cam.get_editor_property("camera_component")
    component.set_editor_property("field_of_view", float(camera["fov_h"]))
    component.set_editor_property("constrain_aspect_ratio", False)
    name = camera["name"]
    STATE["shot_after"] = time.time()
    task = unreal.AutomationLibrary.take_high_res_screenshot(
        int(camera["W"]), int(camera["H"]), f"{name}.png", cam, False, False,
        unreal.ComparisonTolerance.LOW, "", 0.5, True)
    STATE["task"] = task
    STATE["shot_name"] = name
    STATE["phase"] = "shot"
    _log(f"shot {name}")


def _collect() -> None:
    out = Path(STATE["job"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    saved = Path(unreal.Paths.project_saved_dir()) / "Screenshots"
    matches = [p for p in saved.rglob(STATE["shot_name"] + ".png") if p.stat().st_mtime >= STATE.get("shot_after", 0) - 1.0]
    matches.sort(key=lambda p: p.stat().st_mtime)
    if not matches:
        _log(f"missing screenshot {STATE['shot_name']}")
        return
    target = out / f"{STATE['shot_name']}.png"
    shutil.copyfile(matches[-1], target)
    _log(f"saved {target}")


def _advance() -> None:
    if STATE["queue_index"] + 1 < len(STATE["queue"]):
        STATE["queue_index"] += 1
        STATE["phase"] = "start"
        return
    _finish()


def _finish() -> None:
    _log("done")
    unreal.EditorPythonScripting.set_keep_python_script_alive(False)
    if STATE["queue_base"].get("quit", False):
        unreal.SystemLibrary.quit_editor()


if __name__ == "__main__":
    unreal.EditorPythonScripting.set_keep_python_script_alive(True)
    unreal.register_slate_post_tick_callback(_tick)
