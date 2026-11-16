"""Save a depth-backend npz. Numpy and cv2 only; this file runs inside a backend venv."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save(path: str | Path, distance: np.ndarray, valid: np.ndarray, normals=None, sky=None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "distance": np.asarray(distance, dtype=np.float32),
        "valid": np.asarray(valid, dtype=bool),
    }
    if normals is not None:
        payload["normals"] = np.asarray(normals, dtype=np.float32)
    if sky is not None:
        payload["sky"] = np.asarray(sky, dtype=bool)
    np.savez_compressed(out, **payload)
