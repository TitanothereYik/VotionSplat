"""List app/ and engine/ modules the live pipeline never imports.

Walks import statements, including ones nested in functions, starting from the
modules the window and the workers actually launch. A string literal matching
``engine.workers.<name>`` in a reachable file is treated as another entry.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = (
    "app.main",
    "engine.workers.pano",
    "engine.workers.caption",
    "engine.workers.detect_fov",
    "engine.workers.hy_expand",
    "engine.workers.hy_hires",
    "engine.workers.hy_export",
    "engine.workers.hy_train_loop",
    "engine.workers.hy_difix",
    "tools.hyworld.run_stage",
    "tools.download_models",
)
WORKER_RE = re.compile(r"engine\.workers\.\w+")


def module_file(name: str) -> Path | None:
    rel = Path(*name.split("."))
    py = ROOT / rel.with_suffix(".py")
    init = ROOT / rel / "__init__.py"
    if py.is_file():
        return py
    if init.is_file():
        return init
    return None


def package_of(path: Path) -> list[str]:
    rel = path.relative_to(ROOT)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        return parts[:-1]
    return parts[:-1]


def names_from(path: Path, tree: ast.AST) -> set[str]:
    pkg = package_of(path)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = pkg[: len(pkg) - (node.level - 1)]
                if node.module:
                    base_parts += node.module.split(".")
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base:
                found.add(base)
            for alias in node.names:
                if alias.name == "*":
                    continue
                found.add(f"{base}.{alias.name}" if base else alias.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.update(WORKER_RE.findall(node.value))
    return found


def resolve(name: str) -> Path | None:
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        hit = module_file(".".join(parts[:i]))
        if hit is not None:
            return hit
    return None


def parents(path: Path) -> list[Path]:
    out = []
    rel = path.relative_to(ROOT)
    folder = rel.parent
    while folder != Path("."):
        init = ROOT / folder / "__init__.py"
        if init.is_file():
            out.append(init)
        if not folder.parent or folder.parent == folder:
            break
        folder = folder.parent
    return out


def reachable() -> set[Path]:
    seen: set[Path] = set()
    queue: list[Path] = []
    for name in ENTRIES:
        hit = module_file(name)
        if hit is None:
            print(f"MISSING entry {name}", file=sys.stderr)
            continue
        queue.append(hit)
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for parent in parents(path):
            if parent not in seen:
                queue.append(parent)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            print(f"SYNTAX {path}: {exc}", file=sys.stderr)
            continue
        for name in names_from(path, tree):
            hit = resolve(name)
            if hit is not None and hit not in seen:
                queue.append(hit)
    return seen


def main() -> int:
    reached = reachable()
    roots = (ROOT / "app", ROOT / "engine")
    missing = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if path not in reached:
                missing.append(path.relative_to(ROOT).as_posix())
    print(f"reachable {len(reached)}")
    print(f"unreachable {len(missing)}")
    for item in missing:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
