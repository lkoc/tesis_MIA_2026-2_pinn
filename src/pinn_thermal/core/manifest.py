"""Manifiestos de ejecución, huellas SHA-256 y escritura atómica segura en Windows y Linux.

Este módulo es **100 % libre de PyTorch**.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

PACKAGE_ROOT: Path = Path(__file__).resolve().parents[1]


def sha256_bytes(data: bytes) -> str:
    """Devuelve el resumen hexadecimal SHA-256 de un bloque binario."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Calcula el SHA-256 de un archivo en disco."""
    return sha256_bytes(Path(path).read_bytes())


def package_sources() -> list[Path]:
    """Lista ordenada de todos los archivos fuente Python del paquete ``pinn_thermal``."""
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def archive_sources(output_dir: str | Path) -> dict[str, str]:
    """Archiva las fuentes de ``pinn_thermal`` bajo su SHA-256 dentro de ``output_dir/source``."""
    archive = Path(output_dir) / "source"
    archive.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for path in package_sources():
        data = path.read_bytes()
        digest = sha256_bytes(data)
        hashes[path.relative_to(PACKAGE_ROOT.parent).as_posix()] = digest
        target = archive / f"{digest}_{path.name}"
        if not target.exists():
            target.write_bytes(data)
    return hashes


def atomic_replace(source: str | Path, target: str | Path, retries: int = 8, initial_delay_s: float = 0.025) -> None:
    """Reemplaza un archivo de forma atómica con reintentos ante bloqueos breves en Windows."""
    src_s, dst_s = os.fspath(source), os.fspath(target)
    for attempt in range(retries):
        try:
            os.replace(src_s, dst_s)
            return
        except PermissionError:
            if os.name != "nt" or attempt + 1 == retries:
                raise
            time.sleep(initial_delay_s * (attempt + 1))


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Escribe un archivo JSON nuevo atómicamente mediante archivo temporal."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=dst.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        atomic_replace(tmp_name, dst)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return dst


def replace_json_atomic(path: str | Path, payload: Any) -> Path:
    """Actualiza atómicamente un JSON existente."""
    return write_json_atomic(path, payload)


def host_versions() -> dict[str, Any]:
    """Recopila versiones relevantes de Python y librerías científicas instaladas."""
    versions: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "scipy", "torch", "plotly", "matplotlib", "deepxde"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return versions


def git_state(root: str | Path | None = None) -> dict[str, Any]:
    """Consulta el estado Git si el directorio forma parte de un repositorio."""
    base = Path(root or PACKAGE_ROOT.parents[1])

    def _run(*args: str) -> str | None:
        try:
            res = subprocess.run(["git", "-C", str(base), *args], capture_output=True, text=True)
            return res.stdout.strip() if res.returncode == 0 else None
        except OSError:
            return None

    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
    }


def build_manifest(
    kind: str,
    command: list[str] | tuple[str, ...],
    inputs: tuple[str | Path, ...] = (),
    seeds: dict[str, Any] | None = None,
    fem_environment: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construye el diccionario de manifiesto reproducible de una ejecución."""
    return {
        "schema": "pinn-v4-manifest-1",
        "kind": kind,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "inputs": {Path(p).name: sha256_file(p) for p in inputs if Path(p).is_file()},
        "seeds": seeds or {},
        "host": host_versions(),
        "fem_environment": fem_environment,
        "git": git_state(),
        "extra": extra or {},
    }


def write_manifest(output_dir: str | Path, manifest: dict[str, Any], with_sources: bool = False) -> Path:
    """Guarda ``manifest.json`` dentro de ``output_dir`` archivando fuentes si se solicita."""
    out = Path(output_dir)
    payload = dict(manifest)
    if with_sources:
        payload["source_sha256"] = archive_sources(out)
    return write_json_atomic(out / "manifest.json", payload)
