"""Adaptador multiplataforma para invocar FEniCSx desde WSL2 (Windows), Linux nativo o Docker.

En Windows usa de forma predeterminada la instalación operativa de WSL2 detectada o
configurada vía variables de entorno, manteniendo sincronizados los códigos puente
sin requerir PyTorch dentro del entorno de elementos finitos.

Variables de entorno soportadas:
- ``PINN_V4_FEM_MODE``: ``auto`` (defecto), ``wsl``, ``native``, ``docker``.
- ``PINN_V4_WSL_DISTRO``: distribución WSL (por defecto ``Ubuntu``).
- ``PINN_V4_FEM_PYTHON``: ruta del intérprete Python dentro de WSL/nativo
  (por defecto ``/home/lkoc/miniforge3/envs/fenicsx/bin/python`` en WSL).
- ``PINN_V4_FEM_DOCKER_IMAGE``: imagen Docker alternativa (defecto ``dolfinx/dolfinx:stable``).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

from pinn_thermal.core.artifact import SolutionArtifact
from pinn_thermal.core.manifest import write_json_atomic
from pinn_thermal.core.problem import ThermalProblem2D, load_problem

DEFAULT_WSL_DISTRO: str = "Ubuntu"
DEFAULT_WSL_PYTHON: str = "/home/lkoc/miniforge3/envs/fenicsx/bin/python"
DEFAULT_DOCKER_IMAGE: str = "dolfinx/dolfinx:stable"


def windows_to_wsl_path(path: str | Path) -> str:
    """Convierte una ruta de Windows (``C:\\...``) a ruta montada WSL (``/mnt/c/...``)."""
    resolved = Path(path).resolve()
    text = str(resolved)
    if len(text) >= 2 and text[1] == ":":
        drive = text[0].lower()
        rest = text[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return text.replace("\\", "/")


class FEniCSxAdapter:
    """Lanzador unificado para obtener soluciones de referencia P2 desde FEniCSx."""

    def __init__(
        self,
        mode: str | None = None,
        wsl_distro: str | None = None,
        fem_python: str | None = None,
        docker_image: str | None = None,
    ) -> None:
        self.mode = mode or os.environ.get("PINN_V4_FEM_MODE", "auto")
        self.wsl_distro = wsl_distro or os.environ.get("PINN_V4_WSL_DISTRO", DEFAULT_WSL_DISTRO)
        self.fem_python = fem_python or os.environ.get("PINN_V4_FEM_PYTHON", DEFAULT_WSL_PYTHON)
        self.docker_image = docker_image or os.environ.get("PINN_V4_FEM_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)
        self._src_root = Path(__file__).resolve().parents[2]

    def detect_backend(self) -> str:
        """Determina el mecanismo de ejecución disponible (``native``, ``wsl`` o ``docker``)."""
        if self.mode != "auto":
            return self.mode
        if importlib.util.find_spec("dolfinx") is not None and importlib.util.find_spec("gmsh") is not None:
            return "native"
        if platform.system() == "Windows" and shutil.which("wsl.exe") is not None:
            return "wsl"
        if shutil.which("docker") is not None:
            return "docker"
        return "native"

    def is_available(self) -> dict[str, Any]:
        """Comprueba si el backend FEniCSx configurado responde correctamente."""
        backend = self.detect_backend()
        try:
            if backend == "native":
                has_dx = importlib.util.find_spec("dolfinx") is not None
                has_gmsh = importlib.util.find_spec("gmsh") is not None
                return {"available": has_dx and has_gmsh, "backend": "native", "python": sys.executable}
            if backend == "wsl":
                proc = subprocess.run(
                    ["wsl.exe", "-d", self.wsl_distro, "--", "test", "-x", self.fem_python],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                return {
                    "available": proc.returncode == 0,
                    "backend": "wsl",
                    "distro": self.wsl_distro,
                    "python": self.fem_python,
                }
            if backend == "docker":
                proc = subprocess.run(
                    ["docker", "image", "inspect", self.docker_image],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                return {"available": proc.returncode == 0, "backend": "docker", "image": self.docker_image}
        except Exception as exc:
            return {"available": False, "backend": backend, "error": str(exc)}
        return {"available": False, "backend": backend}

    def solve(
        self,
        problem_source: str | Path | dict[str, Any] | ThermalProblem2D,
        output_dir: str | Path,
        level: int = 2,
        reuse_if_valid: bool = True,
    ) -> SolutionArtifact:
        """Ejecuta FEniCSx sobre ``problem_source`` y devuelve su ``SolutionArtifact``."""
        problem = load_problem(problem_source)
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        prob_json = out / "problem.json"
        write_json_atomic(prob_json, problem.doc)

        fem_json = out / f"fem_l{int(level)}.json"
        if reuse_if_valid and fem_json.is_file() and fem_json.with_suffix(".npz").is_file():
            art = SolutionArtifact.load(fem_json)
            if art.result.get("physics_sha256") == problem.physics_sha256:
                return art

        backend = self.detect_backend()
        if backend == "native":
            from pinn_thermal.fem.worker import solve_fem_problem

            solve_fem_problem(prob_json, out, level=level)
            return SolutionArtifact.load(fem_json)

        if backend == "wsl":
            wsl_src = windows_to_wsl_path(self._src_root)
            wsl_prob = windows_to_wsl_path(prob_json)
            wsl_out = windows_to_wsl_path(out)
            cmd = [
                "wsl.exe",
                "-d",
                self.wsl_distro,
                "--",
                "env",
                f"PYTHONPATH={wsl_src}",
                "OMP_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1",
                self.fem_python,
                "-m",
                "pinn_thermal.fem.worker",
                "--problem",
                wsl_prob,
                "--out",
                wsl_out,
                "--level",
                str(int(level)),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Falló la ejecución FEniCSx en WSL ({self.wsl_distro}):\n"
                    f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
                )
            return SolutionArtifact.load(fem_json)

        if backend == "docker":
            mount_src = str(self._src_root.resolve())
            mount_out = str(out)
            cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{mount_src}:/pinn_src:ro",
                "-v",
                f"{mount_out}:/pinn_out",
                "-e",
                "PYTHONPATH=/pinn_src",
                self.docker_image,
                "python3",
                "-m",
                "pinn_thermal.fem.worker",
                "--problem",
                "/pinn_out/problem.json",
                "--out",
                "/pinn_out",
                "--level",
                str(int(level)),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"Falló la ejecución FEniCSx en Docker:\n{proc.stderr}")
            return SolutionArtifact.load(fem_json)

        raise ValueError(f"Modo FEM desconocido: {backend!r}; elija entre 'auto', 'wsl', 'native', 'docker'.")
