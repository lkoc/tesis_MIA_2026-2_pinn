"""Gestor de espacio de trabajo compartido entre la interfaz Streamlit, Jupyter y CLI.

Este módulo **no importa ``streamlit``**: permite que los cuadernos Jupyter guarden y
carguen corridas o casos JSON directamente dentro de ``frontend/workspace/`` de forma
que sean visibles al instante en la interfaz gráfica (y viceversa).
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

from pinn_thermal.core.artifact import SolutionArtifact
from pinn_thermal.core.manifest import write_json_atomic
from pinn_thermal.core.problem import CaseCatalog, ThermalProblem2D
from pinn_thermal.core.schema import validate_problem
from pinn_thermal.visualization.geometry_editor import CaseEditorHelper


def default_workspace_root() -> Path:
    """Devuelve ``PINN_V4/frontend/workspace`` independientemente del directorio de trabajo."""
    pkg_root = Path(__file__).resolve().parents[1]
    return pkg_root / "frontend" / "workspace"


class WorkspaceStore:
    """Repositorio compartible de casos de usuario y artefactos de solución."""

    def __init__(self, workspace_root: str | Path | None = None, cases_root: str | Path | None = None) -> None:
        self.root = Path(workspace_root) if workspace_root is not None else default_workspace_root()
        self.user_cases_dir = self.root / "user_cases"
        self.runs_dir = self.root / "runs"
        self.user_cases_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = CaseCatalog(cases_root)

    # ------------------------------------------------------------------
    # Gestión unificada de problemas JSON (catálogo + usuario)
    # ------------------------------------------------------------------

    def list_all_problems(self) -> list[dict[str, Any]]:
        """Lista todos los casos del catálogo curado y de ``user_cases``."""
        entries: list[dict[str, Any]] = []
        for path in sorted(self.user_cases_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                entries.append(
                    {
                        "id": data.get("id", path.stem),
                        "category": "user_cases",
                        "title": data.get("title", path.stem),
                        "family": data.get("family", "generic"),
                        "path": path,
                    }
                )
            except Exception:
                continue
        entries.extend(self.catalog.list_entries())
        return entries

    def load_problem(self, identifier: str | Path) -> ThermalProblem2D:
        """Carga un problema desde archivo directo, ``user_cases`` o catálogo oficial."""
        p = Path(identifier)
        if p.is_file():
            return ThermalProblem2D.from_json(p)
        candidate = self.user_cases_dir / f"{identifier}.json"
        if candidate.is_file():
            return ThermalProblem2D.from_json(candidate)
        for entry in self.list_all_problems():
            if entry["id"] == str(identifier) or f"{entry['category']}/{entry['id']}" == str(identifier):
                return ThermalProblem2D.from_json(entry["path"])
        raise FileNotFoundError(f"No se encontró el problema {identifier!r} en workspace ni catálogo.")

    def save_user_case(self, case_or_doc: CaseEditorHelper | ThermalProblem2D | dict[str, Any], filename: str | None = None) -> Path:
        """Valida y guarda atómicamente un caso JSON dentro de ``workspace/user_cases``."""
        if isinstance(case_or_doc, CaseEditorHelper):
            doc = case_or_doc.doc
        elif isinstance(case_or_doc, ThermalProblem2D):
            doc = case_or_doc.doc
        else:
            doc = dict(case_or_doc)
        validate_problem(doc)
        slug = filename or f"{doc['id']}.json"
        if not slug.lower().endswith(".json"):
            slug = f"{slug}.json"
        target = self.user_cases_dir / Path(slug).name
        write_json_atomic(target, doc)
        return target

    def delete_user_case(self, name_or_id: str) -> None:
        stem = Path(name_or_id).stem
        target = self.user_cases_dir / f"{stem}.json"
        if target.is_file():
            target.unlink()

    # ------------------------------------------------------------------
    # Gestión de artefactos compatibles entre CLI, Streamlit y Jupyter
    # ------------------------------------------------------------------

    def make_run_dir(self, case_id: str, architecture: str, tag: str = "") -> Path:
        """Reserva un identificador único y legible para una nueva corrida."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_tag = f"_{tag}" if tag else ""
        base = f"{case_id}__{architecture}__{stamp}{safe_tag}"
        out = self.runs_dir / base
        suffix = 1
        while out.exists():
            out = self.runs_dir / f"{base}_{suffix}"
            suffix += 1
        return out

    def save_artifact(self, artifact: SolutionArtifact, run_name: str | None = None) -> Path:
        """Persiste un ``SolutionArtifact`` en ``workspace/runs`` para uso transversal."""
        name = run_name or f"{artifact.problem.id}__{artifact.architecture}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        target = self.runs_dir / name
        artifact.save(target)
        return target

    def list_runs(self) -> list[dict[str, Any]]:
        """Enumera todas las corridas válidas guardadas en ``workspace/runs``."""
        runs: list[dict[str, Any]] = []
        if not self.runs_dir.is_dir():
            return runs
        for run_dir in sorted((p for p in self.runs_dir.iterdir() if p.is_dir()), reverse=True):
            res_file = run_dir / "result.json"
            fem_files = sorted(run_dir.glob("fem_l*.json"))
            record: dict[str, Any] = {}
            kind = "unknown"
            if res_file.is_file():
                record = json.loads(res_file.read_text(encoding="utf-8-sig"))
                kind = record.get("kind", "pinn" if (run_dir / "model.pt").is_file() else "fem")
            elif fem_files:
                record = json.loads(fem_files[-1].read_text(encoding="utf-8-sig"))
                kind = "fem"
            else:
                continue

            ev = record.get("evaluation") or {}
            bal = (ev.get("balance") or {}) if isinstance(ev, dict) else {}
            runs.append(
                {
                    "run_id": run_dir.name,
                    "path": run_dir,
                    "kind": kind,
                    "case": record.get("case", run_dir.name),
                    "architecture": record.get("architecture", record.get("method", kind)),
                    "status": record.get("status", "completed"),
                    "Tmax_C": ev.get("Tmax_pred_C", record.get("Tmax_C")),
                    "nrmse_pct": ev.get("nrmse_pct"),
                    "error_Tmax_abs_K": ev.get("error_Tmax_abs_K"),
                    "balance_pct": bal.get("global_derived_pct", record.get("balance_pct")),
                    "elapsed_s": record.get("elapsed_s"),
                    "parameters": record.get("parameters"),
                    "device": record.get("device", "-"),
                }
            )
        return runs

    def load_run(self, run_id_or_path: str | Path, device: str = "cpu", selection: str = "inference") -> SolutionArtifact:
        """Carga cualquier corrida PINN, FEM o RBF desde ``workspace/runs`` o ruta."""
        p = Path(run_id_or_path)
        if not p.exists():
            p = self.runs_dir / str(run_id_or_path)
        return SolutionArtifact.load(p, device=device, selection=selection)

    def delete_run(self, run_id: str) -> None:
        target = self.runs_dir / run_id
        if target.is_dir():
            shutil.rmtree(target)
