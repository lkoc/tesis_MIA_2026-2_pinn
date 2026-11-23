"""Formato único portable de artefactos (``pinn-v4-artifact-1``).

Un ``SolutionArtifact`` unifica los resultados obtenidos desde:
- Entrenamientos PINN ejecutados en CLI, scripts Python, cuadernos Jupyter o Streamlit.
- Referencias FEniCSx generadas en WSL2 / Linux nativo / Docker.
- Inicializaciones discretas RBF-FD.

Cualquier artefacto guardado en un directorio compartido (p. ej. ``frontend/workspace/runs``)
puede cargarse indistintamente desde Jupyter o visualizarse/compararse en Streamlit.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
import torch

from pinn_thermal.core.base_model import BasePINNModel, ModelRegistry
from pinn_thermal.core.device import DTYPES
from pinn_thermal.core.fields import evaluate_field_numpy
from pinn_thermal.core.manifest import atomic_replace, write_json_atomic
from pinn_thermal.core.problem import ThermalProblem2D

ARTIFACT_FORMAT: str = "pinn-v4-artifact-1"
LEGACY_CHECKPOINT_FORMAT: str = "pinn-v2-checkpoint-2"


def cpu_state(value: Any) -> Any:
    """Clona recursivamente tensores y estructuras anidadas hacia memoria CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_state(v) for v in value)
    return copy.deepcopy(value)


def specification_fingerprint(specification: dict[str, Any]) -> str:
    """Resumen SHA-256 determinista de la especificación del artefacto."""
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_specification(
    problem: ThermalProblem2D,
    config: dict[str, Any],
    seeds: dict[str, Any],
    model: BasePINNModel | None = None,
) -> dict[str, Any]:
    """Construye la especificación portable inmutable de un modelo entrenado."""
    resolved = dict(config)
    resolved.setdefault("dtype", "float32")
    spec: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "schema_version": 4,
        "problem": copy.deepcopy(problem.doc),
        "config": resolved,
        "seeds": dict(seeds),
        "physics_sha256": problem.physics_sha256,
        "flux_convention": "q = -k * grad(T); normal exterior",
        "runtime": {"torch": str(torch.__version__)},
    }
    if model is not None:
        spec["compatibility_profile"] = model.compatibility_profile()
    return spec


def _estimate_discrete_field_and_flux(
    problem: ThermalProblem2D,
    dof_xy: np.ndarray,
    dof_T: np.ndarray,
    query_xy: np.ndarray,
    with_flux: bool = False,
) -> dict[str, np.ndarray]:
    """Interpola por región un campo nodal FEM/RBF respetando discontinuidades materiales."""
    pts = np.asarray(query_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
        raise ValueError("query_xy debe ser un arreglo finito de forma (N, 2).")
    labels = np.asarray(problem.owner_index(pts), dtype=np.int64)
    if np.any(labels < 0):
        raise ValueError("Existen puntos de consulta fuera del dominio del problema.")

    dof_pts = np.asarray(dof_xy, dtype=float)[:, :2]
    dof_vals = np.asarray(dof_T, dtype=float).ravel()
    dof_owners = problem.owner(dof_pts[:, 0], dof_pts[:, 1])

    temps = np.empty(len(pts), dtype=float)
    flux = np.empty((len(pts), 2), dtype=float) if with_flux else None

    for region in problem.regions:
        q_mask = labels == region.id
        if not np.any(q_mask):
            continue
        r_pts = pts[q_mask]
        if len(problem.regions) == 1:
            # Con un único material, conserva nodos P2 de bordes curvos ligeramente desplazados por cuerdas
            src_mask = np.ones(len(dof_pts), dtype=bool)
        else:
            src_mask = dof_owners == region.id
            if np.any(dof_owners < 0):
                x0, x1, y0, y1 = problem.bounds
                tol = 1e-3 * max(x1 - x0, y1 - y0, 1.0)
                src_mask = src_mask | ((dof_owners < 0) & region.shape.contains(dof_pts[:, 0], dof_pts[:, 1], tolerance=tol))
        if not np.any(src_mask):
            # Si una región muy delgada no captó nodos estrictamente internos, incluye borde cercano
            x0, x1, y0, y1 = problem.bounds
            tol = 1e-4 * max(x1 - x0, y1 - y0, 1.0)
            src_mask = region.shape.contains(dof_pts[:, 0], dof_pts[:, 1], tolerance=tol)
        if not np.any(src_mask):
            raise ValueError(f"El campo discreto no contiene nodos para la región {region.name!r}.")
        s_xy = dof_pts[src_mask]
        s_T = dof_vals[src_mask]
        if len(s_xy) >= 3:
            interp = LinearNDInterpolator(s_xy, s_T)
            vals = np.asarray(interp(r_pts), dtype=float).ravel()
            missing = ~np.isfinite(vals)
            if np.any(missing):
                vals[missing] = s_T[cKDTree(s_xy).query(r_pts[missing])[1]]
        else:
            vals = s_T[cKDTree(s_xy).query(r_pts)[1]]
        temps[q_mask] = vals

        if with_flux and flux is not None:
            k_vals = region.conductivity_spec.evaluate_numpy(r_pts[:, 0], r_pts[:, 1])
            k_nbr = min(12, len(s_xy))
            if k_nbr >= 6:
                nearest = cKDTree(s_xy).query(r_pts, k=k_nbr)[1]
                grads = np.zeros((len(r_pts), 2), dtype=float)
                for row_idx, (pt, nbr_idx) in enumerate(zip(r_pts, nearest)):
                    delta = s_xy[nbr_idx] - pt
                    design = np.column_stack(
                        [
                            np.ones(len(delta)),
                            delta[:, 0],
                            delta[:, 1],
                            delta[:, 0] ** 2,
                            delta[:, 0] * delta[:, 1],
                            delta[:, 1] ** 2,
                        ]
                    )
                    coeffs = np.linalg.lstsq(design, s_T[nbr_idx], rcond=None)[0]
                    grads[row_idx] = coeffs[1:3]
                flux[q_mask] = -k_vals[:, None] * grads
            else:
                flux[q_mask] = 0.0

    out: dict[str, np.ndarray] = {"T": temps, "region_ids": labels}
    if flux is not None:
        out["q"] = flux
    return out


@dataclass
class SolutionArtifact:
    """Contenedor portable unificado para cualquier solución PINN, FEM o RBF."""

    kind: str  # 'pinn', 'fem', 'rbf'
    problem: ThermalProblem2D
    specification: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    model: BasePINNModel | None = None
    training_state: dict[str, Any] | None = None
    fem_xy: np.ndarray | None = None
    fem_T: np.ndarray | None = None
    source_dir: Path | None = None

    @property
    def id(self) -> str:
        return self.source_dir.name if self.source_dir is not None else self.problem.id

    @property
    def architecture(self) -> str:
        if self.kind == "fem":
            return str(self.result.get("method", "FEniCSx"))
        if self.kind == "rbf":
            return "RBF-FD"
        return str(self.specification.get("config", {}).get("architecture", getattr(self.model, "architecture", "unknown")))

    def predict(self, xy: np.ndarray, with_flux: bool = False, chunk_size: int = 20000) -> dict[str, np.ndarray]:
        """Evalúa temperatura (y flujo opcional) en coordenadas físicas ``(N, 2)``."""
        if self.kind == "pinn" and self.model is not None:
            return self.model.predict(xy, with_flux=with_flux, chunk_size=chunk_size)
        if self.fem_xy is not None and self.fem_T is not None:
            return _estimate_discrete_field_and_flux(
                self.problem, self.fem_xy, self.fem_T, xy, with_flux=with_flux
            )
        raise RuntimeError("El artefacto no contiene un modelo entrenado ni un campo nodal evaluable.")

    def save(
        self,
        directory: str | Path,
        training_state: dict[str, Any] | None = None,
        evaluation_arrays: dict[str, np.ndarray] | None = None,
    ) -> Path:
        """Guarda atómicamente el artefacto completo en ``directory``."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out_dir / "problem.json", self.problem.doc)

        if self.kind == "pinn" and self.model is not None:
            state = training_state if training_state is not None else self.training_state
            spec = dict(self.specification)
            write_json_atomic(out_dir / "specification.json", spec)
            bundle = {
                "format": ARTIFACT_FORMAT,
                "schema_version": 4,
                "specification_sha256": specification_fingerprint(spec),
                "model": cpu_state(self.model.state_dict()),
                "training": cpu_state(state),
                "resume_available": state is not None and "optimizer" in state,
            }
            tmp = out_dir / "model.pt.tmp"
            torch.save(bundle, tmp)
            atomic_replace(tmp, out_dir / "model.pt")
            if self.result:
                write_json_atomic(out_dir / "result.json", self.result)
            if evaluation_arrays:
                tmp_npz = out_dir / "evaluation.tmp.npz"
                with tmp_npz.open("wb") as stream:
                    np.savez_compressed(stream, **evaluation_arrays)
                atomic_replace(tmp_npz, out_dir / "evaluation.npz")
        elif self.fem_xy is not None and self.fem_T is not None:
            write_json_atomic(out_dir / "result.json", self.result)
            np.savez_compressed(out_dir / "field.npz", dof_xy=self.fem_xy, dof_T=self.fem_T)
        self.source_dir = out_dir
        return out_dir

    @classmethod
    def load(
        cls,
        path_or_dir: str | Path,
        device: str | torch.device = "cpu",
        selection: str = "inference",
    ) -> SolutionArtifact:
        """Carga un artefacto desde carpeta de corrida, ``model.pt`` o registro ``fem_*.json``."""
        import pinn_thermal.models  # noqa: F401 - asegura modelos registrados

        target = Path(path_or_dir)
        # Caso 1: archivo JSON de resultado FEM
        if target.is_file() and target.suffix.lower() == ".json" and target.stem.startswith("fem_"):
            record = json.loads(target.read_text(encoding="utf-8-sig"))
            prob_path = target.parent / "problem.json"
            if not prob_path.is_file():
                prob_path = target.parent / "input_problem.json"
            if prob_path.is_file():
                problem = ThermalProblem2D.from_json(prob_path)
            elif "problem" in record:
                problem = ThermalProblem2D.from_json(record["problem"])
            else:
                raise FileNotFoundError(f"No se encontró problem.json junto a {target}.")
            npz_path = target.with_suffix(".npz")
            data = np.load(npz_path)
            return cls(
                kind="fem",
                problem=problem,
                specification={"format": "fem-record", "physics_sha256": record.get("physics_sha256")},
                result=record,
                fem_xy=np.asarray(data["dof_xy"], dtype=float),
                fem_T=np.asarray(data["dof_T"], dtype=float),
                source_dir=target.parent,
            )

        # Caso 2: carpeta de corrida con field.npz / fem_*.json o model.pt
        if target.is_dir():
            fem_jsons = sorted(target.glob("fem_l*.json"))
            if (target / "model.pt").is_file():
                ckpt_path = target / "model.pt"
            elif fem_jsons:
                return cls.load(fem_jsons[-1], device=device, selection=selection)
            elif (target / "field.npz").is_file() and (target / "problem.json").is_file():
                problem = ThermalProblem2D.from_json(target / "problem.json")
                data = np.load(target / "field.npz")
                res = json.loads((target / "result.json").read_text(encoding="utf-8-sig")) if (target / "result.json").is_file() else {}
                return cls(
                    kind=res.get("kind", "fem"),
                    problem=problem,
                    result=res,
                    fem_xy=np.asarray(data["dof_xy"], dtype=float),
                    fem_T=np.asarray(data["dof_T"], dtype=float),
                    source_dir=target,
                )
            else:
                raise FileNotFoundError(f"La carpeta {target} no contiene 'model.pt' ni resultados FEM reconocibles.")
        else:
            ckpt_path = target

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        fmt = payload.get("format")
        if fmt not in (ARTIFACT_FORMAT, LEGACY_CHECKPOINT_FORMAT):
            raise ValueError(f"Formato de checkpoint no soportado: {fmt!r}.")

        spec_path = ckpt_path.parent / "specification.json"
        if not spec_path.is_file():
            spec_path = ckpt_path.with_suffix(".json")
        specification = json.loads(spec_path.read_text(encoding="utf-8-sig"))
        if fmt == ARTIFACT_FORMAT and specification_fingerprint(specification) != payload["specification_sha256"]:
            raise ValueError("La especificación JSON no coincide con el hash SHA-256 del checkpoint.")

        problem = ThermalProblem2D.from_json(specification["problem"])
        config = specification["config"]
        dtype = DTYPES.get(config.get("dtype", "float32"), torch.float32)

        dev = torch.device(device) if not isinstance(device, torch.device) else device
        chain_cfg = specification.get("chain")
        with torch.random.fork_rng(devices=[]):
            if chain_cfg is not None:
                from pinn_thermal.core.transfer import ChainedPINN

                src_spec = chain_cfg["source_specification"]
                src_prob = ThermalProblem2D.from_json(src_spec["problem"])
                src_cfg = src_spec["config"]
                src_model = ModelRegistry.build(
                    problem=src_prob,
                    architecture=src_cfg.get("base_architecture", src_cfg["architecture"]),
                    width=int(src_cfg["width"]),
                    depth=int(src_cfg["depth"]),
                    activation=src_cfg.get("activation", "tanh"),
                    input_normalization=src_cfg.get("input_normalization"),
                    trace_parameterization=src_cfg.get("trace_parameterization", "standard"),
                    soil_envelope_gain=float(src_cfg.get("soil_envelope_gain", 1.0)),
                ).to(device=dev, dtype=dtype)
                src_art = cls(kind="pinn", problem=src_prob, specification=src_spec, model=src_model)
                tgt_base = ModelRegistry.build(
                    problem=problem,
                    architecture=chain_cfg["base_architecture"],
                    width=int(config["width"]),
                    depth=int(config["depth"]),
                    activation=config.get("activation", "tanh"),
                    input_normalization=config.get("input_normalization"),
                    trace_parameterization=config.get("trace_parameterization", "standard"),
                    soil_envelope_gain=float(config.get("soil_envelope_gain", 1.0)),
                ).to(device=dev, dtype=dtype)
                model = ChainedPINN(
                    source_artifact=src_art,
                    target_model=tgt_base,
                    chain_mode=chain_cfg["chain_mode"],
                    freeze_source=bool(chain_cfg.get("freeze_source", True)),
                ).to(device=dev, dtype=dtype)
            else:
                model = ModelRegistry.build(
                    problem=problem,
                    architecture=config["architecture"],
                    width=int(config["width"]),
                    depth=int(config["depth"]),
                    activation=config.get("activation", "tanh"),
                    input_normalization=config.get("input_normalization"),
                    trace_parameterization=config.get("trace_parameterization", "standard"),
                    soil_envelope_gain=float(config.get("soil_envelope_gain", 1.0)),
                ).to(device=dev, dtype=dtype)

        if selection == "latest":
            training = payload.get("training")
            if training is None:
                raise ValueError("El artefacto no contiene estado de entrenamiento ('latest').")
            weights = training["model"]
        elif selection == "inference":
            weights = payload["model"]
        else:
            raise ValueError("selection debe ser 'inference' o 'latest'.")

        model.load_state_dict(weights, strict=True)
        model.eval()

        res_path = ckpt_path.parent / "result.json"
        result = json.loads(res_path.read_text(encoding="utf-8-sig")) if res_path.is_file() else {}
        return cls(
            kind="pinn",
            problem=problem,
            specification=specification,
            result=result,
            model=model,
            training_state=payload.get("training"),
            source_dir=ckpt_path.parent,
        )

    def export_portable_zip(self, zip_path: str | Path) -> Path:
        """Empaqueta el artefacto en un archivo ZIP autoconstituido y portable."""
        dst = Path(zip_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            if self.source_dir is not None and (self.source_dir / "model.pt").is_file():
                for fname in ("model.pt", "specification.json", "result.json", "problem.json", "manifest.json", "evaluation.npz"):
                    src_f = self.source_dir / fname
                    if src_f.is_file():
                        shutil.copy2(src_f, tmp_path / fname)
            else:
                self.save(tmp_path)
            with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for item in sorted(tmp_path.rglob("*")):
                    if item.is_file():
                        zf.write(item, item.relative_to(tmp_path))
        return dst

    @classmethod
    def import_portable_zip(
        cls,
        zip_path: str | Path,
        extract_dir: str | Path,
        device: str | torch.device = "cpu",
    ) -> SolutionArtifact:
        """Desempaqueta un archivo ZIP portable en ``extract_dir`` y carga su ``SolutionArtifact``."""
        dst = Path(extract_dir)
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dst)
        return cls.load(dst, device=device)
