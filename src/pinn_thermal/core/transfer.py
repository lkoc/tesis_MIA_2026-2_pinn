"""Compatibilidad entre artefactos, transferencia de pesos, transferencia funcional y encadenamiento.

Cubre los cuatro mecanismos requeridos para reutilizar entrenamientos previos:
1. **Reanudación exacta**: mismo problema, modelo, optimizador, scheduler y nubes.
2. **Transferencia de pesos (``WeightTransfer``)**: permite continuar el entrenamiento
   iniciado por otra red incluso si los JSON son diferentes, siempre que compartan las
   características críticas (arquitectura/subredes, ancho, profundidad, activación, normalización,
   y subdominios equivalentes).
3. **Transferencia funcional (``FunctionalTransfer``)**: permite transferir el campo
   aprendido por **cualquier** modelo previo (PINN, FEM o RBF-FD) hacia **cualquier**
   arquitectura destino mediante mínimos cuadrados (``readout`` / ``readout_gls``) o
   preentrenamiento supervisado (``pretrain``), aun cuando cambien arquitectura o regiones.
4. **Encadenamiento directo (``ChainedPINN``)**: usa la salida de un modelo anterior
   como campo base residual (``T = T_prev + delta_T``) o como entrada adicional
   normalizada manteniendo derivadas diferenciables por autograd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

from pinn_thermal.core.artifact import SolutionArtifact
from pinn_thermal.core.base_model import BasePINNModel, ModelMetadata, ModelRegistry
from pinn_thermal.core.physics import gradients
from pinn_thermal.core.problem import RegionSpec, ThermalProblem2D


@dataclass
class CompatibilityReport:
    """Resultado auditable del análisis de compatibilidad entre un artefacto origen y uno destino."""

    compatible: bool
    level: str  # 'exact', 'weights_compatible', 'partial_weights', 'functional_only'
    matched_keys: list[str] = field(default_factory=list)
    skipped_keys: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    same_physics: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "level": self.level,
            "same_physics": self.same_physics,
            "matched_keys": list(self.matched_keys),
            "skipped_keys": list(self.skipped_keys),
            "reasons": list(self.reasons),
        }


class CompatibilityChecker:
    """Evalúa si dos redes/problemas comparten las características críticas para transferir pesos."""

    @staticmethod
    def check(source: SolutionArtifact, target_model: BasePINNModel) -> CompatibilityReport:
        same_physics = source.problem.physics_sha256 == target_model.problem.physics_sha256
        if source.kind != "pinn" or source.model is None:
            return CompatibilityReport(
                compatible=False,
                level="functional_only",
                same_physics=same_physics,
                reasons=[f"El artefacto origen es de tipo {source.kind!r}; solo admite transferencia funcional."],
            )

        src_prof = source.model.compatibility_profile()
        tgt_prof = target_model.compatibility_profile()
        reasons: list[str] = []

        if src_prof["partition"] != tgt_prof["partition"]:
            reasons.append(f"Partición distinta: origen={src_prof['partition']!r}, destino={tgt_prof['partition']!r}.")
        if src_prof["formulation"] != tgt_prof["formulation"]:
            reasons.append(f"Formulación distinta: origen={src_prof['formulation']!r}, destino={tgt_prof['formulation']!r}.")
        if src_prof["activation"] != tgt_prof["activation"]:
            reasons.append(f"Activación distinta: origen={src_prof['activation']!r}, destino={tgt_prof['activation']!r}.")
        if src_prof["width"] != tgt_prof["width"] or src_prof["depth"] != tgt_prof["depth"]:
            reasons.append(
                f"Dimensiones distintas: origen=({src_prof['width']}x{src_prof['depth']}), "
                f"destino=({tgt_prof['width']}x{tgt_prof['depth']})."
            )
        if src_prof["input_normalization"] != tgt_prof["input_normalization"]:
            reasons.append(
                f"Normalización de entrada distinta: origen={src_prof['input_normalization']!r}, "
                f"destino={tgt_prof['input_normalization']!r}."
            )
        if src_prof["domain_shape"] != tgt_prof["domain_shape"]:
            reasons.append(
                f"Geometría de dominio incompatible: origen={src_prof['domain_shape']!r}, destino={tgt_prof['domain_shape']!r}."
            )

        src_subnets = source.model.subnetwork_map()
        tgt_subnets = target_model.subnetwork_map()
        matched: list[str] = []
        skipped: list[str] = []

        # Mapeo directo o por tipo de región ('soil', 'conductor', 'material') si difiere el nombre/id
        for t_key, t_mod in tgt_subnets.items():
            cand_key: str | None = t_key if t_key in src_subnets else None
            if cand_key is None and t_key.startswith("region:"):
                t_name = t_key.split(":", 1)[1]
                t_kind = tgt_prof["region_kinds"].get(t_name)
                for s_key in src_subnets:
                    if s_key.startswith("region:"):
                        s_name = s_key.split(":", 1)[1]
                        if src_prof["region_kinds"].get(s_name) == t_kind:
                            cand_key = s_key
                            break
            if cand_key is None:
                skipped.append(f"{t_key} (sin subred equivalente en origen)")
                continue
            s_state = src_subnets[cand_key].state_dict()
            t_state = t_mod.state_dict()
            shapes_ok = (
                set(s_state) == set(t_state)
                and all(s_state[k].shape == t_state[k].shape for k in t_state)
            )
            if shapes_ok:
                matched.append(f"{cand_key} -> {t_key}")
            else:
                skipped.append(f"{t_key} (formas de tensores incompatibles con {cand_key})")

        if not reasons and matched and not skipped and same_physics and src_prof["architecture"] == tgt_prof["architecture"]:
            return CompatibilityReport(
                compatible=True,
                level="exact",
                same_physics=True,
                matched_keys=matched,
                skipped_keys=skipped,
                reasons=["Arquitectura y física idénticas; compatible incluso para reanudación exacta."],
            )
        if not reasons and matched and not skipped:
            return CompatibilityReport(
                compatible=True,
                level="weights_compatible",
                same_physics=same_physics,
                matched_keys=matched,
                skipped_keys=skipped,
                reasons=["Características críticas y dimensiones de subredes compatibles."],
            )
        if not reasons and matched:
            return CompatibilityReport(
                compatible=True,
                level="partial_weights",
                same_physics=same_physics,
                matched_keys=matched,
                skipped_keys=skipped,
                reasons=["Compatibilidad parcial: se transferirán subredes coincidentes."],
            )
        return CompatibilityReport(
            compatible=False,
            level="functional_only",
            same_physics=same_physics,
            matched_keys=matched,
            skipped_keys=skipped,
            reasons=reasons or ["No existen subredes de forma compatible; utilice transferencia funcional."],
        )


class WeightTransfer:
    """Copia segura de parámetros entrenables compatibles entre dos modelos PINN."""

    @staticmethod
    def apply(source: SolutionArtifact, target_model: BasePINNModel) -> dict[str, Any]:
        report = CompatibilityChecker.check(source, target_model)
        if not report.compatible or source.model is None:
            raise ValueError(
                "No es posible transferir pesos directamente entre los modelos indicados: "
                + "; ".join(report.reasons)
            )

        src_subnets = source.model.subnetwork_map()
        tgt_subnets = target_model.subnetwork_map()
        tgt_prof = target_model.compatibility_profile()
        src_prof = source.model.compatibility_profile()

        copied_tensors = 0
        pairs_copied: list[str] = []
        with torch.no_grad():
            for t_key, t_mod in tgt_subnets.items():
                cand_key: str | None = t_key if t_key in src_subnets else None
                if cand_key is None and t_key.startswith("region:"):
                    t_kind = tgt_prof["region_kinds"].get(t_key.split(":", 1)[1])
                    for s_key in src_subnets:
                        if s_key.startswith("region:") and src_prof["region_kinds"].get(s_key.split(":", 1)[1]) == t_kind:
                            cand_key = s_key
                            break
                if cand_key is None:
                    continue
                s_state = src_subnets[cand_key].state_dict()
                t_state = t_mod.state_dict()
                if set(s_state) == set(t_state) and all(s_state[k].shape == t_state[k].shape for k in t_state):
                    t_mod.load_state_dict({k: v.to(dtype=t_state[k].dtype, device=t_state[k].device) for k, v in s_state.items()})
                    copied_tensors += len(t_state)
                    pairs_copied.append(f"{cand_key} -> {t_key}")

            # Transferir offsets si coinciden en longitud y estructura
            src_offsets = getattr(source.model, "offsets", None)
            tgt_offsets = getattr(target_model, "offsets", None)
            if (
                isinstance(src_offsets, nn.ParameterList)
                and isinstance(tgt_offsets, nn.ParameterList)
                and len(src_offsets) == len(tgt_offsets)
            ):
                for s_off, t_off in zip(src_offsets, tgt_offsets):
                    if s_off.shape == t_off.shape:
                        t_off.copy_(s_off.to(dtype=t_off.dtype, device=t_off.device))
                        copied_tensors += 1

        return {
            "method": "weight_transfer",
            "compatibility": report.to_dict(),
            "pairs_copied": pairs_copied,
            "copied_tensors": copied_tensors,
        }


def _estimate_gradients_kdtree(xy_source: np.ndarray, t_source: np.ndarray, query_xy: np.ndarray, neighbours: int = 12) -> np.ndarray:
    """Estima ``grad(T)`` local mediante ajuste cuadrático por región."""
    k = min(neighbours, len(xy_source))
    if k < 6:
        return np.zeros((len(query_xy), 2), dtype=float)
    nearest = cKDTree(xy_source).query(query_xy, k=k)[1]
    grads = np.zeros((len(query_xy), 2), dtype=float)
    for row, (pt, idx) in enumerate(zip(query_xy, nearest)):
        delta = xy_source[idx] - pt
        design = np.column_stack(
            [np.ones(len(delta)), delta[:, 0], delta[:, 1], delta[:, 0] ** 2, delta[:, 0] * delta[:, 1], delta[:, 1] ** 2]
        )
        coeffs = np.linalg.lstsq(design, t_source[idx], rcond=None)[0]
        grads[row] = coeffs[1:3]
    return grads


class FunctionalTransfer:
    """Transfiere el campo térmico desde cualquier artefacto origen hacia cualquier modelo PINN destino.

    Al operar exclusivamente en el espacio físico ``(x, y) -> (T, q)``, permite conectar
    libremente modelos de arquitecturas distintas (p. ej. ``M0 -> M1``, ``M1 -> CUSPR``,
    ``FEM -> M1``, ``FBPINN -> M1_EXACT``, ``M1 -> M2``) y problemas cuyo esquema de
    regiones haya cambiado.
    """

    @classmethod
    def apply(
        cls,
        source: SolutionArtifact,
        target_model: BasePINNModel,
        method: str = "pretrain",
        steps: int = 150,
        lr: float = 2e-3,
        points_per_region: int = 256,
        seed: int = 11,
        gradient_weight: float = 1.0,
        ridge: float = 0.0,
    ) -> dict[str, Any]:
        """Inicializa ``target_model`` replicando el campo térmico de ``source``."""
        if method not in ("readout", "readout_gls", "pretrain"):
            raise ValueError("method debe ser 'readout', 'readout_gls' o 'pretrain'.")

        problem = target_model.problem
        rng = np.random.default_rng(int(seed))
        region_points = {
            r.id: problem.sample_region_interior(r, int(points_per_region), rng)
            for r in problem.regions
        }

        # Consultar el campo del artefacto origen dentro de cada región del problema destino
        need_grad = (method == "readout_gls") or bool(target_model.mixed)
        pred_T: dict[int, np.ndarray] = {}
        pred_grad: dict[int, np.ndarray] = {}
        pred_flux: dict[int, np.ndarray] = {}

        for r in problem.regions:
            pts = region_points[r.id]
            res = source.predict(pts, with_flux=need_grad)
            pred_T[r.id] = res["T"]
            if need_grad:
                k_target = r.conductivity_spec.evaluate_numpy(pts[:, 0], pts[:, 1])
                if "q" in res:
                    pred_flux[r.id] = res["q"]
                    pred_grad[r.id] = -res["q"] / np.maximum(k_target[:, None], 1e-12)
                else:
                    grads = _estimate_gradients_kdtree(pts, res["T"], pts)
                    pred_grad[r.id] = grads
                    pred_flux[r.id] = -k_target[:, None] * grads

        if method in ("readout", "readout_gls") and not target_model.mixed:
            try:
                report = cls._readout_projection(
                    target_model,
                    region_points,
                    pred_T,
                    gradients=pred_grad if method == "readout_gls" else None,
                    gradient_weight=gradient_weight,
                    ridge=ridge,
                )
                report["transfer_method"] = method
                report["source_kind"] = source.kind
                report["source_architecture"] = source.architecture
                return report
            except Exception as exc:
                # Si una variante específica (p. ej. múltiples ventanas FBPINN o salidas combinadas)
                # no admite proyección lineal directa de última capa, cae limpiamente a pretrain supervisado.
                fallback_report = cls._pretrain_fit(
                    target_model, region_points, pred_T, pred_flux if target_model.mixed else None, steps=steps, lr=lr
                )
                fallback_report["fallback_from"] = f"{method} ({exc})"
                fallback_report["source_kind"] = source.kind
                fallback_report["source_architecture"] = source.architecture
                return fallback_report

        report = cls._pretrain_fit(
            target_model, region_points, pred_T, pred_flux if target_model.mixed else None, steps=steps, lr=lr
        )
        report["source_kind"] = source.kind
        report["source_architecture"] = source.architecture
        return report

    @staticmethod
    def _readout_projection(
        model: BasePINNModel,
        region_points: dict[int, np.ndarray],
        region_T: dict[int, np.ndarray],
        gradients: dict[int, np.ndarray] | None = None,
        gradient_weight: float = 1.0,
        ridge: float = 0.0,
    ) -> dict[str, Any]:
        """Proyección por mínimos cuadrados ordinarios o generalizados sobre la capa de salida."""
        ref = next(model.parameters())
        dtype, device = ref.dtype, ref.device
        problem = model.problem
        net_model = getattr(model, "base_model", model)
        networks = getattr(net_model, "networks", None)
        if not isinstance(networks, nn.ModuleList) or len(networks) == 0:
            raise ValueError("El modelo destino no expone ModuleList 'networks' para proyección readout.")

        groups: dict[tuple[int, int], list[RegionSpec]] = {}
        for region in problem.regions:
            idx = 0 if net_model.partition == "global" and len(networks) == 1 else region.id
            if idx >= len(networks):
                raise ValueError("El número de subredes no corresponde al mapeo directo de regiones.")
            net = networks[idx]
            last = net[-1] if hasattr(net, "__getitem__") else None
            if not isinstance(last, nn.Linear):
                raise ValueError("La última capa de la subred no es nn.Linear.")
            # En redes globales multicabezal (M0, M0R, CUSPR), cada región posee su propio canal de salida
            ch = region.id if (idx == 0 and last.out_features == len(problem.regions) and len(problem.regions) > 1) else 0
            groups.setdefault((idx, ch), []).append(region)

        value_errors, grad_errors, weight_norms = [], [], []
        for (idx, ch), regions in groups.items():
            network = networks[idx]
            last = network[-1]
            blocks: dict[int, dict[str, Any]] = {}
            val_rows, val_targets, g_rows, g_targets = [], [], [], []
            for region in regions:
                xy = torch.as_tensor(region_points[region.id], dtype=dtype, device=device).requires_grad_(True)
                target_v = torch.as_tensor(region_T[region.id], dtype=torch.float64, device=device).reshape(-1, 1)
                captured: list[tuple[torch.Tensor, torch.Tensor]] = []
                hook = last.register_forward_hook(lambda _mod, inp, out: captured.append((inp[0], out)))
                try:
                    base = model.field(region, xy)[0].double().reshape(-1, 1)
                finally:
                    hook.remove()
                design_v = None
                raw_contrib = torch.zeros_like(base)
                for feat, raw in captured:
                    sens = torch.autograd.grad(base.sum(), raw, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    if sens is None:
                        continue
                    sc = sens[:, ch:ch + 1].double()
                    aug = torch.cat([feat, torch.ones(len(xy), 1, dtype=feat.dtype, device=device)], dim=1).double()
                    contrib = sc * aug
                    design_v = contrib if design_v is None else design_v + contrib
                    raw_contrib = raw_contrib + sc * raw[:, ch:ch + 1].double()
                if design_v is None:
                    raise ValueError("No se pudo trazar la sensibilidad de la capa de salida.")
                offset = base - raw_contrib
                val_rows.append(design_v)
                val_targets.append(target_v - offset)
                blk: dict[str, Any] = {"xy": xy, "values": target_v, "gradient": None}
                if gradients is not None:
                    length = float(np.sqrt(problem.areas[region.id]))
                    jac = torch.stack(
                        [torch.autograd.grad(design_v[:, u].sum(), xy, retain_graph=True)[0] for u in range(design_v.shape[1])],
                        dim=2,
                    ).double()
                    off_g = torch.autograd.grad(offset.sum(), xy, retain_graph=True)[0].double()
                    ax_r, ax_t = [], []
                    for ax in range(2):
                        ax_r.append(gradient_weight * length * jac[:, ax, :])
                        g_true = torch.as_tensor(gradients[region.id][:, ax], dtype=torch.float64, device=device).reshape(-1, 1)
                        ax_t.append(gradient_weight * length * (g_true - off_g[:, ax:ax + 1]))
                    blk["gradient"] = (torch.cat(ax_r), torch.cat(ax_t))
                    g_rows.append(blk["gradient"][0])
                    g_targets.append(blk["gradient"][1])
                blocks[region.id] = blk

            design = torch.cat(val_rows + g_rows)
            target = torch.cat(val_targets + g_targets)
            if ridge > 0:
                w_cols = design.shape[1] - 1
                pen = torch.cat(
                    [float(np.sqrt(ridge)) * torch.eye(w_cols, dtype=torch.float64, device=device), torch.zeros(w_cols, 1, dtype=torch.float64, device=device)],
                    dim=1,
                )
                design = torch.cat([design, pen])
                target = torch.cat([target, torch.zeros(w_cols, 1, dtype=torch.float64, device=device)])
            theta = torch.linalg.lstsq(design, target).solution.reshape(-1)
            with torch.no_grad():
                last.weight[ch].copy_(theta[:-1].to(dtype=last.weight.dtype, device=last.weight.device))
                last.bias[ch].copy_(theta[-1].to(dtype=last.bias.dtype, device=last.bias.device))
            weight_norms.append(float(theta[:-1].detach().norm()))
            for region in regions:
                blk = blocks[region.id]
                with torch.no_grad():
                    pred = model.field(region, blk["xy"].detach())[0].double().reshape(-1, 1)
                value_errors.append((pred - blk["values"]).reshape(-1))
                if blk["gradient"] is not None:
                    gd, gt = blk["gradient"]
                    grad_errors.append((gd @ theta.reshape(-1, 1) - gt).reshape(-1))

        err = torch.cat(value_errors)
        rep: dict[str, Any] = {
            "steps": 0,
            "points": int(len(err)),
            "weight_norm": max(weight_norms) if weight_norms else 0.0,
            "rms_K": float(err.pow(2).mean().sqrt()),
            "max_K": float(err.abs().max()),
        }
        if gradients is not None and grad_errors:
            rep["gradient_rms_K"] = float(torch.cat(grad_errors).detach().pow(2).mean().sqrt())
        return rep

    @staticmethod
    def _pretrain_fit(
        model: BasePINNModel,
        region_points: dict[int, np.ndarray],
        region_T: dict[int, np.ndarray],
        region_flux: dict[int, np.ndarray] | None = None,
        steps: int = 150,
        lr: float = 2e-3,
    ) -> dict[str, Any]:
        """Ajuste rápido mediante Adam para transferir temperatura (y flujo en M2) a cualquier modelo."""
        ref = next(model.parameters())
        tensors: list[tuple[RegionSpec, torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
        total_pts = 0
        for region in model.problem.regions:
            xy_t = torch.as_tensor(region_points[region.id], dtype=ref.dtype, device=ref.device)
            t_t = torch.as_tensor(region_T[region.id], dtype=ref.dtype, device=ref.device).reshape(-1, 1)
            q_t = (
                torch.as_tensor(region_flux[region.id], dtype=ref.dtype, device=ref.device)
                if region_flux is not None and region.id in region_flux
                else None
            )
            tensors.append((region, xy_t, t_t, q_t))
            total_pts += len(xy_t)

        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        losses: list[float] = []
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            terms = []
            for region, xy_t, t_t, q_t in tensors:
                pts = xy_t.detach().requires_grad_(True)
                pred_t, pred_q = model.field(region, pts)
                terms.append(((pred_t - t_t) / max(model.problem.T_scale, 1e-12)).square().mean())
                if model.mixed and pred_q is not None and q_t is not None:
                    f_scale = max(model.problem.scales(region)["flux"], 1e-12)
                    terms.append(((pred_q - q_t) / f_scale).square().mean())
            loss = sum(terms) / max(len(terms), 1)
            losses.append(float(loss.detach()))
            loss.backward()
            opt.step()

        # Error final respecto al campo fuente
        errs = []
        with torch.no_grad():
            for region, xy_t, t_t, _ in tensors:
                pred_t, _ = model.field(region, xy_t)
                errs.append((pred_t - t_t).detach().cpu().numpy().ravel())
        cat = np.concatenate(errs)
        return {
            "transfer_method": "pretrain",
            "steps": int(steps),
            "points": int(total_pts),
            "initial_loss": losses[0] if losses else 0.0,
            "final_loss": losses[-1] if losses else 0.0,
            "rms_K": float(np.sqrt(np.mean(cat ** 2))),
            "max_K": float(np.max(np.abs(cat))),
        }


class ChainedPINN(BasePINNModel):
    """Encadena un modelo previo entrenado como punto de partida activo de un modelo nuevo.

    Modos disponibles:
    - ``residual``: ``T(x, y) = T_prev(x, y) + T_corr(x, y)``, manteniendo derivadas completas
      por autograd respecto de ``(x, y)`` para que las pérdidas PDE y de flujo actúen sobre
      el campo físico total.
    - ``feature``: proyecta una compuerta aprendible que combina la predicción previa normalizada
      con la red residual destino, permitiendo refinar sin partir desde cero.
    """

    model_kind = "chained"

    def __init__(
        self,
        source_artifact: SolutionArtifact,
        target_model: BasePINNModel,
        chain_mode: str = "residual",
        freeze_source: bool = True,
    ) -> None:
        if chain_mode not in ("residual", "feature"):
            raise ValueError("chain_mode debe ser 'residual' o 'feature'.")
        if source_artifact.kind != "pinn" or source_artifact.model is None:
            raise ValueError("ChainedPINN requiere un artefacto PINN diferenciable; para FEM use FunctionalTransfer.")
        super().__init__(
            problem=target_model.problem,
            architecture=f"CHAINED({target_model.architecture})",
            width=target_model.width,
            depth=target_model.depth,
            activation=target_model.activation,
            partition=target_model.partition,
            formulation=target_model.formulation,
            input_normalization=target_model.input_normalization,
        )
        self.chain_mode = chain_mode
        self.freeze_source = bool(freeze_source)
        self.exact_constraints = bool(target_model.exact_constraints)
        self.exact_boundary = bool(target_model.exact_boundary)

        # Clonar / reubicar modelo origen al mismo dispositivo y dtype del modelo destino
        ref = next(target_model.parameters())
        self.source_model: BasePINNModel = source_artifact.model.to(device=ref.device, dtype=ref.dtype)
        for param in self.source_model.parameters():
            param.requires_grad_(not self.freeze_source)

        self.target_model: BasePINNModel = target_model
        # En modo residual centramos la corrección para evitar sumar T_ref dos veces
        self.correction_gain = nn.Parameter(torch.tensor(1.0, dtype=ref.dtype, device=ref.device))
        if chain_mode == "feature":
            self.feature_gate = nn.Linear(2, 1).to(dtype=ref.dtype, device=ref.device)
            with torch.no_grad():
                self.feature_gate.weight.copy_(torch.tensor([[1.0, 1.0]], dtype=ref.dtype, device=ref.device))
                self.feature_gate.bias.zero_()

    def _source_field(self, region: RegionSpec, xy: torch.Tensor) -> torch.Tensor:
        """Evalúa la temperatura previa en los puntos físicos ``xy`` manejando diferencias de regiones."""
        src_prob = self.source_model.problem
        # Mapeo directo por nombre si existe; si no, por región propietaria geométrica en el origen
        if region.name in src_prob.region_by_name and len(src_prob.regions) == len(self.problem.regions):
            src_region = src_prob.region_by_name[region.name]
            t_prev, _ = self.source_model.field(src_region, xy)
            return t_prev
        xy_np = xy.detach().cpu().numpy()
        labels_np = src_prob.owner_index(xy_np)
        out = torch.empty((len(xy), 1), dtype=xy.dtype, device=xy.device)
        for src_reg in src_prob.regions:
            mask = labels_np == src_reg.id
            if np.any(mask):
                out[mask] = self.source_model.field(src_reg, xy[mask])[0]
        if np.any(labels_np < 0):
            # Puntos fuera del dominio origen (si el nuevo dominio fue ampliado): usan región de fondo 0
            out[labels_np < 0] = self.source_model.field(src_prob.regions[0], xy[labels_np < 0])[0]
        return out

    def field(self, region: RegionSpec, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        t_prev = self._source_field(region, xy)
        t_tgt, q_tgt = self.target_model.field(region, xy)
        delta_t = t_tgt - self.problem.T_ref
        if self.chain_mode == "residual":
            t_total = t_prev + self.correction_gain * delta_t
            return t_total, q_tgt
        t_prev_norm = (t_prev - self.problem.T_ref) / max(self.problem.T_scale, 1e-12)
        t_tgt_norm = delta_t / max(self.problem.T_scale, 1e-12)
        combined = self.feature_gate(torch.cat([t_prev_norm, t_tgt_norm], dim=1))
        return self.problem.T_ref + self.problem.T_scale * combined, q_tgt

    def subnetwork_map(self) -> dict[str, nn.Module]:
        return self.target_model.subnetwork_map()
