"""Evaluación numérica independiente y auditoría física completa.

Contiene tres componentes desacoplados:
- ``ErrorMetrics``: métricas de error térmico respecto a una referencia analítica o FEM.
- ``PhysicsAuditor``: verificación independiente de residuos PDE, ley constitutiva,
  saltos de temperatura y flujo en interfaces, condiciones de frontera y balance energético.
- ``AcceptanceGate``: puertas de aceptación parametrizables (estándar y tesis estricta).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pinn_thermal.core.base_model import BasePINNModel
from pinn_thermal.core.objective import flux_fields
from pinn_thermal.core.physics import heat_residual, mixed_residual
from pinn_thermal.core.problem import ThermalProblem2D

AUDIT_SEED: int = 909091
PERCENTILES: tuple[int, ...] = (50, 90, 95, 99)

CRITERIA_STANDARD: dict[str, float] = {"nrmse_pct": 10.0, "Tmax_rise_pct": 10.0}
CRITERIA_EXPLICIT_CAMPAIGN: dict[str, float] = {
    "nrmse_pct": 5.0,
    "Tmax_rise_pct": 5.0,
    "balance_pct": 2.0,
    "interface_T_jump_max_K": 0.1,
    "interface_flux_jump_rms_pct": 1.0,
}
CRITERIA_STRICT_THESIS: dict[str, float] = {
    "error_Tmax_abs_K": 0.05,
    "nrmse_pct": 0.25,
    "balance_pct": 0.5,
}


class ErrorMetrics:
    """Cálculo de errores de temperatura frente a solución exacta o campo FEM interpolado."""

    @staticmethod
    def hot_mask(labels: np.ndarray, conductor_ids: list[int] | tuple[int, ...]) -> np.ndarray:
        return np.isin(labels, list(conductor_ids)) if conductor_ids else np.ones(len(labels), dtype=bool)

    @classmethod
    def temperature_summary(
        cls,
        T: np.ndarray,
        xy: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        names: list[str],
        conductor_ids: list[int],
        hot: np.ndarray,
    ) -> dict[str, Any]:
        hot_indices = np.flatnonzero(hot)
        idx = hot_indices[int(np.argmax(T[hot]))]
        summary: dict[str, Any] = {
            "Tmax_C": float(T[idx]),
            "hotspot_xy": [float(xy[idx, 0]), float(xy[idx, 1])],
        }
        conductor_means: dict[str, float] = {}
        for rid in conductor_ids:
            sel = labels == rid
            if not np.any(sel):
                continue
            rw = weights[sel]
            avg = np.average(T[sel], weights=rw) if rw.sum() > 0 else np.mean(T[sel])
            conductor_means[names[rid]] = float(avg)
        summary["conductor_mean_C"] = conductor_means
        return summary

    @classmethod
    def compute(
        cls,
        T: np.ndarray,
        reference: np.ndarray,
        xy: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        names: list[str],
        level_C: float,
        conductor_ids: list[int],
        temperature_scale: float | None = None,
    ) -> dict[str, Any]:
        """Calcula errores globales, percentiles y errores desagregados por región."""
        hot = cls.hot_mask(labels, conductor_ids)
        error = T - reference
        pred = cls.temperature_summary(T, xy, labels, weights, names, conductor_ids, hot)
        ref = cls.temperature_summary(reference, xy, labels, weights, names, conductor_ids, hot)
        rise_ref = ref["Tmax_C"] - level_C
        rise_pred = pred["Tmax_C"] - level_C
        zero_rise = abs(rise_ref) <= 1e-12 * max(abs(ref["Tmax_C"]), 1.0)
        norm_k = temperature_scale if zero_rise else abs(rise_ref)
        if norm_k is None or norm_k <= 0.0:
            raise ValueError("Se requiere una escala térmica positiva cuando la elevación de referencia es nula.")

        rmse = float(np.sqrt(np.average(error ** 2, weights=weights)))
        mae = float(np.average(np.abs(error), weights=weights))
        abs_err = np.abs(error)
        site = np.flatnonzero(hot)[int(np.argmax(reference[hot]))]
        site_err = float(abs(T[site] - reference[site]))

        report: dict[str, Any] = {
            "level_C": float(level_C),
            "rise_ref_K": float(rise_ref),
            "Tmax_pred_C": pred["Tmax_C"],
            "Tmax_ref_C": ref["Tmax_C"],
            "error_Tmax_abs_K": float(abs(pred["Tmax_C"] - ref["Tmax_C"])),
            "error_rise_pct": float(100.0 * abs(rise_pred - rise_ref) / norm_k),
            "error_Tmax_site_K": site_err,
            "error_rise_site_pct": float(100.0 * site_err / norm_k),
            "temperature_normalization": "characteristic_scale" if zero_rise else "reference_rise",
            "temperature_normalization_K": float(norm_k),
            "hotspot_pred_xy": pred["hotspot_xy"],
            "hotspot_ref_xy": ref["hotspot_xy"],
            "hotspot_distance_m": float(np.hypot(*(np.array(pred["hotspot_xy"]) - np.array(ref["hotspot_xy"])))),
            "rmse_K": rmse,
            "mae_K": mae,
            "nrmse_pct": float(100.0 * rmse / norm_k),
            "max_error_K": float(abs_err.max()),
            "error_percentiles_K": {str(p): float(np.percentile(abs_err, p)) for p in PERCENTILES},
            "conductor_mean_pred_C": pred["conductor_mean_C"],
            "conductor_mean_ref_C": ref["conductor_mean_C"],
            "conductor_mean_error_K": {n: pred["conductor_mean_C"][n] - ref["conductor_mean_C"][n] for n in pred["conductor_mean_C"]},
            "region_rmse_K": {},
            "region_rmse_rise_pct": {},
            "region_max_error_K": {},
        }
        for i, name in enumerate(names):
            sel = labels == i
            if np.any(sel):
                val = float(np.sqrt(np.mean(error[sel] ** 2)))
                report["region_rmse_K"][name] = val
                report["region_rmse_rise_pct"][name] = float(100.0 * val / norm_k)
                report["region_max_error_K"][name] = float(abs_err[sel].max())
        return report


class PhysicsAuditor:
    """Auditoría física independiente usando semillas y nubes ajenas al entrenamiento."""

    @staticmethod
    def _tensor(model: BasePINNModel, array: np.ndarray) -> torch.Tensor:
        ref = next(model.parameters())
        return torch.as_tensor(np.asarray(array), dtype=ref.dtype, device=ref.device).clone().detach().requires_grad_(True)

    @classmethod
    def audit(
        cls,
        model: BasePINNModel,
        n_interior: int = 2048,
        n_layer: int = 1024,
        n_face: int = 512,
        n_boundary: int = 512,
        seed: int = AUDIT_SEED,
    ) -> dict[str, Any]:
        """Evalúa residuos PDE, constitutivo, saltos en interfaces, contorno y balance energético."""
        problem = model.problem
        names = {r.id: r.name for r in problem.regions}
        pde: dict[str, float] = {}
        constitutive: dict[str, float] = {}

        for region in problem.regions:
            n_pts = n_interior if region.kind == "soil" else n_layer
            xy = cls._tensor(model, problem.sample_interior(region, n_pts, seed + 7919 * region.id, "uniform"))
            T, predicted = model.field(region, xy)
            k = problem.conductivity(region, xy)
            source = problem.source(region, xy).detach()
            scales = problem.scales(region)
            if model.mixed and predicted is not None:
                res, law = mixed_residual(T, predicted, xy, k, source)
                constitutive[names[region.id]] = float(torch.sqrt(torch.mean((law / scales["flux"]) ** 2)).detach())
            else:
                res = heat_residual(T, xy, k, source)
            pde[names[region.id]] = float(torch.sqrt(torch.mean((res / scales["divergence"]) ** 2)).detach())

        derived_balance: dict[int, float] = {r.id: -problem.region_power[r.id] for r in problem.regions}
        law_balance = dict(derived_balance)
        interfaces: list[dict[str, Any]] = []

        for face in problem.faces(n_face):
            curve = face.xy if face.xy_right is None else 0.5 * (face.xy + face.xy_right)
            xy = cls._tensor(model, curve)
            normal = cls._tensor(model, face.normal)
            left, right = problem.regions[face.left], problem.regions[face.right]
            tl, ql, dl, _ = flux_fields(model, left, xy)
            tr, qr, dr, _ = flux_fields(model, right, xy)
            qln, qrn = (ql * normal).sum(1, keepdim=True), (qr * normal).sum(1, keepdim=True)
            dln, drn = (dl * normal).sum(1, keepdim=True), (dr * normal).sum(1, keepdim=True)
            scale = max(problem.scales(left)["flux"], problem.scales(right)["flux"], 1e-12)
            jump_T = (tl - tr).detach().cpu().numpy()
            jump_q = (qln - qrn).detach().cpu().numpy()
            jump_d = (dln - drn).detach().cpu().numpy()
            interfaces.append(
                {
                    "name": face.name,
                    "T_jump_rms_K": float(np.sqrt(np.mean(jump_T ** 2))),
                    "T_jump_max_K": float(np.abs(jump_T).max()),
                    "flux_jump_rms_pct": float(100.0 * np.sqrt(np.mean(jump_q ** 2)) / scale),
                    "flux_jump_max_pct": float(100.0 * np.abs(jump_q).max() / scale),
                    "derived_flux_jump_rms_pct": float(100.0 * np.sqrt(np.mean(jump_d ** 2)) / scale),
                    "flux_scale_W_m2": float(scale),
                }
            )
            derived_balance[left.id] += face.weight * float(dln.sum().detach())
            derived_balance[right.id] -= face.weight * float(drn.sum().detach())
            law_balance[left.id] += face.weight * float(qln.sum().detach())
            law_balance[right.id] -= face.weight * float(qrn.sum().detach())

        boundary: list[dict[str, Any]] = []
        for piece in problem.pieces(n_boundary):
            xy = cls._tensor(model, piece.xy)
            normal = cls._tensor(model, piece.normal)
            region = problem.regions[piece.region]
            t_pred, _q, derived, _ = flux_fields(model, region, xy)
            dn = (derived * normal).sum(1, keepdim=True)
            bc = piece.bc
            scale = max(problem.scales(region)["flux"], 1e-12)
            if bc["type"] == "dirichlet":
                err = (t_pred - bc["value"](xy)).detach().cpu().numpy()
                flux_out = dn
            elif bc["type"] == "neumann":
                err = ((dn - bc["value"](xy)) / scale).detach().cpu().numpy()
                flux_out = dn
            else:
                robin = bc["h"] * (t_pred - bc["T_inf"](xy))
                err = ((dn - robin) / scale).detach().cpu().numpy()
                flux_out = robin
            boundary.append(
                {
                    "side": piece.side,
                    "region": names[region.id],
                    "type": bc["type"],
                    "rms": float(np.sqrt(np.mean(err ** 2))),
                    "unit": "K" if bc["type"] == "dirichlet" else "flux_scale",
                }
            )
            derived_balance[region.id] += piece.weight * float(dn.sum().detach())
            law_balance[region.id] += piece.weight * float(flux_out.sum().detach())

        p_scale = max(problem.power_scale, 1e-12)
        balance = {
            "region_derived_pct": {names[i]: float(100.0 * abs(v) / p_scale) for i, v in derived_balance.items()},
            "region_constitutive_pct": {names[i]: float(100.0 * abs(v) / p_scale) for i, v in law_balance.items()},
            "global_derived_pct": float(100.0 * abs(sum(derived_balance.values())) / p_scale),
            "global_constitutive_pct": float(100.0 * abs(sum(law_balance.values())) / p_scale),
            "power_scale_W_m": float(p_scale),
            "global_derived_W_m": float(abs(sum(derived_balance.values()))),
            "region_derived_W_m": {names[i]: float(abs(v)) for i, v in derived_balance.items()},
            "normalization": "integrated_source" if any(problem.region_power.values()) else "characteristic_scale",
        }
        return {
            "pde_rms": pde,
            "constitutive_rms": constitutive,
            "interfaces": interfaces,
            "boundary": boundary,
            "balance": balance,
        }


class AcceptanceGate:
    """Evalúa umbrales de aceptación sobre un reporte de auditoría y métricas."""

    @staticmethod
    def evaluate(
        report: dict[str, Any],
        criteria: dict[str, float] = CRITERIA_STANDARD,
    ) -> dict[str, Any]:
        gates: dict[str, bool] = {}
        if "nrmse_pct" not in report:
            return {"gates": {}, "criteria": dict(criteria), "accepted": None, "state": "no_reference"}

        if "nrmse_pct" in criteria:
            gates["nrmse_pct"] = bool(report["nrmse_pct"] <= criteria["nrmse_pct"])
        if "Tmax_rise_pct" in criteria:
            gates["Tmax_rise_pct"] = bool(report["error_rise_site_pct"] <= criteria["Tmax_rise_pct"])
        if "error_Tmax_abs_K" in criteria:
            gates["error_Tmax_abs_K"] = bool(report["error_Tmax_abs_K"] <= criteria["error_Tmax_abs_K"])
        if "balance_pct" in criteria and "balance" in report:
            gates["balance_pct"] = bool(report["balance"]["global_derived_pct"] <= criteria["balance_pct"])
        if "interface_T_jump_max_K" in criteria and report.get("interfaces"):
            worst_t = max(item["T_jump_max_K"] for item in report["interfaces"])
            gates["interface_T_jump_max_K"] = bool(worst_t <= criteria["interface_T_jump_max_K"])
        if "interface_flux_jump_rms_pct" in criteria and report.get("interfaces"):
            worst_q = max(item["derived_flux_jump_rms_pct"] for item in report["interfaces"])
            gates["interface_flux_jump_rms_pct"] = bool(worst_q <= criteria["interface_flux_jump_rms_pct"])

        accepted = all(gates.values()) if gates else False
        return {
            "gates": gates,
            "raw_gates": dict(gates),
            "criteria": dict(criteria),
            "accepted": accepted,
            "state": "accepted" if accepted else "rejected",
        }


def evaluate_model(
    model: BasePINNModel,
    reference: np.ndarray | None = None,
    reference_label: str = "supplied",
    include_audit: bool = True,
    criteria: dict[str, float] = CRITERIA_STANDARD,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evalúa integralmente un modelo entrenado devolviendo ``(reporte, arreglos_evaluacion)``."""
    problem = model.problem
    xy, labels, weights = problem.evaluation()
    ref_tensor = next(model.parameters())
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            ev_xy = torch.as_tensor(xy, dtype=ref_tensor.dtype, device=ref_tensor.device)
            ev_labels = torch.as_tensor(labels, dtype=torch.long, device=ref_tensor.device)
            T = model.temperature(ev_xy, ev_labels).detach().cpu().numpy().ravel()
    finally:
        if was_training:
            model.train()

    truth = problem.reference(xy, labels) if reference is None else np.asarray(reference, dtype=float).ravel()
    names = [r.name for r in problem.regions]
    report: dict[str, Any] = dict(PhysicsAuditor.audit(model)) if include_audit else {}
    arrays: dict[str, np.ndarray] = {"xy": xy, "labels": labels, "weights": weights, "T": T}

    if truth is not None:
        report.update(
            ErrorMetrics.compute(
                T,
                truth,
                xy,
                labels,
                weights,
                names,
                problem.T_level,
                problem.conductors(),
                temperature_scale=problem.T_scale,
            )
        )
        arrays["T_reference"] = truth
    else:
        hot = np.isin(labels, problem.conductors())
        report["Tmax_pred_C"] = float(T[hot].max()) if np.any(hot) else float(T.max())

    report["acceptance"] = AcceptanceGate.evaluate(report, criteria=criteria) if include_audit else {"state": "not_audited", "gates": {}}
    report["reference_label"] = problem.reference_label if reference is None else reference_label
    return report, arrays
