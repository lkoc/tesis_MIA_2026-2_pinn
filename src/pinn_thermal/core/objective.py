"""Función objetivo física adimensional compartida por todos los modelos PINN.

Evalúa cada término normalizado por las escalas físicas propias de cada subregión:
- ``pde``: residuo ``-div(k grad T) - Q`` en cada región (o ``div(q) - Q`` en M2).
- ``constitutive``: ley de Fourier ``q + k grad T = 0`` (solo en formulaciones mixtas).
- ``interface_temperature``: continuidad de temperatura ``[T] = 0`` entre regiones vecinas.
- ``dirichlet``, ``neumann``, ``robin``: condiciones de contorno externas.
- ``energy``: balance energético regional entre potencia generada y flujos de borde.

Ningún término lee soluciones FEM ni referencias analíticas.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from pinn_thermal.core.attention import BaseAttention, build_attention
from pinn_thermal.core.base_model import BasePINNModel
from pinn_thermal.core.physics import derived_flux, gradients, heat_residual, mixed_residual
from pinn_thermal.core.problem import BoundaryPiece, InterfaceCloud, RegionSpec

DEFAULT_WEIGHTS: dict[str, float] = {
    "pde": 1.0,
    "constitutive": 1.0,
    "interface_temperature": 100.0,
    "dirichlet": 100.0,
    "neumann": 10.0,
    "robin": 10.0,
    "energy": 0.0,
}
LOSS_TERMS: tuple[str, ...] = tuple(DEFAULT_WEIGHTS)
FOCUS_CRITERIA: tuple[str, ...] = ("residual", "gradient", "mixed")


def flux_fields(
    model: BasePINNModel,
    region: RegionSpec,
    xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Devuelve ``(T, flujo_primario, flujo_derivado_Fourier, k)`` en ``xy``."""
    T, predicted = model.field(region, xy)
    k = model.problem.conductivity(region, xy)
    derived = derived_flux(T, xy, k)
    return T, (derived if predicted is None else predicted), derived, k


class PhysicsObjective:
    """Evaluador adimensional de residuos físicos con soporte para muestreo focalizado y currículum."""

    def __init__(
        self,
        model: BasePINNModel,
        n_interior: int = 512,
        n_layer: int = 256,
        n_face: int = 128,
        n_boundary: int = 128,
        seed: int = 11,
        sampling: str = "mixed",
        weights: dict[str, float] | None = None,
        attention: str | None = None,
        exact_energy: bool = False,
        focused: bool = False,
        focus_criterion: str = "residual",
        focus_surfaces: bool = True,
        initial_fraction: float | None = None,
    ) -> None:
        self.model = model
        self.problem = problem = model.problem
        self.seed = int(seed)
        self.sampling = sampling
        self.focus_criterion = focus_criterion
        self.focus_surfaces = bool(focus_surfaces)
        if focus_criterion not in FOCUS_CRITERIA:
            raise ValueError(f"focus_criterion debe ser uno de {FOCUS_CRITERIA}.")

        self.counts = {r.id: (int(n_interior) if r.kind == "soil" else int(n_layer)) for r in problem.regions}
        if initial_fraction is not None and (not np.isfinite(initial_fraction) or not 0.0 < initial_fraction <= 1.0):
            raise ValueError("initial_fraction debe ser finito y estar en (0, 1].")
        if initial_fraction is not None and attention is not None:
            raise ValueError("El crecimiento progresivo de puntos no es compatible con atención adaptativa.")

        self.face_count = int(n_face)
        self.boundary_count = int(n_boundary)
        self.weights = dict(DEFAULT_WEIGHTS, **(weights or {}))
        unknown = set(self.weights) - set(LOSS_TERMS)
        if unknown or any(not np.isfinite(v) or v < 0 for v in self.weights.values()):
            raise ValueError(f"Pesos inválidos en función objetivo: {sorted(unknown)}.")

        self.exact_constraints = bool(getattr(model, "exact_constraints", False))
        self.exact_energy = bool(exact_energy) and self.exact_constraints
        if self.exact_constraints:
            if self.weights["pde"] <= 0:
                raise ValueError("Los modelos exactos requieren peso positivo en el término 'pde'.")
            kept = ("pde", "energy") if self.exact_energy else ("pde",)
            self.weights = {name: (self.weights[name] if name in kept else 0.0) for name in LOSS_TERMS}

        self.focused = bool(focused)
        ref = next(model.parameters())
        self.tensor = lambda arr: torch.as_tensor(np.asarray(arr), dtype=ref.dtype, device=ref.device).clone().detach().requires_grad_(True)

        init_counts = {
            r.id: (
                self.counts[r.id]
                if initial_fraction is None
                else max(1, int(np.ceil(self.counts[r.id] * initial_fraction)))
            )
            for r in problem.regions
        }
        self.interior: dict[int, torch.Tensor] = {
            r.id: self.tensor(
                self._focused_points(r, seed + 7919 * r.id, init_counts[r.id])
                if self.focused
                else problem.sample_interior(r, init_counts[r.id], seed + 7919 * r.id, sampling)
            )
            for r in problem.regions
        }
        self.sources: dict[int, torch.Tensor] = {
            r.id: problem.source(r, self.interior[r.id]).detach() for r in problem.regions
        }
        self.faces: list[tuple[InterfaceCloud, torch.Tensor, torch.Tensor, torch.Tensor]] = (
            self._surface_clouds("face", n_face, seed + 104729)
            if self.focused and self.focus_surfaces
            else [
                self._surface_entry("face", f, f.xy, f.normal, self._right_xy(f))
                for f in problem.faces(n_face)
            ]
        )
        self.pieces: list[tuple[BoundaryPiece, torch.Tensor, torch.Tensor]] = (
            self._surface_clouds("piece", n_boundary, seed + 209759)
            if self.focused and self.focus_surfaces
            else [(p, self.tensor(p.xy), self.tensor(p.normal)) for p in problem.pieces(n_boundary)]
        )
        self.attention: BaseAttention | None = (
            None
            if attention is None
            else build_attention(
                attention,
                {r.name: len(self.interior[r.id]) for r in problem.regions},
                dtype=ref.dtype,
                device=ref.device,
            )
        )
        self.residuals: dict[str, torch.Tensor] = {}
        self.resampling_count: int = 0
        self._full_interior = self.interior
        self._full_sources = self.sources
        self.point_fraction: float = 1.0
        self.optimized_pde: torch.Tensor = torch.zeros((), dtype=ref.dtype, device=ref.device)

    @staticmethod
    def _right_xy(face: InterfaceCloud) -> np.ndarray:
        return face.xy if face.xy_right is None else face.xy_right

    def _surface_entry(
        self,
        kind: str,
        surface: Any,
        xy: np.ndarray,
        normal: np.ndarray,
        xy_right: np.ndarray | None = None,
    ) -> Any:
        entry = (surface, self.tensor(xy), self.tensor(normal))
        if kind != "face":
            return entry
        return entry + (self.tensor(self._right_xy(surface) if xy_right is None else xy_right),)

    @staticmethod
    def _active_indices(point_count: int, fraction: float, focused: bool, device: torch.device) -> torch.Tensor:
        if point_count == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        active = max(1, min(point_count, int(np.ceil(point_count * fraction))))
        if not focused:
            return torch.arange(active, device=device)
        focused_count = point_count // 2
        active_focused = min(focused_count, (active + 1) // 2)
        active_uniform = active - active_focused
        indices = [torch.arange(active_focused, device=device)]
        if active_uniform:
            indices.append(torch.arange(focused_count, focused_count + active_uniform, device=device))
        return torch.cat(indices)

    def set_point_fraction(self, fraction: float) -> None:
        """Activa un subconjunto anidado de puntos interiores conservando superficies completas."""
        frac = float(fraction)
        if not np.isfinite(frac) or not 0.0 < frac <= 1.0:
            raise ValueError("point_fraction debe ser finito y estar en (0, 1].")
        self.point_fraction = frac
        self.interior, self.sources = {}, {}
        for region in self.problem.regions:
            full_xy = self._full_interior[region.id]
            idx = self._active_indices(len(full_xy), frac, self.focused, full_xy.device)
            self.interior[region.id] = full_xy.index_select(0, idx).detach().clone().requires_grad_(True)
            self.sources[region.id] = self._full_sources[region.id].index_select(0, idx).detach().clone()

    def grow_interior(self, fraction: float, stage: int) -> None:
        """Agrega progresivamente nuevos puntos interiores hasta alcanzar la fracción deseada."""
        if self.attention is not None:
            raise ValueError("El crecimiento incremental no soporta atención adaptativa.")
        frac = float(fraction)
        if not np.isfinite(frac) or not 0.0 < frac <= 1.0:
            raise ValueError("point_fraction debe ser finito y estar en (0, 1].")
        for region in self.problem.regions:
            target = max(1, int(np.ceil(self.counts[region.id] * frac)))
            missing = target - len(self._full_interior[region.id])
            if missing <= 0:
                continue
            seed = self.seed + 7919 * region.id + 1_000_003 * (int(stage) + 1)
            new_pts = (
                self._focused_points(region, seed, missing)
                if self.focused
                else self.problem.sample_interior(region, missing, seed, self.sampling)
            )
            pts = self.tensor(np.vstack((self._full_interior[region.id].detach().cpu().numpy(), new_pts)))
            self._full_interior[region.id] = pts
            self._full_sources[region.id] = self.problem.source(region, pts).detach()
        self.point_fraction = 1.0
        self.interior = dict(self._full_interior)
        self.sources = dict(self._full_sources)

    def resample(self, seed: int | None = None) -> int:
        """Renueva la nube de colocación manteniendo cobertura uniforme por material."""
        frac = self.point_fraction
        resolved_seed = self.seed + (self.resampling_count + 1) * 1_000_003 if seed is None else int(seed)
        if self.focused:
            self.interior = {
                r.id: self.tensor(self._focused_points(r, resolved_seed + 7919 * r.id)) for r in self.problem.regions
            }
        else:
            self.interior = {
                r.id: self.tensor(self.problem.sample_interior(r, self.counts[r.id], resolved_seed + 7919 * r.id, self.sampling))
                for r in self.problem.regions
            }
        self.sources = {r.id: self.problem.source(r, self.interior[r.id]).detach() for r in self.problem.regions}
        if self.focused and self.focus_surfaces:
            self.faces = self._surface_clouds("face", self.face_count, resolved_seed + 104729)
            self.pieces = self._surface_clouds("piece", self.boundary_count, resolved_seed + 209759)
        if self.attention is not None:
            ref = next(self.model.parameters())
            self.attention = build_attention(
                self.attention.kind,
                {r.name: self.counts[r.id] for r in self.problem.regions},
                dtype=ref.dtype,
                device=ref.device,
                rate=self.attention.rate,
                decay=self.attention.decay,
            )
        self.residuals = {}
        self._full_interior = self.interior
        self._full_sources = self.sources
        self.set_point_fraction(frac)
        self.resampling_count += 1
        return resolved_seed

    def _surface_clouds(self, kind: str, count: int, seed: int) -> list[Any]:
        ref = next(self.model.parameters())
        if kind == "face":
            uniform = self.problem.faces(count)
            candidates = {f.name: f for f in self.problem.faces(max(2, 16 * count))}
        else:
            uniform = self.problem.pieces(count)
            candidates = dict(enumerate(self.problem.pieces(max(2, 16 * count))))
        result = []
        for idx, base in enumerate(uniform):
            group = candidates.get(base.name if kind == "face" else idx)
            if group is None or len(base.xy) < 2:
                result.append(self._surface_entry(kind, base, base.xy, base.normal))
                continue
            xy = torch.as_tensor(group.xy, dtype=ref.dtype, device=ref.device).requires_grad_(True)
            normal = torch.as_tensor(group.normal, dtype=ref.dtype, device=ref.device)
            xy_right = None
            if kind == "face":
                xy_right = torch.as_tensor(self._right_xy(group), dtype=ref.dtype, device=ref.device).requires_grad_(True)
            score = self._surface_score(kind, base, xy, normal, xy_right).detach().cpu().numpy()
            focused_count = min(len(base.xy) // 2, len(group.xy))
            selected = np.argsort(-score)[:focused_count]
            uniform_count = len(base.xy) - focused_count
            sel_xy = np.vstack([group.xy[selected], base.xy[:uniform_count]])
            sel_normal = np.vstack([group.normal[selected], base.normal[:uniform_count]])
            if kind == "face":
                sel_right = np.vstack([self._right_xy(group)[selected], self._right_xy(base)[:uniform_count]])
                surf = replace(base, xy=sel_xy, normal=sel_normal, xy_right=sel_right)
                result.append(self._surface_entry(kind, surf, sel_xy, sel_normal, sel_right))
            else:
                surf = replace(base, xy=sel_xy, normal=sel_normal)
                result.append(self._surface_entry(kind, surf, sel_xy, sel_normal))
        return result

    def _surface_score(
        self,
        kind: str,
        surface: Any,
        xy: torch.Tensor,
        normal: torch.Tensor,
        xy_right: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kind == "face":
            left, right = self.problem.regions[surface.left], self.problem.regions[surface.right]
            xy_r = xy if xy_right is None else xy_right
            tl, ql, _, _ = flux_fields(self.model, left, xy)
            tr, qr, _, _ = flux_fields(self.model, right, xy_r)
            t_err = (tl - tr).abs() / max(self.problem.T_scale, 1e-12)
            f_scale = max(self.problem.scales(left)["flux"], self.problem.scales(right)["flux"], 1e-12)
            q_err = ((ql * normal).sum(1, keepdim=True) - (qr * normal).sum(1, keepdim=True)).abs() / f_scale
            residual = (t_err + q_err).flatten()
            g_l = torch.linalg.vector_norm(gradients(tl, xy)[:, :2], dim=1) / max(
                self.problem.scales(left)["temperature"] / max(np.sqrt(self.problem.area(left)), 1e-12), 1e-12
            )
            g_r = torch.linalg.vector_norm(gradients(tr, xy_r)[:, :2], dim=1) / max(
                self.problem.scales(right)["temperature"] / max(np.sqrt(self.problem.area(right)), 1e-12), 1e-12
            )
            grad_score = 0.5 * (g_l + g_r)
        else:
            region = self.problem.regions[surface.region]
            t_pred, q_pred, _, _ = flux_fields(self.model, region, xy)
            qn = (q_pred * normal).sum(1, keepdim=True)
            bc = surface.bc
            scale = max(self.problem.scales(region)["flux"], 1e-12)
            if bc["type"] == "dirichlet":
                residual = ((t_pred - bc["value"](xy)).abs() / max(self.problem.T_scale, 1e-12)).flatten()
            elif bc["type"] == "neumann":
                residual = ((qn - bc["value"](xy)).abs() / scale).flatten()
            else:
                residual = ((qn - bc["h"] * (t_pred - bc["T_inf"](xy))).abs() / scale).flatten()
            g_scale = max(
                self.problem.scales(region)["temperature"] / max(np.sqrt(self.problem.area(region)), 1e-12), 1e-12
            )
            grad_score = torch.linalg.vector_norm(gradients(t_pred, xy)[:, :2], dim=1) / g_scale

        if self.focus_criterion == "residual":
            return residual
        if self.focus_criterion == "gradient":
            return grad_score
        return 0.5 * (
            residual / residual.mean().clamp_min(1e-12) + grad_score / grad_score.mean().clamp_min(1e-12)
        )

    def _importance_score(self, region: RegionSpec, xy: torch.Tensor) -> torch.Tensor:
        T, _ = self.model.field(region, xy)
        scales = self.problem.scales(region)
        res = (
            heat_residual(T, xy, self.problem.conductivity(region, xy), self.problem.source(region, xy).detach())
            / scales["divergence"]
        ).detach().abs().flatten()
        if self.focus_criterion == "residual":
            return res
        grad = gradients(T, xy)[:, :2]
        g_scale = scales["temperature"] / max(np.sqrt(self.problem.area(region)), 1e-12)
        g_score = (torch.linalg.vector_norm(grad, dim=1) / g_scale).detach().abs()
        if self.focus_criterion == "gradient":
            return g_score
        return 0.5 * (res / res.mean().clamp_min(1e-12) + g_score / g_score.mean().clamp_min(1e-12))

    def _focused_points(self, region: RegionSpec, seed: int, count: int | None = None) -> np.ndarray:
        n = self.counts[region.id] if count is None else int(count)
        pool = np.asarray(self.problem.focused_interior_pool(region, 16 * n, seed, self.sampling))
        ref = next(self.model.parameters())
        xy = torch.as_tensor(pool, dtype=ref.dtype, device=ref.device).requires_grad_(True)
        score = self._importance_score(region, xy).cpu().numpy()
        important = pool[np.argsort(-score)[: n // 2]]
        fresh = np.asarray(self.problem.sample_interior(region, n - len(important), seed + 1, self.sampling))
        return np.vstack([important, fresh])

    def update_attention(self) -> None:
        if self.attention is not None:
            for name, r2 in self.residuals.items():
                self.attention.update(name, r2)

    def state_dict(self) -> dict[str, Any]:
        """Serializa nubes interiores, fuentes y estado de atención para reanudación exacta."""
        return {
            "interior": {k: v.detach().cpu().clone() for k, v in self._full_interior.items()},
            "sources": {k: v.detach().cpu().clone() for k, v in self._full_sources.items()},
            "point_fraction": float(self.point_fraction),
            "resampling_count": int(self.resampling_count),
            "attention": None if self.attention is None else self.attention.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restaura nubes interiores y estado de atención."""
        ref = next(self.model.parameters())
        self._full_interior = {
            k: v.to(ref).detach().clone().requires_grad_(True) for k, v in state["interior"].items()
        }
        self._full_sources = {k: v.to(ref).detach().clone() for k, v in state["sources"].items()}
        self.resampling_count = int(state["resampling_count"])
        if (self.attention is None) != (state["attention"] is None):
            raise ValueError("Incompatibilidad de atención adaptativa al restaurar el objetivo.")
        if self.attention is not None and state["attention"] is not None:
            self.attention.load_state_dict(state["attention"], ref_tensor=ref)
        self.set_point_fraction(state.get("point_fraction", 1.0))
        self.residuals = {}

    def __call__(self) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        model, problem = self.model, self.problem
        zero = next(model.parameters()).new_zeros(())
        parts: dict[str, torch.Tensor] = {name: zero for name in LOSS_TERMS}
        region_pde: dict[str, torch.Tensor] = {}
        weighted_pde = zero
        balance: dict[int, torch.Tensor | float] = {r.id: -problem.region_power[r.id] for r in problem.regions}
        constitutive = zero

        for region in problem.regions:
            xy = self.interior[region.id]
            T, predicted = model.field(region, xy)
            k = problem.conductivity(region, xy)
            scales = problem.scales(region)
            if model.mixed:
                if predicted is None:
                    raise ValueError(f"El modelo mixto {model.architecture!r} devolvió flujo None.")
                residual, law = mixed_residual(T, predicted, xy, k, self.sources[region.id])
                constitutive = constitutive + torch.mean((law / scales["flux"]) ** 2)
            else:
                residual = heat_residual(T, xy, k, self.sources[region.id])
            r2 = (residual / scales["divergence"]) ** 2
            region_pde[region.name] = torch.mean(r2)
            parts["pde"] = parts["pde"] + region_pde[region.name]
            if self.attention is not None:
                self.residuals[region.name] = r2.detach()
                weighted_pde = weighted_pde + torch.mean(self.attention.weight(region.name) * r2)

        skip_pieces = self.exact_constraints and not self.exact_energy
        for face, xy, normal, xy_right in self.faces:
            left, right = problem.regions[face.left], problem.regions[face.right]
            tl, ql, dl, _ = flux_fields(model, left, xy)
            tr, qr, dr, _ = flux_fields(model, right, xy_right)
            if model.mixed:
                constitutive = constitutive + torch.mean(((ql - dl) / problem.scales(left)["flux"]) ** 2)
                constitutive = constitutive + torch.mean(((qr - dr) / problem.scales(right)["flux"]) ** 2)
            qln = (ql * normal).sum(1, keepdim=True)
            qrn = (qr * normal).sum(1, keepdim=True)
            parts["interface_temperature"] = parts["interface_temperature"] + torch.mean(((tl - tr) / problem.T_scale) ** 2)
            balance[left.id] = balance[left.id] + face.weight * qln.sum()
            balance[right.id] = balance[right.id] - face.weight * qrn.sum()

        for piece, xy, normal in (() if skip_pieces else self.pieces):
            region = problem.regions[piece.region]
            t_pred, q_pred, derived, _ = flux_fields(model, region, xy)
            scale = problem.scales(region)["flux"]
            if model.mixed:
                constitutive = constitutive + torch.mean(((q_pred - derived) / scale) ** 2)
            qn = (q_pred * normal).sum(1, keepdim=True)
            bc = piece.bc
            if getattr(model, "exact_boundary", False):
                pass
            elif bc["type"] == "dirichlet":
                parts["dirichlet"] = parts["dirichlet"] + torch.mean(((t_pred - bc["value"](xy)) / problem.T_scale) ** 2)
            elif bc["type"] == "neumann":
                parts["neumann"] = parts["neumann"] + torch.mean(((qn - bc["value"](xy)) / scale) ** 2)
            else:
                parts["robin"] = parts["robin"] + torch.mean(((qn - bc["h"] * (t_pred - bc["T_inf"](xy))) / scale) ** 2)
            balance[region.id] = balance[region.id] + piece.weight * qn.sum()

        parts["constitutive"] = constitutive
        if self.exact_constraints and not self.exact_energy:
            parts["energy"] = zero
            total = sum(self.weights[name] * parts[name] for name in ("pde", "constitutive", "interface_temperature"))
        else:
            parts["energy"] = sum((value / problem.power_scale) ** 2 for value in balance.values())
            if self.exact_constraints:
                total = sum(self.weights[name] * parts[name] for name in ("pde", "energy", "constitutive", "interface_temperature"))
            else:
                total = sum(self.weights[name] * value for name, value in parts.items())

        if self.attention is not None:
            # Conserva el valor escalar comparable de PDE, aplicando gradiente ponderado por atención
            total = total + self.weights["pde"] * (
                (weighted_pde - weighted_pde.detach()) - (parts["pde"] - parts["pde"].detach())
            )
        self.optimized_pde = (weighted_pde if self.attention is not None else parts["pde"]).detach()
        return total, parts, region_pde
