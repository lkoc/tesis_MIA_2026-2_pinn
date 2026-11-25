"""Arquitecturas PINN con imposición analítica exacta de fronteras e interfaces.

Incluye tres familias complementarias inspiradas en formulaciones por funciones de
distancia (McFall & Mahan, 2009; Sheng & Yang PFNN, 2021) y parametrización exacta
de trazas de interfaz:
1. ``M1_EXACT`` (``ExactInterfaceCablePINN`` / ``ExactBoundaryPINN``): continuidad C^0
   analítica sobre cada capa concéntrica del cable y Dirichlet exacto en fronteras externas.
2. ``M0_EXACT_BC``, ``M1_EXACT_BC``, ``DCSNN_EXACT_BC``: imposición exacta de condiciones
   Dirichlet conservando entrenamiento débil sobre interfaces internas y bordes Neumann/Robin.
3. ``M0_EXACT``, ``CUSP_EXACT``, ``DCSNN_EXACT``, ``FBPINN_EXACT``: envoltorio exacto de
   frontera aplicado sobre arquitecturas continuas o enriquecidas existentes.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pinn_thermal.core.base_model import BasePINNModel, ModelMetadata, ModelRegistry, build_mlp
from pinn_thermal.core.fields import evaluate_field_torch
from pinn_thermal.core.problem import CableLayer, RegionSpec, ThermalProblem2D
from pinn_thermal.models.baseline import M0GlobalPINN, M1SubdomainPINN
from pinn_thermal.models.cusp import CuspPINN
from pinn_thermal.models.dcsnn import DCSNNPINN
from pinn_thermal.models.fbpinn import FBPINNPINN


def _dirichlet_sides(problem: ThermalProblem2D) -> list[str]:
    return [side for side, cond in problem.doc["boundaries"].items() if cond["type"] == "dirichlet"]


def _all_boundaries_dirichlet(problem: ThermalProblem2D) -> bool:
    bounds = problem.doc["boundaries"]
    return all(cond["type"] == "dirichlet" for cond in bounds.values())


def _boundary_lift_and_distance(problem: ThermalProblem2D, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcula ``(g_D(x, y), d_D(x, y))`` donde ``d_D = 0`` exactamente sobre bordes Dirichlet.

    Soporta dominios rectangulares y anulares con cualquier combinación de bordes Dirichlet
    totales o parciales (p. ej. con otros lados Neumann o Robin).
    """
    d_sides = _dirichlet_sides(problem)
    if not d_sides:
        return xy.new_full((len(xy), 1), float(problem.T_ref)), xy.new_ones((len(xy), 1))

    bounds = problem.doc["boundaries"]
    if problem.domain_shape == "annulus":
        dom = problem.doc["domain"]
        center = xy.new_tensor(dom["center"])
        ri = float(dom["inner_radius_m"])
        ro = float(dom["outer_radius_m"])
        ri_safe = max(ri, 1e-12)
        log_ratio = max(math.log(max(ro, ri_safe * 1.000001) / ri_safe), 1e-12)
        r = torch.linalg.vector_norm(xy[:, :2] - center, dim=1, keepdim=True).clamp_min(1e-12)
        frac = torch.log(r / ri_safe) / log_ratio
        has_in = "inner" in d_sides
        has_out = "outer" in d_sides
        if has_in and has_out:
            t_in = evaluate_field_torch(bounds["inner"]["value"], xy[:, 0:1], xy[:, 1:2])
            t_out = evaluate_field_torch(bounds["outer"]["value"], xy[:, 0:1], xy[:, 1:2])
            g_d = (1.0 - frac) * t_in + frac * t_out
            d_d = 4.0 * frac * (1.0 - frac)
            return g_d, d_d
        if has_in:
            t_in = evaluate_field_torch(bounds["inner"]["value"], xy[:, 0:1], xy[:, 1:2])
            return t_in, frac
        t_out = evaluate_field_torch(bounds["outer"]["value"], xy[:, 0:1], xy[:, 1:2])
        return t_out, 1.0 - frac

    # Dominio rectangular: distancia normalizada respecto a cada borde Dirichlet activo
    x0, x1, y0, y1 = problem.bounds
    dx = max(x1 - x0, 1e-12)
    dy = max(y1 - y0, 1e-12)
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    phi: dict[str, torch.Tensor] = {
        "left": (x - x0) / dx,
        "right": (x1 - x) / dx,
        "bottom": (y - y0) / dy,
        "top": (y1 - y) / dy,
    }
    vals: dict[str, torch.Tensor] = {
        side: evaluate_field_torch(bounds[side]["value"], x, y) for side in d_sides
    }

    if len(d_sides) == 1:
        s0 = d_sides[0]
        return vals[s0], phi[s0]

    # Interpolación inversa normalizada sobre las distancias phi_s: en phi_s = 0 vale exactamente vals[s]
    eps = 1e-18
    d_d = xy.new_ones((len(xy), 1))
    inv_weights = []
    for s in d_sides:
        d_d = d_d * phi[s]
        others = xy.new_ones((len(xy), 1))
        for other in d_sides:
            if other != s:
                others = others * (phi[other] ** 2)
        inv_weights.append(others)

    denom = sum(inv_weights) + eps
    g_d = sum(w * vals[s] for w, s in zip(inv_weights, d_sides)) / denom
    # Escala O(1) en el centro del rectángulo según ejes restringidos
    scale = (4.0 if ("left" in d_sides and "right" in d_sides) else 1.0) * (
        4.0 if ("bottom" in d_sides and "top" in d_sides) else 1.0
    )
    return g_d, d_d * scale


def _side_owner_regions(problem: ThermalProblem2D) -> dict[str, tuple[int, ...]]:
    """Mapea cada frontera declarada del dominio a los IDs de las regiones en contacto."""
    owners: dict[str, list[int]] = {}
    for seg in problem.sample_boundary_regions(4, rows=1):
        side = str(seg["side"])
        reg_id = int(seg["region"])
        lst = owners.setdefault(side, [])
        if reg_id not in lst:
            lst.append(reg_id)
    return {side: tuple(ids) for side, ids in owners.items()}


def _boundary_conductivity(
    problem: ThermalProblem2D,
    region: RegionSpec,
    side: str,
    xy_p: torch.Tensor,
    side_owners: dict[str, tuple[int, ...]],
) -> torch.Tensor:
    """Conductividad de la región propietaria del borde (preserva [T]=0 en ansatz globales)."""
    ids = side_owners.get(side, ())
    if not ids or region.id in ids:
        return problem.conductivity(region, xy_p)
    return problem.conductivity(problem.regions[ids[0]], xy_p)


def _apply_exact_natural_bcs(
    problem: ThermalProblem2D,
    region: RegionSpec,
    xy: torch.Tensor,
    t0_fn,
    side_owners: dict[str, tuple[int, ...]],
    interior_zero_factor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Impone analíticamente condiciones Neumann y Robin sobre un campo base ``t0_fn`` Dirichlet-exacto.

    Aplica extensión de Hermite de primer orden localizado con distancia acotada
    ``psi = l_s * tanh(phi_s / l_s)`` y ponderación racional tipo Shepard
    (Sukumar & Srivastava, 2022), asegurando exactitud simultánea en bordes Dirichlet,
    Neumann y Robin sin requerir términos de penalización en la pérdida.
    """
    bounds = problem.doc["boundaries"]
    d_sides = [s for s, c in bounds.items() if c["type"] == "dirichlet"]
    m_sides = [s for s, c in bounds.items() if c["type"] in ("neumann", "robin")]
    t0 = t0_fn(xy)
    if not m_sides:
        return t0

    is_annulus = problem.domain_shape == "annulus"
    if is_annulus:
        dom = problem.doc["domain"]
        center = xy.new_tensor(dom["center"])
        ri, ro = float(dom["inner_radius_m"]), float(dom["outer_radius_m"])
        ri_safe = max(ri, 1e-12)
        ro_safe = max(ro, ri_safe * 1.000001)
        span = max(ro_safe - ri_safe, 1e-12)
        log_ratio = max(math.log(ro_safe / ri_safe), 1e-12)
        delta = xy[:, :2] - center
        r = torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp_min(1e-12)
        unit = delta / r
        eta = torch.log(r / ri_safe) / log_ratio
        phi_norm = {"inner": eta, "outer": 1.0 - eta}
        scale_out = max(min(ro_safe, 0.25 * span), 1e-12)
        psi_map = {
            "inner": ri_safe * torch.log(r / ri_safe),
            "outer": scale_out * torch.tanh((ro_safe * torch.log(ro_safe / r)) / scale_out),
        }
        w_loc_map: dict[str, torch.Tensor] | None = None
        proj = {"inner": center + ri * unit, "outer": center + ro * unit}
    else:
        x0, x1, y0, y1 = problem.bounds
        dx, dy = max(x1 - x0, 1e-12), max(y1 - y0, 1e-12)
        x, y = xy[:, 0:1], xy[:, 1:2]
        phi_phys = {"left": x - x0, "right": x1 - x, "bottom": y - y0, "top": y1 - y}
        phi_norm = {"left": (x - x0) / dx, "right": (x1 - x) / dx, "bottom": (y - y0) / dy, "top": (y1 - y) / dy}
        l_loc = {"left": 0.25 * dx, "right": 0.25 * dx, "bottom": 0.25 * dy, "top": 0.25 * dy}
        psi_map = {
            s: max(l_loc[s], 1e-12) * torch.tanh(phi_phys[s] / max(l_loc[s], 1e-12))
            for s in m_sides
        }
        w_loc_map = {
            s: torch.exp(-((phi_phys[s] / max(l_loc[s], 1e-12)) ** 2))
            for s in m_sides
        }
        proj = {
            "left": torch.cat([xy.new_full((len(xy), 1), x0), y], dim=1),
            "right": torch.cat([xy.new_full((len(xy), 1), x1), y], dim=1),
            "bottom": torch.cat([x, xy.new_full((len(xy), 1), y0)], dim=1),
            "top": torch.cat([x, xy.new_full((len(xy), 1), y1)], dim=1),
        }

    d_d2 = xy.new_ones((len(xy), 1))
    for s in d_sides:
        d_d2 = d_d2 * (phi_norm[s] ** 2)
    if interior_zero_factor is not None:
        d_d2 = d_d2 * interior_zero_factor

    jets: list[torch.Tensor] = []
    p_sides: list[torch.Tensor] = []
    for s in m_sides:
        bc = bounds[s]
        xy_p = proj[s]
        t_s = t0_fn(xy_p)
        k_s = _boundary_conductivity(problem, region, s, xy_p, side_owners).clamp_min(1e-12)
        if bc["type"] == "neumann":
            qn = evaluate_field_torch(bc["q_n_W_m2"], xy_p[:, 0:1], xy_p[:, 1:2])
            slope = qn / k_s
        else:
            h = float(bc["h_W_m2K"])
            t_inf = evaluate_field_torch(bc["T_inf_C"], xy_p[:, 0:1], xy_p[:, 1:2])
            slope = (h / k_s) * (t_s - t_inf)
        jets.append(t_s + psi_map[s] * slope)

        p = d_d2 if w_loc_map is None else (d_d2 * w_loc_map[s])
        for other in m_sides:
            if other != s:
                p = p * (phi_norm[other] ** 2)
        p_sides.append(p)

    p_int = xy.new_ones((len(xy), 1))
    for s in m_sides:
        p_int = p_int * (phi_norm[s] ** 2)

    denom = (p_int + sum(p_sides)).clamp_min(1e-24)
    num = p_int * t0 + sum(p * jet for p, jet in zip(p_sides, jets))
    return num / denom


def _harmonic_trace_basis(
    xy: torch.Tensor,
    center: tuple[float, float],
    outer_radius: float,
    harmonics: int,
) -> torch.Tensor:
    """Base armónica polinomial ``(r/R)^k cos(k theta), (r/R)^k sin(k theta)`` no singular en ``r = 0``."""
    dx = (xy[:, 0:1] - float(center[0])) / max(float(outer_radius), 1e-12)
    dy = (xy[:, 1:2] - float(center[1])) / max(float(outer_radius), 1e-12)
    cols = [xy.new_ones((len(xy), 1))]
    # Recurrencia compleja z^k = (dx + i dy) * z^{k-1}: C^inf en el origen sin división por r
    c_k = xy.new_ones((len(xy), 1))
    s_k = xy.new_zeros((len(xy), 1))
    for _ in range(harmonics):
        c_next = c_k * dx - s_k * dy
        s_next = c_k * dy + s_k * dx
        c_k, s_k = c_next, s_next
        cols.extend([c_k, s_k])
    return torch.cat(cols, dim=1)


def _angular_trace_basis(xy: torch.Tensor, center: tuple[float, float], harmonics: int) -> torch.Tensor:
    """Base trigonométrica en la superficie ``theta = atan2(y - cy, x - cx)`` para trazas exteriores."""
    dx = xy[:, 0:1] - float(center[0])
    dy = xy[:, 1:2] - float(center[1])
    radius = torch.linalg.vector_norm(torch.cat([dx, dy], dim=1), dim=1, keepdim=True).clamp_min(1e-12)
    cos_t = dx / radius
    sin_t = dy / radius
    cols = [xy.new_ones((len(xy), 1))]
    c_k = xy.new_ones((len(xy), 1))
    s_k = xy.new_zeros((len(xy), 1))
    for _ in range(harmonics):
        c_next = c_k * cos_t - s_k * sin_t
        s_next = c_k * sin_t + s_k * cos_t
        c_k, s_k = c_next, s_next
        cols.extend([c_k, s_k])
    return torch.cat(cols, dim=1)


class ExactInterfaceCablePINN(BasePINNModel):
    """PINN exacta para cables subterráneos concéntricos (``M1_EXACT`` cuando aplica).

    Garantiza analíticamente:
    1. Continuidad de temperatura :math:`[T] = 0` en todas las interfaces circulares
       mediante expansión armónica compartida por cada radio de interfaz.
    2. Condición de contorno Dirichlet exacta en bordes externos del suelo.
    """

    model_kind = "exact_cable_interface"

    def __init__(
        self,
        problem: ThermalProblem2D,
        architecture: str = "M1_EXACT",
        width: int = 32,
        depth: int = 3,
        activation: str = "tanh",
        input_normalization: str | None = None,
        trace_parameterization: str = "standard",
        soil_envelope_gain: float = 1.0,
        harmonics: int = 8,
        **_extra: object,
    ) -> None:
        if not problem.supports_exact_cable_interfaces:
            raise ValueError("ExactInterfaceCablePINN requiere un problema de cables con dominio rectangular único.")
        super().__init__(
            problem=problem,
            architecture=architecture,
            width=width,
            depth=depth,
            activation=activation,
            partition="region",
            formulation="strong",
            input_normalization=input_normalization or "region_coordinates",
        )
        self.trace_parameterization = str(trace_parameterization)
        self.soil_envelope_gain = float(soil_envelope_gain)
        self.harmonics = int(harmonics)
        self._side_owners = _side_owner_regions(problem)
        self.exact_boundary = True
        self.exact_constraints = True

        self.networks = nn.ModuleList()
        for region in problem.regions:
            in_features = problem.input_size(region, "region")
            self.networks.append(build_mlp(in_features, self.width, self.depth, 1, self.activation))

        # Mapa de cada región conductora/aislante -> CableLayer y tabla de trazas por interfaz circular
        self._layer_by_region: dict[int, CableLayer] = {}
        self._trace_index_by_key: dict[tuple[int, int], int] = {}
        n_traces = 0
        for cable in problem.cable_layout:
            for l_idx, layer in enumerate(cable.layers):
                self._layer_by_region[layer.region_id] = layer
                self._trace_index_by_key[(cable.index, l_idx)] = n_traces
                n_traces += 1

        n_coeffs = 1 + 2 * self.harmonics
        self.trace_coefficients = nn.Parameter(torch.zeros((n_traces, n_coeffs), dtype=torch.float32))
        self.trace_scale = max(
            [problem.T_scale, *(problem.amplitude(r, "region") for r in problem.regions if r.kind != "soil")]
        )

    def _trace_temperature(self, trace_idx: int, xy: torch.Tensor, center: tuple[float, float], radius: float, polynomial: bool) -> torch.Tensor:
        if polynomial:
            basis = _harmonic_trace_basis(xy, center, radius, self.harmonics)
        else:
            basis = _angular_trace_basis(xy, center, self.harmonics)
        coeff = self.trace_coefficients[trace_idx].to(dtype=xy.dtype, device=xy.device).reshape(-1, 1)
        return self.problem.T_ref + self.trace_scale * (basis @ coeff)

    def field(self, region: RegionSpec, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        amplitude = self.problem.amplitude(region, "region")

        if region.id in self._layer_by_region:
            features = self.input_coordinates(region, xy)
            raw = self.networks[region.id](features)
            layer = self._layer_by_region[region.id]
            cable = self.problem.cable_layout[layer.cable_index]
            l_idx = next(i for i, item in enumerate(cable.layers) if item.region_id == region.id)
            outer_trace_idx = self._trace_index_by_key[(cable.index, l_idx)]
            center = xy.new_tensor(layer.center)

            t_out = self._trace_temperature(outer_trace_idx, xy, layer.center, layer.outer_radius, polynomial=(layer.inner_radius == 0.0))
            if layer.inner_radius == 0.0:
                r2 = ((xy[:, :2] - center) ** 2).sum(dim=1, keepdim=True)
                bubble = 1.0 - r2 / (max(layer.outer_radius, 1e-12) ** 2)
                return t_out + amplitude * bubble * raw, None

            r = torch.linalg.vector_norm(xy[:, :2] - center, dim=1, keepdim=True)
            inner_trace_idx = self._trace_index_by_key[(cable.index, l_idx - 1)]
            t_in = self._trace_temperature(inner_trace_idx, xy, layer.center, layer.outer_radius, polynomial=False)
            if self.trace_parameterization == "fourier":
                # Proyección directa sobre radios reales r/R para modos angulares suaves en anillo
                t_in = self._trace_temperature(inner_trace_idx, xy, layer.center, layer.inner_radius, polynomial=False)
            span = max(layer.outer_radius - layer.inner_radius, 1e-12)
            frac = (r - layer.inner_radius) / span
            base = (1.0 - frac) * t_in + frac * t_out
            bubble = 4.0 * frac * (1.0 - frac)
            return base + amplitude * bubble * raw, None

        # Región de suelo circundante: acopla exacto a cada superficie externa de cable y bordes externos
        def _soil_dirichlet_base(pts: torch.Tensor) -> torch.Tensor:
            feats = self.input_coordinates(region, pts)
            raw_pts = self.networks[region.id](feats)
            g_d, d_d = _boundary_lift_and_distance(self.problem, pts)
            surf_dists = []
            cable_temps = []
            for cable in self.problem.cable_layout:
                c = pts.new_tensor(cable.center)
                r_c = torch.linalg.vector_norm(pts[:, :2] - c, dim=1, keepdim=True)
                s_c = r_c - cable.radius
                surf_dists.append(s_c)
                outer_idx = self._trace_index_by_key[(cable.index, len(cable.layers) - 1)]
                cable_temps.append(self._trace_temperature(outer_idx, pts, cable.center, cable.radius, polynomial=False))

            # Pesos de Shepard suaves por cable vs frontera externa (sin división por cero)
            pi_d = pts.new_ones((len(pts), 1))
            for s_c in surf_dists:
                pi_d = pi_d * (s_c ** 2)

            pi_cables = []
            for i, cable in enumerate(self.problem.cable_layout):
                w = (d_d ** 2) * torch.exp(-self.soil_envelope_gain * surf_dists[i] / max(cable.radius, 1e-12))
                for j, s_j in enumerate(surf_dists):
                    if j != i:
                        w = w * (s_j ** 2)
                pi_cables.append(w)

            denom = (pi_d + sum(pi_cables)).clamp_min(1e-24)
            base_pts = (pi_d * g_d + sum(pi_c * t_c for pi_c, t_c in zip(pi_cables, cable_temps))) / denom

            cable_bubble = pts.new_ones((len(pts), 1))
            for s_c, cable in zip(surf_dists, self.problem.cable_layout):
                cable_bubble = cable_bubble * (1.0 - torch.exp(-self.soil_envelope_gain * s_c / max(cable.radius, 1e-12)))

            return base_pts + amplitude * d_d * cable_bubble * raw_pts

        x0, x1, y0, y1 = self.problem.bounds
        dom_span = max(x1 - x0, y1 - y0, 1e-12)
        cable_zero = xy.new_ones((len(xy), 1))
        for cable in self.problem.cable_layout:
            c = xy.new_tensor(cable.center)
            r_c = torch.linalg.vector_norm(xy[:, :2] - c, dim=1, keepdim=True)
            cable_zero = cable_zero * (((r_c - cable.radius) / dom_span) ** 2)

        temp = _apply_exact_natural_bcs(
            self.problem,
            region,
            xy,
            _soil_dirichlet_base,
            self._side_owners,
            interior_zero_factor=cable_zero,
        )
        return temp, None


class ExactBoundaryPINN(BasePINNModel):
    """Envoltorio exacto de fronteras Dirichlet sobre cualquier arquitectura base."""

    model_kind = "exact_boundary"
    exact_boundary = True

    def __init__(
        self,
        problem: ThermalProblem2D,
        architecture: str,
        base_model: BasePINNModel,
    ) -> None:
        super().__init__(
            problem=problem,
            architecture=architecture,
            width=base_model.width,
            depth=base_model.depth,
            activation=base_model.activation,
            partition=base_model.partition,
            formulation=base_model.formulation,
            input_normalization=base_model.input_normalization,
        )
        self.base_model = base_model
        self._side_owners = _side_owner_regions(problem)
        self.exact_boundary = True
        # Solo elimina pérdidas de interfaz cuando es dominio de región única (sin interfaces)
        self.exact_constraints = len(problem.regions) == 1

    def field(self, region: RegionSpec, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        q_out: list[torch.Tensor | None] = [None]

        def _dirichlet_exact(pts: torch.Tensor) -> torch.Tensor:
            g_d, d_d = _boundary_lift_and_distance(self.problem, pts)
            t_raw, q_raw = self.base_model.field(region, pts)
            if pts is xy:
                q_out[0] = q_raw
            return g_d + d_d * (t_raw - self.problem.T_ref)

        temp = _apply_exact_natural_bcs(self.problem, region, xy, _dirichlet_exact, self._side_owners)
        return temp, q_out[0]

    def subnetwork_map(self) -> dict[str, nn.Module]:
        return self.base_model.subnetwork_map()

    @property
    def offsets(self) -> nn.ParameterList | None:
        return getattr(self.base_model, "offsets", None)


class ExactCorrectorPINN(ExactBoundaryPINN):
    """Variante exacta orientada a familias continuas globales (``M0_EXACT``, ``CUSP_EXACT``, etc.)."""

    model_kind = "exact_corrector"

    def __init__(
        self,
        problem: ThermalProblem2D,
        architecture: str,
        base_model: BasePINNModel,
    ) -> None:
        super().__init__(problem=problem, architecture=architecture, base_model=base_model)
        # Cuando base_model es global continuo escalar único (CUSP, M0), [T]=0 se satisface globalmente
        is_globally_continuous = architecture in ("CUSP_EXACT", "M0_EXACT")
        self.exact_constraints = is_globally_continuous


def _build_m1_exact(
    problem: ThermalProblem2D,
    architecture: str = "M1_EXACT",
    width: int = 32,
    depth: int = 3,
    activation: str = "tanh",
    input_normalization: str | None = None,
    trace_parameterization: str = "standard",
    soil_envelope_gain: float = 1.0,
    **_extra: object,
) -> BasePINNModel:
    if problem.supports_exact_cable_interfaces:
        return ExactInterfaceCablePINN(
            problem=problem,
            architecture=architecture,
            width=width,
            depth=depth,
            activation=activation,
            input_normalization=input_normalization,
            trace_parameterization=trace_parameterization,
            soil_envelope_gain=soil_envelope_gain,
        )
    base = M1SubdomainPINN(
        problem=problem,
        architecture="M1",
        width=width,
        depth=depth,
        activation=activation,
        input_normalization=input_normalization,
    )
    return ExactBoundaryPINN(problem=problem, architecture=architecture, base_model=base)


ModelRegistry.register_entry(
    "M1_EXACT",
    ModelMetadata(
        name="M1_EXACT",
        family="exact_constraints",
        partition="region",
        formulation="strong",
        authors="Sheng & Yang (PFNN) + parametrización armónica de trazas de tesis",
        year=2025,
        doi="10.1016/j.jcp.2020.110085",
        summary=(
            "Imposición analítica exacta tanto de fronteras externas Dirichlet como de la "
            "continuidad de temperatura [T]=0 sobre cada interfaz circular concéntrica."
        ),
        modifications=(
            "Emplea trazas armónicas no singulares (z^k) en el núcleo conductor, interpolación "
            "radial suave en coronas aislantes y pesos Shepard en el suelo sin división por cero."
        ),
        supports_exact_constraints=True,
        supports_mixed_flux=False,
        tags=("exact-bc", "exact-interface", "subdomain"),
    ),
    _build_m1_exact,
)

for _arch, _cls, _authors, _doi, _desc in (
    ("M0_EXACT_BC", M0GlobalPINN, "McFall & Mahan / PFNN sobre M0", "10.1016/j.jcp.2020.110085", "M0 con fronteras Dirichlet impuestas exactamente por distancia normalizada."),
    ("M1_EXACT_BC", M1SubdomainPINN, "McFall & Mahan / PFNN sobre M1", "10.1016/j.jcp.2020.110085", "M1 con fronteras Dirichlet externas exactas y continuidad débil en interfaces."),
    ("DCSNN_EXACT_BC", DCSNNPINN, "Hu et al. + PFNN sobre DCSNN", "10.1016/j.jcp.2022.111576", "DCSNN con fronteras Dirichlet exactas."),
):
    ModelRegistry.register_entry(
        _arch,
        ModelMetadata(
            name=_arch,
            family="exact_boundary",
            partition="global" if _cls in (M0GlobalPINN, DCSNNPINN) else "region",
            formulation="strong",
            authors=_authors,
            year=2025,
            doi=_doi,
            summary=_desc,
            modifications="Envuelve la arquitectura base con elevación algebraica Dirichlet exact_boundary.",
            supports_exact_constraints=False,
            supports_mixed_flux=False,
            tags=("exact-bc",),
        ),
        lambda problem, architecture=_arch, width=32, depth=3, activation="tanh", input_normalization=None, _base_cls=_cls, **_kw: ExactBoundaryPINN(
            problem=problem,
            architecture=architecture,
            base_model=_base_cls(
                problem=problem,
                architecture=_base_cls.__name__,
                width=width,
                depth=depth,
                activation=activation,
                input_normalization=input_normalization,
            ),
        ),
    )

for _arch, _cls, _authors, _doi, _desc in (
    ("M0_EXACT", M0GlobalPINN, "Raissi / PFNN + Hermite-Shepard sobre M0", "10.1016/j.jcp.2020.110085", "M0 con fronteras Dirichlet, Neumann y Robin impuestas analíticamente."),
    ("CUSP_EXACT", CuspPINN, "Tseng et al. + PFNN + Hermite-Shepard sobre CUSP", "10.1016/j.jcp.2023.112359", "CUSP con continuidad global exacta [T]=0 y fronteras Dirichlet/Neumann/Robin exactas."),
    ("DCSNN_EXACT", DCSNNPINN, "Hu et al. + PFNN + Hermite-Shepard sobre DCSNN", "10.1016/j.jcp.2022.111576", "DCSNN con imposición exacta de fronteras Dirichlet, Neumann y Robin."),
    ("FBPINN_EXACT", FBPINNPINN, "Moseley et al. + PFNN + Hermite-Shepard sobre FBPINN", "10.1007/s10444-023-10065-9", "FBPINN envuelto con imposición exacta de bordes Dirichlet, Neumann y Robin."),
):
    ModelRegistry.register_entry(
        _arch,
        ModelMetadata(
            name=_arch,
            family="exact_corrector",
            partition="region" if _cls is FBPINNPINN else "global",
            formulation="strong",
            authors=_authors,
            year=2025,
            doi=_doi,
            summary=_desc,
            modifications="Combina la función trial exacta Dirichlet/Neumann/Robin con los bloques base de la arquitectura.",
            supports_exact_constraints=_arch in ("CUSP_EXACT", "M0_EXACT"),
            supports_mixed_flux=False,
            tags=("exact-bc", "exact-corrector"),
        ),
        lambda problem, architecture=_arch, width=32, depth=3, activation="tanh", input_normalization=None, _base_cls=_cls, **_kw: ExactCorrectorPINN(
            problem=problem,
            architecture=architecture,
            base_model=_base_cls(
                problem=problem,
                architecture=_base_cls.__name__,
                width=width,
                depth=depth,
                activation=activation,
                input_normalization=input_normalization,
            ),
        ),
    )
