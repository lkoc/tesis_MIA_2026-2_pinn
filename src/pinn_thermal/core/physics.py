"""Operadores diferenciales físicos y modelo electromagnético de efecto piel.

Implementa con diferenciación automática de PyTorch:
- Gradiente espacial/temporal de primer orden (conservando el grafo de cómputo).
- Operador de divergencia con conductividad térmica variable: ``div(k(x, y) * grad(T))``.
- Residuos en forma fuerte estacionaria y transitoria: ``-div(k grad T) - Q`` y ``rho*cp*dT/dt - div(k grad T) - Q``.
- Residuo de formulación mixta de primer orden ``(T, q)``: ``div(q) - Q`` y ley constitutiva ``q + k grad T = 0``.
- Condiciones de frontera Neumann y Robin y cálculo cilíndrico analítico de efecto piel.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import jv
import torch

MU0: float = 4.0 * math.pi * 1e-7


def gradients(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Calcula ``d(field)/d(coords)`` manteniendo ``create_graph=True`` para derivadas superiores."""
    if not coords.requires_grad:
        raise ValueError("El tensor de coordenadas requiere requires_grad=True para diferenciación automática.")
    if not field.requires_grad:
        return coords * 0.0
    grad = torch.autograd.grad(
        field,
        coords,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad + coords * 0.0


def _as_column_tensor(value: torch.Tensor | float, ref: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        t = torch.tensor(float(value), device=ref.device, dtype=ref.dtype)
    else:
        t = value
    if t.ndim == 0:
        t = t.view(1, 1)
    if t.shape[0] == 1 and ref.shape[0] > 1:
        t = t.expand(ref.shape[0], 1)
    return t


def laplace_variable_k(
    temperature: torch.Tensor,
    coords: torch.Tensor,
    k: torch.Tensor | float,
    spatial_cols: tuple[int, int] = (0, 1),
) -> torch.Tensor:
    """Evalúa en forma de divergencia ``d/dx(k dT/dx) + d/dy(k dT/dy)``."""
    c0, c1 = spatial_cols
    k_t = _as_column_tensor(k, coords)
    grad_t = gradients(temperature, coords)
    flux_x = k_t * grad_t[:, c0:c0 + 1]
    flux_y = k_t * grad_t[:, c1:c1 + 1]
    d_fx = gradients(flux_x, coords)[:, c0:c0 + 1]
    d_fy = gradients(flux_y, coords)[:, c1:c1 + 1]
    return d_fx + d_fy


def derived_flux(temperature: torch.Tensor, coords: torch.Tensor, k: torch.Tensor | float) -> torch.Tensor:
    """Calcula el vector de flujo térmico de Fourier ``q = -k * grad(T)`` en ``(x, y)``."""
    k_t = _as_column_tensor(k, coords)
    return -k_t * gradients(temperature, coords)[:, :2]


def heat_residual(
    temperature: torch.Tensor,
    coords: torch.Tensor,
    k: torch.Tensor | float,
    source: torch.Tensor | float,
    capacity: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Residuo fuerte de la ecuación del calor ``-div(k grad T) - Q = 0`` (o transitorio si hay columna ``t``)."""
    storage: torch.Tensor | float = 0.0
    if coords.shape[1] == 3:
        if capacity is None:
            raise ValueError("La ecuación transitoria del calor requiere rho*cp positivo.")
        cap_t = _as_column_tensor(capacity, coords)
        if bool(torch.any(cap_t <= 0)):
            raise ValueError("La capacidad calorífica volumétrica rho*cp debe ser positiva.")
        storage = cap_t * gradients(temperature, coords)[:, 2:3]
    q_t = _as_column_tensor(source, coords)
    return storage - laplace_variable_k(temperature, coords, k) - q_t


def mixed_residual(
    temperature: torch.Tensor,
    flux: torch.Tensor,
    coords: torch.Tensor,
    k: torch.Tensor | float,
    source: torch.Tensor | float,
    capacity: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residuo mixto de primer orden: devuelve ``(conservación_energía, ley_constitutiva)``."""
    grad_t = gradients(temperature, coords)
    k_t = _as_column_tensor(k, coords)
    constitutive = flux + k_t * grad_t[:, :2]
    div_q = gradients(flux[:, 0:1], coords)[:, 0:1] + gradients(flux[:, 1:2], coords)[:, 1:2]
    storage: torch.Tensor | float = 0.0
    if coords.shape[1] == 3:
        if capacity is None:
            raise ValueError("La ecuación transitoria mixta requiere rho*cp positivo.")
        cap_t = _as_column_tensor(capacity, coords)
        storage = cap_t * grad_t[:, 2:3]
    q_t = _as_column_tensor(source, coords)
    return storage + div_q - q_t, constitutive


def neumann_residual(
    temperature: torch.Tensor,
    coords: torch.Tensor,
    normal: torch.Tensor,
    target_flux: torch.Tensor | float = 0.0,
    k_val: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Residuo de condición Neumann ``(-k grad T) . n - q_n = 0``."""
    qn = (derived_flux(temperature, coords, k_val) * normal).sum(dim=1, keepdim=True)
    return qn - target_flux


def robin_residual(
    temperature: torch.Tensor,
    coords: torch.Tensor,
    normal: torch.Tensor,
    k_val: torch.Tensor | float,
    h: float,
    T_inf: torch.Tensor | float,
) -> torch.Tensor:
    """Residuo de condición convectiva de Robin ``(-k grad T) . n - h (T - T_inf) = 0``."""
    qn = (derived_flux(temperature, coords, k_val) * normal).sum(dim=1, keepdim=True)
    return qn - float(h) * (temperature - T_inf)


# ---------------------------------------------------------------------------
# Cálculo electromagnético auxiliar: efecto piel en conductor circular macizo
# ---------------------------------------------------------------------------

def skin_factor(
    resistance_dc_ohm_m: float,
    radius_m: float,
    frequency_Hz: float,
    mu_r: float = 1.0,
) -> float:
    """Factor de incremento AC/DC por efecto piel en un conductor cilíndrico sólido aislado."""
    if not all(np.isfinite(v) for v in (resistance_dc_ohm_m, radius_m, frequency_Hz, mu_r)):
        raise ValueError("Los parámetros eléctricos deben ser finitos.")
    if min(resistance_dc_ohm_m, radius_m, mu_r) <= 0 or frequency_Hz < 0:
        raise ValueError("Se requiere R_dc > 0, radio > 0, mu_r > 0 y frecuencia >= 0.")
    if frequency_Hz == 0.0:
        return 1.0
    rho = resistance_dc_ohm_m * math.pi * radius_m ** 2
    delta = math.sqrt(rho / (math.pi * frequency_Hz * MU0 * mu_r))
    z = (1.0 - 1.0j) * radius_m / delta
    if abs(z) < 1e-3:
        return 1.0 + abs(z) ** 4 / 192.0
    return float(np.real(z * jv(0, z) / (2.0 * jv(1, z))))


def skin_source_profile(
    xy: np.ndarray,
    center: tuple[float, float] | list[float],
    radius_m: float,
    current_A: float,
    resistance_dc_ohm_m: float,
    frequency_Hz: float,
    mu_r: float = 1.0,
) -> np.ndarray:
    """Perfil radial de pérdidas volumétricas [W/m^3] por efecto piel mediante funciones de Bessel."""
    pts = np.asarray(xy, dtype=float)
    area = math.pi * radius_m ** 2
    if frequency_Hz == 0.0:
        power = (current_A ** 2) * resistance_dc_ohm_m
        return np.full((len(pts), 1), power / area, dtype=float)
    rho = resistance_dc_ohm_m * area
    delta = math.sqrt(rho / (math.pi * frequency_Hz * MU0 * mu_r))
    gamma = (1.0 - 1.0j) / delta
    r = np.minimum(np.linalg.norm(pts[:, :2] - np.asarray(center, dtype=float), axis=1), radius_m)
    current_density = current_A * gamma * jv(0, gamma * r) / (2.0 * math.pi * radius_m * jv(1, gamma * radius_m))
    return (rho * np.abs(current_density) ** 2).reshape(-1, 1)
