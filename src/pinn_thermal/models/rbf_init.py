"""Inicializadores discretos híbridos: aproximación variacional RBF/P1 sin malla externa y FEM.

Permiten arrancar cualquier arquitectura PINN cerca del equilibrio térmico antes de optimizar
con Adam/L-BFGS:
- ``_phs_stencil_weights``: pesos locales de diferenciación radial poliharmónica (PHS + poli).
- ``RBFFDSolver``: solucionador discreto multidominio totalmente autocontenido (NumPy + SciPy)
  que discretiza geometrías circulares concéntricas mediante malla polar estructurada y regiones
  generales mediante triangulación Delaunay filtrada, garantizando continuidad exacta en interfaces
  y estabilidad estricta en conductores con altos contrastes de conductividad.
- ``RBFFDInitializer``: convierte la solución nodal en un ``SolutionArtifact`` de tipo ``rbf``
  y la transfiere a cualquier modelo PINN destino mediante ``FunctionalTransfer``.
- ``FEMInitializer``: invoca FEniCSx (WSL2/nativo/Docker) y proyecta su campo hacia la PINN.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.spatial import Delaunay, cKDTree

from pinn_thermal.core.artifact import SolutionArtifact
from pinn_thermal.core.base_model import BasePINNModel
from pinn_thermal.core.fields import evaluate_field_numpy
from pinn_thermal.core.objective import PhysicsObjective
from pinn_thermal.core.problem import RegionSpec, ThermalProblem2D
from pinn_thermal.core.transfer import FunctionalTransfer


def _phs_stencil_weights(center: np.ndarray, neighbors: np.ndarray, order: int = 5, degree: int = 2) -> np.ndarray:
    """Pesos RBF-FD PHS ``r^order`` aumentados con polinomio 2D de grado 1 o 2.

    Devuelve un arreglo de forma ``(M, 6)`` con operadores sobre ``neighbors`` para:
    ``[valor, d/dx, d/dy, d2/dx2, d2/dxdy, d2/dy2]``.
    """
    m = len(neighbors)
    dx = neighbors[:, 0] - center[0]
    dy = neighbors[:, 1] - center[1]
    scale = max(float(np.max(np.hypot(dx, dy))), 1e-12)
    ux, uy = dx / scale, dy / scale

    rx = ux[:, None] - ux[None, :]
    ry = uy[:, None] - uy[None, :]
    r = np.hypot(rx, ry)
    phi = np.power(r, order)

    if degree == 2 and m >= 8:
        poly = np.column_stack([np.ones(m), ux, uy, ux ** 2, ux * uy, uy ** 2])
    else:
        poly = np.column_stack([np.ones(m), ux, uy])
    n_p = poly.shape[1]

    mat = np.zeros((m + n_p, m + n_p), dtype=np.float64)
    mat[:m, :m] = phi
    mat[:m, m:] = poly
    mat[m:, :m] = poly.T

    rhs = np.zeros((m + n_p, 6), dtype=np.float64)
    r0 = np.hypot(ux, uy)
    safe_r0 = np.where(r0 > 1e-12, r0, 1.0)
    mask0 = r0 > 1e-12
    rhs[:m, 0] = np.power(r0, order)
    rhs[:m, 1] = np.where(mask0, order * np.power(safe_r0, max(order - 2, 1)) * (-ux), 0.0) / scale
    rhs[:m, 2] = np.where(mask0, order * np.power(safe_r0, max(order - 2, 1)) * (-uy), 0.0) / scale
    rhs[:m, 3] = np.where(
        mask0,
        order * np.power(safe_r0, order - 4) * ((order - 1) * ux ** 2 + uy ** 2),
        0.0,
    ) / (scale ** 2)
    rhs[:m, 4] = np.where(
        mask0,
        order * (order - 2) * np.power(safe_r0, order - 4) * ux * uy,
        0.0,
    ) / (scale ** 2)
    rhs[:m, 5] = np.where(
        mask0,
        order * np.power(safe_r0, order - 4) * (ux ** 2 + (order - 1) * uy ** 2),
        0.0,
    ) / (scale ** 2)

    rhs[m + 0, 0] = 1.0
    rhs[m + 1, 1] = 1.0 / scale
    rhs[m + 2, 2] = 1.0 / scale
    if n_p == 6:
        rhs[m + 3, 3] = 2.0 / (scale ** 2)
        rhs[m + 4, 4] = 1.0 / (scale ** 2)
        rhs[m + 5, 5] = 2.0 / (scale ** 2)

    sol = np.linalg.lstsq(mat, rhs, rcond=1e-10)[0]
    return sol[:m, :]


class RBFFDSolver:
    """Solucionador discreto rápido sin dependencias externas para inicializar PINNs 2D."""

    def __init__(self, n_per_region: int = 360, stencil_size: int = 18, eps_grad: float = 1e-4) -> None:
        self.n_per_region = int(n_per_region)
        self.stencil_size = int(stencil_size)
        self.eps_grad = float(eps_grad)

    @staticmethod
    def _polar_disk_mesh(cx: float, cy: float, r_out: float, n_r: int, n_th: int) -> tuple[np.ndarray, np.ndarray]:
        angles = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
        pts = [[cx, cy]]
        for ir in range(1, n_r + 1):
            rad = r_out * ir / float(n_r)
            for th in angles:
                pts.append([cx + rad * math.cos(th), cy + rad * math.sin(th)])
        tris: list[list[int]] = []
        for j in range(n_th):
            jn = (j + 1) % n_th
            tris.append([0, 1 + j, 1 + jn])
        for ir in range(1, n_r):
            r0 = 1 + (ir - 1) * n_th
            r1 = 1 + ir * n_th
            for j in range(n_th):
                jn = (j + 1) % n_th
                a, b = r0 + j, r0 + jn
                c, d = r1 + j, r1 + jn
                tris.append([a, c, d])
                tris.append([a, d, b])
        return np.asarray(pts, dtype=np.float64), np.asarray(tris, dtype=np.int64)

    @staticmethod
    def _polar_annulus_mesh(
        cx: float, cy: float, ri: float, ro: float, n_r: int, n_th: int
    ) -> tuple[np.ndarray, np.ndarray]:
        angles = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
        pts = []
        for ir in range(n_r + 1):
            rad = ri + (ro - ri) * ir / float(n_r)
            for th in angles:
                pts.append([cx + rad * math.cos(th), cy + rad * math.sin(th)])
        tris: list[list[int]] = []
        for ir in range(n_r):
            r0 = ir * n_th
            r1 = (ir + 1) * n_th
            for j in range(n_th):
                jn = (j + 1) % n_th
                a, b = r0 + j, r0 + jn
                c, d = r1 + j, r1 + jn
                tris.append([a, c, d])
                tris.append([a, d, b])
        return np.asarray(pts, dtype=np.float64), np.asarray(tris, dtype=np.int64)

    @staticmethod
    def _dedup_points(pts: np.ndarray, tol: float) -> np.ndarray:
        if len(pts) <= 1:
            return pts
        tree = cKDTree(pts)
        kept_mask = np.ones(len(pts), dtype=bool)
        for i in range(len(pts)):
            if not kept_mask[i]:
                continue
            hits = tree.query_ball_point(pts[i], r=tol)
            for j in hits:
                if j > i:
                    kept_mask[j] = False
        return pts[kept_mask]

    def _planar_region_mesh(
        self,
        problem: ThermalProblem2D,
        region: RegionSpec,
        n_th: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Genera triangulación Delaunay filtrada respetando huecos de cables y subregiones."""
        x0_dom, x1_dom, y0_dom, y1_dom = problem.bounds
        span = max(x1_dom - x0_dom, y1_dom - y0_dom, 1.0)
        shape_tol = 1e-8 * span

        clouds: list[np.ndarray] = [
            problem.sample_region_interior(region, max(64, self.n_per_region), rng)
        ]

        # Puntos de borde de la propia región (esquinas y aristas)
        s = region.shape
        n_edge = max(12, int(np.sqrt(max(self.n_per_region, 16))))
        if s.shape_type == "rectangle":
            bx0, bx1, by0, by1 = s.bounds  # type: ignore[attr-defined]
            t = np.linspace(0.0, 1.0, n_edge + 1)
            edges = np.vstack(
                [
                    np.c_[bx0 + (bx1 - bx0) * t, np.full_like(t, by0)],
                    np.c_[bx0 + (bx1 - bx0) * t, np.full_like(t, by1)],
                    np.c_[np.full_like(t, bx0), by0 + (by1 - by0) * t],
                    np.c_[np.full_like(t, bx1), by0 + (by1 - by0) * t],
                ]
            )
            clouds.append(edges)
        elif s.shape_type == "polygon":
            verts = np.asarray(s.vertices, dtype=np.float64)  # type: ignore[attr-defined]
            seg_pts = []
            for k in range(len(verts)):
                p0, p1 = verts[k], verts[(k + 1) % len(verts)]
                t = np.linspace(0.0, 1.0, max(4, n_edge // 2), endpoint=False)[:, None]
                seg_pts.append(p0 + t * (p1 - p0))
            clouds.append(np.vstack(seg_pts))

        # Añadir bordes de otras regiones (cables, rellenos, estratos) que intersectan/tocan esta región
        angles = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
        for other in problem.regions:
            if other.id == region.id:
                continue
            os_shape = other.shape
            if os_shape.shape_type in ("disk", "annulus"):
                cx, cy = os_shape.center  # type: ignore[attr-defined]
                r_out = float(
                    os_shape.radius_m  # type: ignore[attr-defined]
                    if os_shape.shape_type == "disk"
                    else os_shape.outer_radius_m  # type: ignore[attr-defined]
                )
                ring = np.c_[cx + r_out * np.cos(angles), cy + r_out * np.sin(angles)]
                keep = s.contains(ring[:, 0], ring[:, 1], tolerance=shape_tol)
                if np.any(keep):
                    clouds.append(ring[keep])
                # Anillos log-radiales de transición en el suelo alrededor del contorno externo del cable
                if other.priority > region.priority and (
                    other.shape.shape_type == "disk"
                    or not any(
                        c.layers[-1].region_id != other.id and c.layers[-1].outer_radius > r_out
                        and math.hypot(c.center[0] - cx, c.center[1] - cy) < 1e-9
                        for c in problem.cable_layout
                    )
                ):
                    for rad in r_out * np.geomspace(1.25, 14.0, 6):
                        buf = np.c_[cx + rad * np.cos(angles), cy + rad * np.sin(angles)]
                        buf_keep = problem.owner(buf[:, 0], buf[:, 1]) == region.id
                        if np.any(buf_keep):
                            clouds.append(buf[buf_keep])
            elif os_shape.shape_type == "rectangle" and other.priority > region.priority:
                ox0, ox1, oy0, oy1 = os_shape.bounds  # type: ignore[attr-defined]
                t = np.linspace(0.0, 1.0, max(10, n_edge // 2) + 1)
                sub_edges = np.vstack(
                    [
                        np.c_[ox0 + (ox1 - ox0) * t, np.full_like(t, oy0)],
                        np.c_[ox0 + (ox1 - ox0) * t, np.full_like(t, oy1)],
                        np.c_[np.full_like(t, ox0), oy0 + (oy1 - oy0) * t],
                        np.c_[np.full_like(t, ox1), oy0 + (oy1 - oy0) * t],
                    ]
                )
                keep = s.contains(sub_edges[:, 0], sub_edges[:, 1], tolerance=shape_tol)
                if np.any(keep):
                    clouds.append(sub_edges[keep])

        # Si dos regiones rectangulares vecinas (como soil_lower/soil_upper) comparten segmentos donde
        # el borde exterior intersecta el interior, agregar subdivisiones comunes sobre fronteras del dominio
        for seg in problem.sample_boundary_regions(n_edge, rows=1):
            if seg["region"] == region.id:
                clouds.append(seg["xy"])

        merged = self._dedup_points(np.vstack(clouds), tol=1e-7 * span)
        # Mantener solo nodos que pertenecen o tocan la frontera exacta de esta región
        boundary_keep = s.contains(merged[:, 0], merged[:, 1], tolerance=shape_tol)
        # Excluir nodos estrictamente interiores a regiones de MAYOR prioridad que ocultan a esta región
        higher_blocking = np.zeros(len(merged), dtype=bool)
        for other in problem.regions:
            if other.priority > region.priority:
                # Permitir nodos sobre el borde exacto del bloqueador (boundary_distance <= tol)
                bdist = other.shape.boundary_distance(merged[:, 0], merged[:, 1])
                inside_higher = other.shape.contains(merged[:, 0], merged[:, 1], tolerance=0.0) & (bdist > shape_tol)
                higher_blocking |= inside_higher
        pts = merged[boundary_keep & ~higher_blocking]

        tri = Delaunay(pts)
        kept_tris: list[np.ndarray] = []
        for simplex in tri.simplices:
            p0, p1, p2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
            if area <= 1e-16:
                continue
            cent = (p0 + p1 + p2) / 3.0
            if int(problem.owner(np.array([cent[0]]), np.array([cent[1]]))[0]) == region.id:
                kept_tris.append(simplex)
        if not kept_tris:
            raise RuntimeError(f"La región {region.name!r} no generó triángulos válidos en RBFFDSolver.")
        return pts, np.asarray(kept_tris, dtype=np.int64)

    def solve(self, problem: ThermalProblem2D, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
        """Resuelve ``-div(k grad T) = Q`` sobre malla híbrida polar-Delaunay SPD y retorna ``(xy, T)``."""
        rng = np.random.default_rng(int(seed))
        n_th = max(24, min(36, int(np.sqrt(max(self.n_per_region, 16))) * 2))

        region_meshes: list[tuple[RegionSpec, np.ndarray, np.ndarray]] = []
        raw_pts: list[np.ndarray] = []
        raw_tri_groups: list[tuple[RegionSpec, np.ndarray]] = []
        offset = 0

        for region in problem.regions:
            s = region.shape
            if s.shape_type == "disk":
                l_pts, l_tris = self._polar_disk_mesh(*s.center, float(s.radius_m), n_r=4, n_th=n_th)  # type: ignore[attr-defined]
            elif s.shape_type == "annulus":
                n_r = 10 if problem.domain_shape == "annulus" else 4
                l_pts, l_tris = self._polar_annulus_mesh(
                    *s.center, float(s.inner_radius_m), float(s.outer_radius_m), n_r=n_r, n_th=n_th  # type: ignore[attr-defined]
                )
            else:
                l_pts, l_tris = self._planar_region_mesh(problem, region, n_th=n_th, rng=rng)
            region_meshes.append((region, l_pts, l_tris))
            raw_pts.append(l_pts)
            raw_tri_groups.append((region, l_tris + offset))
            offset += len(l_pts)

        all_raw_xy = np.vstack(raw_pts)
        x0, x1, y0, y1 = problem.bounds
        span = max(x1 - x0, y1 - y0, 1.0)
        merge_tol = 1e-7 * min(span, 1.0)

        # Fusionar vértices compartidos sobre interfaces materiales para continuidad exacta [T]=0
        tree = cKDTree(all_raw_xy)
        canon_map = np.arange(len(all_raw_xy), dtype=np.int64)
        for i in range(len(all_raw_xy)):
            if canon_map[i] != i:
                continue
            for j in tree.query_ball_point(all_raw_xy[i], r=merge_tol):
                if j > i and canon_map[j] == j:
                    canon_map[j] = i

        # Compactar únicamente los DOFs referenciados por al menos un triángulo activo
        remapped_groups: list[tuple[RegionSpec, np.ndarray]] = []
        used_raw_ids = set()
        for region, tris in raw_tri_groups:
            mapped = canon_map[tris]
            remapped_groups.append((region, mapped))
            used_raw_ids.update(mapped.ravel().tolist())

        sorted_used = np.array(sorted(used_raw_ids), dtype=np.int64)
        old_to_dof = np.full(len(all_raw_xy), -1, dtype=np.int64)
        old_to_dof[sorted_used] = np.arange(len(sorted_used), dtype=np.int64)
        dof_xy = all_raw_xy[sorted_used]
        n_dofs = len(dof_xy)

        final_groups = [(region, old_to_dof[tris]) for region, tris in remapped_groups]

        # Ensamblaje variacional SPD P1 por triángulos y conteo de aristas para identificar bordes externos
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        rhs = np.zeros(n_dofs, dtype=np.float64)
        edge_count: dict[tuple[int, int], int] = {}

        for region, tris in final_groups:
            p0 = dof_xy[tris[:, 0]]
            p1 = dof_xy[tris[:, 1]]
            p2 = dof_xy[tris[:, 2]]
            cents = (p0 + p1 + p2) / 3.0
            k_vals = np.asarray(region.conductivity_spec.evaluate_numpy(cents[:, 0], cents[:, 1]), dtype=np.float64)
            q_vals = np.asarray(region.source_spec.evaluate_numpy(cents[:, 0], cents[:, 1]), dtype=np.float64)

            det = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
            areas = 0.5 * np.abs(det)
            # Gradientes de funciones base P1: dphi_a/dx = b_a / det, dphi_a/dy = c_a / det
            b_coef = np.column_stack([p1[:, 1] - p2[:, 1], p2[:, 1] - p0[:, 1], p0[:, 1] - p1[:, 1]])
            c_coef = np.column_stack([p2[:, 0] - p1[:, 0], p0[:, 0] - p2[:, 0], p1[:, 0] - p0[:, 0]])

            for t_idx in range(len(tris)):
                tri_ids = tris[t_idx]
                area = float(areas[t_idx])
                if area <= 1e-18:
                    continue
                k_e = float(k_vals[t_idx])
                q_e = float(q_vals[t_idx])
                d2 = 4.0 * area
                for a in range(3):
                    ia = int(tri_ids[a])
                    rhs[ia] += q_e * area / 3.0
                    for b in range(3):
                        ib = int(tri_ids[b])
                        stiff = k_e * (b_coef[t_idx, a] * b_coef[t_idx, b] + c_coef[t_idx, a] * c_coef[t_idx, b]) / d2
                        rows.append(ia)
                        cols.append(ib)
                        vals.append(float(stiff))
                for u_i, v_i in ((tri_ids[0], tri_ids[1]), (tri_ids[1], tri_ids[2]), (tri_ids[2], tri_ids[0])):
                    key = (min(int(u_i), int(v_i)), max(int(u_i), int(v_i)))
                    edge_count[key] = edge_count.get(key, 0) + 1

        # Fronteras externas: aristas pertenecientes a un solo triángulo cuyo punto medio toca un borde exterior
        bc_tol = 1e-5 * span
        bnd_edges = [edge for edge, cnt in edge_count.items() if cnt == 1]

        def _classify_point(x_v: float, y_v: float) -> str | None:
            if problem.domain_shape == "annulus":
                dom = problem.doc["domain"]
                r = math.hypot(x_v - float(dom["center"][0]), y_v - float(dom["center"][1]))
                if abs(r - float(dom["inner_radius_m"])) <= bc_tol:
                    return "inner"
                if abs(r - float(dom["outer_radius_m"])) <= bc_tol:
                    return "outer"
                return None
            if abs(y_v - y0) <= bc_tol:
                return "bottom"
            if abs(y_v - y1) <= bc_tol:
                return "top"
            if abs(x_v - x0) <= bc_tol:
                return "left"
            if abs(x_v - x1) <= bc_tol:
                return "right"
            return None

        # Integrales de contorno Neumann y Robin sobre aristas de frontera externa
        boundaries = problem.doc["boundaries"]
        for u_i, v_i in bnd_edges:
            su = _classify_point(float(dof_xy[u_i, 0]), float(dof_xy[u_i, 1]))
            sv = _classify_point(float(dof_xy[v_i, 0]), float(dof_xy[v_i, 1]))
            side = su if (su is not None and su == sv) else None
            if side is None:
                continue
            mid = 0.5 * (dof_xy[u_i] + dof_xy[v_i])
            bdoc = boundaries[side]
            length = float(np.linalg.norm(dof_xy[u_i] - dof_xy[v_i]))
            if bdoc["type"] == "neumann":
                qn = float(evaluate_field_numpy(bdoc["q_n_W_m2"], np.array([mid[0]]), np.array([mid[1]]))[0])
                rhs[u_i] -= 0.5 * length * qn
                rhs[v_i] -= 0.5 * length * qn
            elif bdoc["type"] == "robin":
                h = float(bdoc["h_W_m2K"])
                t_inf = float(evaluate_field_numpy(bdoc["T_inf_C"], np.array([mid[0]]), np.array([mid[1]]))[0])
                rows.extend([u_i, u_i, v_i, v_i])
                cols.extend([u_i, v_i, u_i, v_i])
                vals.extend([h * length / 3.0, h * length / 6.0, h * length / 6.0, h * length / 3.0])
                rhs[u_i] += 0.5 * h * length * t_inf
                rhs[v_i] += 0.5 * h * length * t_inf

        mat = sparse.lil_matrix((n_dofs, n_dofs), dtype=np.float64)
        for r_i, c_i, v_i in zip(rows, cols, vals):
            mat[r_i, c_i] += v_i

        # Imposición fuerte de condiciones Dirichlet en los nodos de borde correspondientes
        dirichlet_applied = 0
        for vid in range(n_dofs):
            xv, yv = float(dof_xy[vid, 0]), float(dof_xy[vid, 1])
            side = _classify_point(xv, yv)
            if side is not None and boundaries[side]["type"] == "dirichlet":
                val = float(evaluate_field_numpy(boundaries[side]["value"], np.array([xv]), np.array([yv]))[0])
                mat.rows[vid] = [vid]
                mat.data[vid] = [1.0]
                rhs[vid] = val
                dirichlet_applied += 1

        if dirichlet_applied == 0 and all(b["type"] == "neumann" for b in boundaries.values()):
            mat.rows[0] = [0]
            mat.data[0] = [1.0]
            rhs[0] = float(problem.T_ref)

        t_sol = spla.spsolve(mat.tocsr(), rhs)
        return dof_xy, np.asarray(t_sol, dtype=np.float64)


class RBFFDInitializer:
    """Inicializa una red PINN resolviendo previamente una aproximación discreta rápida."""

    def __init__(self, n_per_region: int = 360, stencil_size: int = 18, steps: int = 120) -> None:
        self.solver = RBFFDSolver(n_per_region=n_per_region, stencil_size=stencil_size)
        self.steps = int(steps)

    def apply(
        self,
        model: BasePINNModel,
        objective: PhysicsObjective,
        transfer: str = "readout",
        seed: int = 11,
    ) -> dict[str, Any]:
        dof_xy, dof_t = self.solver.solve(model.problem, seed=seed)
        rbf_art = SolutionArtifact(
            kind="rbf",
            problem=model.problem,
            result={"method": "RBF-FD", "nodes": int(len(dof_xy))},
            fem_xy=dof_xy,
            fem_T=dof_t,
        )
        method = transfer if transfer in ("readout", "readout_gls", "pretrain") else "pretrain"
        report = FunctionalTransfer.apply(rbf_art, model, method=method, steps=self.steps, seed=seed)
        report["initializer"] = "rbf"
        report["rbf_nodes"] = int(len(dof_xy))
        return report


class FEMInitializer:
    """Inicializa una red PINN a partir de una corrida rápida de referencia FEniCSx."""

    def __init__(self, level: int = 1, steps: int = 150) -> None:
        self.level = int(level)
        self.steps = int(steps)

    def apply(
        self,
        model: BasePINNModel,
        objective: PhysicsObjective,
        output_dir: str | Path,
        transfer: str = "readout",
    ) -> dict[str, Any]:
        from pinn_thermal.fem.runner import FEniCSxAdapter

        adapter = FEniCSxAdapter()
        fem_art = adapter.solve(model.problem, level=self.level, output_dir=output_dir)
        method = transfer if transfer in ("readout", "readout_gls", "pretrain") else "pretrain"
        report = FunctionalTransfer.apply(fem_art, model, method=method, steps=self.steps)
        report["initializer"] = "fem"
        report["fem_level"] = self.level
        return report
