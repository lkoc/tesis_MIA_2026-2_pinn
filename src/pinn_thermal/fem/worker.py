"""Trabajador independiente de elementos finitos (Gmsh + DolfinX + PETSc MUMPS P2).

Este módulo está diseñado para ejecutarse dentro de cualquier intérprete que tenga
``gmsh``, ``dolfinx``, ``ufl``, ``mpi4py`` y ``petsc4py`` (p. ej. el entorno ``fenicsx``
en WSL2 Linux, Linux nativo o contenedor Docker), **sin requerir PyTorch en absoluto**.

Uso por línea de comandos::

    python -m pinn_thermal.fem.worker --problem path/to/problem.json --out path/to/out_dir --level 2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from pinn_thermal.core.fields import FieldSpec, evaluate_field_numpy
from pinn_thermal.core.manifest import atomic_replace, write_json_atomic
from pinn_thermal.core.problem import RegionSpec, ThermalProblem2D
from pinn_thermal.fem.mesh_resolution import MeshSizingPlan, build_mesh_plan
from pinn_thermal.fem.ufl_backend import UFLMathBackend


def _create_shape_occ(occ: Any, region: RegionSpec) -> int:
    s = region.shape
    if s.shape_type == "rectangle":
        x0, x1, y0, y1 = s.bounds  # type: ignore[attr-defined]
        return occ.addRectangle(float(x0), float(y0), 0.0, float(x1 - x0), float(y1 - y0))
    if s.shape_type == "disk":
        cx, cy = s.center  # type: ignore[attr-defined]
        r = float(s.radius_m)  # type: ignore[attr-defined]
        return occ.addDisk(float(cx), float(cy), 0.0, r, r)
    if s.shape_type == "annulus":
        cx, cy = s.center  # type: ignore[attr-defined]
        ri = float(s.inner_radius_m)  # type: ignore[attr-defined]
        ro = float(s.outer_radius_m)  # type: ignore[attr-defined]
        outer = occ.addDisk(float(cx), float(cy), 0.0, ro, ro)
        inner = occ.addDisk(float(cx), float(cy), 0.0, ri, ri)
        out, _ = occ.cut([(2, outer)], [(2, inner)], removeObject=True, removeTool=True)
        return out[0][1]
    if s.shape_type == "polygon":
        verts = s.vertices  # type: ignore[attr-defined]
        pts = [occ.addPoint(float(vx), float(vy), 0.0) for vx, vy in verts]
        lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        loop = occ.addCurveLoop(lines)
        return occ.addPlaneSurface([loop])
    raise ValueError(f"Tipo de forma no soportado en malla FEM: {s.shape_type!r}.")


def _build_gmsh_model(problem: ThermalProblem2D, plan: MeshSizingPlan):
    import gmsh  # type: ignore

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(problem.id)
    occ = gmsh.model.occ

    # Construir regiones ordenadas de menor a mayor prioridad
    ordered = sorted(problem.regions, key=lambda r: (r.priority, r.id))
    tools: list[tuple[int, int]] = []
    for reg in ordered:
        tag = _create_shape_occ(occ, reg)
        tools.append((2, tag))

    if len(tools) > 1:
        _, out_map = occ.fragment([tools[0]], tools[1:], removeObject=True, removeTool=True)
    else:
        out_map = [[tools[0]]]
    occ.synchronize()

    # Clasificar cada cara resultante usando el mapa genealógico de occ.fragment en orden de prioridad creciente
    tag_owner: dict[int, int] = {}
    for reg, children in zip(ordered, out_map):
        for dim, surf_tag in children:
            if dim == 2:
                tag_owner[surf_tag] = reg.id

    region_surfaces: dict[int, list[int]] = {r.id: [] for r in problem.regions}
    for surf_tag, owner in tag_owner.items():
        region_surfaces[owner].append(surf_tag)

    for region in problem.regions:
        surf_list = region_surfaces[region.id]
        if not surf_list:
            raise RuntimeError(f"La región {region.name!r} no generó entidades de superficie válidas en Gmsh.")
        gmsh.model.addPhysicalGroup(2, surf_list, tag=region.id + 1, name=region.name)

    # Identificar fronteras externas (dim=1) por geometría del dominio
    side_list = list(problem.doc["boundaries"].keys())
    side_to_tag = {side: i + 1 for i, side in enumerate(side_list)}
    side_curves: dict[str, list[int]] = {side: [] for side in side_list}
    x0, x1, y0, y1 = problem.bounds
    span = max(x1 - x0, y1 - y0, 1.0)
    tol = 1e-6 * span

    for dim, curve_tag in gmsh.model.getEntities(dim=1):
        # Evaluar un punto paramétrico sobre la propia curva (getCenterOfMass devuelve el centro en círculos cerrados)
        tmin, tmax = gmsh.model.getParametrizationBounds(dim, curve_tag)
        t_mid = 0.5 * (float(tmin[0]) + float(tmax[0]))
        cx, cy, _ = gmsh.model.getValue(dim, curve_tag, [t_mid])
        if problem.domain_shape == "annulus":
            dom = problem.doc["domain"]
            center = dom["center"]
            ri, ro = float(dom["inner_radius_m"]), float(dom["outer_radius_m"])
            tol_r = min(tol, 0.25 * max(ro - ri, 1e-9))
            r = math.hypot(cx - center[0], cy - center[1])
            if abs(r - ri) <= tol_r:
                side_curves["inner"].append(curve_tag)
            elif abs(r - ro) <= tol_r:
                side_curves["outer"].append(curve_tag)
        else:
            if abs(cy - y0) <= tol:
                side_curves["bottom"].append(curve_tag)
            elif abs(cy - y1) <= tol:
                side_curves["top"].append(curve_tag)
            elif abs(cx - x0) <= tol:
                side_curves["left"].append(curve_tag)
            elif abs(cx - x1) <= tol:
                side_curves["right"].append(curve_tag)

    missing_sides = [side for side, curves in side_curves.items() if not curves]
    if missing_sides:
        raise RuntimeError(
            f"No se encontraron curvas de frontera en Gmsh para los lados {missing_sides} "
            f"del dominio {problem.domain_shape!r}."
        )
    for side, curves in side_curves.items():
        gmsh.model.addPhysicalGroup(1, curves, tag=side_to_tag[side], name=f"bc_{side}")

    # Tamaño de malla guiado por campo Background Minimum de Distance + Threshold por región
    gmsh.model.mesh.field.add("Constant", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", plan.domain_lc)
    gmsh.model.mesh.field.setNumber(1, "VOut", plan.domain_lc)
    field_ids = [1]
    next_fid = 2

    outer_dom_curves = set(side_curves.get("outer", [])) if problem.domain_shape == "annulus" else set()

    for region in problem.regions:
        if region.shape.shape_type in ("disk", "annulus") or region.kind == "conductor":
            curves = gmsh.model.getBoundary([(2, s) for s in region_surfaces[region.id]], combined=False, oriented=False, recursive=False)
            curve_tags = sorted({abs(tag) for _, tag in curves})
            stripped_outer = False
            if outer_dom_curves:
                filtered_tags = [tag for tag in curve_tags if tag not in outer_dom_curves]
                stripped_outer = len(filtered_tags) < len(curve_tags)
                curve_tags = filtered_tags
            if curve_tags:
                d_id = next_fid
                t_id = next_fid + 1
                next_fid += 2
                gmsh.model.mesh.field.add("Distance", d_id)
                gmsh.model.mesh.field.setNumbers(d_id, "CurvesList", curve_tags)
                gmsh.model.mesh.field.setNumber(d_id, "Sampling", 100)

                lc_reg = plan.region_lc[region.id]
                if stripped_outer:
                    radius_hint = float(getattr(region.shape, "inner_radius_m", span * 0.05))
                    dist_max = max(6.0 * radius_hint, 0.25 * span)
                else:
                    radius_hint = float(getattr(region.shape, "outer_radius_m", getattr(region.shape, "radius_m", span * 0.05)))
                    dist_max = max(4.0 * radius_hint, 0.25 * span)
                gmsh.model.mesh.field.add("Threshold", t_id)
                gmsh.model.mesh.field.setNumber(t_id, "InField", d_id)
                gmsh.model.mesh.field.setNumber(t_id, "SizeMin", lc_reg)
                gmsh.model.mesh.field.setNumber(t_id, "SizeMax", plan.domain_lc)
                gmsh.model.mesh.field.setNumber(t_id, "DistMin", 0.5 * radius_hint)
                gmsh.model.mesh.field.setNumber(t_id, "DistMax", dist_max)
                field_ids.append(t_id)

    # También refinar cerca de interfaces internas rectangulares/estratificadas
    if len(problem.regions) > 1 and all(r.shape.shape_type == "rectangle" for r in problem.regions):
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", plan.min_lc)
    else:
        gmsh.model.mesh.field.add("Min", next_fid)
        gmsh.model.mesh.field.setNumbers(next_fid, "FieldsList", field_ids)
        gmsh.model.mesh.field.setAsBackgroundMesh(next_fid)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", plan.min_lc * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", plan.domain_lc)

    gmsh.model.mesh.generate(2)
    return gmsh, side_to_tag


def solve_fem_problem(
    problem_source: str | Path | dict[str, Any] | ThermalProblem2D,
    output_dir: str | Path,
    level: int = 2,
) -> dict[str, Any]:
    """Resuelve el problema en DolfinX Lagrange P2 y escribe ``fem_l{level}.json/.npz``."""
    from dolfinx import fem  # type: ignore
    from dolfinx.fem.petsc import LinearProblem  # type: ignore
    try:
        from dolfinx.io import gmsh as gmshio  # type: ignore  # DolfinX >= 0.10
    except ImportError:
        from dolfinx.io import gmshio  # type: ignore  # DolfinX <= 0.9
    from mpi4py import MPI  # type: ignore
    import ufl  # type: ignore

    started = time.perf_counter()
    problem = problem_source if isinstance(problem_source, ThermalProblem2D) else ThermalProblem2D.from_json(problem_source)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "problem.json", problem.doc)

    plan = build_mesh_plan(problem, level=level)
    gmsh, side_to_tag = _build_gmsh_model(problem, plan)
    try:
        mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2)
        if hasattr(mesh_data, "mesh"):
            domain = mesh_data.mesh
            cell_tags = mesh_data.cell_tags
            facet_tags = mesh_data.facet_tags
        else:
            domain, cell_tags, facet_tags = mesh_data
    finally:
        gmsh.finalize()

    if facet_tags is None:
        raise RuntimeError(
            f"Gmsh/DolfinX no devolvió etiquetas de facetas (facet_tags=None) para el caso {problem.id!r}."
        )

    v_space = fem.functionspace(domain, ("Lagrange", 2))
    u = ufl.TrialFunction(v_space)
    v = ufl.TestFunction(v_space)
    coords = ufl.SpatialCoordinate(domain)
    dx_meas = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    ds_meas = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    backend = UFLMathBackend(ufl)

    # Forma débil por región: sum_r int_{Omega_r} k_r grad(u).grad(v) dx_r = sum_r int_{Omega_r} Q_r v dx_r
    a_terms = []
    l_terms = []
    for region in problem.regions:
        tag = region.id + 1
        k_ufl = region.conductivity_spec.evaluate_ufl(coords[0], coords[1], backend, fem, v_space)
        q_ufl = region.source_spec.evaluate_ufl(coords[0], coords[1], backend, fem, v_space)
        a_terms.append(k_ufl * ufl.dot(ufl.grad(u), ufl.grad(v)) * dx_meas(tag))
        l_terms.append(q_ufl * v * dx_meas(tag))

    # Condiciones de contorno externas
    bcs = []
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)

    for side, cond in problem.doc["boundaries"].items():
        tag = side_to_tag[side]
        ctype = cond["type"]
        if ctype == "dirichlet":
            g_fn = fem.Function(v_space)
            g_fn.interpolate(lambda z, spec=cond["value"]: evaluate_field_numpy(spec, z[0], z[1]).ravel())
            facets = facet_tags.find(tag)
            if len(facets):
                dofs = fem.locate_dofs_topological(v_space, fdim, facets)
                bcs.append(fem.dirichletbc(g_fn, dofs))
        elif ctype == "neumann":
            # q_n = -k dT/dn -> k dT/dn = -q_n -> pasa restando al lado derecho L(v)
            qn_ufl = FieldSpec.from_dict(cond["q_n_W_m2"]).evaluate_ufl(
                coords[0], coords[1], backend, fem, v_space
            )
            l_terms.append(-qn_ufl * v * ds_meas(tag))
        elif ctype == "robin":
            h_val = float(cond["h_W_m2K"])
            t_inf_ufl = FieldSpec.from_dict(cond["T_inf_C"]).evaluate_ufl(
                coords[0], coords[1], backend, fem, v_space
            )
            a_terms.append(h_val * u * v * ds_meas(tag))
            l_terms.append(h_val * t_inf_ufl * v * ds_meas(tag))

    a_form = sum(a_terms)
    l_form = sum(l_terms)

    petsc_opts = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    try:
        lin = LinearProblem(
            a_form,
            l_form,
            bcs=bcs,
            petsc_options=petsc_opts,
            petsc_options_prefix="pinn_v4_fem_",
        )
    except TypeError:
        lin = LinearProblem(a_form, l_form, bcs=bcs, petsc_options=petsc_opts)

    uh = lin.solve()

    dof_xy = np.asarray(v_space.tabulate_dof_coordinates()[:, :2], dtype=np.float64)
    dof_t = np.asarray(uh.x.array, dtype=np.float64)

    # Auditoría energética FEM global mediante integrales de dominio y frontera
    n_vec = ufl.FacetNormal(domain)
    power_W = 0.0
    for region in problem.regions:
        tag = region.id + 1
        q_ufl = region.source_spec.evaluate_ufl(coords[0], coords[1], backend, fem, v_space)
        power_W += float(fem.assemble_scalar(fem.form(q_ufl * dx_meas(tag))))

    dg0 = fem.functionspace(domain, ("DG", 0))
    k_field = fem.Function(dg0)
    k_field.interpolate(lambda z: problem.field_numpy("conductivity", z[0], z[1]).ravel())
    side_fluxes = [
        float(fem.assemble_scalar(fem.form((-k_field * ufl.dot(ufl.grad(uh), n_vec)) * ds_meas(tag))))
        for tag in side_to_tag.values()
    ]
    boundary_flux_W = float(sum(side_fluxes))
    throughput_W = 0.5 * float(sum(abs(flux_val) for flux_val in side_fluxes))

    denom = max(abs(power_W), abs(boundary_flux_W), throughput_W, 1e-12)
    balance_pct = 100.0 * abs(power_W - boundary_flux_W) / denom

    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    num_dofs = len(dof_t)
    elapsed_s = time.perf_counter() - started

    npz_path = out / f"fem_l{plan.level}.npz"
    tmp_npz = out / f"fem_l{plan.level}.tmp.npz"
    with tmp_npz.open("wb") as stream:
        np.savez_compressed(stream, dof_xy=dof_xy, dof_T=dof_t)
    atomic_replace(tmp_npz, npz_path)

    record = {
        "kind": "fem",
        "method": f"FEniCSx-Gmsh-P2-L{plan.level}",
        "case": problem.id,
        "level": plan.level,
        "dofs": int(num_dofs),
        "cells": int(num_cells),
        "Tmin_C": float(np.min(dof_t)),
        "Tmax_C": float(np.max(dof_t)),
        "power_W": float(power_W),
        "boundary_flux_W": float(boundary_flux_W),
        "balance_pct": float(balance_pct),
        "physics_sha256": problem.physics_sha256,
        "elapsed_s": float(elapsed_s),
        "npz": npz_path.name,
    }
    json_path = out / f"fem_l{plan.level}.json"
    write_json_atomic(json_path, record)
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Solucionador FEniCSx P2 (libre de PyTorch) para PINN_V4.")
    parser.add_argument("--problem", required=True, help="Ruta al archivo problem.json.")
    parser.add_argument("--out", required=True, help="Directorio donde guardar fem_l{level}.json/.npz.")
    parser.add_argument("--level", type=int, default=2, help="Nivel de refinamiento de malla (1..4).")
    args = parser.parse_args(argv)
    record = solve_fem_problem(args.problem, args.out, level=args.level)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
