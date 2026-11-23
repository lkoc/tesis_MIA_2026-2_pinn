"""Visualizadores interactivos (Plotly) y estáticos de publicación (Matplotlib).

Separados por completo del núcleo numérico:
- :class:`InteractivePlotter`: genera figuras Plotly con zoom fluido, lupa, inspección por
  cursor (coordenadas, material, ``k``, ``Q``, temperatura), superposición geométrica
  editable y dibujo/anotación sobre el plano 2D.
- :class:`StaticPlotter`: produce láminas de alta resolución (PNG/PDF/SVG) mediante el
  backend no interactivo ``Agg`` para incluir directamente en informes y tesis LaTeX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

from pinn_thermal.core.problem import ThermalProblem2D  # noqa: E402
from pinn_thermal.visualization.grids import FieldGrid2D, GridSampler  # noqa: E402

PALETTE = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#6d597a", "#457b9d", "#8ab17d"]


class InteractivePlotter:
    """Construye figuras Plotly interactivas con zoom, anotaciones y edición visual."""

    @staticmethod
    def _require_plotly():
        try:
            import plotly.graph_objects as go  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "El paquete 'plotly' es necesario para el visor interactivo. Instálelo con 'pip install plotly'."
            ) from exc
        return go

    @classmethod
    def geometry_figure(
        cls,
        problem: ThermalProblem2D,
        background: str = "conductivity",
        nx: int = 140,
        ny: int = 140,
        editable: bool = True,
        annotations: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Vista interactiva del dominio geométrico con mapa de ``k(x,y)``, ``Q(x,y)`` o regiones."""
        go = cls._require_plotly()
        maps = GridSampler.sample_problem_maps(problem, nx=nx, ny=ny)

        if background == "source":
            z = maps["source"]
            title_cb = "Q [W/m³]"
            colorscale = "Inferno"
        elif background == "regions":
            z = maps["region_ids"].astype(float)
            title_cb = "ID Región"
            colorscale = "Portland"
        else:
            z = maps["conductivity"]
            title_cb = "k [W/(m·K)]"
            colorscale = "Viridis"

        # Texto enriquecido al pasar el cursor
        region_names = np.array([r.name for r in problem.regions] + ["Fuera"])
        ids_safe = np.where(maps["region_ids"] >= 0, maps["region_ids"], len(problem.regions))
        names_grid = region_names[ids_safe]
        custom = np.dstack([names_grid, maps["conductivity"], maps["source"]])

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=maps["x"],
                y=maps["y"],
                z=z,
                customdata=custom,
                colorscale=colorscale,
                colorbar={"title": title_cb},
                hovertemplate=(
                    "x=%{x:.4f} m<br>y=%{y:.4f} m<br>"
                    "Región: %{customdata[0]}<br>"
                    "k=%{customdata[1]:.4g} W/(m·K)<br>"
                    "Q=%{customdata[2]:.4g} W/m³<extra></extra>"
                ),
            )
        )

        # Contornos vectoriales de cada región
        shapes = []
        for idx, region in enumerate(problem.regions):
            s = region.shape
            color = PALETTE[idx % len(PALETTE)]
            if s.shape_type == "rectangle":
                bx0, bx1, by0, by1 = s.bounds  # type: ignore[attr-defined]
                shapes.append(
                    {"type": "rect", "x0": bx0, "x1": bx1, "y0": by0, "y1": by1, "line": {"color": color, "width": 2}}
                )
            elif s.shape_type == "disk":
                cx, cy = s.center  # type: ignore[attr-defined]
                r = s.radius_m  # type: ignore[attr-defined]
                shapes.append(
                    {"type": "circle", "x0": cx - r, "x1": cx + r, "y0": cy - r, "y1": cy + r, "line": {"color": "white", "width": 2}}
                )
            elif s.shape_type == "annulus":
                cx, cy = s.center  # type: ignore[attr-defined]
                for r in (s.inner_radius_m, s.outer_radius_m):  # type: ignore[attr-defined]
                    shapes.append(
                        {"type": "circle", "x0": cx - r, "x1": cx + r, "y0": cy - r, "y1": cy + r, "line": {"color": "white", "width": 1.5, "dash": "dot"}}
                    )
            elif s.shape_type == "polygon":
                verts = s.vertices  # type: ignore[attr-defined]
                xs = [v[0] for v in verts] + [verts[0][0]]
                ys = [v[1] for v in verts] + [verts[0][1]]
                fig.add_trace(
                    go.Scatter(x=xs, y=ys, mode="lines", name=region.name, line={"color": color, "width": 2})
                )

        # Anotaciones guardadas en el JSON o añadidas por el usuario
        all_notes = list(problem.doc.get("annotations", []))
        if annotations:
            all_notes.extend(annotations)
        plotly_annots = []
        for note in all_notes:
            ax, ay = float(note.get("x", 0.0)), float(note.get("y", 0.0))
            txt = str(note.get("text", ""))
            plotly_annots.append(
                {
                    "x": ax,
                    "y": ay,
                    "text": txt,
                    "showarrow": True,
                    "arrowhead": 2,
                    "bgcolor": "rgba(255,255,255,0.85)",
                    "bordercolor": "#264653",
                    "font": {"size": 12, "color": "#1d3557"},
                }
            )

        dragmode = "drawrect" if editable else "zoom"
        fig.update_layout(
            title=f"Visor Geométrico: {problem.title}",
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            yaxis={"scaleanchor": "x", "scaleratio": 1},
            shapes=shapes,
            annotations=plotly_annots,
            dragmode=dragmode,
            newshape={"line": {"color": "#ff006e", "width": 2, "dash": "dash"}},
            modebar_add=["drawrect", "drawcircle", "drawopenpath", "eraseshape"],
            height=620,
            template="plotly_white",
        )
        return fig

    @classmethod
    def temperature_figure(
        cls,
        grid: FieldGrid2D,
        problem: ThermalProblem2D,
        title: str = "Campo Térmico 2D",
        show_flux_quiver: bool = False,
        show_error: bool = False,
        annotations: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Mapa térmico interactivo Plotly con isotermas, flujo opcional y error respecto a referencia."""
        go = cls._require_plotly()
        if show_error and grid.error_map is not None:
            z = grid.error_map
            cb_title = "ΔT = T - T_ref [K]"
            cscale = "RdBu_r"
        else:
            z = grid.T
            cb_title = "T [°C]"
            cscale = "Turbo"

        custom = np.dstack([grid.region_ids, grid.conductivity])
        fig = go.Figure()
        fig.add_trace(
            go.Contour(
                x=grid.x,
                y=grid.y,
                z=z,
                colorscale=cscale,
                colorbar={"title": cb_title},
                contours={"coloring": "heatmap", "showlabels": True},
                customdata=custom,
                hovertemplate=(
                    "x=%{x:.4f} m, y=%{y:.4f} m<br>"
                    "Valor=%{z:.4f}<br>Región ID=%{customdata[0]}<br>"
                    "k=%{customdata[1]:.3g}<extra></extra>"
                ),
            )
        )

        # Flechas submuestreadas de flujo térmico q = (qx, qy)
        if show_flux_quiver and grid.qx is not None and grid.qy is not None:
            step = max(len(grid.x) // 16, 1)
            gx, gy = np.meshgrid(grid.x[::step], grid.y[::step])
            qx_s = grid.qx[::step, ::step]
            qy_s = grid.qy[::step, ::step]
            mag = np.hypot(qx_s, qy_s)
            scale = 0.04 * max(problem.bounds[1] - problem.bounds[0], 1e-6) / max(float(np.nanmax(mag)), 1e-12)
            xs_seg, ys_seg = [], []
            for i in range(gx.shape[0]):
                for j in range(gx.shape[1]):
                    if np.isfinite(qx_s[i, j]) and np.isfinite(qy_s[i, j]):
                        x0_q, y0_q = gx[i, j], gy[i, j]
                        xs_seg.extend([x0_q, x0_q + scale * qx_s[i, j], None])
                        ys_seg.extend([y0_q, y0_q + scale * qy_s[i, j], None])
            fig.add_trace(
                go.Scatter(x=xs_seg, y=ys_seg, mode="lines", name="Flujo q", line={"color": "white", "width": 1.2})
            )

        all_notes = list(problem.doc.get("annotations", []))
        if annotations:
            all_notes.extend(annotations)
        plotly_annots = [
            {
                "x": float(n.get("x", 0.0)),
                "y": float(n.get("y", 0.0)),
                "text": str(n.get("text", "")),
                "showarrow": True,
                "arrowhead": 2,
                "bgcolor": "rgba(255,255,255,0.85)",
            }
            for n in all_notes
        ]

        fig.update_layout(
            title=title,
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            yaxis={"scaleanchor": "x", "scaleratio": 1},
            annotations=plotly_annots,
            modebar_add=["drawrect", "drawcircle", "drawopenpath", "eraseshape"],
            height=620,
            template="plotly_white",
        )
        return fig

    @classmethod
    def line_cut_figure(cls, cut_data: dict[str, Any], title: str = "Perfil Térmico 1D") -> Any:
        """Gráfico interactivo Plotly comparando múltiples soluciones a lo largo de una recta."""
        go = cls._require_plotly()
        fig = go.Figure()
        for name, values in cut_data["series"].items():
            fig.add_trace(go.Scatter(x=cut_data["distance"], y=values, mode="lines", name=name))
        fig.update_layout(
            title=title,
            xaxis_title="Distancia sobre el corte [m]",
            yaxis_title="Temperatura [°C]",
            template="plotly_white",
            height=430,
        )
        return fig

    @classmethod
    def convergence_figure(cls, runs: dict[str, list[dict[str, Any]]], metric: str = "total") -> Any:
        """Curvas logarítmicas interactivas de convergencia durante el entrenamiento."""
        go = cls._require_plotly()
        fig = go.Figure()
        for label, history in runs.items():
            steps = [row["step"] for row in history if metric in row]
            vals = [max(float(row[metric]), 1e-18) for row in history if metric in row]
            if steps:
                fig.add_trace(go.Scatter(x=steps, y=vals, mode="lines", name=label))
        fig.update_layout(
            title=f"Historial de Convergencia ({metric})",
            xaxis_title="Iteración",
            yaxis_title="Pérdida",
            yaxis_type="log",
            template="plotly_white",
            height=420,
        )
        return fig


class StaticPlotter:
    """Generador estático de alta resolución con Matplotlib para reportes y publicación."""

    @staticmethod
    def plot_geometry(problem: ThermalProblem2D, output_path: str | Path | None = None, dpi: int = 180):
        fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=dpi)
        maps = GridSampler.sample_problem_maps(problem, nx=180, ny=180)
        pcm = ax.pcolormesh(maps["x"], maps["y"], maps["conductivity"], shading="auto", cmap="viridis")
        fig.colorbar(pcm, ax=ax, label="Conductividad k [W/(m·K)]")
        for idx, region in enumerate(problem.regions):
            s = region.shape
            if s.shape_type == "disk":
                ax.add_patch(Circle(s.center, s.radius_m, fill=False, edgecolor="white", linewidth=1.5))  # type: ignore[attr-defined]
            elif s.shape_type == "annulus":
                ax.add_patch(Circle(s.center, s.outer_radius_m, fill=False, edgecolor="white", linestyle="--", linewidth=1.2))  # type: ignore[attr-defined]
            elif s.shape_type == "rectangle" and idx > 0:
                bx0, bx1, by0, by1 = s.bounds  # type: ignore[attr-defined]
                ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, edgecolor="white", linewidth=1.2))
            elif s.shape_type == "polygon":
                ax.add_patch(Polygon(s.vertices, closed=True, fill=False, edgecolor="white", linewidth=1.4))  # type: ignore[attr-defined]
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Geometría y Heterogeneidad — {problem.title}")
        fig.tight_layout()
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=dpi)
        return fig

    @staticmethod
    def plot_temperature_field(
        grid: FieldGrid2D,
        problem: ThermalProblem2D,
        title: str = "Distribución Térmica 2D",
        output_path: str | Path | None = None,
        dpi: int = 180,
    ):
        has_err = grid.error_map is not None
        cols = 2 if has_err else 1
        fig, axes = plt.subplots(1, cols, figsize=(6.5 * cols, 5.0), dpi=dpi)
        ax0 = axes[0] if has_err else axes
        cf = ax0.contourf(grid.x, grid.y, grid.T, levels=28, cmap="turbo")
        ax0.contour(grid.x, grid.y, grid.T, levels=12, colors="k", linewidths=0.4, alpha=0.45)
        fig.colorbar(cf, ax=ax0, label="T [°C]")
        ax0.set_aspect("equal")
        ax0.set_xlabel("x [m]")
        ax0.set_ylabel("y [m]")
        ax0.set_title(title)

        if has_err and grid.error_map is not None:
            ax1 = axes[1]
            vmax = max(float(np.nanmax(np.abs(grid.error_map))), 1e-9)
            ce = ax1.contourf(grid.x, grid.y, grid.error_map, levels=28, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            fig.colorbar(ce, ax=ax1, label="T_pred - T_ref [K]")
            ax1.set_aspect("equal")
            ax1.set_xlabel("x [m]")
            ax1.set_ylabel("y [m]")
            ax1.set_title("Mapa de Error Puntual [K]")

        for ax in (axes if has_err else [ax0]):
            for cable in problem.cable_layout:
                ax.add_patch(Circle(cable.center, cable.radius, fill=False, edgecolor="white", linewidth=1.0))

        fig.tight_layout()
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=dpi)
        return fig
