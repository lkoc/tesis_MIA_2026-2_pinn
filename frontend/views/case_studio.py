"""Vista 1: Estudio visual y de texto para diseñar o modificar casos JSON 2D."""

from __future__ import annotations

import json
import streamlit as st  # type: ignore

from pinn_thermal.visualization.geometry_editor import CaseEditorHelper, load_material_presets
from pinn_thermal.visualization.plots import InteractivePlotter
from frontend.state import WorkspaceStore


def _heterogeneity_widget(prefix: str, field_name: str, default_val: float = 1.0):
    mode = st.selectbox(
        f"Modo de definición para {field_name}",
        ["Constante", "Expresión f(x, y)", "Lógica condicional (piecewise)", "Tabla 1D", "Tabla 2D"],
        key=f"{prefix}_mode",
    )
    if mode == "Constante":
        val = st.number_input(f"Valor constante de {field_name}", value=float(default_val), format="%.6g", key=f"{prefix}_cval")
        return CaseEditorHelper.field_constant(val)
    if mode == "Expresión f(x, y)":
        expr = st.text_input(
            f"Expresión segura en (x, y) para {field_name}",
            value="1.0 + 0.3 * exp(-((x)**2 + (y + 1.0)**2))",
            key=f"{prefix}_expr",
        )
        return CaseEditorHelper.field_expression(expr)
    if mode == "Lógica condicional (piecewise)":
        cond = st.text_input("Condición lógica `when`", value="y <= -1.0 and abs(x) < 0.5", key=f"{prefix}_cond")
        v1 = st.number_input("Valor si se cumple", value=0.5, key=f"{prefix}_pv1")
        v2 = st.number_input("Valor por defecto (`default`)", value=1.2, key=f"{prefix}_pv2")
        return CaseEditorHelper.field_piecewise([(cond, v1)], v2)
    if mode == "Tabla 1D":
        axis = st.selectbox("Eje", ["y", "x"], key=f"{prefix}_t1_axis")
        coords_txt = st.text_input("Coordenadas (separadas por coma)", value="-2.0, -1.0, 0.0", key=f"{prefix}_t1_coords")
        vals_txt = st.text_input("Valores correspondientes", value="0.7, 1.1, 1.5", key=f"{prefix}_t1_vals")
        interp = st.selectbox("Interpolación", ["linear", "nearest", "step"], key=f"{prefix}_t1_int")
        coords = [float(v.strip()) for v in coords_txt.split(",") if v.strip()]
        vals = [float(v.strip()) for v in vals_txt.split(",") if v.strip()]
        return CaseEditorHelper.field_table_1d(axis, coords, vals, interpolation=interp)
    # Tabla 2D
    xs_txt = st.text_input("Eje x (coma)", value="-1.5, 0.0, 1.5", key=f"{prefix}_t2_x")
    ys_txt = st.text_input("Eje y (coma)", value="-2.0, -1.0, 0.0", key=f"{prefix}_t2_y")
    mat_txt = st.text_area("Matriz values[iy][ix] (filas JSON)", value="[[0.8, 0.9, 0.8],\n [1.1, 1.3, 1.1],\n [1.5, 1.7, 1.5]]", key=f"{prefix}_t2_m")
    xs = [float(v.strip()) for v in xs_txt.split(",") if v.strip()]
    ys = [float(v.strip()) for v in ys_txt.split(",") if v.strip()]
    return CaseEditorHelper.field_table_2d(xs, ys, json.loads(mat_txt))


def render_case_studio(store: WorkspaceStore) -> None:
    st.header("1. Estudio de Casos 2D, Heterogeneidad y Visor Interactivo")

    if "editor_doc" not in st.session_state:
        st.session_state["editor_doc"] = CaseEditorHelper.template_buried_single_cable()

    problems = store.list_all_problems()
    labels = [f"[{p['category']}] {p['id']} — {p['title']}" for p in problems]

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        sel = st.selectbox("Cargar caso existente para editar", range(len(problems)), format_func=lambda i: labels[i]) if problems else 0
    with c2:
        if problems and st.button("Cargar seleccionado", use_container_width=True):
            st.session_state["editor_doc"] = store.load_problem(problems[sel]["path"]).doc
            st.rerun()
    with c3:
        if st.button("Plantilla: Cable XLPE", use_container_width=True):
            st.session_state["editor_doc"] = CaseEditorHelper.template_buried_single_cable()
            st.rerun()
    with c4:
        if st.button("Plantilla: Suelo Tabla 1D", use_container_width=True):
            st.session_state["editor_doc"] = CaseEditorHelper.template_two_layer_soil()
            st.rerun()

    helper = CaseEditorHelper(st.session_state["editor_doc"])
    materials = {
        f"{grp['title']} — {item['name']}": item
        for grp in load_material_presets().get("groups", [])
        for item in grp.get("items", [])
    }

    tab_vis, tab_edit, tab_json = st.tabs(["🗺️ Visor Interactivo (Zoom y Anotaciones)", "🧱 Constructor Gráfico de Materiales y Cables", "📝 Editor JSON Directo"])

    with tab_edit:
        col_a, col_b = st.columns(2)
        with col_a:
            new_id = st.text_input("Identificador (`id`)", value=helper.doc.get("id", "nuevo_caso"))
            new_title = st.text_input("Título descriptivo", value=helper.doc.get("title", ""))
            families = ["cable", "mms", "annulus", "generic"]
            cur_family = helper.doc.get("family") or "cable"
            fam_idx = families.index(cur_family) if cur_family in families else families.index("generic")
            new_family = st.selectbox("Familia", families, index=fam_idx)
            if st.button("Aplicar metadatos"):
                helper.set_metadata(new_id, new_title, new_family)
                st.session_state["editor_doc"] = helper.doc
                st.success("Metadatos actualizados.")

            st.subheader("Agregar / Modificar Región Heterogénea")
            reg_id = st.text_input("ID de región", value="zona_heterogenea")
            reg_kind = st.selectbox("Tipo de región", ["soil", "conductor", "material"])
            reg_prio = st.number_input("Prioridad (mayor cubre menor)", min_value=0, max_value=999, value=1)
            shape_type = st.selectbox("Geometría", ["rectangle", "disk", "annulus"])
            if shape_type == "rectangle":
                bx = st.text_input("Límites [x0, x1, y0, y1]", value="-0.4, 0.4, -1.3, -0.7")
                shape = {"type": "rectangle", "bounds": [float(v) for v in bx.split(",")]}
            elif shape_type == "disk":
                cx = st.number_input("Centro cx [m]", value=0.25)
                cy = st.number_input("Centro cy [m]", value=-1.0)
                rm = st.number_input("Radio r [m]", value=0.02, format="%.5f")
                shape = {"type": "disk", "center": [cx, cy], "radius_m": rm}
            else:
                cx = st.number_input("Centro cx [m]", value=0.0, key="ann_cx")
                cy = st.number_input("Centro cy [m]", value=-1.0, key="ann_cy")
                ri = st.number_input("Radio interior ri [m]", value=0.018, format="%.5f")
                ro = st.number_input("Radio exterior ro [m]", value=0.036, format="%.5f")
                shape = {"type": "annulus", "center": [cx, cy], "inner_radius_m": ri, "outer_radius_m": ro}

        with col_b:
            mat_keys = ["(personalizado)"] + list(materials.keys())
            mat_pick = st.selectbox("Preset de material térmico", mat_keys)
            default_k = float(materials[mat_pick]["k"]) if mat_pick in materials else 1.5
            k_spec = _heterogeneity_widget(f"k_ed_{mat_pick}", "conductividad k [W/(m·K)]", default_k)
            q_spec = _heterogeneity_widget("q_ed", "fuente Q [W/m³]", 0.0)
            if st.button("Insertar / Actualizar Región en JSON", type="primary"):
                mat_name = materials[mat_pick]["name"] if mat_pick in materials else ""
                helper.upsert_region(reg_id, reg_kind, int(reg_prio), shape, k_spec, q_spec, material=mat_name)
                st.session_state["editor_doc"] = helper.doc
                st.rerun()

            st.subheader("Anotaciones Interactivas")
            ax = st.number_input("Coordenada x anotación [m]", value=0.0, key="note_x")
            ay = st.number_input("Coordenada y anotación [m]", value=-1.0, key="note_y")
            atxt = st.text_input("Texto de anotación", value="Punto de control térmico", key="note_txt")
            n1, n2 = st.columns(2)
            with n1:
                if st.button("Añadir anotación"):
                    helper.add_annotation(ax, ay, atxt)
                    st.session_state["editor_doc"] = helper.doc
                    st.rerun()
            with n2:
                if st.button("Limpiar anotaciones"):
                    helper.clear_annotations()
                    st.session_state["editor_doc"] = helper.doc
                    st.rerun()

    with tab_json:
        edited_text = st.text_area("Documento JSON (`pinn-v4-problem-1`)", value=helper.to_json_text(), height=500)
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Aplicar JSON desde texto", use_container_width=True):
                try:
                    updated = CaseEditorHelper.from_json_text(edited_text)
                    st.session_state["editor_doc"] = updated.doc
                    st.success("JSON válido cargado en el visor.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error de formato o validación: {exc}")
        with b2:
            st.download_button(
                "Descargar problem.json",
                data=edited_text,
                file_name=f"{helper.doc.get('id', 'caso')}.json",
                mime="application/json",
                use_container_width=True,
            )

    with tab_vis:
        errs = helper.validate()
        if errs:
            st.error(f"El caso presenta errores de validación: {errs[0]}")
        else:
            prob = helper.build_problem()
            layer_mode = st.radio("Capa de mapa", ["conductivity", "source", "regions"], horizontal=True)
            fig = InteractivePlotter.geometry_figure(prob, background=layer_mode, editable=True)
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
            st.caption("Sugerencia: use la rueda del ratón para hacer zoom, pase el cursor para inspeccionar k(x,y) y Q(x,y), o dibuje formas guía sobre la barra superior.")

    if not errs:
        if st.button("💾 Guardar caso en Workspace compartido (visible para Entrenamiento y Jupyter)", type="primary"):
            saved = store.save_user_case(helper)
            st.success(f"Guardado en: `{saved}`")
