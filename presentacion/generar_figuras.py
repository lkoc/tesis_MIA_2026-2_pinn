"""Genera figuras vectoriales PDF reproducibles para ``presentacion_pinn_v4.tex``.

Lee en modo solo lectura los archivos JSON reales de ``notebooks_campaign/campaign_workspace/``
y produce los gráficos en ``presentacion/figuras/``.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRES_DIR = Path(__file__).resolve().parent
ROOT_DIR = PRES_DIR.parent
WS_DIR = ROOT_DIR / "notebooks_campaign" / "campaign_workspace"
FIG_DIR = PRES_DIR / "figuras"

NAVY = "#102A43"
TEAL = "#0F766E"
AMBER = "#D97706"
CORAL = "#B91C1C"
BLUE = "#1D4ED8"
PURPLE = "#6D28D9"
SLATE = "#475569"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
    }
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def make_stage2_energy_impact(promo: dict) -> Path:
    """Compara pasadas CON energy (100.0) vs SIN energy (0.0) en Etapa 2."""
    archs = ["M0R", "M1", "FBPINN", "M2", "DCSNN", "CUSPR"]
    by_arch_mode: dict[tuple[str, str], dict] = {
        (r["architecture"], r["energy_mode"]): r for r in promo.get("ranking", [])
    }

    with_score = [by_arch_mode[(a, "with_energy")]["mean_score"] for a in archs]
    no_score = [by_arch_mode[(a, "no_energy")]["mean_score"] for a in archs]
    with_bal = [max(by_arch_mode[(a, "with_energy")]["max_balance_pct"], 0.01) for a in archs]
    no_bal = [max(by_arch_mode[(a, "no_energy")]["max_balance_pct"], 0.01) for a in archs]

    x = np.arange(len(archs))
    w = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)

    b1 = ax1.bar(x - w / 2, with_score, width=w, color=TEAL, label="CON energy (100.0)")
    b2 = ax1.bar(x + w / 2, no_score, width=w, color=AMBER, label="SIN energy (0.0)")
    ax1.set_title("Puntaje log1p compuesto (menor es mejor)")
    ax1.set_xticks(x, archs, fontweight="bold")
    ax1.set_ylabel("mean_log_score")
    ax1.set_ylim(0, max(no_score) * 1.18)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, facecolor="white")
    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.22, f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)

    b3 = ax2.bar(x - w / 2, with_bal, width=w, color=TEAL, label="CON energy (100.0)")
    b4 = ax2.bar(x + w / 2, no_bal, width=w, color=CORAL, label="SIN energy (0.0)")
    ax2.axhline(50.0, color=CORAL, linestyle="--", linewidth=1.3, label="Compuerta sanidad (50%)")
    ax2.axhline(0.5, color=NAVY, linestyle=":", linewidth=1.2, label="Umbral STRICT (0.5%)")
    ax2.set_yscale("log")
    ax2.set_title("Peor desbalance entre 4 casos (escala log)")
    ax2.set_xticks(x, archs, fontweight="bold")
    ax2.set_ylabel("max_balance_pct (%) [log]")
    ax2.set_ylim(0.2, max(max(with_bal), max(no_bal)) * 2.5)
    ax2.grid(axis="y", which="both", linestyle="--", alpha=0.35)
    ax2.legend(frameon=True, facecolor="white", loc="upper left", fontsize=8.5)
    for bars in (b3, b4):
        for bar in bars:
            val = bar.get_height()
            lbl = f"{val:.1f}%" if val < 100 else f"{val:.0f}%"
            ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.12, lbl, ha="center", va="bottom", fontsize=8)

    out = FIG_DIR / "fig_stage2_energy_impact.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_stage2_promoted_cases(promo: dict, s2: dict) -> Path:
    """Compara NRMSE, error Tmax y balance por caso pivote en los 5 modelos promovidos."""
    promoted = promo["promoted_models"]
    cases = ["mms_sine", "cable_disk", "xlpe_discrete_layers", "aras_flat"]
    case_labels = ["mms_sine", "cable_disk", "xlpe_discrete", "aras_flat"]
    colors = {
        "M0R": TEAL,
        "M1": BLUE,
        "FBPINN": PURPLE,
        "M2": AMBER,
        "DCSNN": SLATE,
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), constrained_layout=True)
    specs = [
        ("nrmse_pct", "NRMSE (%)", "NRMSE por caso (%)", 0.25),
        ("error_Tmax_abs_K", "Error |dTmax| (K)", "Error en Tmax por caso (K)", 0.05),
        ("balance_pct", "Desbalance (%)", "Desbalance energético (%)", 0.5),
    ]

    x = np.arange(len(cases))
    n = len(promoted)
    w = 0.78 / n

    for ax, (metric, ylabel, title, strict_lim) in zip(axes, specs):
        for i, arch in enumerate(promoted):
            vals = [float(s2[f"{c}:{arch}:with_energy"][metric]) for c in cases]
            offset = (i - (n - 1) / 2) * w
            ax.bar(
                x + offset,
                vals,
                width=w * 0.94,
                color=colors.get(arch, NAVY),
                label=arch if ax is axes[0] else None,
            )
        ax.axhline(strict_lim, color=CORAL, linestyle="--", linewidth=1.1, label=f"STRICT ({strict_lim})" if ax is axes[0] else None)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, case_labels, rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0].legend(frameon=True, facecolor="white", ncol=2, loc="upper left", fontsize=8.5)
    out = FIG_DIR / "fig_stage2_promoted_cases.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_stage3_preliminar(promo: dict, s3: dict) -> Path:
    """Resume medianas por arquitectura sobre las corridas ya completadas de Etapa 3."""
    promoted = promo["promoted_models"]
    by_arch: dict[str, list[dict]] = defaultdict(list)
    cases_done: set[str] = set()
    for rec in s3.values():
        arch = rec.get("architecture")
        if arch in promoted and rec.get("status") == "completed":
            by_arch[arch].append(rec)
            if rec.get("case"):
                cases_done.add(str(rec["case"]))

    archs = [a for a in promoted if by_arch[a]]
    n_total = sum(len(by_arch[a]) for a in archs)

    med_nrmse = [statistics.median(r["nrmse_pct"] for r in by_arch[a] if r.get("nrmse_pct") is not None) for a in archs]
    med_dt = [statistics.median(r["error_Tmax_abs_K"] for r in by_arch[a] if r.get("error_Tmax_abs_K") is not None) for a in archs]
    med_bal = [statistics.median(r["balance_pct"] for r in by_arch[a] if r.get("balance_pct") is not None) for a in archs]

    bar_colors = [TEAL if a == "M0R" else BLUE if a == "M1" else PURPLE if a == "FBPINN" else SLATE if a == "DCSNN" else AMBER for a in archs]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8), constrained_layout=True)

    panels = [
        (med_nrmse, "Mediana NRMSE (%)", "Mediana NRMSE (%)", 0.25),
        (med_dt, "Mediana |dTmax| (K)", "Mediana Error Tmax (K)", 0.05),
        (med_bal, "Mediana Desbalance (%)", "Mediana Desbalance (%)", 0.50),
    ]
    x = np.arange(len(archs))
    for ax, (vals, ylabel, title, lim) in zip(axes, panels):
        bars = ax.bar(x, vals, width=0.58, color=bar_colors)
        ax.axhline(lim, color=CORAL, linestyle="--", linewidth=1.1, label=f"STRICT ({lim})")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, archs, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.22 if vals else 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=True, facecolor="white", loc="upper left", fontsize=8.5)
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        f"Etapa 3 en curso (presupuesto corto 1000+50): {n_total} corridas completadas en {len(cases_done)} casos",
        fontsize=12.5,
        fontweight="bold",
        color=NAVY,
    )
    out = FIG_DIR / "fig_stage3_preliminar.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    promo = _load_json(WS_DIR / "promoted_models.json")
    s2 = _load_json(WS_DIR / "stage2_selection_summary.json")
    s3_path = WS_DIR / "stage3_multicase_smoke" / "summary.json"
    s3 = _load_json(s3_path) if s3_path.is_file() else {}

    outs = [
        make_stage2_energy_impact(promo),
        make_stage2_promoted_cases(promo, s2),
        make_stage3_preliminar(promo, s3),
    ]
    for p in outs:
        print(f"[OK] Generada figura: {p.relative_to(ROOT_DIR)} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
