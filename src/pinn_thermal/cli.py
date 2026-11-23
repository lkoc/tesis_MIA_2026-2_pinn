"""Interfaz de línea de comandos (``pinn-thermal`` / ``python -m pinn_thermal``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from pinn_thermal.core.ampacity import AmpacitySolver
from pinn_thermal.core.artifact import SolutionArtifact
from pinn_thermal.core.base_model import ModelRegistry
from pinn_thermal.core.device import DeviceManager
from pinn_thermal.core.evaluation import evaluate_model
from pinn_thermal.core.parallel import ParallelRunner, RunTask
from pinn_thermal.core.problem import CaseCatalog, load_problem
from pinn_thermal.core.trainer import PINNTrainer, TrainConfig
from pinn_thermal.fem.runner import FEniCSxAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinn-thermal",
        description="Librería modular PINN para conducción térmica 2D en cables heterogéneos.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Muestra hardware PyTorch detectado, estado FEniCSx, modelos y casos.")
    sub.add_parser("list-models", help="Lista todas las arquitecturas PINN registradas con autores.")
    sub.add_parser("list-cases", help="Lista todos los casos curados en benchmarks/ y thesis_curated/.")

    p_val = sub.add_parser("validate", help="Valida un archivo problem.json o identificador del catálogo.")
    p_val.add_argument("problem", help="Ruta JSON o ID de caso.")

    p_tr = sub.add_parser("train", help="Entrena un modelo PINN sobre un archivo problem.json o ID.")
    p_tr.add_argument("--problem", required=True, help="Ruta JSON o ID de caso.")
    p_tr.add_argument("--arch", default="M1_EXACT", help="Arquitectura PINN registrada.")
    p_tr.add_argument("--out", required=True, help="Directorio destino del artefacto.")
    p_tr.add_argument("--adam-steps", type=int, default=1000)
    p_tr.add_argument("--lbfgs-steps", type=int, default=60)
    p_tr.add_argument("--width", type=int, default=32)
    p_tr.add_argument("--depth", type=int, default=3)
    p_tr.add_argument("--activation", default="tanh")
    p_tr.add_argument("--device", default="auto")
    p_tr.add_argument("--dtype", default="float32")
    p_tr.add_argument("--initializer", default=None)
    p_tr.add_argument("--resume-from", default=None)
    p_tr.add_argument("--warm-start", default=None)
    p_tr.add_argument("--transfer-mode", default="auto")
    p_tr.add_argument("--chain-from", default=None)
    p_tr.add_argument("--chain-mode", default="residual")
    p_tr.add_argument("--focused", action="store_true")
    p_tr.add_argument("--ampacity", type=float, default=None, help="Calcula además ampacidad para esta Tmax [°C].")
    p_tr.add_argument("--overwrite", action="store_true")

    p_xf = sub.add_parser("transfer", help="Transfiere o encadena un artefacto existente hacia un modelo nuevo.")
    p_xf.add_argument("--source", required=True, help="Ruta del artefacto origen (PINN, FEM o RBF).")
    p_xf.add_argument("--target-problem", required=True, help="JSON o ID del problema destino.")
    p_xf.add_argument("--target-arch", default="M1")
    p_xf.add_argument("--out", required=True)
    p_xf.add_argument("--mode", default="auto", choices=["auto", "weights", "functional_readout", "functional_readout_gls", "functional_pretrain", "chain_residual", "chain_feature"])
    p_xf.add_argument("--adam-steps", type=int, default=400)
    p_xf.add_argument("--lbfgs-steps", type=int, default=40)
    p_xf.add_argument("--overwrite", action="store_true")

    p_ev = sub.add_parser("eval", help="Evalúa un artefacto PINN guardado (opcionalmente contra FEM/otro artefacto).")
    p_ev.add_argument("--artifact", required=True)
    p_ev.add_argument("--reference", default=None, help="Ruta opcional a artefacto FEM/PINN de referencia.")

    p_fem = sub.add_parser("fem", help="Resuelve referencia P2 con FEniCSx (WSL2 / nativo / Docker).")
    p_fem.add_argument("--problem", required=True)
    p_fem.add_argument("--out", required=True)
    p_fem.add_argument("--level", type=int, default=2)

    p_par = sub.add_parser("parallel", help="Ejecuta múltiples pares problema x arquitectura en paralelo.")
    p_par.add_argument("--problems", nargs="+", required=True)
    p_par.add_argument("--archs", nargs="+", required=True)
    p_par.add_argument("--out-root", required=True)
    p_par.add_argument("--workers", type=int, default=None)
    p_par.add_argument("--adam-steps", type=int, default=500)
    p_par.add_argument("--lbfgs-steps", type=int, default=30)

    p_ui = sub.add_parser("ui", help="Inicia la interfaz gráfica web Streamlit.")
    p_ui.add_argument("--port", type=int, default=8501)
    return parser


def main(argv: list[str] | None = None) -> int:
    import pinn_thermal.models  # noqa: F401 - registra arquitecturas

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        payload = {
            "device": DeviceManager.describe(),
            "fenicsx": FEniCSxAdapter().is_available(),
            "models": ModelRegistry.list_models(),
            "cases_count": len(CaseCatalog().list_entries()),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "list-models":
        print(json.dumps(ModelRegistry.describe_all(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "list-cases":
        entries = [{k: (str(v) if isinstance(v, Path) else v) for k, v in e.items()} for e in CaseCatalog().list_entries()]
        print(json.dumps(entries, indent=2, ensure_ascii=False))
        return 0

    if args.command == "validate":
        prob = load_problem(args.problem)
        print(
            json.dumps(
                {
                    "valid": True,
                    "id": prob.id,
                    "family": prob.family,
                    "regions": [r.name for r in prob.regions],
                    "physics_sha256": prob.physics_sha256,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "fem":
        art = FEniCSxAdapter().solve(args.problem, output_dir=args.out, level=args.level)
        print(json.dumps(art.result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "train":
        cfg = TrainConfig(
            adam_steps=args.adam_steps,
            lbfgs_steps=args.lbfgs_steps,
            width=args.width,
            depth=args.depth,
            activation=args.activation,
            device=args.device,
            dtype=args.dtype,
            initializer=args.initializer,
            focused=args.focused,
            resample_every=200 if args.focused else 0,
            resume_from=args.resume_from,
            warm_start_from=args.warm_start,
            transfer_mode=args.transfer_mode,
            chain_from=args.chain_from,
            chain_mode=args.chain_mode,
        )
        art = PINNTrainer(cfg).train(args.problem, args.arch, args.out, overwrite=args.overwrite)
        summary = {"status": art.result.get("status"), "best_loss": art.result.get("best_loss"), "evaluation": art.result.get("evaluation")}
        if args.ampacity is not None:
            amp = AmpacitySolver(target_temp_C=args.ampacity).solve_with_pinn(
                art.problem, architecture=args.arch, config=cfg, work_dir=Path(args.out) / "ampacity"
            )
            summary["ampacity"] = {
                "converged": amp.converged,
                "I_mid_A": amp.I_mid_A,
                "power_W_m": amp.power_W_m,
                "Tmax_at_mid_C": amp.Tmax_at_mid_C,
            }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "transfer":
        is_chain = args.mode.startswith("chain_")
        cfg = TrainConfig(
            adam_steps=args.adam_steps,
            lbfgs_steps=args.lbfgs_steps,
            warm_start_from=None if is_chain else args.source,
            transfer_mode="auto" if is_chain else args.mode,
            chain_from=args.source if is_chain else None,
            chain_mode=args.mode.replace("chain_", "", 1) if is_chain else "residual",
        )
        art = PINNTrainer(cfg).train(args.target_problem, args.target_arch, args.out, overwrite=args.overwrite)
        print(json.dumps({"status": art.result.get("status"), "transfer_report": art.result.get("transfer_report")}, indent=2, ensure_ascii=False))
        return 0

    if args.command == "eval":
        art = SolutionArtifact.load(args.artifact)
        if art.kind != "pinn" or art.model is None:
            print(json.dumps(art.result, indent=2, ensure_ascii=False))
            return 0
        ref_arr = None
        ref_lbl = "supplied"
        if args.reference is not None:
            ref_art = SolutionArtifact.load(args.reference)
            pts, _, _ = art.problem.evaluation()
            ref_arr = ref_art.predict(pts)["T"]
            ref_lbl = args.reference
        rep, _ = evaluate_model(art.model, reference=ref_arr, reference_label=ref_lbl)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0

    if args.command == "parallel":
        out_root = Path(args.out_root)
        tasks = [
            RunTask(
                name=f"{Path(p).stem}__{a}",
                problem_source=p,
                architecture=a,
                output_dir=str(out_root / f"{Path(p).stem}__{a}"),
                config_kwargs={"adam_steps": args.adam_steps, "lbfgs_steps": args.lbfgs_steps},
                overwrite=True,
            )
            for p in args.problems
            for a in args.archs
        ]
        results = ParallelRunner(max_workers=args.workers).run(tasks)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0

    if args.command == "ui":
        app_path = Path(__file__).resolve().parents[2] / "frontend" / "app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(args.port)])

    return 1
