"""Pruebas de compatibilidad entre modelos, reanudación, transferencia y encadenamiento."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from pinn_thermal.core.artifact import SolutionArtifact
from pinn_thermal.core.base_model import ModelRegistry
from pinn_thermal.core.problem import ThermalProblem2D
from pinn_thermal.core.trainer import PINNTrainer, TrainConfig
from pinn_thermal.core.transfer import CompatibilityChecker


def test_all_continuation_and_transfer_routes(tmp_path: Path, mms_problem: ThermalProblem2D) -> None:
    pts, _, _ = mms_problem.evaluation(per_region=25)

    # 0) Entrenar modelo base M1
    base_dir = tmp_path / "base_m1"
    cfg_base = TrainConfig(adam_steps=30, lbfgs_steps=0, width=16, depth=2, device="cpu")
    art_base = PINNTrainer(cfg_base).train(mms_problem, "M1", base_dir, overwrite=True)

    # 1) Reanudación exacta
    res_dir = tmp_path / "resume_m1"
    cfg_res = TrainConfig(adam_steps=50, lbfgs_steps=0, width=16, depth=2, device="cpu", resume_from=base_dir)
    art_res = PINNTrainer(cfg_res).train(mms_problem, "M1", res_dir, overwrite=True)
    assert art_res.result["transfer_report"]["method"] == "exact_resume"
    assert len(art_res.result["history"]) > len(art_base.result["history"])

    # 2) Transferencia de pesos hacia un JSON MODIFICADO pero compatible en características críticas
    doc_mod = copy.deepcopy(mms_problem.doc)
    doc_mod["id"] = "mms_constant_modified_source"
    doc_mod["regions"][0]["source"] = 1.2 * float(doc_mod["regions"][0]["source"]) if isinstance(doc_mod["regions"][0]["source"], (int, float)) else {"expression": "1.2 * (" + doc_mod["regions"][0]["source"]["expression"] + ")"}
    prob_mod = ThermalProblem2D.from_json(doc_mod)

    tgt_cand = ModelRegistry.build(prob_mod, "M1", width=16, depth=2)
    comp = CompatibilityChecker.check(art_base, tgt_cand)
    assert comp.compatible and comp.level == "weights_compatible" and comp.same_physics is False

    warm_dir = tmp_path / "warm_m1"
    art_warm = PINNTrainer(
        TrainConfig(adam_steps=20, lbfgs_steps=0, width=16, depth=2, device="cpu", warm_start_from=base_dir, transfer_mode="weights")
    ).train(prob_mod, "M1", warm_dir, overwrite=True)
    assert art_warm.result["transfer_report"]["method"] == "weight_transfer"

    # 3) Transferencia funcional entre arquitecturas incompatibles (M1 -> CUSPR y M1 -> M2 mixto)
    for tgt_arch, tmode in [("CUSPR", "functional_readout"), ("M2", "functional_pretrain")]:
        tdir = tmp_path / f"xf_{tgt_arch}"
        art_xf = PINNTrainer(
            TrainConfig(
                adam_steps=20, lbfgs_steps=0, width=16, depth=2, device="cpu",
                warm_start_from=base_dir, transfer_mode=tmode, transfer_steps=25,
            )
        ).train(prob_mod, tgt_arch, tdir, overwrite=True)
        assert np.all(np.isfinite(art_xf.predict(pts)["T"]))

    # 4) Encadenamiento directo ChainedPINN (residual y feature) + recarga portable
    for cmode in ("residual", "feature"):
        cdir = tmp_path / f"chain_{cmode}"
        art_ch = PINNTrainer(
            TrainConfig(
                adam_steps=20, lbfgs_steps=0, width=16, depth=2, device="cpu",
                chain_from=base_dir, chain_mode=cmode,
            )
        ).train(prob_mod, "CUSPR", cdir, overwrite=True)
        assert art_ch.architecture == "CHAINED(CUSPR)"
        reloaded = SolutionArtifact.load(cdir)
        assert np.allclose(art_ch.predict(pts)["T"], reloaded.predict(pts)["T"], atol=1e-6)

    # 5) Exportar e importar ZIP portable entre máquinas
    zip_path = art_base.export_portable_zip(tmp_path / "portable.zip")
    imported = SolutionArtifact.import_portable_zip(zip_path, tmp_path / "unpacked")
    assert np.allclose(art_base.predict(pts)["T"], imported.predict(pts)["T"], atol=1e-6)
