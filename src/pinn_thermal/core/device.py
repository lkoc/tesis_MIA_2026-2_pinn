"""Gestión transparente de dispositivos de cómputo (CPU, CUDA/ROCm, Intel XPU, NPU, MPS, DirectML).

Permite que el cambio entre CPU y cualquier acelerador soportado por PyTorch sea
completamente transparente con solo instalar la distribución correspondiente de ``torch``
o seleccionar ``device="auto"`` / ``device="gpu"``.
"""

from __future__ import annotations

import os
import random
from typing import Any
import warnings

import numpy as np
import torch

DTYPES: dict[str, torch.dtype] = {"float32": torch.float32, "float64": torch.float64}

_ORIG_TENSOR_DOT = torch.Tensor.dot
_ORIG_TORCH_DOT = torch.dot
_XPU_DOT_PATCHED: bool = False


def _xpu_safe_tensor_dot(self: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """Producto punto 1D nativo en XPU para evitar el fallback CPU de ``aten::dot``."""
    if (
        self.device.type == "xpu"
        and tensor.device.type == "xpu"
        and not self.is_complex()
        and not tensor.is_complex()
    ):
        if self.dim() != 1 or tensor.dim() != 1:
            raise RuntimeError(f"1D tensors expected, but got {self.dim()}D and {tensor.dim()}D tensors")
        if self.numel() != tensor.numel():
            raise RuntimeError(
                f"inconsistent tensor size, expected tensor [{self.numel()}] and src [{tensor.numel()}] "
                "to have the same number of elements"
            )
        if self.dtype != tensor.dtype:
            raise RuntimeError(f"expected both vectors to have same dtype, but found {self.dtype} and {tensor.dtype}")
        return torch.sum(self * tensor)
    return _ORIG_TENSOR_DOT(self, tensor)


def _xpu_safe_torch_dot(
    input: torch.Tensor,
    tensor: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Envoltorio nativo de ``torch.dot`` para tensores 1D en XPU sin fallback a CPU."""
    if (
        input.device.type == "xpu"
        and tensor.device.type == "xpu"
        and not input.is_complex()
        and not tensor.is_complex()
    ):
        res = _xpu_safe_tensor_dot(input, tensor)
        if out is not None:
            return out.copy_(res)
        return res
    if out is not None:
        return _ORIG_TORCH_DOT(input, tensor, out=out)
    return _ORIG_TORCH_DOT(input, tensor)


class DeviceManager:
    """Detector y gestor unificado de hardware y precisión numérica en PyTorch."""

    @staticmethod
    def patch_xpu_ops() -> bool:
        """Parchea operadores faltantes en Intel XPU (``aten::dot``) usando kernels nativos XPU.

        En PyTorch XPU (p. ej. 2.9+xpu), ``torch.optim.LBFGS`` llama repetidamente a
        ``Tensor.dot()``, cuyo kernel ``aten::dot`` cae a CPU. Reemplazarlo por
        ``torch.sum(a * b)`` mantiene el 100 % de las operaciones de L-BFGS en XPU
        sin advertencias ni copias XPU<->CPU.
        """
        global _XPU_DOT_PATCHED
        xpu_mod = getattr(torch, "xpu", None)
        if xpu_mod is None or not xpu_mod.is_available():
            return False
        if not _XPU_DOT_PATCHED:
            torch.Tensor.dot = _xpu_safe_tensor_dot  # type: ignore[method-assign]
            torch.dot = _xpu_safe_torch_dot  # type: ignore[assignment]
            _XPU_DOT_PATCHED = True
        return True

    @staticmethod
    def available_backends() -> dict[str, dict[str, Any]]:
        """Inspecciona todos los backends de PyTorch disponibles en la máquina actual."""
        backends: dict[str, dict[str, Any]] = {}

        # NVIDIA CUDA o AMD ROCm (PyTorch ROCm expone el mismo namespace torch.cuda con torch.version.hip)
        cuda_ok = bool(torch.cuda.is_available())
        hip_ver = getattr(torch.version, "hip", None)
        backends["cuda"] = {
            "available": cuda_ok,
            "count": torch.cuda.device_count() if cuda_ok else 0,
            "kind": "rocm" if hip_ver else "nvidia_cuda",
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda_ok else [],
        }

        # Intel XPU (Arc / Iris Xe / Data Center GPU)
        xpu_mod = getattr(torch, "xpu", None)
        xpu_ok = bool(xpu_mod is not None and xpu_mod.is_available())
        backends["xpu"] = {
            "available": xpu_ok,
            "count": xpu_mod.device_count() if xpu_ok else 0,
            "devices": [xpu_mod.get_device_name(i) for i in range(xpu_mod.device_count())] if xpu_ok else [],
        }

        # Unidades NPU (p. ej. Huawei Ascend via torch_npu o backends vendor compatibles)
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is None:
            try:
                import torch_npu  # type: ignore[import-not-found]
                npu_mod = getattr(torch_npu, "npu", None) or getattr(torch, "npu", None)
            except ImportError:
                npu_mod = None
        npu_ok = bool(npu_mod is not None and npu_mod.is_available())
        backends["npu"] = {
            "available": npu_ok,
            "count": npu_mod.device_count() if npu_ok and hasattr(npu_mod, "device_count") else (1 if npu_ok else 0),
            "devices": (
                [npu_mod.get_device_name(i) for i in range(npu_mod.device_count())]
                if npu_ok and hasattr(npu_mod, "get_device_name")
                else (["NPU"] if npu_ok else [])
            ),
        }

        # Apple Metal Performance Shaders (MPS)
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        mps_ok = bool(mps_backend is not None and mps_backend.is_available())
        backends["mps"] = {
            "available": mps_ok,
            "count": 1 if mps_ok else 0,
            "devices": ["Apple MPS"] if mps_ok else [],
        }

        # Microsoft DirectML opcional
        dml_ok = False
        try:
            import torch_directml  # type: ignore[import-not-found]
            dml_ok = bool(torch_directml.is_available())
        except ImportError:
            dml_ok = False
        backends["dml"] = {"available": dml_ok, "count": 1 if dml_ok else 0, "devices": ["DirectML"] if dml_ok else []}

        backends["cpu"] = {"available": True, "count": os.cpu_count() or 1, "devices": ["CPU"]}
        return backends

    @classmethod
    def accelerator_priority(cls) -> list[str]:
        """Devuelve la lista ordenada de aceleradores activos disponibles."""
        info = cls.available_backends()
        order = ("cuda", "xpu", "npu", "mps", "dml")
        return [name for name in order if info[name]["available"]]

    @classmethod
    def resolve(cls, requested: str | torch.device = "auto") -> torch.device:
        """Resuelve la solicitud de dispositivo hacia un objeto ``torch.device`` válido.

        Valores admitidos:
        - ``"auto"``: selecciona el primer acelerador disponible (CUDA/ROCm > XPU > NPU > MPS > DML) o CPU.
        - ``"gpu"``: exige algún acelerador activo (lanza ``RuntimeError`` si solo hay CPU).
        - ``"cpu"``, ``"cuda"``, ``"cuda:N"``, ``"rocm"``, ``"xpu"``, ``"xpu:N"``, ``"npu"``, ``"mps"``, ``"dml"``.
        """
        if isinstance(requested, torch.device):
            if requested.type == "xpu":
                cls.patch_xpu_ops()
            return requested
        req = str(requested).strip().lower()
        info = cls.available_backends()

        if req == "auto":
            active = cls.accelerator_priority()
            if not active:
                return torch.device("cpu")
            chosen = active[0]
            if chosen == "dml":
                import torch_directml  # type: ignore[import-not-found]
                return torch_directml.device()
            if chosen == "xpu":
                cls.patch_xpu_ops()
            return torch.device(chosen)

        if req == "gpu":
            active = cls.accelerator_priority()
            if not active:
                raise RuntimeError("Se solicitó device='gpu', pero PyTorch no detectó ningún acelerador activo.")
            chosen = active[0]
            if chosen == "dml":
                import torch_directml  # type: ignore[import-not-found]
                return torch_directml.device()
            if chosen == "xpu":
                cls.patch_xpu_ops()
            return torch.device(chosen)

        if req == "cpu":
            return torch.device("cpu")

        if req in ("rocm", "cuda") or req.startswith("cuda:"):
            if not info["cuda"]["available"]:
                raise RuntimeError(f"Se solicitó {requested!r}, pero CUDA/ROCm no está disponible en este PyTorch.")
            return torch.device("cuda" if req == "rocm" else req)

        if req == "xpu" or req.startswith("xpu:"):
            if not info["xpu"]["available"]:
                raise RuntimeError(f"Se solicitó {requested!r}, pero Intel XPU no está disponible.")
            cls.patch_xpu_ops()
            return torch.device(req)

        if req == "npu" or req.startswith("npu:"):
            if not info["npu"]["available"]:
                raise RuntimeError(f"Se solicitó {requested!r}, pero NPU no está disponible.")
            return torch.device(req)

        if req == "mps":
            if not info["mps"]["available"]:
                raise RuntimeError("Se solicitó 'mps', pero Apple MPS no está disponible.")
            return torch.device("mps")

        if req == "dml":
            if not info["dml"]["available"]:
                raise RuntimeError("Se solicitó 'dml', pero torch-directml no está instalado o disponible.")
            import torch_directml  # type: ignore[import-not-found]
            return torch_directml.device()

        raise ValueError(
            f"Dispositivo no soportado: {requested!r}. "
            "Use 'auto', 'gpu', 'cpu', 'cuda', 'rocm', 'xpu', 'npu', 'mps' o 'dml'."
        )

    @classmethod
    def supports_dtype(cls, device: torch.device, dtype: torch.dtype) -> bool:
        """Comprueba empíricamente si el dispositivo ejecuta operaciones básicas con ``dtype``."""
        if dtype == torch.float32:
            return True
        if device.type == "mps":
            return False
        try:
            a = torch.ones((2, 2), device=device, dtype=dtype)
            _ = (a @ a).sum().item()
            return True
        except Exception:
            return False

    @classmethod
    def resolve_dtype(
        cls,
        device: torch.device,
        requested: str | torch.dtype = "float32",
        allow_fallback: bool = True,
    ) -> torch.dtype:
        """Devuelve el ``torch.dtype`` efectivo; degrada con aviso si el hardware no soporta FP64."""
        if isinstance(requested, torch.dtype):
            dtype = requested
        else:
            key = str(requested).strip().lower()
            if key not in DTYPES:
                raise ValueError(f"dtype debe ser 'float32' o 'float64'; recibido {requested!r}.")
            dtype = DTYPES[key]

        if cls.supports_dtype(device, dtype):
            return dtype

        msg = f"El dispositivo {device} no soporta {dtype}; se utilizará float32."
        if not allow_fallback:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return torch.float32

    @staticmethod
    def synchronize(device: torch.device | str | None) -> None:
        """Sincroniza el flujo asíncrono de ejecución cuando el dispositivo es un acelerador."""
        if device is None:
            return
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(dev)
        elif dev.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        elif dev.type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
        elif dev.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    @staticmethod
    def seed_all(seed: int, device: torch.device | None = None) -> None:
        """Fija semillas deterministas en Python, NumPy y PyTorch (CPU y acelerador)."""
        s = int(seed)
        random.seed(s)
        np.random.seed(s % (2 ** 32))
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(s)
        if device is not None and device.type == "npu" and hasattr(torch, "npu"):
            torch.npu.manual_seed_all(s)

    @staticmethod
    def capture_rng(device: torch.device | None = None) -> dict[str, Any]:
        """Captura el estado completo de los generadores seudoaleatorios."""
        np_state = np.random.get_state()
        state: dict[str, Any] = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "numpy": (
                np_state[0],
                torch.as_tensor(np_state[1].astype(np.int64)),
                np_state[2],
                np_state[3],
                np_state[4],
            ),
        }
        if torch.cuda.is_available():
            state["cuda"] = [t.cpu() for t in torch.cuda.get_rng_state_all()]
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            state["xpu"] = [t.cpu() for t in torch.xpu.get_rng_state_all()]
        return state

    @staticmethod
    def restore_rng(state: dict[str, Any]) -> None:
        """Restaura el estado exacto de los generadores seudoaleatorios."""
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"].cpu())
        np_state = state["numpy"]
        np.random.set_state(
            (np_state[0], np_state[1].cpu().numpy().astype(np.uint32), np_state[2], np_state[3], np_state[4])
        )
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
        if "xpu" in state and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.set_rng_state_all([item.cpu() for item in state["xpu"]])

    @staticmethod
    def configure_threads(num_threads: int | None = None) -> int:
        """Configura el número de hilos intra-op de CPU."""
        if num_threads is not None and num_threads > 0:
            torch.set_num_threads(int(num_threads))
        return int(torch.get_num_threads())

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Devuelve un resumen legible del entorno PyTorch y dispositivos detectados."""
        return {
            "torch_version": str(torch.__version__),
            "backends": cls.available_backends(),
            "recommended_device": str(cls.resolve("auto")),
            "cpu_threads": int(torch.get_num_threads()),
        }


# Activa automáticamente el parche nativo XPU al importar el módulo si XPU está disponible.
DeviceManager.patch_xpu_ops()
