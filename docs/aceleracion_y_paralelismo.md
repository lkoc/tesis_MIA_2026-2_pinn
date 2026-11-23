# Aceleración Transparente de Hardware y Paralelismo

## 1. Cambio Transparente de Hardware (`DeviceManager`)

`PINN_V4` nunca fija llamadas específicas a un único fabricante (sin `.cuda()` rígidos en el código). Toda selección de dispositivo y precisión pasa por `pinn_thermal.core.device.DeviceManager`:

| Backend PyTorch instalado | Resolución `device="auto"` | Ejemplo de instalación PyTorch externa |
| :--- | :--- | :--- |
| NVIDIA GPU | `cuda:0` | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| AMD GPU | `cuda:0` (ROCm HIP) | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` |
| Intel Arc / Iris Xe | `xpu:0` | Paquete PyTorch XPU / IPEX estándar |
| Huawei Ascend NPU | `npu:0` | `torch_npu` |
| Apple Silicon | `mps` | `pip install torch` en macOS arm64 |
| DirectML | `privateuseone:0` | `torch-directml` |
| CPU multipropósito | `cpu` | Cualquier compilación de `torch` |

En este proyecto basta con sustituir la rueda (*wheel*) de `torch` correspondiente al acelerador del equipo y utilizar `device="auto"` (o indicar explícitamente `--device xpu`, `--device cuda:0`, etc.).

### Política de Precisión Numérica (`float32` vs `float64`)
Algunos aceleradores (como ciertas GPUs integradas Intel XPU o MPS) no admiten operaciones completas en doble precisión (`float64`). `DeviceManager.resolve_dtype()` ejecuta una prueba real sobre el acelerador seleccionado; si `float64` no es soportado y `allow_fallback=True`, emite una advertencia informativa y degrada limpiamente a `float32` sin abortar la corrida.

---

## 2. Ejecución Paralela Multi-Proceso (`ParallelRunner`)

Para campañas con múltiples casos JSON, múltiples semillas o comparaciones entre arquitecturas (`M1`, `CUSPR`, `M0R`, `M2`, `M1_EXACT`):

- Utiliza `concurrent.futures.ProcessPoolExecutor` con contexto **`spawn`** seguro tanto en Windows como en Linux/macOS.
- En modo **CPU**, divide equitativamente los hilos intra-operador (`torch.set_num_threads`) entre procesos para evitar sobre-suscripción de OpenMP.
- En modo **multi-GPU/multi-XPU**, distribuye las tareas por turno rotatorio (*round-robin*) asignando un índice de dispositivo distinto a cada trabajador.

```python
from pinn_thermal.core.parallel import ParallelRunner, RunTask

tasks = [
    RunTask("xlpe_m1", "xlpe_discrete_layers", "M1", "salidas/xlpe_m1"),
    RunTask("xlpe_cuspr", "xlpe_discrete_layers", "CUSPR", "salidas/xlpe_cuspr"),
    RunTask("xlpe_exact", "xlpe_discrete_layers", "M1_EXACT", "salidas/xlpe_exact"),
]
results = ParallelRunner(max_workers=2).run(tasks)
```
