# Configuración de FEniCSx (DolfinX + Gmsh P2)

Las soluciones de referencia por elementos finitos de `PINN_V4` emplean **Gmsh + DolfinX + PETSc MUMPS** con elementos cuadráticos de Lagrange (**P2**).

## 1. Diseño Libre de PyTorch en el Trabajador FEM

El paquete separa intencionalmente:
- `pinn_thermal.core.{expressions,fields,shapes,schema,problem,manifest}` + `pinn_thermal.fem.worker`: **no importan `torch`**.
- `pinn_thermal.fem.runner.FEniCSxAdapter`: orquesta el lanzamiento desde el entorno principal (donde sí corre PyTorch).

Gracias a esta separación, el entorno `fenicsx` no necesita tener instalado PyTorch ni acceso GPU.

---

## 2. Operación Actual en Windows mediante WSL2 (Configuración por Defecto)

En esta máquina la referencia FEM funciona mediante la instalación existente en WSL2 (`Ubuntu`):
- Distribución: `Ubuntu`
- Intérprete Python WSL: `/home/lkoc/miniforge3/envs/fenicsx/bin/python`

Al invocar `FEniCSxAdapter().solve(problem, output_dir, level=2)` o `pinn-thermal fem ...`, `PINN_V4` traduce automáticamente las rutas Windows (`C:\...` -> `/mnt/c/...`), apunta `PYTHONPATH` hacia `PINN_V4/src` y ejecuta `python -m pinn_thermal.fem.worker` dentro de WSL2 de manera totalmente autocontenida.

Si su instalación WSL usa otro nombre de distribución o ruta de Conda/Mamba, puede sobreescribirla mediante variables de entorno:

```powershell
$env:PINN_V4_FEM_MODE = "wsl"
$env:PINN_V4_WSL_DISTRO = "Ubuntu"
$env:PINN_V4_FEM_PYTHON = "/home/usuario/miniforge3/envs/fenicsx/bin/python"
```

---

## 3. Opciones Alternativas para Otras Máquinas (Linux Nativo o Docker)

Al trasladar `PINN_V4` a otro computador puede elegir cualquiera de estos backends vía `PINN_V4_FEM_MODE`:

1. **Linux / macOS Nativo (`PINN_V4_FEM_MODE=native`)**:
   Instale el entorno completo desde Conda-Forge (`mamba create -n pinn-fenicsx -c conda-forge fenics-dolfinx mpich pyvista python-gmsh petsc4py`) e instale `PINN_V4` dentro con `pip install -e . --no-deps`. En modo `auto`, `FEniCSxAdapter` detectará `dolfinx` y `gmsh` locales automáticamente sin usar WSL.
2. **Contenedor Docker (`PINN_V4_FEM_MODE=docker`)**:
   Con Docker Desktop o Docker Engine operativo y la imagen `dolfinx/dolfinx:stable`, el adaptador monta `PINN_V4/src` y la carpeta de salida directamente en el contenedor y ejecuta `pinn_thermal.fem.worker` sin instalaciones adicionales en el sistema anfitrión.
