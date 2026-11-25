# Detalle Técnico e Ingeniería de las Implementaciones PINN (`PINN_V4` / `pinn-thermal`)

Este documento técnico describe el diseño matemático, la fidelidad respecto a la literatura fundacional, las modificaciones introducidas para problemas térmicos multiescala y el resultado de la auditoría integral aplicada a **todas** las arquitecturas PINN registradas en `pinn_thermal`.

---

## 1. Principios Arquitectónicos y Generalidad POO

Todo resolvedor PINN del proyecto obedece un contrato estricto orientado a objetos sin lógica *ad hoc* por caso:

1. **Fuente única de verdad (`pinn-v4-problem-1` JSON)**: cada modelo recibe exclusivamente una instancia `ThermalProblem2D` construida desde JSON (constantes, expresiones simbólicas `(x, y)`, reglas `piecewise`, tablas 1D/2D, estratos `layers`, parches `patch`, geometrías `rectangle`, `disk`, `annulus`, `polygon` y condiciones límite `dirichlet`, `neumann`, `robin`).
2. **Interfaz uniforme (`BasePINNModel` y `ModelRegistry`)**: toda arquitectura hereda de `BasePINNModel`, almacena sus bloques entrenables en `self.networks` (`nn.ModuleList`) / `self.offsets` (`nn.ParameterList`), implementa `field(region, xy) -> (T, q_opcional)` y expone `subnetwork_map()` para `WeightTransfer`, `FunctionalTransfer`, `ChainedPINN` y `RBFFDInitializer`.
3. **Transparencia de hardware y paralelismo**: las arquitecturas operan íntegramente en tensores PyTorch agnósticos al dispositivo (`cpu`, `cuda`, `xpu`, `rocm`, `npu`, `mps`, `dml` vía `DeviceManager`, incluyendo el parche nativo `aten::dot` para `L-BFGS` en Intel XPU) y son serializables/ejecutables en paralelo con procesos `spawn` mediante `ParallelRunner`.

---

## 2. Núcleo Físico Adimensional y Correcciones de Auditoría

### 2.1 Función Objetivo Física Compartida (`PhysicsObjective`)
Para $\Omega = \bigcup_{r=0}^{R-1} \Omega_r$, todos los modelos optimizan la suma ponderada de residuos adimensionales:

| Término | Definición matemática normalizada | Peso por defecto | Activo en `exact_constraints=True` |
| :--- | :--- | :---: | :---: |
| `pde` | $(-\nabla\cdot(k_r\nabla T_r) - Q_r)/d_{\text{scale},r}$ (o $\nabla\cdot\mathbf{q}_r - Q_r$ en `M2`) | `1.0` | Sí |
| `constitutive` | $(\mathbf{q}_r + k_r\nabla T_r)/q_{\text{scale},r}$ (solo formulaciones mixtas) | `1.0` | Sí (si `mixed=True`) |
| `interface_temperature` | $(T_i - T_j)/T_{\text{scale}}$ en $\Gamma_{ij}$ | `100.0` | No (analítico) |
| `interface_flux` | $(\mathbf{q}_i\cdot\mathbf{n} - \mathbf{q}_j\cdot\mathbf{n})/f_{\text{scale},\Gamma}$ en $\Gamma_{ij}$ | `100.0` | **Sí** (incluso en `M1_EXACT`/`CUSP_EXACT`) |
| `dirichlet` | $(T - g_D)/T_{\text{scale}}$ en $\partial\Omega_D$ | `100.0` | No si `exact_boundary=True` |
| `neumann` | $(\mathbf{q}\cdot\mathbf{n} - q_n)/q_{\text{scale},r}$ en $\partial\Omega_N$ | `10.0` | No si `exact_boundary=True` |
| `robin` | $(\mathbf{q}\cdot\mathbf{n} - h(T - T_\infty))/q_{\text{scale},r}$ en $\partial\Omega_R$ | `10.0` | No si `exact_boundary=True` |
| `energy` | $\bigl(-\int_{\Omega_r} Q_r\,dA + \int_{\partial\Omega_r} \mathbf{q}\cdot\mathbf{n}\,ds\bigr)/P_{\text{scale}}$ | `0.0` (screening dual `100.0` / `0.0`) | Opcional (`exact_energy=True`) |

> **Por qué `interface_flux` es obligatorio incluso en modelos con `[T]=0` exacto:** sin penalizar el salto de flujo normal $[\mathbf{q}\cdot\mathbf{n}]_{\Gamma} = 0$, los dos lados de una interfaz cable-suelo quedan débilmente acoplados o desacoplados y pueden colapsar a la solución trivial del suelo ($T \approx T_{\text{ext}}$ con residuo PDE $\approx 0$ en el suelo pero salto de flujo $\sim 30\%$).

### 2.2 Hallazgos y Correcciones Estructurales Aplicadas en el Núcleo
1. **Áreas exactas de capas delgadas (`Shape2D.area()` + muestreo local de respaldo en `ThermalProblem2D`)**:
   - *Fallo detectado*: en dominios grandes con estratos múltiples (p. ej. `xlpe_discrete_layers`, suelo $8\times 4\text{ m}$ con pantalla metálica anular de espesor $1\text{ mm}$), la malla global de pre-cálculo ($96\times 96$) dejaba `0` puntos dentro de capas delgadas y asignaba un área mínima degenerada $10^{-12}\text{ m}^2$, inflando `flux_scale` hasta $\sim 1.9\times 10^{10}$ y `divergence` a $10^{16}$.
   - *Corrección general*: se añadió el método abstracto `Shape2D.area()` a todas las geometrías (`RectangleShape`, `DiskShape`, `AnnulusShape`, `PolygonShape`). Cuando la malla global obtiene `< 64` impactos en una región, `ThermalProblem2D` muestrea `4096` puntos locales con `shape.sample_local()` para calcular el área y las propiedades físicas promedio exactas sin supuestos específicos del caso.
2. **Escalamiento pasivo de Fourier en capas sin fuente (`ThermalProblem2D.scales`)**:
   - En regiones con $Q_r = 0$ en presencia de fuentes activas ($P_{\text{abs}} > 0$), `length = min(sqrt(area), shape.characteristic_size())` y la escala térmica local usa $\Delta T_{\text{local}, r} = \min\bigl(T_{\text{scale}}, \max(\frac{2\pi P_{\text{abs}} L_r^2}{k_r A_r}, 10^{-6} T_{\text{scale}})\bigr)$, manteniendo `flux_scale` y `divergence` en órdenes físicamente coherentes en anillos delgados conductores/aislantes.
3. **Regularización $C^\infty$ en el centro de conductores (`r = 0`)**:
   - *Fallo detectado*: `torch.linalg.vector_norm(xy - center)` en `_source_distance_features` y en `level_sets("circle")` tiene derivada singular $O(1/r)$ y Laplaciano singular $O(1/r^2)$ en `r = 0`, generando `NaN` al evaluar gradientes/Hessianos en el centro exacto de discos fuente.
   - *Corrección*: `_source_distance_features` usa `0.5 * torch.log1p(r2 / rm2)` (sin `.clamp(-1, 1)` que introducía quiebres $C^0$ en $r=r_m$) y `level_sets("circle")` usa $\sqrt{\|\mathbf{x}-\mathbf{c}\|^2 + \varepsilon_r^2} - \sqrt{R^2 + \varepsilon_r^2}$ con $\varepsilon_r = \max(0.1 R, 10^{-6})$, ambas estrictamente $C^\infty$ en todo $\mathbb{R}^2$.

---

## 3. Catálogo Técnico Detallado por Arquitectura PINN

### 3.1 Línea Base y Escalamiento Multiescala (`models/baseline.py`, `models/dcsnn.py`)

#### `M0` (`M0GlobalPINN`)
- **Paper fundacional**: Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. *Journal of Computational Physics*, 378, 686–707. DOI: `10.1016/j.jcp.2018.10.045`.
- **Formulación original**: única red MLP global $\mathcal{N}(\mathbf{x};\theta)$ que predice un campo escalar continuo $T(\mathbf{x})$ en todo $\Omega$ penalizando los residuos PDE y BCs en colocación.
- **Auditoría y corrección**: la versión previa usaba una salida multi-cabezal por región alimentada con coordenadas locales normalizadas por el radio del conductor ($\nabla \sim 1/r_{\text{cond}}$) mientras escalaba la salida por $T_{\text{scale}}$ global, generando pérdidas iniciales de $\sim 2.2\times 10^8$ y gradientes $\sim 6\times 10^8$. Se corrigió a una **red escalar continua única** fiel a Raissi et al. (2019) alimentada únicamente con `normalized_xy(xy)`, con un único sesgo global entrenable `offsets[0]`.
- **Capacidad física**: resuelve con alta precisión problemas `mms_*` suaves; en cables multicapa actúa como línea base clásica sin picos de gradiente artificiales (subestima picos extremos de flujo en conductores compactos al no descomponer escalas, rol que cumplen `M0R`, `M1`, `CUSPR` y `M1_EXACT`).

#### `M0R` (`RegionScaledGlobalPINN`) *(Retenida en tesis)*
- **Base fundacional**: Raissi et al. (2019), DOI `10.1016/j.jcp.2018.10.045`, con extensión multiescala propia de la tesis.
- **Formulación**: red compartida `global` con codificación por región (`partition="global"`) y $R$ cabezales de salida, cada uno escalado por su amplitud física local $\Delta T_{\text{local},r}$ y sesgo `offsets[region.id]`:
  $$T_r(\mathbf{x}) = T_{\text{ref}} + T_{\text{scale}}\,b_r + \Delta T_{\text{local}, r}\,\mathcal{N}_r(\phi_{\text{global}}(\mathbf{x})).$$
- **Capacidad física**: elimina el sesgo de escala de `M0` en `cable_disk`, `xlpe_*`, `aras_flat` y suelos estratificados/heterogéneos.

#### `M1` (`M1SubdomainPINN`) *(Retenida en tesis)*
- **Papers fundacionales**:
  - Jagtap, A. D., Kharazmi, E., & Karniadakis, G. E. (2020). *Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems*. *Computer Methods in Applied Mechanics and Engineering*, 365, 113028. DOI: `10.1016/j.cma.2020.113028` (**cPINN**).
  - Jagtap, A. D., & Karniadakis, G. E. (2020). *Extended physics-informed neural networks (XPINNs): A generalized space-time domain decomposition based deep learning framework for nonlinear differential equations*. *Communications in Computational Physics*, 28(5), 2002–2041. DOI: `10.4208/cicp.OA-2020-0164` (**XPINN**).
- **Formulación y modificaciones**: asigna una subred independiente $\mathcal{N}_r$ por subregión física $\Omega_r$ con mapeo local propio: cartesiano normalizado + distancias log-radiales en `rectangle`, cartesiano normalizado en `disk`/`polygon`, y coordenadas polares periódicas $(\tilde{\rho}, \cos\theta, \sin\theta)$ sin salto en $\theta = \pm\pi$ en `annulus`.
- **Capacidad física**: soporta naturalmente cualquier geometría, conductividad heterogénea, capas múltiples y planos trifásicos (`aras_flat`, `xlpe_discrete_layers`, `kim_*`, `annulus`, `mms_*`).

#### `M2` (`M2MixedFluxPINN`) *(Retenida en tesis)*
- **Paper fundacional**: Gladstone, R. J., Nabian, M. A., Viana, F. A., & Meidani, H. (2022). *FO-PINN: A First-Order formulation for Physics Informed Neural Networks*. `arXiv:2211.09321`.
- **Formulación y modificaciones**: cada subred regional predice 3 salidas $(T, q_x, q_y)$ escaladas con $\Delta T_{\text{local},r}$ y `scales(region)["flux"]`. Resuelve el sistema de primer orden $\nabla\cdot\mathbf{q} - Q = 0$ y $\mathbf{q} + k\nabla T = 0$, reduciendo derivadas de orden 2 a orden 1.
- **Auditoría**: tras corregir `Shape2D.area()` y `scales()` para capas delgadas en `xlpe_discrete_layers`, la pérdida inicial de `M2` bajó de `7.21 × 10⁸` a escala normal ($\sim 10^2$–$10^3$).

---

### 3.2 Métodos Enriquecidos para Interfaces y Descomposición de Dominio (`models/cusp.py`, `models/dcsnn.py`, `models/fbpinn.py`)

#### `CUSP` (`CuspPINN`) y `CUSPR` (`CuspRegionScaledPINN`) *(CUSPR retenida en tesis)*
- **Paper fundacional**: Tseng, Y.-H., Lin, T.-S., Hu, W.-F., & Lai, M.-C. (2023). *A cusp-capturing PINN for elliptic interface problems*. *Journal of Computational Physics*, 488, 112359. DOI: `10.1016/j.jcp.2023.112359` (`arXiv:2210.08424`).
- **Formulación original**: una única red continua global enriquece su entrada con el valor absoluto $|\phi_\Gamma(\mathbf{x})|$ de las funciones de nivel hacia cada interfaz material. Como $|\phi_\Gamma|$ es $C^0$ con cambio de pendiente en $\Gamma$, captura quiebres exactos en $\nabla T$ manteniendo $[T]=0$ analíticamente.
- **Modificaciones**:
  - `CUSP`: extracción automática de nivel cero desde el JSON para círculos, coronas, subrectángulos, polígonos y planos internos, con escala global unificada `max(T_scale, amplitudes)`.
  - **`CUSPR`** (aporte original de la tesis): conserva la base enriquecida con cúspides $|\phi_\Gamma|/L_{\text{dom}}$ y añade cabezales/amplitudes regionales $\Delta T_{\text{local},r}$ + sesgos por región, resolviendo disparidades multiescala en cables subterráneos.
- **Auditoría**: la regularización $C^\infty$ de `level_sets("circle")` y `_source_distance_features` eliminó los `NaN` de segunda derivada en `r = 0`.

#### `DCSNN` (`DCSNNPINN`)
- **Paper fundacional**: Hu, W.-F., Lin, T.-S., & Lai, M.-C. (2022). *A discontinuity capturing shallow neural network for elliptic interface problems*. *Journal of Computational Physics*, 469, 111576. DOI: `10.1016/j.jcp.2022.111576`.
- **Formulación original**: extiende el dominio $\mathbb{R}^d$ mediante una coordenada aumentada discontinua $z(\mathbf{x})$ que etiqueta cada subdominio junto con funciones de nivel, permitiendo representar soluciones discontinuas por tramos o con saltos de derivada usando una red compartida.
- **Modificaciones y auditoría**: se generalizó $z(\mathbf{x})$ a un selector *one-hot* de longitud arbitraria $R$ y nivel sets normalizados desde JSON. En la auditoría se detectó que `DCSNNPINN` compartía un único sesgo escalar `offsets[0]` entre todas las regiones; se actualizó a `offsets[region.id]` por región manteniendo total fidelidad multiregión.

#### `FBPINN` (`FBPINNPINN`)
- **Paper fundacional**: Moseley, B., Markham, A., & Nissen-Meyer, T. (2023). *Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations*. *Advances in Computational Mathematics*, 49, 62. DOI: `10.1007/s10444-023-10065-9` (`github.com/benmoseley/FBPINNs`).
- **Formulación original**: descompone regiones extensas en subdominios solapados con normalización local en $[-1, 1]$ por tile y ponderación suave por partición de la unidad $\sum_j \hat{\omega}_j(\mathbf{x}) = 1$, sin caer a cero en la frontera externa global del dominio.
- **Auditoría y corrección**: la implementación previa usaba ventanas coseno $\cos^2(\frac{\pi}{2} u)$ que caían a `0` en los 4 bordes externos del rectángulo y añadía `+ 1e-6`, produciendo `weight_sum ≈ 4e-6`, discontinuidades $C^1$ en el Laplaciano ($\sim 10^5$) y pérdidas iniciales de $1.87\times 10^7$ (`mms_sine`) a $5.71\times 10^7$ (`cable_disk`). Se reimplementó con ventanas tensorizadas **smootherstep $C^2$** ($6t^5 - 15t^4 + 10t^3$) que solo transicionan en los solapamientos interiores y valen `1.0` hacia los bordes globales del rectángulo, sin `+ 1e-6`. Con ello la pérdida inicial descendió a $\sim 10^3$ y en solo `400+20` pasos convergió a `NRMSE = 0.36%` en `mms_sine` y `best_loss = 0.0586` en `cable_disk`.

---

### 3.3 Imposición Analítica Exacta de Fronteras e Interfaces (`models/exact.py`)

- **Papers fundacionales**:
  - McFall, K. S., & Mahan, J. R. (2009). *Artificial neural network method for solution of boundary value problems with exact satisfaction of arbitrary boundary conditions*. *IEEE Transactions on Neural Networks*, 20(8), 1221–1233. DOI: `10.1109/TNN.2009.2020735`.
  - Sheng, H., & Yang, C. (2021). *PFNN: A penalty-free neural network method for solving a class of second-order boundary-value problems on complex geometries*. *Journal of Computational Physics*, 428, 110085. DOI: `10.1016/j.jcp.2020.110085`.
  - Sukumar, N., & Srivastava, A. (2022). *Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks*. *Computer Methods in Applied Mechanics and Engineering*, 389, 114333. DOI: `10.1016/j.cma.2021.114333`.

#### Corrección Crítica en Auditoría: Forzamiento Analítico Real de Dirichlet, Neumann y Robin
Previamente, `_boundary_lift_and_distance` imponía de forma exacta únicamente los lados `dirichlet`, mientras `PhysicsObjective` omitía todas las condiciones de contorno cuando `exact_boundary=True`, dejando lados `neumann` y `robin` sin imposición exacta ni débil en casos como `coaxial_robin`, `annulus`, `symmetric_half`, `two_layer_slab` y `kim_v5_summer`.

Se implementó el operador general **`_apply_exact_natural_bcs`** basado en extensión de Hermite localizada + ponderación racional de Shepard (Sukumar & Srivastava, 2022):
1. Partiendo de un campo base $T_0(\mathbf{x})$ ya exacto en los bordes Dirichlet ($T_0|_{\partial\Omega_D} = g_D$, usando coordenada armónica $\eta = \ln(r/r_i)/\ln(r_o/r_i)$ en coronas `annulus`), para cada lado $s \in \partial\Omega_N \cup \partial\Omega_R$ se proyecta ortogonalmente $\mathbf{x} \mapsto \mathbf{x}_s \in \Gamma_s$ (sobre caras cartesianas o radios interior/exterior en coronas `annulus`).
2. Como $\mathbf{q}\cdot\mathbf{n} = -k\,\partial_n T = k\,\partial_{\phi_s} T$ (siendo $\phi_s$ la distancia normal hacia el interior del dominio):
   - **Neumann**: $\partial_{\phi_s} T|_{\Gamma_s} = q_{n,s}(\mathbf{x}_s) / k_s(\mathbf{x}_s)$.
   - **Robin**: $\partial_{\phi_s} T|_{\Gamma_s} = \frac{h_s}{k_s(\mathbf{x}_s)}\bigl(T_0(\mathbf{x}_s) - T_{\infty, s}(\mathbf{x}_s)\bigr)$.
3. Se construye el jet de Hermite acotado $\tilde{H}_s(\mathbf{x}) = T_0(\mathbf{x}_s) + \psi_s(\mathbf{x})\,\partial_{\phi_s} T|_{\Gamma_s}$ (con $\psi_s = \ell_s \tanh(\phi_s / \ell_s)$ en rectángulos y funciones radiales armónicas $\psi_{\text{in}} = r_i \ln(r/r_i)$ sin corte gaussiano en `annulus`) y se combina con $T_0(\mathbf{x})$ mediante pesos racionales que anulan cualquier lado natural sobre bordes Dirichlet e interfaces de cable (`interior_zero_factor`).
4. Resultado verificado en `float64`: **errores en frontera $< 10^{-12}$** en bordes Dirichlet, Neumann y Robin simultáneamente sin entrenar términos de frontera.

#### Variantes registradas en `models/exact.py`
- **`M1_EXACT` (`ExactInterfaceCablePINN` + fallback `ExactBoundaryPINN`)**:
  - En problemas de cables con dominio rectangular (`supports_exact_cable_interfaces=True`): parametriza cada interfaz circular mediante expansión trigonométrica/armónica compartida $\tau_k(\theta)$ (garantizando $[T]=0$ exacto), usa burbuja polinomial $1 - (r/R)^2$ en el núcleo (`disk`, $C^\infty$ en $r=0$), interpolación radial en coronas (`annulus`), pesos de Shepard entre cables en el suelo y `_apply_exact_natural_bcs` para imponer todas las BCs externas (Dirichlet, Neumann y Robin).
  - En otros dominios (como `annulus` o geometrías poligonales): construye `ExactBoundaryPINN(M1SubdomainPINN)` con BCs exactas.
- **`M0_EXACT_BC`, `M1_EXACT_BC`, `DCSNN_EXACT_BC`** (`ExactBoundaryPINN`): envuelven la arquitectura base con imposición analítica exacta de todas las fronteras externas (`exact_boundary=True`), conservando las condiciones débiles en interfaces internas si $R > 1$.
- **`M0_EXACT`, `CUSP_EXACT`, `DCSNN_EXACT`, `FBPINN_EXACT`** (`ExactCorrectorPINN`): combinan la base global/particionada con `exact_boundary=True`. En `CUSP_EXACT` y `M0_EXACT` (campos escalares continuos globales), $[T]=0$ y las BCs externas se cumplen analíticamente (`exact_constraints=True`), entrenando únicamente `pde`, `interface_flux` y opcionalmente `energy`.

---

### 3.4 Adaptadores Externos DeepXDE (`models/deepxde_adapter.py`)

- **`DeepXDE_PFNN` y `DeepXDE_FNN`**: Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). *DeepXDE: A deep learning library for solving differential equations*. *SIAM Review*, 63(1), 208–228. DOI: `10.1137/19M1274067`. Envuelven `dde.nn.PFNN` / `dde.nn.FNN` en backend PyTorch con la codificación por región de `ThermalProblem2D` y amplitudes $\Delta T_{\text{local},r}$.

---

## 4. Capacidad Física por Familia de Problemas JSON

| Familia de casos | `M0` | `M0R` | `M1` | `M2` | `CUSP` | `CUSPR` | `DCSNN` | `FBPINN` | `M1_EXACT` | `CUSP_EXACT` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mms_*` (analíticos 1 región, Dirichlet/Robin)** | Alta | Alta | Alta | Alta | Alta | Alta | Alta | Alta | Alta (BC exact) | Alta (BC exact) |
| **`cable_disk` / `xlpe_single` / `xlpe_backfill`** | Básica | Alta | Alta | Alta | Media | Alta | Alta | Alta | Muy alta | Media-alta |
| **`xlpe_discrete_layers` / `two_layer_slab` / `tabla_*`** | Básica | Alta | Alta | Alta | Alta | Alta | Alta | Alta | Muy alta (BC exact) | Alta (BC exact) |
| **`aras_flat` / `aras_single` / `coaxial_robin` / `symmetric_*`** | Básica | Alta | Alta | Alta | Media | Alta | Alta | Alta | Muy alta | Media-alta |
| **`annulus` (Neumann interior + Dirichlet exterior)** | Media | Alta | Alta | Alta | Media | Alta | Alta | Alta | Alta (BC exact) | Alta (BC exact) |
| **`kim_*` (55 regiones, suelo heterogéneo / verano / PCA)** | Limitada | Alta | Alta | Alta | Limitada | Alta | Alta | Alta | Muy alta | Limitada |

> **Observación de capacidad**: `CUSP` y `M0` utilizan una única amplitud global $T_{\text{scale}}$ (`CUSP` puede exhibir saltos de flujo iniciales mayores en problemas con muchos conductores/capas de alta conductividad), mientras que sus versiones escaladas por región (`CUSPR`, `M0R`), descompuestas (`M1`, `M2`, `FBPINN`), aumentadas (`DCSNN`) y exactas por interfaz (`M1_EXACT`) equilibran los órdenes de magnitud desde el paso 0.
