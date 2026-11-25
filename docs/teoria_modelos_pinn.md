# Fundamento Teórico de los Modelos PINN, Autores Originales y Modificaciones

## 1. Planteamiento Físico General

Todos los modelos de `PINN_V4` resuelven el problema estacionario de conducción térmica 2D en medios compuestos $\Omega = \bigcup_{r=0}^{R-1} \Omega_r \subset \mathbb{R}^2$:

$$
-\nabla \cdot \bigl( k_r(\mathbf{x}) \nabla T_r(\mathbf{x}) \bigr) = Q_r(\mathbf{x}), \qquad \mathbf{x} \in \Omega_r
$$

sujeto en la frontera externa $\partial\Omega$ a condiciones:
- **Dirichlet**: $T(\mathbf{x}) = g_D(\mathbf{x})$.
- **Neumann** (flujo prescrito respecto a la normal exterior $\mathbf{n}$): $-k(\mathbf{x})\,\nabla T(\mathbf{x})\cdot\mathbf{n} = q_n(\mathbf{x})$.
- **Robin** (convección): $-k(\mathbf{x})\,\nabla T(\mathbf{x})\cdot\mathbf{n} = h \bigl(T(\mathbf{x}) - T_\infty(\mathbf{x})\bigr)$.

Y en toda interfaz interna $\Gamma_{ij} = \partial\Omega_i \cap \partial\Omega_j$ entre materiales distintos:

$$
[T]_{\Gamma_{ij}} = T_i - T_j = 0, \qquad [q_n]_{\Gamma_{ij}} = (-k_i \nabla T_i + k_j \nabla T_j)\cdot\mathbf{n} = 0
$$

> **Acoplamiento obligatorio de flujo en interfaz (`interface_flux`)**: `PhysicsObjective` incluye con peso `100.0` el término adimensional $[q_n]_{\Gamma_{ij}} / f_{\text{scale}, \Gamma}$ en **todas** las arquitecturas (incluidas `M1_EXACT` y `CUSP_EXACT` que ya satisfacen $[T]=0$ analíticamente), evitando el colapso de problemas multiregión hacia la solución espuria del suelo exterior.

### Escalamiento Físico Adimensional y Geometría General (`ThermalProblem2D.scales`)

En cables subterráneos, un conductor de cobre de radio $r \sim 0.005{-}0.01\text{ m}$ genera $Q \sim 10^5\text{ W/m}^3$ con $k \approx 380{-}400\text{ W/(m}\cdot\text{K)}$, mientras las pantallas delgadas tienen espesores milimétricos sin generación ($Q=0$) y el suelo exterior tiene extensión métrica con $k \approx 0.5{-}1.5\text{ W/(m}\cdot\text{K)}$. Para acondicionar cualquier geometría JSON sin soluciones *ad hoc*:

1. `Shape2D.area()` provee áreas exactas y `ThermalProblem2D` utiliza muestreo local de respaldo en regiones pequeñas o delgadas que la malla global no capture con suficiente densidad.
2. Con $L_r = \min\bigl(\sqrt{A_r}, L_{\text{car}, r}\bigr)$, se define la escala térmica local de cada región:

$$
\Delta T_{\text{local}, r} = \begin{cases}
|Q_r| L_r^2 / k_{\text{ref}, r} & \text{si } |Q_r| > 0 \\
\min\Bigl(T_{\text{scale}},\, \max\Bigl(\frac{2\pi P_{\text{abs}} L_r^2}{k_{\text{ref}, r} A_r},\, 10^{-6} T_{\text{scale}}\Bigr)\Bigr) & \text{si } Q_r = 0 \text{ y } P_{\text{abs}} > 0 \\
T_{\text{scale}} & \text{en otro caso}
\end{cases}
$$

$$
q_{\text{scale}, r} = \frac{k_{\text{ref}, r} \Delta T_{\text{local}, r}}{L_r}, \qquad
d_{\text{scale}, r} = \frac{k_{\text{ref}, r} \Delta T_{\text{local}, r}}{L_r^2}
$$

3. Las distancias log-radiales a discos fuente (`0.5 * log1p(r^2 / r_m^2)`) y las funciones de nivel circulares ($\sqrt{\|\mathbf{x}-\mathbf{c}\|^2 + \varepsilon_r^2} - \sqrt{R^2 + \varepsilon_r^2}$) son estrictamente $C^\infty$ incluso en el centro $r = 0$.

---

## 2. Descripción de Arquitecturas, Autores Originales y Modificaciones

Para la auditoría matemática completa, tablas comparativas y parámetros por modelo consulte [`docs/detalle_tecnico_implementaciones_pinn.md`](detalle_tecnico_implementaciones_pinn.md).

### `M0` — PINN Global Clásica (`M0GlobalPINN`)
- **Autores originales**: Raissi, Perdikaris & Karniadakis (2019), *"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations"*, *Journal of Computational Physics*, DOI `10.1016/j.jcp.2018.10.045`.
- **Fundamento y modificaciones**: red MLP escalar continua única sobre coordenadas globales normalizadas `(x, y)` del dominio, con escala uniforme $T_{\text{scale}}$ y sesgo global único `offsets[0]`, fiel a la formulación clásica de Raissi et al. (2019).

### `M0R` — PINN Global Escalada por Región (`RegionScaledGlobalPINN`) *(Retenida en tesis)*
- **Base original**: Raissi et al. (2019), DOI `10.1016/j.jcp.2018.10.045`.
- **Modificación implementada**: conserva una red compartida global `partition="global"`, pero escala cada cabezal regional por su temperatura característica $\Delta T_{\text{local}, r}$ y sesgo `offsets[region.id]`, eliminando la subestimación de puntos calientes en conductores compactos.

### `M1` — Descomposición por Subdominios (`M1SubdomainPINN`) *(Retenida en tesis)*
- **Autores originales**: Jagtap, Kharazmi & Karniadakis (2020, cPINN, DOI `10.1016/j.cma.2020.113028`) y Jagtap & Karniadakis (2020, XPINN, DOI `10.4208/cicp.OA-2020-0164`).
- **Fundamento y modificaciones**: asigna una subred independiente $\mathcal{N}_r$ por región $\Omega_r$ acoplada en $\Gamma_{ij}$ mediante $[T]=0$ y $[q_n]=0$. En coronas circulares (`annulus`) utiliza $(\tilde{\rho}, \cos\theta, \sin\theta)$ periódicas y en el suelo incorpora características log-radiales hacia los cables.

### `M2` — Formulación Mixta de Primer Orden (`M2MixedFluxPINN`) *(Retenida en tesis)*
- **Autores originales**: Gladstone, Nabian, Viana & Meidani (2022), *"FO-PINN: A First-Order formulation for Physics Informed Neural Networks"*, `arXiv:2211.09321`.
- **Fundamento y modificaciones**: predice $(T, q_x, q_y)$ en primer orden minimizando simultáneamente $\mathbf{q} + k\nabla T = 0$ y $\nabla\cdot\mathbf{q} = Q$, con adimensionalización conjunta por subregión.

### `CUSP` y `CUSPR` — Enriquecimiento Cúspide (`CuspPINN`, `CuspRegionScaledPINN`) *(CUSPR retenida en tesis)*
- **Autores originales**: Tseng, Lin, Hu & Lai (2023), *"A cusp-capturing PINN for elliptic interface problems"*, *Journal of Computational Physics*, DOI `10.1016/j.jcp.2023.112359`.
- **Fundamento y modificaciones**: incorpora $|\phi_\Gamma(\mathbf{x})|$ hacia cada interfaz (extraída automáticamente del JSON para círculos, coronas, subrectángulos, polígonos y planos). En **`CUSPR`** (aporte original de la tesis), cada región dispone de su propia amplitud $\Delta T_{\text{local}, r}$ y sesgo entrenable sobre la base enriquecida con cúspides.

### `DCSNN` — Red Capturadora de Discontinuidades (`DCSNNPINN`)
- **Autores originales**: Hu, Lin & Lai (2022), *"A discontinuity capturing shallow neural network for elliptic interface problems"*, *Journal of Computational Physics*, DOI `10.1016/j.jcp.2022.111576`.
- **Fundamento y modificaciones**: extiende la entrada espacial con un selector *one-hot* por región $z(\mathbf{x})$ y funciones de nivel regularizadas, permitiendo ramas suaves distintas por subdominio con amplitudes y sesgos `offsets[region.id]` propios.

### `FBPINN` — Redes Neuronales de Base Finita (`FBPINNPINN`)
- **Autores originales**: Moseley, Markham & Nissen-Meyer (2023), *"Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations"*, *Advances in Computational Mathematics*, DOI `10.1007/s10444-023-10065-9`.
- **Fundamento y modificaciones**: particiona regiones rectangulares en grilla solapada $2\times 2$ con normalización local $[-1, 1]$ por tile y subredes dedicadas para capas circulares. Utiliza ventanas tensorizadas **smootherstep $C^2$** ($6t^5 - 15t^4 + 10t^3$) que solo decaen en los solapamientos interiores y valen `1.0` en los bordes globales del dominio, formando una partición de la unidad exacta ($\sum_j w_j = 1$) sin singularidades en el Laplaciano.

### `M1_EXACT`, `*_EXACT_BC` y `*_EXACT` — Imposición Analítica Exacta de Fronteras e Interfaces
- **Autores originales**:
  - McFall & Mahan (2009), *IEEE Transactions on Neural Networks*, DOI `10.1109/TNN.2009.2020735`.
  - Sheng & Yang (2021, *PFNN*), *Journal of Computational Physics*, DOI `10.1016/j.jcp.2020.110085`.
  - Sukumar & Srivastava (2022), *Computer Methods in Applied Mechanics and Engineering*, DOI `10.1016/j.cma.2021.114333`.
- **Fundamento y modificaciones**:
  1. **Imposición exacta unificada de Dirichlet, Neumann y Robin (`_boundary_lift_and_distance` + `_apply_exact_natural_bcs`)**: garantiza $T|_{\partial\Omega_D} = g_D$, $-k\nabla T\cdot\mathbf{n}|_{\partial\Omega_N} = q_n$ y $-k\nabla T\cdot\mathbf{n}|_{\partial\Omega_R} = h(T - T_\infty)$ a precisión de máquina mediante extensión de Hermite localizada ($\psi_s = \ell_s \tanh(\phi_s/\ell_s)$ en rectángulos y coordenadas armónicas logarítmicas $\eta = \ln(r/r_i)/\ln(r_o/r_i)$, $\psi_{\text{in}} = r_i\ln(r/r_i)$ en coronas `annulus`) y ponderación racional tipo Shepard.
  2. **`ExactInterfaceCablePINN` (`M1_EXACT`)**: parametriza cada interfaz circular mediante trazas armónicas compartidas $\tau_k(\theta)$ (con $[T]=0$ exacto), usa burbuja polinomial $1 - (r/R)^2$ en el núcleo (`disk`, $C^\infty$ en $r = 0$), interpolación radial en coronas y pesos de Shepard en el suelo sin interferir con las condiciones límite exactas.

### Métodos Complementarios Integrados
- **Atención adaptativa RBA**: Anagnostopoulos, Toscano, Stergiopulos & Karniadakis (2024), CMAME, DOI `10.1016/j.cma.2024.116805`.
- **Atención autoadaptativa SA**: McClenny & Braga-Neto (2023), *J. Comput. Phys.* (`arXiv:2009.04544`).
- **Adaptador DeepXDE**: Lu, Meng, Mao & Karniadakis (2021), *SIAM Review*, DOI `10.1137/19M1274067`.
- **Inicialización discreta RBF-FD**: colocación radial poliharmónica $r^5$ aumentada con polinomios cuadráticos completos por región, acoplada globalmente vía matriz dispersa `scipy.sparse.linalg.spsolve`.
- **Referencia geotérmica/cables heterogéneos (`kim_*`)**: casos adaptados a partir de Kim et al. (2025), *Geothermics*, DOI `10.1016/j.geothermics.2024.103151`.
