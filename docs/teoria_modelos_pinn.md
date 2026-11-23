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

### Escalamiento Físico Adimensional por Subregión

En cables subterráneos, un conductor de cobre de radio $r \sim 0.01\text{ m}$ genera $Q \sim 10^5\text{ W/m}^3$ con $k \approx 380\text{ W/(m}\cdot\text{K)}$, mientras el suelo exterior tiene $Q = 0$, extensión métrica y $k \approx 0.5{-}1.5\text{ W/(m}\cdot\text{K)}$. Para acondicionar las pérdidas PDE sin sesgar la optimización, `ThermalProblem2D.scales(region)` define para cada región de área $A_r$ y escala longitudinal $L_r = \sqrt{A_r}$:

$$
\Delta T_{\text{local}, r} = \begin{cases} |Q_r| L_r^2 / k_{\text{ref}, r} & \text{si } |Q_r| > 0 \\ T_{\text{scale}} & \text{si } Q_r = 0 \end{cases}, \qquad
q_{\text{scale}, r} = \frac{k_{\text{ref}, r} \Delta T_{\text{local}, r}}{L_r}, \qquad
d_{\text{scale}, r} = \frac{k_{\text{ref}, r} \Delta T_{\text{local}, r}}{L_r^2}
$$

---

## 2. Descripción de Arquitecturas, Autores Originales y Modificaciones

### `M0` — PINN Global Clásica (`M0GlobalPINN`)
- **Autores originales**: Raissi, Perdikaris & Karniadakis (2019), *"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations"*, Journal of Computational Physics, DOI `10.1016/j.jcp.2018.10.045`.
- **Fundamento**: una única red MLP aproxima globalmente la solución usando minimización fuerte de los residuos PDE y condiciones de contorno en colocación aleatoria.
- **Modificaciones en este trabajo**: se agrega codificación aumentada con coordenadas normalizadas locales rellenadas, selector *one-hot* de región y características log-radiales de distancia hacia cada fuente circular (`log(d / r_m)`), más sesgos regionales entrenables `offsets`.

### `M0R` — PINN Global Escalada por Región (`RegionScaledGlobalPINN`) *(Retenida en tesis)*
- **Base original**: Raissi et al. (2019), DOI `10.1016/j.jcp.2018.10.045`.
- **Modificación implementada**: conserva la red compartida global de `M0`, pero multiplica la salida normalizada de cada cabeza regional por la escala térmica local $\Delta T_{\text{local}, r} = |Q_r| L_r^2 / k_r$ en lugar de una única escala de frontera $T_{\text{scale}}$. Esto elimina la subestimación sistemática de los puntos calientes en conductores compactos.

### `M1` — Descomposición por Subdominios (`M1SubdomainPINN`) *(Retenida en tesis)*
- **Autores originales**: Jagtap, Kharazmi & Karniadakis (2020, cPINN, DOI `10.1016/j.cma.2020.113028`) y Jagtap & Karniadakis (2020, XPINN, DOI `10.4208/cicp.OA-2020-0164`).
- **Fundamento**: particiona $\Omega$ en subdominios disjuntos $\Omega_r$, asignando una red neuronal independiente $\mathcal{N}_r$ a cada material homogéneo o heterogéneo continuo, acopladas débilmente en $\Gamma_{ij}$ por continuidad de temperatura y flujo normal.
- **Modificaciones**: mapeo automático de coordenadas locales según la geometría (`rectangle`, `disk`, `annulus`, `polygon`). En coronas circulares (`annulus`) utiliza $(\tilde{\rho}, \cos\theta, \sin\theta)$ periódicas sin discontinuidad de ángulo en $\theta = \pm\pi$, y en el suelo incorpora características log-radiales hacia los cables para capturar el decaimiento logarítmico tipo Kennelly.

### `M2` — Formulación Mixta de Primer Orden (`M2MixedFluxPINN`) *(Retenida en tesis)*
- **Autores originales**: Gladstone, Nabian, Viana & Meidani (2022), *"FO-PINN: A First-Order formulation for Physics Informed Neural Networks"*, `arXiv:2211.09321`.
- **Fundamento**: introduce explícitamente el flujo $\mathbf{q} = (q_x, q_y)$ como salida neuronal junto con $T$, descomponiendo la EDP de segundo orden en ecuaciones constitutiva y de conservación de primer orden ($\mathbf{q} + k\nabla T = 0$ y $\nabla \cdot \mathbf{q} = Q$). Reduce la necesidad de segundas derivadas por diferenciación automática y facilita el acople de flujo en interfaces.
- **Modificaciones**: adimensionalización conjunta de $(T, q_x, q_y)$ por subregión con pesos adaptables en `PhysicsObjective`.

### `CUSP` y `CUSPR` — Enriquecimiento Cúspide (`CuspPINN`, `CuspRegionScaledPINN`) *(CUSPR retenida en tesis)*
- **Autores originales**: Tseng, Lin, Hu & Lai (2023), *"A cusp-capturing PINN for elliptic interface problems"*, Journal of Computational Physics, DOI `10.1016/j.jcp.2023.112359`.
- **Fundamento**: incorpora al vector de entrada el valor absoluto $|\phi_\Gamma(\mathbf{x})|$ de la función de nivel hacia cada interfaz material. Como $|\phi|$ es continua ($C^0$) pero su derivada cambia de signo sobre $\Gamma$, una red MLP suave puede representar quiebres exactos en $\nabla T$ manteniendo $[T] = 0$ sin descomposición explícita en subredes.
- **Modificaciones**: extracción automática de nivel cero para círculos, coronas, subrectángulos, polígonos y planos declarados en JSON. En **`CUSPR`** (aporte original de la tesis), la arquitectura global se extiende con salidas por región escaladas con $\Delta T_{\text{local}, r}$, unificando la captura geométrica de cúspides con el balance de potencias extremas en cables subterráneos.

### `DCSNN` — Red Capturadora de Discontinuidades (`DCSNNPINN`)
- **Autores originales**: Hu, Lin & Lai (2022), *"A discontinuity capturing shallow neural network for elliptic interface problems"*, Journal of Computational Physics, DOI `10.1016/j.jcp.2022.111576`.
- **Fundamento**: añade coordenadas aumentadas discontinuas mediante un indicador binario/por-piezas de región $z(\mathbf{x})$ acoplado con funciones de nivel, permitiendo que una red compartida represente múltiples ramas suaves separadas a través de interfaces.
- **Modificaciones**: generalización dinámica a cualquier número de regiones $R \ge 1$ leídas desde `ThermalProblem2D`.

### `FBPINN` — Redes Neuronales de Base Finita (`FBPINNPINN`)
- **Autores originales**: Moseley, Markham & Nissen-Meyer (2023), *"Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations"*, Advances in Computational Mathematics, DOI `10.1007/s10444-023-10065-9`.
- **Fundamento**: particiona dominios extensos en ventanas espaciales solapadas con normalización local $[-1, 1]$ y ponderación suave por partición de la unidad $\hat{\omega}_j(\mathbf{x}) = \omega_j(\mathbf{x}) / \sum_m \omega_m(\mathbf{x})$, mitigando el sesgo espectral en dominios grandes.
- **Modificaciones**: descomposición híbrida automática en grilla solapada coseno $2\times 2$ para el suelo rectangular combinada con subredes dedicadas para las capas concéntricas de cables.

### `M1_EXACT`, `*_EXACT_BC` y `*_EXACT` — Restricciones Analíticas Exactas
- **Autores originales**: McFall & Mahan (2009) y Sheng & Yang (2021, *PFNN: Penalty-free neural network method for solving a class of second-order boundary-value problems on complex geometries*, Journal of Computational Physics, DOI `10.1016/j.jcp.2020.110085`).
- **Fundamento**: construye ansatz $T(\mathbf{x}) = g_D(\mathbf{x}) + d_D(\mathbf{x}) \mathcal{N}(\mathbf{x})$ donde $d_D = 0$ exactamente sobre bordes Dirichlet $\partial\Omega_D$.
- **Modificaciones de tesis en `ExactInterfaceCablePINN` (`M1_EXACT`)**:
  1. Parametriza cada interfaz circular cable-aislamiento o cable-suelo mediante una traza armónica angular compartida $\tau_k(\theta)$.
  2. Dentro de los núcleos circulares (`disk`, $r = 0 \in \Omega$) utiliza recurrencia polinomial compleja $z^m = ((x - c_x + i(y - c_y))/R)^m$, totalmente regular ($C^\infty$) en $r = 0$ sin singularidades $1/r$ al diferenciar con `autograd`.
  3. En la región suelo exterior enlaza múltiples cables concéntricos y condiciones Dirichlet arbitrarias mediante pesos interpolantes racionales suaves tipo Shepard sin división por cero ni pérdida de derivabilidad en $r = R_c$.

### Métodos Complementarios Integrados
- **Atención adaptativa RBA**: Anagnostopoulos, Toscano, Stergiopulos & Karniadakis (2024), *Residual-based attention in physics-informed neural networks*, Computer Methods in Applied Mechanics and Engineering, DOI `10.1016/j.cma.2024.116805`.
- **Atención autoadaptativa SA**: McClenny & Braga-Neto (2023), *Self-Adaptive Physics-Informed Neural Networks*, Journal of Computational Physics (`arXiv:2009.04544`).
- **Adaptador DeepXDE**: Lu, Meng, Mao & Karniadakis (2021), *DeepXDE: A deep learning library for solving differential equations*, SIAM Review, DOI `10.1137/19M1274067`.
- **Inicialización discreta RBF-FD**: colocación radial poliharmónica $r^5$ aumentada con polinomios cuadráticos completos por región, acoplada globalmente vía matriz dispersa `scipy.sparse.linalg.spsolve`.
- **Referencia geotérmica/cables heterogéneos (`kim_*`)**: casos adaptados a partir de Kim et al. (2025), *Geothermics*, DOI `10.1016/j.geothermics.2024.103151`.
