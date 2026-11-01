# Taylor (1993) Revisited: Technical Monograph

This document gives the model, the identification results with proofs, a methodological audit of the original paper, the computational design, and the complete results behind `README.md`. Every number in the tables is written by the scripts in `src/` to `results/`; the audit script `src/05_audit.py` checks each number in this file against those outputs.

---

## 1. Model and propositions

### 1.1 Environment

Time is discrete and quarterly, $t \in \mathbb{Z}$. Uncertainty is described by a probability space $(\Omega, \mathcal{F}, P)$ carrying three mutually independent i.i.d. standard normal innovation sequences $\{\varepsilon^d_t, \varepsilon^u_t, \varepsilon^v_t\}$, with $\mathcal{F}_t$ the filtration they generate. All variables are deviations from a deterministic steady state with inflation target $\pi^* = 2$ and are measured in annualized percentage points. A representative household with log-linearized Euler equation, monopolistically competitive firms with Calvo pricing, and a central bank that follows an interest rate rule give the standard three-equation system:

$$x_t = \mathbb{E}_t x_{t+1} - s\left(i_t - \mathbb{E}_t \pi_{t+1}\right) + d_t \qquad \text{(IS)}$$

$$\pi_t = \beta\, \mathbb{E}_t \pi_{t+1} + \kappa x_t + u_t \qquad \text{(Phillips curve)}$$

$$i_t = \rho_i i_{t-1} + (1-\rho_i)\left(\phi_\pi \bar\pi_t + \phi_y x_t\right) + v_t, \qquad \bar\pi_t = \tfrac14 \textstyle\sum_{j=0}^{3}\pi_{t-j} \qquad \text{(rule)}$$

with $z_t = \rho_z z_{t-1} + \sigma_z \varepsilon^z_t$ for $z \in \{d, u, v\}$. Here $x$ is the output gap, $\pi$ quarterly inflation at an annual rate, $i$ the policy rate, $s > 0$ the interest elasticity of demand (in annualized units, so $s = 1/(4\sigma)$ for a quarterly intertemporal elasticity $1/\sigma$), and $\kappa > 0$ the slope of the Phillips curve. Taylor's (1993) rule is the special case $\rho_i = 0$, $\phi_\pi = 1.5$, $\phi_y = 0.5$, with the neutral nominal rate $r^* + \pi^* = 4$ in levels. An equilibrium is a stochastic process for $(x_t, \pi_t, i_t)$, adapted to $\mathcal{F}_t$, that satisfies the three equations and is bounded in mean square.

Stack the predetermined states $k_t = (d_t, u_t, v_t, i_{t-1}, \pi_{t-1}, \pi_{t-2}, \pi_{t-3})'$ and the jump variables $p_t = (x_t, \pi_t)'$. Substituting the rule into the IS curve and into the law of motion for $i_{t-1}$ gives $A\, \mathbb{E}_t w_{t+1} = B\, w_t$ for $w_t = (k_t', p_t')'$, with $\det A = \beta \neq 0$.

### 1.2 Determinacy

**Proposition 1 (Taylor principle).** Let $\rho_i = 0$ and let the rule respond to contemporaneous inflation. A unique bounded equilibrium exists if and only if
$$\kappa(\phi_\pi - 1) + (1-\beta)\phi_y > 0 .$$

*Proof.* With no endogenous predetermined variables, the shocks enter only as forcing terms, and determinacy depends on the homogeneous system $\mathbb{E}_t q_{t+1} = M q_t$ for $q_t = (x_t, \pi_t)'$. From the Phillips curve, $\mathbb{E}_t \pi_{t+1} = (\pi_t - \kappa x_t)/\beta$. Substituting this and $i_t = \phi_\pi\pi_t + \phi_y x_t$ into the IS curve,
$$M = \begin{pmatrix} 1 + s\phi_y + s\kappa/\beta & s(\phi_\pi - 1/\beta) \\ -\kappa/\beta & 1/\beta \end{pmatrix}.$$
Both variables are jumps, so a unique bounded solution requires both eigenvalues of $M$ outside the unit circle (Blanchard and Kahn 1980). For a real $2\times 2$ matrix with determinant $D$ and trace $T$, both roots lie outside the unit circle if and only if $|D| > 1$ and $|T| < 1 + D$. Direct computation gives
$$D = \frac{1 + s\phi_y + s\kappa\phi_\pi}{\beta}, \qquad T = 1 + s\phi_y + \frac{s\kappa + 1}{\beta}.$$
Since all parameters are positive and $\beta < 1$, $D > 1$ and $T > 0$, so the binding condition is $1 + D - T > 0$. Then
$$1 + D - T = \frac{s}{\beta}\left[\kappa(\phi_\pi - 1) + (1-\beta)\phi_y\right],$$
which is positive exactly when the stated inequality holds. $\blacksquare$

This is the condition in Bullard and Mitra (2002). With $\phi_y = 0$ it reduces to $\phi_\pi > 1$, the Taylor principle. The unit tests `test_taylor_principle_boundary` and `test_bullard_mitra_generalized_principle` confirm that the numerical solver switches from determinacy to indeterminacy on both sides of this boundary.

### 1.3 What a Taylor-rule regression estimates

**Proposition 2 (Cochrane 2011, frictionless case).** Suppose output is exogenous and the real rate is constant, so the Fisher equation $i_t = \mathbb{E}_t\pi_{t+1}$ holds, and policy follows $i_t = \phi\pi_t + v_t$ with $v_t = \rho v_{t-1} + \sigma\varepsilon_t$, $|\rho| < 1 < |\phi|$. Then (a) the unique bounded equilibrium is $\pi_t = v_t/(\rho - \phi)$; (b) $i_t = \rho\pi_t$ for every $t$, so the OLS slope of $i_t$ on $\pi_t$ converges to $\rho$ and its bias for $\phi$ is $\rho - \phi$; (c) the distribution of the observables depends on $(\phi, \sigma)$ only through $\sigma/(\phi - \rho)$, so $\phi$ is not identified.

*Proof.* Combining the two equations gives $\mathbb{E}_t\pi_{t+1} = \phi\pi_t + v_t$. The general solution is $\pi_t = \pi^p_t + c\,\phi^t$ for a particular solution $\pi^p$ and an arbitrary $\mathcal{F}_0$-measurable $c$; since $|\phi| > 1$ boundedness forces $c = 0$. Try $\pi^p_t = a v_t$: then $a\rho v_t = \phi a v_t + v_t$, so $a = 1/(\rho - \phi)$, which proves (a). For (b), $i_t = \mathbb{E}_t\pi_{t+1} = a\rho v_t = \rho\pi_t$ holds identically, so the population regression coefficient is $\mathrm{Cov}(i,\pi)/\mathrm{Var}(\pi) = \rho$. For (c), the observables are $\pi_t = a v_t$ and $i_t = \rho\pi_t$; $\pi_t$ is a Gaussian AR(1) with persistence $\rho$ and innovation standard deviation $|a|\sigma = \sigma/|\phi - \rho|$, so any $(\phi', \sigma')$ with the same ratio generates the same joint distribution. $\blacksquare$

The test `test_cochrane_frictionless_ols_recovers_rho_not_phi` verifies (a) and (b) to machine precision. In `results/key_numbers.json`, simulations with $\phi \in \{1.5, 3\}$ and $\rho = 0.7$ return OLS slopes of 0.70 in both cases.

**Proposition 3 (OLS limit in the three-equation model).** In a determinate equilibrium, $p_t = F k_t$, $i_t = G k_t$, and $k_{t+1} = P k_t + Q\varepsilon_{t+1}$. Let $X_t = H_X k_t$ collect the regressors and $\Sigma_k$ solve $\Sigma_k = P\Sigma_k P' + QQ'$. The OLS coefficient from regressing $i_t$ on $X_t$ converges to
$$\operatorname*{plim}_{T\to\infty} \hat b = \left(H_X\Sigma_k H_X'\right)^{-1} H_X\Sigma_k G',$$
which differs from the structural coefficients by $\left(H_X\Sigma_k H_X'\right)^{-1} H_X\Sigma_k e_3\,$, the projection of the policy shock $v_t = e_3'k_t$ on the regressors.

*Proof.* The state is a stable Gaussian VAR(1), so it is ergodic and sample second moments converge to their population counterparts $\Sigma_k$. The OLS formula then gives the first expression. Writing $i_t = b' X_t + v_t$ along the equilibrium path and applying the same limit to $\hat b - b = (\sum X_tX_t')^{-1}\sum X_t v_t$ gives the second. The bias is zero only if $H_X\Sigma_k e_3 = 0$, which fails whenever $\sigma_v > 0$, because $\bar\pi_t$ and $x_t$ load on $v_t$ through $F$. $\blacksquare$

`src/core/nk.py::ols_plim` implements this formula. The test `test_analytic_plim_matches_simulation` compares it with OLS on 200,000 simulated quarters.

**Proposition 4 (lagged instruments with persistent policy shocks).** If $v_t = \rho_v v_{t-1} + \sigma_v\varepsilon^v_t$ with $\rho_v \neq 0$, then for any instrument $z_{t-j}$ that is $\mathcal{F}_{t-j}$-measurable, $\mathbb{E}[z_{t-j}v_t] = \rho_v^{\,j}\,\mathbb{E}[z_{t-j}v_{t-j}]$, which is nonzero whenever $z$ responds to the contemporaneous policy shock.

*Proof.* Iterating, $v_t = \rho_v^j v_{t-j} + \sum_{h=0}^{j-1}\rho_v^h\sigma_v\varepsilon^v_{t-h}$. The innovations after $t-j$ are independent of $\mathcal{F}_{t-j}$, so $\mathbb{E}[z_{t-j}v_t] = \rho_v^j\mathbb{E}[z_{t-j}v_{t-j}]$. For $z_{t-j} = i_{t-j}$, $\mathbb{E}[i_{t-j}v_{t-j}] = G\Sigma_k e_3$, which is nonzero because $\partial i_t/\partial v_t = G_3 \neq 0$. $\blacksquare$

The moment conditions behind forward-looking GMM estimates in the style of Clarida, Galí and Gertler (2000), which use lags of the policy rate, inflation and the gap as instruments, therefore fail whenever deviations from the rule are serially correlated. Rudebusch (2002) argues from the term structure that they are.

**Proposition 5 (real-time data and errors in variables).** Suppose the Fed sets $i_t = b' X^{rt}_t + e_t$ on real-time data $X^{rt}_t$ and the econometrician uses revised data $X^{rev}_t$, with $\mathbb{E}[X^{rev}_t e_t] = 0$. Then
$$\operatorname*{plim}\hat b^{rev} = \left(\mathbb{E}X^{rev}_tX^{rev\prime}_t\right)^{-1}\mathbb{E}X^{rev}_tX^{rt\prime}_t\; b .$$
If revisions are pure news ($X^{rev} = X^{rt} + \eta$ with $\eta \perp X^{rt}$), the revised-data coefficients are attenuated. If they are pure noise ($X^{rt} = X^{rev} + \eta$ with $\eta \perp X^{rev}$), they are consistent.

*Proof.* Substitute $i_t = b'X^{rt}_t + e_t$ into the OLS formula on $X^{rev}$ and take limits. Under news, $\mathbb{E}X^{rev}X^{rt\prime} = \mathbb{E}X^{rt}X^{rt\prime}$, which is smaller than $\mathbb{E}X^{rev}X^{rev\prime}$ in the positive semidefinite order. Under noise, $\mathbb{E}X^{rev}X^{rt\prime} = \mathbb{E}X^{rev}X^{rev\prime}$. $\blacksquare$

The data fall between the two cases, and the formula with sample moments is checked against the actual revised-data estimates in Section 4.3.

### 1.4 The bridge from structure to estimand

In the rule, $\theta = (\phi_\pi, \phi_y, \rho_i)$ are structural policy parameters and the disturbance is the policy shock $v_t$, which collects discretionary deviations, omitted targets such as financial conditions, and errors in the econometrician's measurement of the Fed's information. Equilibrium makes $\bar\pi_t$ and $x_t$ functions of $v_t$ (Proposition 3), so the reduced-form regression coefficient mixes $\theta$ with $(s, \kappa, \beta)$ and with the shock processes $(\rho_d, \rho_u, \rho_v, \sigma_d, \sigma_u, \sigma_v)$. An instrument identifies $\phi_\pi$ only if it shifts $\bar\pi_t$ through $d_t$ or $u_t$ and is independent of $v_t$ at all leads and lags that enter the error. The structural model identifies $\phi_\pi$ through cross-equation restrictions and the assumption that the three shocks are mutually independent AR(1) processes. Neither route escapes Proposition 2's lesson that the off-equilibrium threat behind determinacy is not observed.

---

## 2. Audit of the original paper

| Item | Taylor (1993) |
|---|---|
| Design | A calibrated rule chosen to summarize earlier model-based research; no parameter is estimated (p. 202) |
| Sample | 1987 to 1992, quarterly, shown graphically in Figure 1 |
| Data vintage | Data available in late 1992; real GDP in 1987 dollars (Figure 2) |
| Inflation | GDP deflator, four quarters ending in the current quarter (Figure 3) |
| Output gap | Percent deviation of real GDP from a trend growing 2.2 percent per year, 1984:1 to 1992:3 (p. 202); the trend's level is not stated |
| Policy rate | Federal funds rate; the averaging convention is not stated |
| Inference | None. The fit is judged visually; no standard errors, fit statistics, or tests are reported |
| Missing data | Not applicable |

| Headline statistic | Value | Location | Population quantity | What it does not establish |
|---|---|---|---|---|
| Inflation coefficient | 1.5 | p. 202, eq. (1) | A chosen policy response | That the Fed responded this way, or that this response is identified from data |
| Gap coefficient | 0.5 | p. 202, eq. (1) | A chosen policy response | Same; also depends on the trend chosen for the gap |
| Inflation target | 2 percent | p. 202 | A normative target | The Fed's actual target in 1987 to 1992 |
| Neutral real rate | 2 percent | p. 202 | Close to trend growth of 2.2 percent | The equilibrium real rate in later decades |
| "Fits remarkably well" | No statistic | p. 202 | Visual agreement with Figure 1 | Stability of the relationship, causal response, or determinacy |

Read as a regression, the paper's implicit estimator is the rule evaluated at fixed coefficients, and its implicit claim is that $i_t - (1 + 1.5\bar\pi_t + 0.5x_t)$ is small. The probability limit of any estimated version is given by Proposition 3, and its asymptotic variance under serial correlation is the HAC sandwich $\mathrm{Avar}(\hat b) = \Sigma_{XX}^{-1}\,\Omega\,\Sigma_{XX}^{-1}$ with $\Omega = \sum_{h=-\infty}^{\infty}\mathbb{E}[X_tX_{t-h}'v_tv_{t-h}]$. In the estimated model the OLS limit for the static rule is 1.26 against a structural value of 3.11 (Section 4.4). The estimators in the horse race address this bias in different ways. Real-time data address measurement error (Proposition 5). GMM with lagged instruments addresses simultaneity but fails under Proposition 4. The external instrument is valid under its exclusion restriction but weak. The structural model identifies $\phi_\pi$ under the model's restrictions.

---

## 3. Data and computation

### 3.1 Sources and sample construction

`src/01_data_pipeline.py` downloads every source with exponential-backoff retries, records the URL, retrieval time, row count and SHA-256 hash in `data/raw/manifest.json`, and builds the quarterly panel `data/processed/analytical_panel.parquet`. Monthly series are averaged within the quarter. The 1992 vintage comes from column `ROUTPUT92Q4` and `P92Q4` of the Philadelphia Fed Real-Time Data Set, the first vintage containing 1992Q3. Taylor's trend has slope $\ln(1.022)/4$ per quarter; I set its intercept so the gap averages zero over 1984Q1 to 1992Q3, which is the least-squares intercept given the slope. The Greenbook gap uses the 1987 to 2013 meeting-level file for its current-quarter estimate and the 1996 to 2020 file for later quarters. The two agree almost perfectly where they overlap (correlation 0.9999, asserted in the pipeline). For each quarter I keep the Greenbook closest to mid-quarter. Real-time inflation is the mean of the Greenbook's annualized quarterly deflator inflation over the four quarters ending in the current quarter, matching the timing of the revised four-quarter rate. Känzig's monthly series are summed within the quarter.

Sample construction (`results/sample_waterfall.csv`):

| Step | Observations | First | Last |
|---|---|---|---|
| raw FRED FEDFUNDS | 867 | 1954-07-01 | 2026-09-01 |
| raw FRED GDPC1 | 319 | 1947-01-01 | 2026-07-01 |
| raw FRED GDPDEF | 319 | 1947-01-01 | 2026-07-01 |
| raw FRED GDPPOT | 352 | 1949-01-01 | 2036-10-01 |
| raw FRED PCEPILFE | 813 | 1959-01-01 | 2026-09-01 |
| raw FRED CPIAUCSL | 956 | 1947-01-01 | 2026-09-01 |
| raw FRED UNRATE | 944 | 1948-01-01 | 2026-09-01 |
| raw FRED NROU | 352 | 1949-01-01 | 2036-10-01 |
| raw RTDSM ROUTPUT vintage 92Q4 | 135 | 1959Q1 | 1992Q3 |
| raw RTDSM P vintage 92Q4 | 135 | 1959Q1 | 1992Q3 |
| Greenbook real-time gap, one vintage per quarter | 134 | 1987Q3 | 2020Q4 |
| Greenbook real-time deflator inflation, one vintage per quarter | 198 | 1967Q1 | 2020Q3 |
| Kanzig oil supply series, quarterly sums | 204 | 1975Q1 | 2025Q4 |
| merged quarterly panel (policy rate observed) | 271 | 1959Q1 | 2026Q3 |
| core variables non-missing (rate, inflation, CBO gap) | 267 | 1960Q1 | 2026Q3 |
| zero-lower-bound quarters (rate below 0.25) | 36 | 2009Q1 | 2022Q1 |
| estimation sample taylor_1987_1992 | 23 | 1987Q1 | 1992Q3 |
| estimation sample greenspan_bernanke | 82 | 1987Q3 | 2007Q4 |
| estimation sample pre_volcker | 58 | 1965Q1 | 1979Q2 |
| estimation sample cgg_pre_volcker | 78 | 1960Q1 | 1979Q2 |
| estimation sample cgg_volcker_greenspan | 70 | 1979Q3 | 1996Q4 |
| estimation sample volcker_greenspan | 114 | 1979Q3 | 2007Q4 |
| estimation sample post_2008 | 75 | 2008Q1 | 2026Q3 |
| estimation sample full | 247 | 1965Q1 | 2026Q3 |
| estimation sample full_nonzlb | 211 | 1965Q1 | 2026Q3 |
| estimation sample realtime_greenbook | 98 | 1987Q3 | 2020Q1 |
| estimation sample oil_iv | 134 | 1983Q3 | 2025Q4 |

Distributional moments (`results/summary_stats.csv`; kurtosis is not excess kurtosis):

| Variable (1965Q1 onward) | N | Mean | SD | Skewness | Kurtosis | P5 | P10 | P25 | P50 | P75 | P90 | P95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| federal funds rate | 247 | 4.93 | 3.68 | 0.81 | 3.87 | 0.09 | 0.14 | 1.93 | 4.87 | 6.61 | 9.45 | 11.18 |
| deflator inflation, 4 quarters | 247 | 3.48 | 2.27 | 1.29 | 4.02 | 1.10 | 1.44 | 1.90 | 2.73 | 4.43 | 7.06 | 8.68 |
| deflator inflation, quarterly annualized | 247 | 3.49 | 2.43 | 1.28 | 4.50 | 0.96 | 1.27 | 1.97 | 2.78 | 4.33 | 7.19 | 8.60 |
| CBO output gap | 247 | −0.21 | 2.30 | −0.41 | 3.97 | −4.21 | −3.01 | −1.54 | −0.10 | 1.27 | 2.27 | 3.46 |
| CPI inflation, 4 quarters | 247 | 3.94 | 2.78 | 1.52 | 5.40 | 1.18 | 1.50 | 2.15 | 3.23 | 4.74 | 7.76 | 10.36 |
| unemployment gap | 247 | 0.47 | 1.65 | 1.14 | 5.14 | −1.90 | −1.14 | −0.57 | 0.06 | 1.24 | 2.74 | 4.03 |
| Greenbook inflation, real time | 198 | 3.42 | 2.39 | 1.46 | 4.37 | 0.98 | 1.29 | 1.82 | 2.50 | 4.12 | 7.66 | 8.95 |
| Greenbook gap, real time | 134 | −1.11 | 2.61 | −0.77 | 2.89 | −6.54 | −4.85 | −2.79 | −0.40 | 0.67 | 1.93 | 2.23 |
| OPEC surprise, quarterly sum | 204 | 0.00 | 2.51 | −0.31 | 8.15 | −4.41 | −2.31 | −0.25 | 0.00 | 0.67 | 2.25 | 3.38 |
| oil news shock, quarterly sum | 204 | 0.00 | 1.07 | −0.38 | 6.03 | −1.54 | −1.14 | −0.64 | −0.01 | 0.63 | 1.18 | 1.82 |

### 3.2 Estimators and numerical settings

**HAC inference.** All time-series standard errors use the Newey-West Bartlett kernel with bandwidth $\lfloor 4(T/100)^{2/9}\rfloor$ and a $T/(T-k)$ degrees-of-freedom correction. The test `test_ols_hac_matches_statsmodels` matches statsmodels to $10^{-8}$. Long-run responses $b/(1-\rho)$ use the delta method.

**GMM.** Two-step efficient GMM with a centered HAC weighting matrix and Hansen's $J$. Instruments are a constant and four lags each of the funds rate, four-quarter inflation and the CBO gap. The test `test_gmm_recovers_parameter_and_j_has_correct_size` checks that the $J$ test rejects about 5 percent of the time under valid instruments.

**2SLS and weak instruments.** For one endogenous regressor, the Montiel Olea and Pflueger (2013) effective first-stage statistic is
$$F_{\text{eff}} = \frac{\hat\pi'\tilde Z'\tilde Z\hat\pi}{\operatorname{tr}\!\left(\tilde Z'\tilde Z\,\hat V_{\hat\pi}\right)},$$
with $\tilde Z$ the instruments after partialling out exogenous controls and $\hat V_{\hat\pi}$ the HAC covariance of the first-stage coefficients. The 5 percent critical value for a worst-case bias of 10 percent in the simplified conservative test is the 95th percentile of a noncentral $\chi^2_1(1/\tau)$ with $\tau = 0.1$, which the code computes as 23.1. Anderson-Rubin confidence sets invert the HAC Wald test of the instruments in the regression of $i_t - \phi_y x_t - b_0\bar\pi_t$ on the instruments, over a grid of 2,501 points from −10 to 15.

**Tobit.** The zero lower bound is treated as left-censoring at 0.25 percent; the likelihood is maximized by BFGS and standard errors come from the inverse numerical Hessian.

**Structural break.** The Quandt-Andrews sup-Wald statistic uses HAC Wald tests of a shift in all coefficients with 15 percent trimming. Its asymptotic null distribution, the supremum of a normalized squared Brownian bridge, is simulated with 1,000 steps and 4,000 replications instead of read from tables.

**Model solution.** `src/core/nk.py` forms $M = A^{-1}B$, computes its left eigenvectors $\ell$, and imposes $\ell' w_t = 0$ for each eigenvalue with modulus above $1 + 10^{-10}$. With exactly two such eigenvalues these restrictions give $p_t = Fk_t$. The state transition is $P = M_{kk} + M_{kp}F$. The unconditional state covariance solves a discrete Lyapunov equation.

**Minimum-distance estimation.** The data moments are the 21 distinct elements of the autocovariances at lags 0, 1, 2 and 4 of the CBO gap, annualized quarterly deflator inflation and the funds rate, over 1987Q3 to 2007Q4. Their covariance matrix $\Omega$ comes from a moving-block bootstrap with blocks of 8 quarters and 999 replications. The weight matrix is $W = \mathrm{diag}(\Omega)^{-1}$. Eleven parameters are free and $\beta$ is fixed at 0.99. Indeterminate parameter vectors receive a penalty. The optimizer runs L-BFGS-B from 10 random starts inside the bounds, then polishes the best by bounded Nelder-Mead and a final L-BFGS-B pass. Standard errors use the sandwich $(G'WG)^{-1}G'W\Omega WG(G'WG)^{-1}$ with a central-difference Jacobian. The profile in Figure 4C fixes $\phi_\pi$ at 19 values and re-estimates the other ten parameters from two starts each, the global estimate and the neighbouring profile solution.

**Counterfactual.** Given the estimated model, the three shocks in each quarter are recovered exactly from the observed $(x_t, \pi_t, i_t)$ and the predetermined lags, because the $3\times 3$ loading matrix is invertible. From 2021Q1 the model is re-solved under the alternative rule and simulated with the recovered demand and cost-push shocks and no policy shocks. When the rule would set the rate below 0.125 percent, a one-period policy surprise holds the rate at that floor. The counterfactual regime has no persistent discretionary shocks, so agents expect each floor episode to end the next quarter. Letting the floor shock inherit the estimated persistence of 0.97 makes agents expect the floor to bind almost permanently and sends the model into a deflationary spiral, an artifact of the approximation rather than a result.

**Monte Carlo.** The data-generating processes are the estimated model and a static-rule variant ($\rho_i = 0$), with 2,000 replications at $T = 82$ and 500 at $T = 1000$. The external instrument in the simulations is the cost-push innovation (strong) or that innovation plus Gaussian noise, with the noise scaled so the mean effective $F$ matches the data.

The full analysis from the committed panel ran in 5 minutes on a shared laptop CPU.

---

## 4. Results

### 4.1 Stage 1: Taylor's figure on three data vintages

| Data | N | RMSE (pp) | Mean rate minus rule | Correlation | Largest miss | Quarter of largest miss |
|---|---|---|---|---|---|---|
| 1992Q4 vintage, 2.2% trend gap | 23 | 0.40 | −0.03 | 0.975 | 0.97 | 1987Q1 |
| current vintage, 2.2% trend gap | 23 | 1.17 | 0.28 | 0.766 | 2.80 | 1987Q1 |
| current vintage, CBO output gap | 23 | 1.27 | 1.06 | 0.919 | 2.58 | 1987Q1 |

The vintage replication is the only setting in which the rule's fit is as good as Taylor's figure suggests. The difference between the first and third rows is almost entirely the output gap. Benchmark revisions since 1992, including the move to chain weighting, raised measured growth over 1984 to 1992 and shifted the 1992Q3 gap from −2.36 on the vintage trend to −0.97 on the CBO series.

### 4.2 The horse race

Native units throughout: the coefficients are percentage points of policy rate per percentage point of inflation or gap, which keeps the Taylor-principle threshold at 1. Long-run responses for partial-adjustment rows; HAC standard errors.

| Spec | Description | Period | N | Estimator | φπ | SE | 95% CI | t (φπ = 1) | p (φπ ≤ 1) | Holm p | φy | SE φy | ρ | Fit | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 | Taylor (1993) rule as written | 1987Q1-1992Q3 | 23 | calibrated | 1.50 |  |  |  |  |  | 0.50 |  |  |  | Coefficients chosen, not estimated |
| S1 | Unrestricted rule, Taylor's window, 1992 vintage | 1987Q1-1992Q3 | 23 | OLS, HAC | 1.25 | 0.19 | [0.88, 1.61] | 1.32 | 0.093 | 1.000 | 0.55 | 0.05 |  | R² 0.956 |  |
| S2 | Unrestricted rule, Taylor's window, revised data | 1987Q1-1992Q3 | 23 | OLS, HAC | 1.32 | 0.52 | [0.31, 2.33] | 0.62 | 0.268 | 1.000 | 0.74 | 0.10 |  | R² 0.862 |  |
| S3 | Static rule, Greenspan-Bernanke era | 1987Q3-2007Q4 | 82 | OLS, HAC | 1.23 | 0.37 | [0.51, 1.95] | 0.63 | 0.265 | 1.000 | 0.85 | 0.18 |  | R² 0.494 |  |
| S4 | Partial adjustment, Greenspan-Bernanke era | 1987Q3-2007Q4 | 82 | OLS, HAC, delta method | 1.46 | 0.57 | [0.34, 2.58] | 0.81 | 0.210 | 1.000 | 2.01 | 0.38 | 0.88 | R² 0.971 |  |
| S5 | Partial adjustment, revised data, real-time quarters | 1987Q3-2020Q1 | 98 | OLS, HAC, delta method | 1.71 | 0.54 | [0.65, 2.77] | 1.32 | 0.094 | 1.000 | 2.14 | 0.37 | 0.88 | R² 0.971 |  |
| S6 | Partial adjustment, Greenbook real-time data | 1987Q3-2020Q1 | 98 | OLS, HAC, delta method | 2.26 | 0.63 | [1.03, 3.49] | 2.01 | 0.022 | 0.314 | 1.38 | 0.40 | 0.89 | R² 0.966 |  |
| S7 | Forward-looking, Greenbook one-year forecast | 1987Q3-2020Q1 | 98 | OLS, HAC, delta method | 2.37 | 0.58 | [1.23, 3.51] | 2.35 | 0.009 | 0.141 | 1.05 | 0.28 | 0.87 | R² 0.968 |  |
| S8 | Forward-looking GMM, 1979Q3-2007Q4 | 1979Q3-2007Q4 | 114 | two-step GMM, HAC | 4.12 | 0.87 | [2.41, 5.83] | 3.59 | 0.000 | 0.003 | 1.75 | 0.72 | 0.91 | J p 0.614 | instruments: four lags of rate, inflation, gap |
| S8b | Forward-looking GMM, 1979Q3-1996Q4 (CGG window) | 1979Q3-1996Q4 | 70 | two-step GMM, HAC | 2.90 | 0.55 | [1.81, 3.98] | 3.43 | 0.000 | 0.005 | 0.73 | 0.51 | 0.85 | J p 0.488 | instruments: four lags of rate, inflation, gap |
| S9 | Forward-looking GMM, 1965Q1-1979Q2 | 1965Q1-1979Q2 | 58 | two-step GMM, HAC | 1.77 | 0.31 | [1.16, 2.38] | 2.47 | 0.007 | 0.108 | 0.98 | 0.27 | 0.79 | J p 0.374 | instruments: four lags of rate, inflation, gap |
| S9b | Forward-looking GMM, 1960Q1-1979Q2 (CGG window) | 1961Q1-1979Q2 | 74 | two-step GMM, HAC | 0.87 | 0.15 | [0.58, 1.15] | −0.90 | 0.817 | 1.000 | 0.66 | 0.22 | 0.79 | J p 0.047 | instruments: four lags of rate, inflation, gap |
| S10 | Partial adjustment, pre-Volcker | 1965Q1-1979Q2 | 58 | OLS, HAC, delta method | 1.36 | 0.34 | [0.69, 2.03] | 1.06 | 0.145 | 1.000 | 1.20 | 0.29 | 0.75 | R² 0.851 |  |
| S11 | Partial adjustment, Volcker-Greenspan | 1979Q3-2007Q4 | 114 | OLS, HAC, delta method | 2.09 | 0.30 | [1.51, 2.68] | 3.65 | 0.000 | 0.002 | 1.13 | 0.42 | 0.83 | R² 0.942 |  |
| S12 | Static rule, ZLB as censoring, 1987 onward | 1987Q3-2026Q3 | 157 | Tobit MLE | 0.09 | 0.17 | [−0.25, 0.42] | −5.32 | 1.000 | 1.000 | 1.38 | 0.15 |  |  | 36 censored quarters |
| S13 | Static rule, ZLB as censoring, 2008 onward | 2008Q1-2026Q3 | 75 | Tobit MLE | −0.35 | 0.14 | [−0.62, −0.07] | −9.55 | 1.000 | 1.000 | 1.65 | 0.20 |  |  | 36 censored quarters |
| S14 | OLS on the instrument sample (gap response fixed at 0.5) | 1983Q3-2024Q4 | 130 | OLS, HAC | 0.76 | 0.63 | [−0.46, 1.99] | −0.38 | 0.646 | 1.000 | 0.50 |  |  | R² 0.108 |  |
| S15 | 2SLS, deflator inflation on Kanzig OPEC announcement surprises | 1983Q3-2024Q4 | 130 | 2SLS, HAC | −0.32 | 1.12 | [−2.52, 1.87] | −1.18 | 0.881 | 1.000 | 0.50 |  |  | F_eff 0.47 | AR 95% set [-10.00, 15.00], unbounded within grid |
| S16 | 2SLS, deflator inflation on Kanzig oil supply news shocks | 1983Q3-2024Q4 | 130 | 2SLS, HAC | −0.21 | 0.76 | [−1.70, 1.28] | −1.59 | 0.944 | 1.000 | 0.50 |  |  | F_eff 2.18 | AR 95% set [-10.00, 15.00], unbounded within grid |
| S17 | 2SLS, CPI inflation on Kanzig oil supply news shocks | 1983Q3-2025Q4 | 134 | 2SLS, HAC | −0.12 | 0.43 | [−0.95, 0.72] | −2.63 | 0.996 | 1.000 | 0.50 |  |  | F_eff 7.43 | AR 95% set [-2.12, 1.51] |

The estimates are fragile in ways that line up with the critiques. Moving from a static rule to partial adjustment, from revised to real-time data, from backward to forward-looking inflation, or from one sample boundary to another changes the inflation response by more than its standard error. After a Holm adjustment across the 19 tests of $\phi_\pi \le 1$, only the three Volcker-Greenspan specifications reject. The pre-Volcker estimate falls below one only when the sample starts in 1960, as in Clarida, Galí and Gertler, and the overidentifying restrictions are rejected at the 5 percent level in that case ($J$ p-value 0.047).

### 4.3 Diagnostics and sensitivity

| Test | Statistic | Value | Reference | Note |
|---|---|---|---|---|
| first stage OPEC announcement surprises | Montiel Olea-Pflueger effective F | 0.47 | 23.11 (5% test, 10% worst-case bias) | HAC robust first-stage F 0.36 |
| first stage oil supply news shocks | Montiel Olea-Pflueger effective F | 2.18 | 23.11 (5% test, 10% worst-case bias) | HAC robust first-stage F 1.24 |
| placebo: future OPEC surprises as instruments | effective F | 0.23 | should be near zero | leads t+1..t+4 cannot move current inflation |
| Greenbook gap revision (final CBO minus real time) | mean, SD | 0.32 | 0 if unbiased | SD 1.16 |
| revision correlated with real-time gap (noise view predicts negative) | correlation | −0.74 | 0 under pure news | corr with final gap -0.14 |
| errors-in-variables formula: predicted long-run gap response on revised data | phi_y | 2.20 | actual 2.14 | plim under the hypothesis that the Fed set policy on Greenbook data |
| Quandt-Andrews sup-Wald break in all rule coefficients, 1965-2007 | sup-Wald | 64.30 | 5% critical 13.94 (simulated) | break at 1979Q4, p = 0.0000 |
| leave-one-chair-out, drop Martin | long-run phi_pi | 1.86 | full non-ZLB sample | N 191, SE 0.49 |
| leave-one-chair-out, drop Burns | long-run phi_pi | 2.13 | full non-ZLB sample | N 178, SE 0.34 |
| leave-one-chair-out, drop Miller | long-run phi_pi | 1.77 | full non-ZLB sample | N 206, SE 0.54 |
| leave-one-chair-out, drop Volcker | long-run phi_pi | 1.06 | full non-ZLB sample | N 179, SE 0.35 |
| leave-one-chair-out, drop Greenspan | long-run phi_pi | 1.93 | full non-ZLB sample | N 137, SE 0.53 |
| leave-one-chair-out, drop Bernanke | long-run phi_pi | 1.67 | full non-ZLB sample | N 199, SE 0.48 |
| leave-one-chair-out, drop Yellen | long-run phi_pi | 1.90 | full non-ZLB sample | N 203, SE 0.53 |
| leave-one-chair-out, drop Powell | long-run phi_pi | 1.72 | full non-ZLB sample | N 184, SE 0.48 |
| leave-one-year-out jackknife, Greenspan-Bernanke partial adjustment | range of long-run phi_pi | 0.55 | [1.13, 1.68] | 21 years |

The errors-in-variables formula of Proposition 5, evaluated with sample moments, predicts the revised-data gap response almost exactly but predicts an inflation response of 0.96 against an actual 1.71. The gap result supports the view that the Fed set policy on its staff's gap estimates. The inflation result says that revised inflation carries information beyond the Greenbook nowcast that the Fed was reacting to, for instance from core measures or the staff's forecast, which a single real-time inflation rate does not capture.

Omitted-variable sensitivity in the sense of Oster (2019) or Cinelli and Hazlett (2020) is not meaningful here, because the omitted variable is the policy shock itself, which is correlated with the regressors by construction (Proposition 3). Figure 4B plays the same role. It maps the probability limit of the OLS estimator as the persistence and volatility of policy shocks vary, holding the other structural parameters at their estimates.

### 4.4 Structural estimates

Minimum-distance estimates, 1987Q3 to 2007Q4, annualized units:

| Parameter | Estimate | SE | At bound |
|---|---|---|---|
| s, real-rate elasticity of the gap | 0.164 | 0.215 |  |
| κ, Phillips curve slope | 0.033 | 0.038 |  |
| φπ, inflation response | 3.110 | 1.575 |  |
| φy, gap response | 1.900 | 1.037 |  |
| ρi, smoothing | 0.636 | 0.340 |  |
| ρd, demand shock persistence | 0.878 | 0.065 |  |
| ρu, cost-push persistence | 0.787 | 0.098 |  |
| ρv, policy shock persistence | 0.970 | 0.048 | yes |
| σd | 0.235 | 0.184 |  |
| σu | 0.108 | 0.060 |  |
| σv | 0.214 | 0.294 |  |
| beta (fixed) | 0.990 |  |  |
| objective | 0.804 |  |  |
| N quarters | 82 |  |  |

The estimated policy shock persistence sits at the upper bound of 0.97. The model reads much of the funds rate's inertia, and the low-frequency drift of the neutral rate, as persistent deviations from the rule. That is consistent with Rudebusch's (2002) evidence, and it is also the main reason the structural $\phi_\pi$ is imprecise. The profile in Figure 4C has its minimum at $\phi_\pi$ = 3.2 and rises by at most 0.51 across the grid from 0.6 to 5.0, against a minimum objective of 0.80. The 21 moments barely discriminate between a passive response and a very aggressive one, which is Proposition 2's identification problem reappearing inside the richer model.

### 4.5 Counterfactual 2021 onward

Paths from 2021Q1 to the latest quarter. Output loss is the cumulative shortfall of the gap relative to the actual path over 2021 to 2023, in percent of a year's potential output.

| Path | Rate 2021Q4 | Peak rate | Peak 4-quarter inflation | Mean inflation 2021-2023 | Trough gap | Output loss 2021-2023 | Quarters at ZLB |
|---|---|---|---|---|---|---|---|
| Actual | 0.08 | 5.33 (2023Q4) | 7.8 (2022Q2) | 5.2 | −0.5 (2021Q1) |  |  |
| Taylor (1993) rule, r* = 2 | 1.19 | 4.56 (2024Q2) | 3.2 (2021Q2) | 2.0 | −8.6 (2022Q1) | 16.2 | 0 |
| Taylor (1993) rule, r* = 0.5 | 1.13 | 4.51 (2024Q2) | 3.5 (2021Q2) | 2.6 | −7.4 (2022Q1) | 12.4 | 0 |
| estimated 1987-2007 rule, no deviations | 0.22 | 4.51 (2024Q3) | 3.6 (2022Q2) | 2.5 | −5.5 (2022Q2) | 10.3 | 2 |

### 4.6 Monte Carlo validation

True φπ is the structural estimate. Coverage is for the nominal 95 percent Wald interval; AR coverage is the Anderson-Rubin test's acceptance rate at the true value.

| DGP | T | Replications | Estimator | True φπ | Mean | Bias | Emp. SE | RMSE | Coverage | Analytic plim | J rejects | Mean F_eff | AR coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| estimated NK model (partial adjustment) | 82 | 2000 | OLS, static rule | 3.11 | 1.60 | −1.51 | 0.66 | 1.65 | 0.106 | 1.26 |  |  |  |
| estimated NK model (partial adjustment) | 82 | 2000 | OLS, partial adjustment | 3.11 | 2.43 | −0.68 | 1.65 | 1.79 | 0.501 | 1.66 |  |  |  |
| estimated NK model (partial adjustment) | 82 | 2000 | GMM, lagged instruments | 3.11 | 2.71 | −0.40 | 10.07 | 10.08 | 0.507 |  | 0.875 |  |  |
| estimated NK model (partial adjustment) | 1000 | 500 | OLS, static rule | 3.11 | 1.30 | −1.81 | 0.22 | 1.82 | 0.000 | 1.26 |  |  |  |
| estimated NK model (partial adjustment) | 1000 | 500 | OLS, partial adjustment | 3.11 | 1.77 | −1.34 | 0.33 | 1.38 | 0.002 | 1.66 |  |  |  |
| estimated NK model (partial adjustment) | 1000 | 500 | GMM, lagged instruments | 3.11 | 2.15 | −0.96 | 0.50 | 1.08 | 0.234 |  | 1.000 |  |  |
| estimated NK model, static rule | 82 | 2000 | OLS, static rule | 3.11 | 2.90 | −0.21 | 0.34 | 0.40 | 0.563 | 2.70 |  |  |  |
| estimated NK model, static rule | 82 | 2000 | OLS, partial adjustment | 3.11 | 2.96 | −0.15 | 0.36 | 0.39 | 0.616 | 2.78 |  |  |  |
| estimated NK model, static rule | 82 | 2000 | GMM, lagged instruments | 3.11 | 2.40 | −0.71 | 14.31 | 14.33 | 0.545 |  | 0.638 |  |  |
| estimated NK model, static rule | 82 | 2000 | 2SLS, strong external instrument | 3.11 | 3.13 | 0.02 | 0.30 | 0.30 | 0.828 |  |  | 25.92 |  |
| estimated NK model, static rule | 82 | 2000 | 2SLS, weak external instrument | 3.11 | 3.03 | −0.08 | 0.63 | 0.63 | 0.888 |  |  | 1.99 | 0.892 |
| estimated NK model, static rule | 1000 | 500 | OLS, static rule | 3.11 | 2.72 | −0.39 | 0.14 | 0.42 | 0.042 | 2.70 |  |  |  |
| estimated NK model, static rule | 1000 | 500 | OLS, partial adjustment | 3.11 | 2.80 | −0.31 | 0.14 | 0.34 | 0.162 | 2.78 |  |  |  |
| estimated NK model, static rule | 1000 | 500 | GMM, lagged instruments | 3.11 | 2.43 | −0.68 | 0.36 | 0.76 | 0.306 |  | 1.000 |  |  |
| estimated NK model, static rule | 1000 | 500 | 2SLS, strong external instrument | 3.11 | 3.11 | −0.00 | 0.11 | 0.11 | 0.934 |  |  | 241.70 |  |
| estimated NK model, static rule | 1000 | 500 | 2SLS, weak external instrument | 3.11 | 3.08 | −0.03 | 0.24 | 0.24 | 0.944 |  |  | 9.67 | 0.950 |

Three results matter. First, the analytic limits of Proposition 3 match the simulated means. For the static-rule process at $T = 1000$ the limits are 2.70 and 2.78 against simulated means of 2.72 and 2.80. In the estimated process, where policy shocks are nearly a unit root, convergence is slower: the limits are 1.26 and 1.66 against simulated means of 1.30 and 1.77. Second, GMM with lagged instruments is biased under persistent shocks, as Proposition 4 predicts, and Hansen's $J$ detects the invalid instruments in most samples of 82 quarters. Its failure to reject in the data ($J$ p-value 0.61 for 1979 to 2007) is therefore mild evidence that real-world policy shocks are less persistent than the estimated model says. Third, with an instrument as weak as Känzig's in the data, 2SLS Wald intervals undercover, while the Anderson-Rubin test keeps its size close to nominal.

### 4.7 Auxiliary predictions

The theory makes predictions beyond the headline coefficient, and I test four. If the Volcker appointment changed the regime, a break should appear at 1979: the Quandt-Andrews test puts it at 1979Q4. If the 1970s inflation reflected a passive rule, the inflation response should be below one before 1979 and above one after: it is, on Clarida, Galí and Gertler's windows (0.87 and 2.90), but not when the early sample starts in 1965. If the Fed responded to what it saw, real-time data should fit at least as well as revised data: they do not ($R^2$ 0.966 against 0.971 on the same quarters), which together with Section 4.3 suggests the Fed used more information than the Greenbook nowcast. If central banks look through supply shocks, the response to oil-driven inflation should be smaller than the response to inflation in general: the Anderson-Rubin set for CPI inflation instrumented by oil news, [−2.12, 1.51], excludes the OLS and GMM estimates from the Volcker-Greenspan era.

---

## 5. Design decisions

| Decision | Choice | Alternative considered | Reason |
|---|---|---|---|
| Trend for the 1992 vintage gap | Slope 2.2 percent, intercept so the gap averages zero over 1984Q1 to 1992Q3 | Line through the 1984Q1 observation | Taylor states the slope and window but not the level; least squares is the neutral choice |
| Revised output gap | CBO potential | HP filter, Taylor trend | CBO is the measure used by the Fed and by Clarida, Galí and Gertler's successors; the Taylor trend is reported as a check |
| Inflation measure | GDP deflator, four quarters | Core PCE, CPI | Matches Taylor (1993); CPI is used where oil identification needs it |
| Policy rate | Quarterly average of the effective funds rate | End of quarter | Matches the quarterly averaging of GDP and inflation |
| Greenbook vintage per quarter | Closest to mid-quarter | First meeting of the quarter | Balances information within the quarter |
| Zero lower bound | Rate below 0.25 percent | Shadow rate | Shadow rates are model estimates; censoring uses only observed data |
| IV specification | Gap response fixed at 0.5, one endogenous regressor | Two endogenous regressors | Oil news is too weak to identify two coefficients; fixing the gap response keeps Anderson-Rubin inference valid |
| Instruments | Quarterly sums at lags 0 to 3 | Longer distributed lags | Four-quarter inflation responds within a year to oil news |
| Policy shock persistence bound | 0.97 | Unbounded | Keeps the shock stationary; the estimate at the bound is reported |
| Weight matrix | Diagonal inverse bootstrap variances | Full optimal matrix | 21 moments and 82 quarters make the full inverse unstable |
| Neutral rate in the counterfactual | 2 percent (Taylor) and 0.5 percent | Time-varying estimate | Brackets the range without importing another model |
| Numba | Not used | JIT for the simulation loop | The state has seven elements; the full Monte Carlo is dominated by estimation, not simulation |

Methods from the standard toolkit that I did not use, and why. Difference-in-differences, event studies, HonestDiD and synthetic control need treated and untreated units; there is one Federal Reserve. Regression discontinuity and bunching designs need a running variable with a threshold, and no rule threshold exists here. Double/debiased machine learning and causal forests target treatment effects under selection on observables with many controls; the Taylor rule's problem is simultaneity, not high-dimensional confounding, so cross-fitting nuisance functions would not address it. Shift-share inference does not apply. Wild cluster bootstrap needs clusters. LIML and Fuller estimators reduce bias with moderately weak instruments; with effective $F$ below 8 I rely on Anderson-Rubin sets, which remain valid at any strength. Bai-Perron multiple-break estimation is replaced by the single-break test plus rolling windows, which show the post-2008 instability directly.

---

## 6. Reproduction

```bash
make env        # Python 3.12 virtual environment with pinned requirements
make data       # download and rebuild data/processed/analytical_panel.parquet
make analysis   # Stage 1 to 4 estimates, structural model, Monte Carlo
make figures    # results/fig*.png at 300 DPI
make test       # unit tests (offline, a few minutes)
make check      # quality gate
```

`make analysis` reads only the committed panel and runs offline. The code was run with Python 3.12.8. All randomness flows from one seed in `src/core/__init__.py`.

---

## 7. References

- Anderson, T. W. and H. Rubin (1949). "Estimation of the Parameters of a Single Equation in a Complete System of Stochastic Equations." *Annals of Mathematical Statistics* 20(1): 46–63. https://doi.org/10.1214/aoms/1177730090
- Andrews, D. W. K. (1993). "Tests for Parameter Instability and Structural Change with Unknown Change Point." *Econometrica* 61(4): 821–856. https://doi.org/10.2307/2951764
- Blanchard, O. J. and C. M. Kahn (1980). "The Solution of Linear Difference Models under Rational Expectations." *Econometrica* 48(5): 1305–1311. https://doi.org/10.2307/1912186
- Bullard, J. and K. Mitra (2002). "Learning about Monetary Policy Rules." *Journal of Monetary Economics* 49(6): 1105–1129. https://doi.org/10.1016/S0304-3932(02)00144-7
- Cinelli, C. and C. Hazlett (2020). "Making Sense of Sensitivity: Extending Omitted Variable Bias." *Journal of the Royal Statistical Society Series B* 82(1): 39–67. https://doi.org/10.1111/rssb.12348
- Clarida, R., J. Galí and M. Gertler (1998). "Monetary Policy Rules and Macroeconomic Stability: Evidence and Some Theory." NBER Working Paper 6442. https://www.nber.org/papers/w6442
- Clarida, R., J. Galí and M. Gertler (2000). "Monetary Policy Rules and Macroeconomic Stability: Evidence and Some Theory." *Quarterly Journal of Economics* 115(1): 147–180. https://doi.org/10.1162/003355300554692
- Cochrane, J. H. (2011). "Determinacy and Identification with Taylor Rules." *Journal of Political Economy* 119(3): 565–615. https://doi.org/10.1086/660817
- Croushore, D. and T. Stark (2001). "A Real-Time Data Set for Macroeconomists." *Journal of Econometrics* 105(1): 111–130. https://doi.org/10.1016/S0304-4076(01)00072-0
- Hansen, L. P. (1982). "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica* 50(4): 1029–1054. https://doi.org/10.2307/1912775
- Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian Journal of Statistics* 6(2): 65–70. https://www.jstor.org/stable/4615733
- Känzig, D. R. (2021). "The Macroeconomic Effects of Oil Supply News: Evidence from OPEC Announcements." *American Economic Review* 111(4): 1092–1125. https://doi.org/10.1257/aer.20190964
- Klein, P. (2000). "Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model." *Journal of Economic Dynamics and Control* 24(10): 1405–1423. https://doi.org/10.1016/S0165-1889(99)00045-7
- Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations." *Annals of Statistics* 17(3): 1217–1241. https://doi.org/10.1214/aos/1176347265
- Montiel Olea, J. L. and C. Pflueger (2013). "A Robust Test for Weak Instruments." *Journal of Business and Economic Statistics* 31(3): 358–369. https://doi.org/10.1080/00401706.2013.806694
- Newey, W. K. and K. D. West (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55(3): 703–708. https://doi.org/10.2307/1913610
- Newey, W. K. and K. D. West (1994). "Automatic Lag Selection in Covariance Matrix Estimation." *Review of Economic Studies* 61(4): 631–653. https://doi.org/10.2307/2297912
- Orphanides, A. (2001). "Monetary Policy Rules Based on Real-Time Data." *American Economic Review* 91(4): 964–985. https://doi.org/10.1257/aer.91.4.964
- Orphanides, A. (2003). "Historical Monetary Policy Analysis and the Taylor Rule." *Journal of Monetary Economics* 50(5): 983–1022. https://doi.org/10.1016/S0304-3932(03)00065-5
- Oster, E. (2019). "Unobservable Selection and Coefficient Stability: Theory and Evidence." *Journal of Business and Economic Statistics* 37(2): 187–204. https://doi.org/10.1080/07350015.2016.1227711
- Rudebusch, G. D. (2002). "Term Structure Evidence on Interest Rate Smoothing and Monetary Policy Inertia." *Journal of Monetary Economics* 49(6): 1161–1187. https://doi.org/10.1016/S0304-3932(02)00149-6
- Taylor, J. B. (1993). "Discretion versus Policy Rules in Practice." *Carnegie-Rochester Conference Series on Public Policy* 39: 195–214. https://doi.org/10.1016/0167-2231(93)90009-L
- Taylor, J. B. (2007). "Housing and Monetary Policy." NBER Working Paper 13682. https://www.nber.org/papers/w13682
- Tobin, J. (1958). "Estimation of Relationships for Limited Dependent Variables." *Econometrica* 26(1): 24–36. https://doi.org/10.2307/1907382
