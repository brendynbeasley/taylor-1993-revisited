# Findings

**Verdict: the original finding is attenuated.** Taylor's rule describes Federal Reserve policy closely on the data available in 1992 and over Alan Greenspan's tenure, and poorly in every other era. The coefficient it is built around is not identified by the regressions usually used to estimate it.

**Replication.** On the 1992Q4 data vintage the rule tracks the funds rate over 1987Q1 to 1992Q3 with an RMSE of 0.40 percentage points (correlation 0.975). On today's revised data with the CBO gap the RMSE is 1.27, about 3.2 times larger, almost entirely because the 1992Q3 output gap was revised from −2.36 to −0.97.

**Eras.** Mean funds rate minus rule, revised data:

| Chair | Quarters | Mean deviation | RMSE |
|---|---|---|---|
| Burns | 33 | −3.62 | 4.03 |
| Volcker | 32 | 2.72 | 3.50 |
| Greenspan | 74 | 0.13 | 1.63 |
| Bernanke | 32 | −1.37 | 1.75 |
| Powell (to 2026Q3) | 35 | −3.52 | 5.03 |

A Quandt-Andrews test locates a break in the rule at 1979Q4 (sup-Wald 64.3, simulated 5 percent critical value 13.9). The largest single deviation in the sample is 2022Q2: a funds rate of 0.77 percent against a prescription of 13.03 percent.

**Estimates.** Across 19 specifications the long-run inflation response ranges from −0.35 to 4.12. After a Holm adjustment only the three Volcker-Greenspan specifications reject φπ ≤ 1. Real-time Greenbook data raise the response from 1.71 to 2.26 on the same 98 quarters. On Clarida, Galí and Gertler's windows the GMM estimates are 0.87 and 2.90, against their 0.80 and 1.96.

**Identification.** In the estimated New Keynesian model (φπ = 3.11, SE 1.58) a static OLS Taylor regression converges to 1.26. Oil supply news is too weak to identify φπ for deflator inflation (effective F 0.47 and 2.18). For CPI inflation (F 7.43) the Anderson-Rubin set is [−2.12, 1.51], consistent with the Fed looking through oil-driven inflation.

**Counterfactual.** Following Taylor's rule from 2021Q1, the model's four-quarter inflation peaks at 3.2 percent instead of 7.8, while the output gap bottoms out at −8.6 percent and output over 2021 to 2023 falls short of the actual path by 16.2 percent of a year's potential.

**Named outliers** (`named_entities_table.csv`): Burns in 1974Q4 to 1975Q3 (OPEC embargo; up to 8.77 points below the rule), Volcker in 1982 (up to 6.66 points above it during the disinflation), Powell in 2022 (12.26 points below it in 2022Q2), and Greenspan's 2001 cuts and the 1998 LTCM easing as the largest residuals from the estimated Greenspan-era rule.
