"""Each estimator recovers known parameters from simulated data or matches a reference implementation."""
import numpy as np
import pytest
import statsmodels.api as sm

from core import estimators as est


def ar1(n, rho, rng):
    e = rng.standard_normal(n)
    out = np.empty(n)
    out[0] = e[0]
    for t in range(1, n):
        out[t] = rho * out[t - 1] + e[t]
    return out


def test_ols_hac_matches_statsmodels():
    rng = np.random.default_rng(1)
    n = 300
    x = ar1(n, 0.7, rng)
    y = 1.0 + 2.0 * x + ar1(n, 0.5, rng)
    X = np.column_stack([np.ones(n), x])
    ours = est.ols(y, X, ["c", "x"], lags=4, small_sample=False)
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": False})
    np.testing.assert_allclose(ours.params, ref.params, rtol=1e-10)
    np.testing.assert_allclose(ours.se, ref.bse, rtol=1e-8)


def test_ols_recovers_parameters():
    rng = np.random.default_rng(2)
    n = 20000
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    y = X @ np.array([1.0, 1.5, 0.5]) + rng.standard_normal(n)
    f = est.ols(y, X, ["c", "pi", "y"])
    np.testing.assert_allclose(f.params, [1.0, 1.5, 0.5], atol=0.03)


def iv_dgp(n, rng, strength=1.0):
    z = rng.standard_normal((n, 3))
    common = rng.standard_normal(n)
    v = common + rng.standard_normal(n)
    u = common + rng.standard_normal(n)
    x = strength * z.sum(1) + v
    y = 1.0 + 2.0 * x + u
    return y, x, z


def test_gmm_recovers_parameter_and_j_has_correct_size():
    rng = np.random.default_rng(3)
    y, x, z = iv_dgp(20000, rng)
    g = est.gmm_linear(y, np.column_stack([np.ones(len(y)), x]), np.column_stack([np.ones(len(y)), z]), ["c", "x"])
    assert abs(g.get("x")[0] - 2.0) < 0.03
    pvals = []
    for _ in range(300):
        y, x, z = iv_dgp(500, rng)
        g = est.gmm_linear(y, np.column_stack([np.ones(500), x]), np.column_stack([np.ones(500), z]), ["c", "x"])
        pvals.append(g.extra["J_p"])
    assert 0.02 < np.mean(np.array(pvals) < 0.05) < 0.09


def test_iv_single_and_effective_f():
    rng = np.random.default_rng(4)
    y, x, z = iv_dgp(5000, rng, strength=0.3)
    W = np.ones((len(y), 1))
    g = est.iv_single(y, x, W, z, ["x", "c"], lags=0)
    assert abs(g.get("x")[0] - 2.0) < 0.1
    # Under homoskedasticity the effective F should be close to the conventional first-stage F.
    first = sm.OLS(x, np.column_stack([np.ones(len(x)), z])).fit()
    f_conv = first.f_test(np.eye(4)[1:]).fvalue
    assert abs(g.extra["F_eff"] / float(np.squeeze(f_conv)) - 1) < 0.1


def test_anderson_rubin_set_covers_truth():
    rng = np.random.default_rng(5)
    y, x, z = iv_dgp(2000, rng, strength=0.5)
    W = np.ones((len(y), 1))
    ar = est.anderson_rubin_set(y, x, W, z, np.linspace(0, 4, 401), lags=0)
    assert ar["lo"] <= 2.0 <= ar["hi"]
    assert ar["bounded"]


def test_tobit_recovers_censored_regression():
    rng = np.random.default_rng(6)
    n = 8000
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    ystar = X @ np.array([0.5, 1.0]) + rng.standard_normal(n)
    y = np.maximum(ystar, 0.0)
    t = est.tobit_left(y, X, ["c", "x"], censor=0.0)
    np.testing.assert_allclose(t.params[:2], [0.5, 1.0], atol=0.05)
    assert abs(np.exp(t.params[2]) - 1.0) < 0.05


def test_long_run_delta_method():
    f = est.Fit(["rho", "b"], np.array([0.8, 0.3]), np.diag([0.01, 0.04]), 100)
    val, se = est.long_run(f, "b", "rho")
    assert val == pytest.approx(1.5)
    grad = np.array([0.3 / 0.04, 1 / 0.2])
    assert se == pytest.approx(np.sqrt(grad @ np.diag([0.01, 0.04]) @ grad))


def test_qlr_finds_planted_break():
    rng = np.random.default_rng(7)
    n = 200
    x = rng.standard_normal(n)
    beta = np.where(np.arange(n) < 120, 0.5, 2.0)
    y = 1 + beta * x + 0.5 * rng.standard_normal(n)
    q = est.qlr_break(y, np.column_stack([np.ones(n), x]), lags=0, rng=rng, reps=500)
    assert abs(q["break_index"] - 120) <= 3
    assert q["p_value"] < 0.01


def test_holm():
    np.testing.assert_allclose(est.holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
