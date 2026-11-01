"""The New Keynesian solver satisfies the model's equations and its known analytic properties."""
from dataclasses import replace

import numpy as np
import pytest

from core import nk


def test_taylor_principle_boundary():
    for phi, det in [(0.8, False), (0.99, False), (1.01, True), (2.0, True)]:
        p = nk.Params(phi_pi=phi, phi_y=0.0, rho_i=0.0, four_quarter=False)
        assert nk.solve(p).determinate is det


@pytest.mark.parametrize("phi_y", [0.25, 1.0])
def test_bullard_mitra_generalized_principle(phi_y):
    """Determinacy iff phi_pi + (1 - beta) phi_y / kappa > 1 for a contemporaneous rule."""
    p = nk.Params(phi_y=phi_y, rho_i=0.0, four_quarter=False)
    boundary = 1 - (1 - p.beta) * phi_y / p.kappa
    assert nk.solve(replace(p, phi_pi=boundary + 0.02)).determinate
    assert not nk.solve(replace(p, phi_pi=boundary - 0.02)).determinate


def test_equilibrium_conditions_hold_in_simulation():
    p = nk.Params(rho_i=0.7)
    sol = nk.solve(p)
    s = nk.simulate(sol, 30000, np.random.default_rng(0))
    rule = s["i"] - (p.rho_i * s["ilag"] + (1 - p.rho_i) * (p.phi_pi * s["pibar"] + p.phi_y * s["x"]))
    np.testing.assert_allclose(rule, s["v"], atol=1e-10)
    is_err = s["x"][:-1] - s["x"][1:] + p.s * (s["i"][:-1] - s["pi"][1:]) - s["d"][:-1]
    pc_err = s["pi"][:-1] - p.beta * s["pi"][1:] - p.kappa * s["x"][:-1] - s["u"][:-1]
    for err in (is_err, pc_err):  # expectation errors are unforecastable from date-t states
        assert abs(np.corrcoef(err, s["x"][:-1])[0, 1]) < 0.03
        assert abs(np.corrcoef(err, s["pi"][:-1])[0, 1]) < 0.03


def test_analytic_plim_matches_simulation():
    p = nk.Params(sd_v=1.5, rho_v=0.6)
    sol = nk.solve(p)
    s = nk.simulate(sol, 200000, np.random.default_rng(1))
    X = np.column_stack([np.ones(len(s["i"])), s["pibar"], s["x"]])
    b = np.linalg.lstsq(X, s["i"], rcond=None)[0]
    np.testing.assert_allclose(b[1:], nk.ols_plim(sol, smoothing=False), atol=0.02)


def test_autocov_matches_simulation():
    sol = nk.solve(nk.Params(rho_i=0.5))
    s = nk.simulate(sol, 200000, np.random.default_rng(2))
    Y = np.column_stack([s["x"], s["pi"], s["i"]])
    G = nk.autocov(sol, (0, 1))
    np.testing.assert_allclose(np.cov(Y.T, bias=True), G[0], rtol=0.05, atol=0.02)


def test_cochrane_frictionless_ols_recovers_rho_not_phi():
    rng = np.random.default_rng(3)
    for phi, rho in [(1.5, 0.7), (4.0, 0.7)]:
        load, slope = nk.frictionless_equilibrium(phi, rho)
        v = np.zeros(100000)
        e = rng.standard_normal(len(v))
        for t in range(1, len(v)):
            v[t] = rho * v[t - 1] + e[t]
        pi = load * v
        i = phi * pi + v
        # Fisher equation holds in expectation: i_t = E_t pi_{t+1} = rho * pi_t
        np.testing.assert_allclose(i, rho * pi, atol=1e-10)
        assert np.polyfit(pi, i, 1)[0] == pytest.approx(slope, abs=1e-8)
