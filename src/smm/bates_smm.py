"""
§4.3  Bates SMM Calibration

Model (log-odds space, Δt = 1):
  dX_t = √v_t · dW^X_t  +  J_t · dN_t
  dv_t = κ(θ − v_t)dt   + σ_v · √v_t · dW^v_t
  corr(dW^X, dW^v) = ρ    (baseline: ρ = 0)
  N_t ~ Poisson(λ),  J ~ N(0, σ_J²)

Parameter schedule
------------------
κ : FREE in the main selection run (shared spec across the ladder).  κ only
    enters the simulator through σ_v·√v, so it is identified in Heston and
    Bates and irrelevant (v ≡ θ) in ConstantVol and Merton.  §4.8 re-runs
    the ladder with κ fixed on the grid {1, 5, 10} as a robustness check.
ρ : fixed — 0 (baseline)
θ : fixed — Var(ΔX) / Δt  (with Δt=1: θ = sample variance of increments)
λ : fixed — fraction of |ΔX| > 3·SD (rough jump frequency)
σ_v : FREE — optimizer
σ_J : FREE — optimizer

Nesting guarantee
-----------------
Each model is warm-started at the optimum of the model it nests (Heston at
ConstantVol with σ_v=0; Bates at Heston with σ_J=0 and at Merton with
σ_v=0), so a richer model can never end at a worse objective than the
model it nests — worst case it stays at the warm start.  For this to hold
exactly, ConstantVol is scored by SIMULATION at (σ_v=0, σ_J=0) with the
shared CRN draws, not analytically.

Five moments (computed once on pooled, demeaned real increments)
  [0] Var(ΔX)
  [1] Excess kurtosis(ΔX)
  [2] lag-1 ACF(ΔX²)
  [3] lag-2 ACF(ΔX²)
  [4] 95th-percentile of ΔX

Weighting matrix W
  Diagonal, entries = 1/bootstrap-variance of each moment.
  Computed once, fixed throughout optimisation.

Common random numbers
  Pre-drawn once before optimisation; reused on every objective evaluation.
  Makes the objective deterministic in the parameters so Nelder-Mead converges.

Simulator: Euler–Maruyama with full-truncation (replace v with max(v,0)
           everywhere it appears, including the drift and the sqrt term).

Optimiser: Nelder-Mead with multiple restarts.

J-test: with 5 moments and 2 free params → 3 over-identifying restrictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOMENT_LABELS = [
    "Var(ΔX)",
    "Kurt(ΔX)",
    "ACF(ΔX², lag=1)",
    "ACF(ΔX², lag=2)",
    "95th pct(ΔX)",
]
N_MOMENTS = len(MOMENT_LABELS)
LARGE_PENALTY = 1e9
# Upper bound for free κ — twice the top of the {1,5,10} robustness grid;
# unbounded κ lets Nelder-Mead wander into Euler-unstable territory.
KAPPA_MAX = 20.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SMMResult:
    model: str           # "ConstantVol", "Heston", "Bates", "Merton"
    kappa: float
    theta: float
    rho: float
    lambda_: float
    # Free parameters (nan if not applicable to model)
    sigma_v: float
    sigma_J: float
    # Moments
    moments_real: np.ndarray
    moments_sim: np.ndarray
    moment_labels: list = field(default_factory=lambda: list(MOMENT_LABELS))
    # Fit quality
    objective_value: float = np.nan
    j_stat: float = np.nan
    j_dof: int = 0
    j_pvalue: float = np.nan
    # Inference
    se_sigma_v: float = np.nan
    se_sigma_J: float = np.nan
    se_kappa: float = np.nan
    # Book-keeping
    kappa_free: bool = False
    n_real: int = 0
    n_sim: int = 0
    n_free: int = 2
    converged: bool = True


# ---------------------------------------------------------------------------
# Moment computation
# ---------------------------------------------------------------------------

def _acf_scalar(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return np.nan
    c = np.corrcoef(x[:-lag], x[lag:])
    return float(c[0, 1]) if c.shape == (2, 2) else np.nan


def compute_moments(dx: np.ndarray) -> np.ndarray:
    """Five SMM moments on a demeaned increment array."""
    n = len(dx)
    if n < 10:
        return np.full(N_MOMENTS, np.nan)
    dx2 = dx ** 2
    return np.array([
        float(np.var(dx, ddof=1)),
        float(stats.kurtosis(dx, fisher=True)),
        _acf_scalar(dx2, 1),
        _acf_scalar(dx2, 2),
        float(np.percentile(dx, 95)),
    ])


def pool_demeaned_increments(panel: pd.DataFrame) -> np.ndarray:
    """Concatenate within-contract demeaned ΔX across all contracts."""
    parts: list[np.ndarray] = []
    for _, grp in panel.groupby("contract_id", sort=False):
        dx = grp["delta_X"].dropna().values
        if len(dx) >= 2:
            parts.append(dx - dx.mean())
    return np.concatenate(parts) if parts else np.array([])


# ---------------------------------------------------------------------------
# Euler–Maruyama simulator (pure Python/NumPy, full truncation)
# ---------------------------------------------------------------------------

def _simulate_path(
    sigma_v: float,
    sigma_J: float,
    randoms: dict,
    kappa: float,
    theta: float,
    lambda_: float,
    rho: float,
    n_sim: int,
) -> np.ndarray:
    """Simulate one Bates path and return the ΔX array.

    The variance loop is sequential (v[i+1] depends on v[i]) so it cannot be
    fully vectorised.  If numba is installed, wrapping this function with
    @numba.jit(nopython=True) gives a ~100× speedup; for a research pipeline
    that runs once, the pure-Python version is acceptable.
    """
    dw_x   = randoms["dw_x"]
    dw_v   = randoms["dw_v"]   # already correlated: ρ·dw_x + √(1-ρ²)·dw_v_indep
    u_jump = randoms["u_jump"]
    z_jump = randoms["z_jump"]

    jump_threshold = lambda_   # Δt = 1 so P(jump in step) = λ·1

    v = theta          # initial variance
    dx_sim = np.empty(n_sim)

    for i in range(n_sim):
        v_pos = max(v, 0.0)
        sv    = v_pos ** 0.5

        jump = sigma_J * z_jump[i] if u_jump[i] < jump_threshold else 0.0
        dx_sim[i] = sv * dw_x[i] + jump

        # Full-truncation Euler for v (Δt = 1)
        v = v_pos + kappa * (theta - v_pos) + sigma_v * sv * dw_v[i]
        v = max(v, 0.0)

    return dx_sim


# ---------------------------------------------------------------------------
# Main calibrator
# ---------------------------------------------------------------------------

class BatesSMM:
    """Calibrate the Bates (or nested) model via SMM.

    Parameters
    ----------
    n_sim_multiplier : int
        n_sim = n_sim_multiplier × n_real.  50 in the spec; use 10 while
        debugging for speed, then increase to 50 for the final run.
    n_bootstrap : int
        Bootstrap resamples for the weighting matrix (spec: ~500).
    n_restarts : int
        Nelder-Mead restarts with different starting points.
    rng_seed : int
        Controls reproducibility of bootstrap and CRN draws.
    """

    def __init__(
        self,
        n_sim_multiplier: int = 20,
        n_bootstrap: int = 500,
        n_restarts: int = 3,
        rng_seed: int = 42,
    ):
        self.n_sim_multiplier = n_sim_multiplier
        self.n_bootstrap = n_bootstrap
        self.n_restarts = n_restarts
        self.rng_seed = rng_seed

    # ------------------------------------------------------------------
    # Public API — called by BatesSMM.fit() and NestedLadder
    # ------------------------------------------------------------------

    def prepare(self, panel: pd.DataFrame) -> dict:
        """Pre-compute everything that is fixed across optimisation runs.

        Returns a 'cache' dict with:
          dx_real, m_real, W, theta, lambda_, n_real, n_sim, randoms
        """
        dx_real = pool_demeaned_increments(panel)
        n_real  = len(dx_real)
        n_sim   = self.n_sim_multiplier * n_real

        m_real = compute_moments(dx_real)
        W      = self._bootstrap_weights(dx_real)

        sd_real = float(np.std(dx_real, ddof=1))
        lambda_ = float(np.mean(np.abs(dx_real) > 3 * sd_real))

        # Use truncated variance (increments within 3 SD) so that jump variance
        # doesn't fold into θ and double-count once σ_J enters the simulator.
        # The spec recommends this fix when σ_v is being pushed to a boundary.
        diffusion_mask = np.abs(dx_real) <= 3 * sd_real
        if diffusion_mask.sum() > 10:
            theta = float(np.var(dx_real[diffusion_mask], ddof=1))
        else:
            theta = float(np.var(dx_real, ddof=1))

        randoms = self._draw_randoms(n_sim, rho=0.0)

        return {
            "dx_real": dx_real,
            "m_real":  m_real,
            "W":       W,
            "theta":   theta,
            "lambda_": lambda_,
            "sd_real": sd_real,
            "n_real":  n_real,
            "n_sim":   n_sim,
            "randoms": randoms,
        }

    def fit_bates(
        self,
        cache: dict,
        kappa: float = 5.0,
        rho: float = 0.0,
        verbose: bool = True,
        free_kappa: bool = False,
        warm_starts: "list[np.ndarray] | None" = None,
    ) -> SMMResult:
        """Fit Bates (σ_v and σ_J free; κ free too when free_kappa)."""
        free = ("sigma_v", "sigma_J", "kappa") if free_kappa \
            else ("sigma_v", "sigma_J")
        return self._fit(
            cache, kappa=kappa, rho=rho,
            model="Bates", free=free,
            warm_starts=warm_starts, verbose=verbose,
        )

    def fit_heston(
        self,
        cache: dict,
        kappa: float = 5.0,
        rho: float = 0.0,
        verbose: bool = True,
        free_kappa: bool = False,
        warm_starts: "list[np.ndarray] | None" = None,
    ) -> SMMResult:
        """Fit Heston (σ_v free, σ_J = 0; κ free too when free_kappa)."""
        free = ("sigma_v", "kappa") if free_kappa else ("sigma_v",)
        return self._fit(
            cache, kappa=kappa, rho=rho,
            model="Heston", free=free,
            warm_starts=warm_starts, verbose=verbose,
        )

    def fit_merton(
        self,
        cache: dict,
        kappa: float = 5.0,
        rho: float = 0.0,
        verbose: bool = True,
        warm_starts: "list[np.ndarray] | None" = None,
    ) -> SMMResult:
        """Fit Merton jump-diffusion (σ_J free, σ_v = 0).

        κ is never free here: with σ_v = 0 the variance stays at θ and κ
        drops out of the model entirely.
        """
        return self._fit(
            cache, kappa=kappa, rho=rho,
            model="Merton", free=("sigma_J",),
            warm_starts=warm_starts, verbose=verbose,
        )

    def fit_constant_vol(
        self,
        cache: dict,
        kappa: float = 5.0,
        rho: float = 0.0,
        verbose: bool = True,
    ) -> SMMResult:
        """Evaluate constant-vol diffusion (no free params).

        Moments are SIMULATED at (σ_v=0, σ_J=0) with the shared CRN draws —
        not the analytical N(0, θ) values — so ConstantVol is exactly the
        σ_v=0 restriction of Heston under the same simulator.  Analytical
        moments would break the J(CV) ≥ J(Heston) nesting guarantee by the
        sampling noise of the CRN draws.  (κ is recorded but inert: σ_v=0
        pins v ≡ θ.)
        """
        m_real  = cache["m_real"]
        W       = cache["W"]
        theta   = cache["theta"]
        lambda_ = cache["lambda_"]
        n_real  = cache["n_real"]
        n_sim   = cache["n_sim"]
        randoms = cache["randoms"]

        m_sim = self._simulate_moments(
            0.0, 0.0, kappa, theta, lambda_, rho, n_sim, randoms
        )
        g = m_sim - m_real
        obj = float(g @ W @ g)
        j_stat, j_p = _j_test(g, W, n_real, n_sim, dof=N_MOMENTS)  # 5 restrictions

        if verbose:
            print(f"  ConstantVol: obj={obj:.4f}, J={j_stat:.3f}"
                  f" (χ²({N_MOMENTS}), p={j_p:.3f})")

        return SMMResult(
            model="ConstantVol",
            kappa=kappa, theta=theta, rho=rho,
            lambda_=lambda_, sigma_v=0.0, sigma_J=0.0,
            moments_real=m_real, moments_sim=m_sim,
            objective_value=obj,
            j_stat=j_stat, j_dof=N_MOMENTS, j_pvalue=j_p,
            n_real=n_real, n_sim=n_sim, n_free=0,
        )

    def standard_errors(
        self,
        result: SMMResult,
        cache: dict,
        h_frac: float = 1e-3,
    ) -> tuple[float, float, float]:
        """Finite-difference Jacobian → asymptotic SEs for the free params.

        avar(θ̂) = (G'WG)⁻¹  [simplified: assuming W ≈ Ω⁻¹]

        κ is included in the Jacobian when the fit had κ free, so the σ
        SEs are not conditional on a fixed κ.

        Returns (se_sigma_v, se_sigma_J, se_kappa); nan if not free.
        """
        W       = cache["W"]
        randoms = cache["randoms"]
        kappa   = result.kappa
        theta   = result.theta
        lambda_ = result.lambda_
        rho     = result.rho
        n_sim   = cache["n_sim"]

        sv  = result.sigma_v
        sJ  = result.sigma_J

        model = result.model
        if model == "Bates":
            free_names = ["sigma_v", "sigma_J"]
        elif model == "Heston":
            free_names = ["sigma_v"]
        elif model == "Merton":
            free_names = ["sigma_J"]
        else:
            return np.nan, np.nan, np.nan
        if result.kappa_free:
            free_names.append("kappa")
        base = {"sigma_v": sv, "sigma_J": sJ, "kappa": kappa}
        free_vals = np.array([base[n] for n in free_names])

        n_free = len(free_vals)
        G = np.zeros((N_MOMENTS, n_free))

        for j, val in enumerate(free_vals):
            h = max(abs(val) * h_frac, 1e-6)
            v_hi = free_vals.copy(); v_hi[j] += h
            v_lo = free_vals.copy(); v_lo[j] -= h

            m_hi = self._moments_at(
                v_hi, free_names, sv, sJ,
                kappa, theta, lambda_, rho, n_sim, randoms
            )
            m_lo = self._moments_at(
                v_lo, free_names, sv, sJ,
                kappa, theta, lambda_, rho, n_sim, randoms
            )
            G[:, j] = (m_hi - m_lo) / (2 * h)

        try:
            GWG = G.T @ W @ G
            avar = np.linalg.inv(GWG) / cache["n_real"]
            ses = np.sqrt(np.abs(np.diag(avar)))
        except np.linalg.LinAlgError:
            ses = np.full(n_free, np.nan)

        se_sv = ses[free_names.index("sigma_v")] if "sigma_v" in free_names else np.nan
        se_sJ = ses[free_names.index("sigma_J")] if "sigma_J" in free_names else np.nan
        se_k  = ses[free_names.index("kappa")]   if "kappa" in free_names else np.nan
        return se_sv, se_sJ, se_k

    # ------------------------------------------------------------------
    # Internal: unified optimiser
    # ------------------------------------------------------------------

    def _fit(
        self,
        cache: dict,
        kappa: float,
        rho: float,
        model: str,
        free: tuple[str, ...],
        verbose: bool,
        warm_starts: "list[np.ndarray] | None" = None,
    ) -> SMMResult:
        m_real  = cache["m_real"]
        W       = cache["W"]
        theta   = cache["theta"]
        lambda_ = cache["lambda_"]
        sd_real = cache["sd_real"]
        n_real  = cache["n_real"]
        n_sim   = cache["n_sim"]
        randoms = cache["randoms"]

        n_free = len(free)
        j_dof  = N_MOMENTS - n_free

        # Starting points: warm starts (the nested model's optimum) always
        # run first and are never dropped by the restart budget — they are
        # what guarantees the richer model can't end worse than the model
        # it nests.
        x0_pool = (
            [np.asarray(w, dtype=float) for w in (warm_starts or [])]
            + self._starting_points(free, sd_real)[: self.n_restarts]
        )

        best_x   = x0_pool[0]
        best_val = np.inf
        best_sim_moments = compute_moments(np.zeros(10))

        for i, x0 in enumerate(x0_pool):
            try:
                res = optimize.minimize(
                    self._objective,
                    x0,
                    args=(free, kappa, theta, lambda_, rho, n_sim, randoms,
                          m_real, W),
                    method="Nelder-Mead",
                    options={
                        "maxiter": 10_000,
                        "xatol": 1e-7,
                        "fatol": 1e-9,
                        "adaptive": True,
                    },
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
                    if verbose:
                        print(f"  [{model}] restart {i+1}: "
                              + ", ".join(f"{n}={abs(v):.4f}"
                                         for n, v in zip(free, res.x))
                              + f"  obj={res.fun:.6f}")
            except Exception as e:
                if verbose:
                    print(f"  [{model}] restart {i+1} failed: {e}")

        # Unpack best parameters
        sv_hat = abs(float(best_x[free.index("sigma_v")])) \
            if "sigma_v" in free else 0.0
        sJ_hat = abs(float(best_x[free.index("sigma_J")])) \
            if "sigma_J" in free else 0.0
        kap_hat = abs(float(best_x[free.index("kappa")])) \
            if "kappa" in free else kappa

        m_sim = self._simulate_moments(
            sv_hat, sJ_hat, kap_hat, theta, lambda_, rho, n_sim, randoms
        )
        g = m_sim - m_real
        j_stat, j_p = _j_test(g, W, n_real, n_sim, dof=j_dof)

        if verbose:
            kap_note = f", κ={kap_hat:.4f}" if "kappa" in free else ""
            print(f"  [{model}] → σ_v={sv_hat:.4f}, σ_J={sJ_hat:.4f}{kap_note}, "
                  f"J={j_stat:.3f} (χ²({j_dof}), p={j_p:.3f})")

        return SMMResult(
            model=model,
            kappa=kap_hat, theta=theta, rho=rho,
            lambda_=lambda_,
            sigma_v=sv_hat, sigma_J=sJ_hat,
            moments_real=m_real, moments_sim=m_sim,
            objective_value=best_val,
            j_stat=j_stat, j_dof=j_dof, j_pvalue=j_p,
            kappa_free=("kappa" in free),
            n_real=n_real, n_sim=n_sim, n_free=n_free,
        )

    def _objective(
        self,
        x: np.ndarray,
        free: tuple[str, ...],
        kappa: float,
        theta: float,
        lambda_: float,
        rho: float,
        n_sim: int,
        randoms: dict,
        m_real: np.ndarray,
        W: np.ndarray,
    ) -> float:
        # σ_v = 0 and σ_J = 0 are legitimate boundary values (they ARE the
        # nested restrictions), so only negatives are penalised.  κ must be
        # strictly positive and bounded — see KAPPA_MAX.
        for name, v in zip(free, x):
            if v < 0:
                return LARGE_PENALTY
            if name == "kappa" and (v <= 0 or v > KAPPA_MAX):
                return LARGE_PENALTY

        sv  = x[free.index("sigma_v")] if "sigma_v" in free else 0.0
        sJ  = x[free.index("sigma_J")] if "sigma_J" in free else 0.0
        kap = x[free.index("kappa")]   if "kappa" in free else kappa

        m_sim = self._simulate_moments(sv, sJ, kap, theta, lambda_, rho,
                                       n_sim, randoms)
        if np.any(np.isnan(m_sim)):
            return LARGE_PENALTY
        g = m_sim - m_real
        return float(g @ W @ g)

    def _simulate_moments(
        self,
        sigma_v: float,
        sigma_J: float,
        kappa: float,
        theta: float,
        lambda_: float,
        rho: float,
        n_sim: int,
        randoms: dict,
    ) -> np.ndarray:
        dx = _simulate_path(
            sigma_v, sigma_J, randoms,
            kappa, theta, lambda_, rho, n_sim,
        )
        dx = dx - dx.mean()   # demean to match real increments
        return compute_moments(dx)

    def _moments_at(
        self,
        free_vals: np.ndarray,
        free_names: list[str],
        sv_base: float,
        sJ_base: float,
        kappa: float,
        theta: float,
        lambda_: float,
        rho: float,
        n_sim: int,
        randoms: dict,
    ) -> np.ndarray:
        sv = abs(float(free_vals[free_names.index("sigma_v")])) \
            if "sigma_v" in free_names else sv_base
        sJ = abs(float(free_vals[free_names.index("sigma_J")])) \
            if "sigma_J" in free_names else sJ_base
        kap = abs(float(free_vals[free_names.index("kappa")])) \
            if "kappa" in free_names else kappa
        return self._simulate_moments(sv, sJ, kap, theta, lambda_, rho,
                                      n_sim, randoms)

    # ------------------------------------------------------------------
    # Weighting matrix and random draws
    # ------------------------------------------------------------------

    def _bootstrap_weights(self, dx: np.ndarray) -> np.ndarray:
        """Diagonal W = diag(1/bootstrap-variance of each moment)."""
        rng = np.random.default_rng(self.rng_seed)
        n = len(dx)
        draws = np.zeros((self.n_bootstrap, N_MOMENTS))
        for b in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            dx_b = dx[idx] - dx[idx].mean()
            draws[b] = compute_moments(dx_b)
        variances = np.nanvar(draws, axis=0, ddof=1)
        variances = np.where(variances > 1e-30, variances, 1.0)
        return np.diag(1.0 / variances)

    def _draw_randoms(self, n_sim: int, rho: float) -> dict:
        """Pre-draw all four CRN arrays; fix seed for reproducibility."""
        rng = np.random.default_rng(self.rng_seed + 999)
        dw_x       = rng.standard_normal(n_sim)
        dw_v_indep = rng.standard_normal(n_sim)
        u_jump     = rng.uniform(size=n_sim)
        z_jump     = rng.standard_normal(n_sim)
        dw_v = rho * dw_x + np.sqrt(max(1.0 - rho**2, 0.0)) * dw_v_indep
        return {
            "dw_x":   dw_x,
            "dw_v":   dw_v,
            "u_jump": u_jump,
            "z_jump": z_jump,
        }

    def _starting_points(
        self, free: tuple[str, ...], sd_real: float
    ) -> list[np.ndarray]:
        guesses = {
            "sigma_v": [0.3, 0.1, 0.6],
            "sigma_J": [2.0 * sd_real, sd_real, 3.0 * sd_real],
            "kappa":   [5.0, 1.0, 10.0],
        }
        return [
            np.array([guesses[name][i] for name in free])
            for i in range(3)
        ]


# ---------------------------------------------------------------------------
# Identification check (§4.3)
# ---------------------------------------------------------------------------

def identification_check(
    result: "SMMResult",
    cache: dict,
    h_frac: float = 0.20,
) -> "pd.DataFrame":
    """§4.3 identification check.

    Perturb σ_v and σ_J each by ±h_frac (default ±20%) and report how much
    each moment changes.  The spec requires that σ_v primarily moves the
    ACF-of-squares moments (vol clustering) and σ_J primarily moves the
    tail (95th pct) and kurtosis.  If both parameters move the same moments,
    the point estimates shouldn't be trusted.

    Returns a DataFrame with columns:
      moment, sigma_v_sensitivity, sigma_J_sensitivity, identified_by
    """
    sv    = result.sigma_v
    sJ    = result.sigma_J
    kappa = result.kappa
    theta = result.theta
    lam   = result.lambda_
    rho   = result.rho
    randoms = cache["randoms"]
    n_sim   = cache["n_sim"]

    # Reuse a BatesSMM instance to get _simulate_moments (avoids re-importing
    # _simulate_path which lives in the same module — the circular guard above
    # is no longer needed, but a method call is cleaner).
    _cal = BatesSMM()

    def _moments_at(sv_: float, sJ_: float) -> np.ndarray:
        return _cal._simulate_moments(sv_, sJ_, kappa, theta, lam, rho, n_sim, randoms)

    m_base = _moments_at(sv, sJ)

    rows = []
    for param_name, sv_p, sJ_p in [
        ("sigma_v", sv * (1 + h_frac), sJ),
        ("sigma_v", sv * (1 - h_frac), sJ),
        ("sigma_J", sv, sJ * (1 + h_frac)),
        ("sigma_J", sv, sJ * (1 - h_frac)),
    ]:
        m_pert = _moments_at(sv_p, sJ_p)
        for j, label in enumerate(MOMENT_LABELS):
            base = m_base[j]
            if abs(base) > 1e-10:
                sens = (m_pert[j] - base) / abs(base)
            else:
                sens = m_pert[j] - base
            rows.append({"param": param_name, "moment": label,
                         "sensitivity": float(sens)})

    df = pd.DataFrame(rows)
    # Average absolute sensitivity per (param, moment)
    agg = (
        df.groupby(["param", "moment"])["sensitivity"]
          .apply(lambda x: float(np.mean(np.abs(x))))
          .reset_index(name="abs_sensitivity")
    )
    wide = agg.pivot(index="moment", columns="param", values="abs_sensitivity")
    wide.columns.name = None
    wide = wide.reindex(MOMENT_LABELS)

    sv_col = "sigma_v" if "sigma_v" in wide.columns else None
    sJ_col = "sigma_J" if "sigma_J" in wide.columns else None

    if sv_col and sJ_col:
        wide["identified_by"] = wide.apply(
            lambda r: "sigma_v" if r[sv_col] > r[sJ_col] else "sigma_J",
            axis=1,
        )
    wide = wide.reset_index()

    print("\n=== §4.3 Identification Check ===")
    print(f"  Perturbation: ±{h_frac*100:.0f}% of optimal parameters")
    print(f"  σ_v = {sv:.4f},  σ_J = {sJ:.4f}")
    print()
    print(wide.to_string(index=False, float_format="{:.4f}".format))

    # Warn if identification is poor
    if sv_col and sJ_col:
        acf_moments = [m for m in MOMENT_LABELS if "ACF" in m]
        tail_moments = [m for m in MOMENT_LABELS if "95th" in m or "Kurt" in m]
        acf_rows  = wide[wide["moment"].isin(acf_moments)]
        tail_rows = wide[wide["moment"].isin(tail_moments)]

        sv_drives_acf  = (acf_rows[sv_col]  > acf_rows[sJ_col]).all()
        sJ_drives_tail = (tail_rows[sJ_col] > tail_rows[sv_col]).all()

        if sv_drives_acf and sJ_drives_tail:
            print("\n  Identification: PASS")
            print("    σ_v drives ACF-of-squares; σ_J drives tail + kurtosis")
        else:
            print("\n  Identification: WARNING")
            if not sv_drives_acf:
                print("    σ_v does NOT dominate ACF moments — vol-of-vol hard to pin down")
            if not sJ_drives_tail:
                print("    σ_J does NOT dominate tail + kurtosis — jump size hard to pin down")
            print("    Point estimates may not be trustworthy individually.")

    return wide


# ---------------------------------------------------------------------------
# J-test (standalone — also used by NestedLadder)
# ---------------------------------------------------------------------------

def _j_test(
    g: np.ndarray,
    W: np.ndarray,
    n_real: int,
    n_sim: int,
    dof: int,
) -> tuple[float, float]:
    """Hansen over-identification J-statistic."""
    if dof <= 0:
        return 0.0, 1.0
    scale = 1.0 / (1.0 + n_real / n_sim)
    j_stat = scale * n_real * float(g @ W @ g)
    j_p    = float(1.0 - stats.chi2.cdf(j_stat, df=dof))
    return j_stat, j_p
