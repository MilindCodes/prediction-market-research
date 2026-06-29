"""
§4.4  Nested model ladder and selection
§4.5  Estimates and inference
§4.6  Which moments do the selecting
§4.7  Validation loop
§4.8  Robustness

The nested structure:
  ConstantVol  ⊂  Heston  ⊂  Bates
  (non-nested sibling: Merton)

Difference-in-J test (SMM analog of LR test)
---------------------------------------------
Because simpler models are nested, we get:
  J_restricted − J_full ~ χ²(Δ n_free)

SAME weighting matrix W must be used for all models.
The W is built once inside BatesSMM.prepare() and shared.

Standard errors (§4.5)
-----------------------
avar(θ̂) = (G'WG)⁻¹  (simplified; uses bootstrap W ≈ efficient matrix)
G = ∂m_sim/∂θ by central finite differences.

Robustness (§4.8)
-----------------
Run the full ladder for each κ ∈ {1, 5, 10} and report stability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src.smm.bates_smm import (
    BatesSMM, SMMResult, _j_test, compute_moments, MOMENT_LABELS, N_MOMENTS
)
from src.smm.stylized_facts import StylizedFacts


# ---------------------------------------------------------------------------
# Difference-in-J test
# ---------------------------------------------------------------------------

@dataclass
class DiffJResult:
    """Difference-in-J test between a restricted and a full model."""
    restricted: str
    full: str
    j_restricted: float
    j_full: float
    diff_j: float
    delta_dof: int
    p_value: float
    reject_at_05: bool

    def __str__(self) -> str:
        return (
            f"  {self.restricted} vs {self.full}: "
            f"ΔJ = {self.diff_j:.3f} (χ²({self.delta_dof}), "
            f"p = {self.p_value:.3f})"
            + ("  *** reject restricted" if self.reject_at_05 else "")
        )


def diff_j_test(restricted: SMMResult, full: SMMResult) -> DiffJResult:
    """Run the difference-in-J test.

    Both results MUST have been estimated with the same W matrix.
    delta_dof = n_free_full − n_free_restricted.
    """
    delta_dof = full.n_free - restricted.n_free
    if delta_dof <= 0:
        raise ValueError(
            f"Full model must have more free params than restricted: "
            f"{full.model}({full.n_free}) vs {restricted.model}({restricted.n_free})"
        )
    diff = restricted.j_stat - full.j_stat
    p = float(1.0 - stats.chi2.cdf(max(diff, 0.0), df=delta_dof))
    return DiffJResult(
        restricted=restricted.model,
        full=full.model,
        j_restricted=restricted.j_stat,
        j_full=full.j_stat,
        diff_j=diff,
        delta_dof=delta_dof,
        p_value=p,
        reject_at_05=p < 0.05,
    )


# ---------------------------------------------------------------------------
# Ladder runner
# ---------------------------------------------------------------------------

@dataclass
class LadderResult:
    kappa: float
    constant_vol: SMMResult
    heston: SMMResult
    bates: SMMResult
    merton: SMMResult
    test_sv: DiffJResult    # ConstantVol vs Heston — need for stochastic vol?
    test_jumps: DiffJResult # Heston vs Bates      — need for jumps?
    moment_table: pd.DataFrame


class NestedLadder:
    """Fit all models in the nested ladder and run selection tests.

    Parameters
    ----------
    calibrator : BatesSMM
        Pre-configured calibrator (controls n_sim, bootstrap, restarts).
    kappa_grid : list[float]
        Mean-reversion speeds to iterate over (§4.8 robustness).
    rho : float
        Correlation parameter (fixed at 0 per baseline spec).
    figures_dir : Path | None
        Where to save output figures.
    """

    def __init__(
        self,
        calibrator: BatesSMM | None = None,
        kappa_grid: list[float] | None = None,
        rho: float = 0.0,
        figures_dir: Path | None = None,
    ):
        self.calibrator = calibrator or BatesSMM()
        self.kappa_grid = kappa_grid or [1.0, 5.0, 10.0]
        self.rho = rho
        self.figures_dir = (
            figures_dir or (config.DATA_DIR / "processed" / "figures")
        )
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        panel: pd.DataFrame,
        verbose: bool = True,
    ) -> list[LadderResult]:
        """Run the full nested ladder for all κ values.

        Parameters
        ----------
        panel : pd.DataFrame
            SMM panel (contract_id, t, p, X, delta_X).
        verbose : bool

        Returns
        -------
        list[LadderResult]  — one per κ value.
        """
        # Pre-compute moments, W, randoms (shared across κ values).
        # W is fixed — this is required for diff-in-J validity.
        cache = self.calibrator.prepare(panel)

        if verbose:
            print(f"\n=== §4.4 Nested Ladder ===")
            print(f"  n_real={cache['n_real']}, n_sim={cache['n_sim']}, "
                  f"θ={cache['theta']:.4f}, λ={cache['lambda_']:.4f}")

        results: list[LadderResult] = []
        for kappa in self.kappa_grid:
            if verbose:
                print(f"\n--- κ = {kappa} ---")
            lr = self._fit_one_kappa(cache, kappa, verbose=verbose)
            results.append(lr)

        self._print_robustness_table(results)
        self._save_results_csv(results)
        return results

    def run_fomc_cpi_split(
        self,
        panel: pd.DataFrame,
        kappa: float = 5.0,
        verbose: bool = True,
    ) -> dict[str, LadderResult]:
        """§4.8 robustness: run ladder separately for FOMC and CPI contracts."""
        out: dict[str, LadderResult] = {}
        for series, prefix in [("FOMC", "KXFED"), ("CPI", "KXCPI")]:
            sub = panel[panel["contract_id"].str.startswith(prefix)]
            if len(sub) < 50:
                print(f"  Skipping {series}: only {len(sub)} rows")
                continue
            if verbose:
                print(f"\n--- {series} sub-panel ({len(sub)} rows) ---")
            cache = self.calibrator.prepare(sub)
            lr = self._fit_one_kappa(cache, kappa, verbose=verbose)
            out[series] = lr
        return out

    def validate_loop(
        self,
        panel: pd.DataFrame,
        selected_result: SMMResult,
        verbose: bool = True,
    ) -> None:
        """§4.7: Simulate the selected model and overlay stylised facts.

        Reproduce the §4.2 aggregational-Gaussianity curve and boundary-
        scaling slope on simulated data and overlay with the empirical ones.
        """
        if verbose:
            print(f"\n=== §4.7 Validation Loop ({selected_result.model}) ===")

        cache = self.calibrator.prepare(panel)
        randoms = cache["randoms"]

        n_sim = cache["n_sim"]
        dx_sim = self._draw_sim_increments(selected_result, randoms, n_sim, cache)

        # Build a synthetic panel for StylizedFacts
        syn_panel = self._sim_to_panel(dx_sim, n_contracts=50)
        sf = StylizedFacts(figures_dir=self.figures_dir)
        res_sim = sf.run(syn_panel, tag=f"sim_{selected_result.model}")

        # Overlay: empirical vs simulated kurtosis-by-k
        self._plot_validation_overlay(panel, res_sim, selected_result)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fit_one_kappa(
        self, cache: dict, kappa: float, verbose: bool
    ) -> LadderResult:
        rho = self.rho
        cal = self.calibrator

        cv  = cal.fit_constant_vol(cache, kappa=kappa, verbose=verbose)
        hes = cal.fit_heston(cache, kappa=kappa, rho=rho, verbose=verbose)
        bat = cal.fit_bates(cache, kappa=kappa, rho=rho, verbose=verbose)
        mer = cal.fit_merton(cache, kappa=kappa, rho=rho, verbose=verbose)

        # Standard errors for free-parameter models
        se_sv_h, _      = cal.standard_errors(hes, cache)
        se_sv_b, se_sJ_b= cal.standard_errors(bat, cache)
        _,  se_sJ_m     = cal.standard_errors(mer, cache)
        hes.se_sigma_v  = se_sv_h
        bat.se_sigma_v  = se_sv_b
        bat.se_sigma_J  = se_sJ_b
        mer.se_sigma_J  = se_sJ_m

        test_sv    = diff_j_test(cv, hes)   # need stochastic vol?
        test_jumps = diff_j_test(hes, bat)  # need jumps?

        if verbose:
            print(f"\n  Difference-in-J tests (κ={kappa}):")
            print(test_sv)
            print(test_jumps)

        mtable = self._moment_table(cv, hes, bat, mer)
        if verbose:
            print("\n  Moment table:")
            print(mtable.to_string(index=False, float_format="{:.4f}".format))

        return LadderResult(
            kappa=kappa,
            constant_vol=cv, heston=hes, bates=bat, merton=mer,
            test_sv=test_sv, test_jumps=test_jumps,
            moment_table=mtable,
        )

    @staticmethod
    def _moment_table(*results: SMMResult) -> pd.DataFrame:
        rows = []
        for r in results:
            for j, label in enumerate(MOMENT_LABELS):
                rows.append({
                    "Model": r.model,
                    "Moment": label,
                    "Target": r.moments_real[j],
                    "Achieved": r.moments_sim[j],
                    "Diff": r.moments_sim[j] - r.moments_real[j],
                })
        return pd.DataFrame(rows)

    def _draw_sim_increments(
        self,
        result: SMMResult,
        randoms: dict,
        n_sim: int,
        cache: dict,
    ) -> np.ndarray:
        from src.smm.bates_smm import _simulate_path
        dx = _simulate_path(
            sigma_v=result.sigma_v,
            sigma_J=result.sigma_J,
            randoms=randoms,
            kappa=result.kappa,
            theta=result.theta,
            lambda_=result.lambda_,
            rho=result.rho,
            n_sim=n_sim,
        )
        return dx - dx.mean()

    @staticmethod
    def _sim_to_panel(dx_sim: np.ndarray, n_contracts: int = 50) -> pd.DataFrame:
        """Wrap a long simulated path into a fake multi-contract panel."""
        n = len(dx_sim) // n_contracts
        rows: list[pd.DataFrame] = []
        for i in range(n_contracts):
            dx = dx_sim[i * n: (i + 1) * n]
            X = np.cumsum(np.concatenate([[0.0], dx]))[:-1]
            p = np.clip(1.0 / (1.0 + np.exp(-X)), 0.02, 0.98)
            rows.append(pd.DataFrame({
                "contract_id": f"SIM_{i:04d}",
                "t": np.arange(len(dx)),
                "p": p,
                "X": X,
                "delta_X": dx,
            }))
        return pd.concat(rows, ignore_index=True)

    def _plot_validation_overlay(
        self,
        panel: pd.DataFrame,
        sim_facts: dict,
        result: SMMResult,
    ) -> None:
        """Overlay empirical and simulated aggregational-Gaussianity curves."""
        from src.smm.stylized_facts import StylizedFacts, _decompose_panel

        inc_real = _decompose_panel(panel)

        ks = [1, 2, 4, 8, 16, 32, 64]

        def _kurtosis_by_k(inc: dict) -> dict:
            from scipy import stats as _stats
            out: dict[int, float] = {}
            for k in ks:
                pooled = []
                for dx in inc.values():
                    nb = len(dx) // k
                    if nb < 2:
                        continue
                    blocks = dx[:nb * k].reshape(nb, k).sum(axis=1)
                    sd = blocks.std(ddof=1)
                    if sd > 0:
                        pooled.append((blocks - blocks.mean()) / sd)
                if pooled:
                    flat = np.concatenate(pooled)
                    out[k] = float(_stats.kurtosis(flat, fisher=True))
            return out

        kag_real = _kurtosis_by_k(inc_real)
        kag_sim  = sim_facts["agg_gaussianity"]["kurtosis_by_k"]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # Aggregational Gaussianity overlay
        ax = axes[0]
        ks_r = sorted(kag_real.keys())
        ks_s = sorted(kag_sim.keys())
        ax.plot(ks_r, [kag_real[k] for k in ks_r], "o-b", label="Empirical")
        ax.plot(ks_s, [kag_sim[k] for k in ks_s], "s--r", label=f"Simulated ({result.model})")
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_title("§4.7 Aggregational Gaussianity")
        ax.set_xlabel("k")
        ax.set_ylabel("Excess kurtosis")
        ax.legend()

        # Boundary scaling: empirical slope
        from src.smm.stylized_facts import _predetermined_p
        p_pre = _predetermined_p(panel)
        bs_real = sim_facts["boundary_scaling"]
        ax2 = axes[1]
        ax2.bar(
            ["Empirical slope", f"{result.model} slope"],
            [0.0, bs_real["slope"]],
            color=["steelblue", "salmon"],
        )
        ax2.set_title("§4.7 Boundary scaling slope (simulated)")
        ax2.set_ylabel("OLS slope on p(1-p)")

        plt.tight_layout()
        out = self.figures_dir / f"validation_{result.model}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Validation figure: {out}")

    def _print_robustness_table(self, results: list[LadderResult]) -> None:
        print("\n=== §4.8 Robustness across κ grid ===")
        header = f"{'κ':>6} {'σ_v(H)':>10} {'σ_v(B)':>10} {'σ_J(B)':>10}"
        header += f" {'J(CV)':>8} {'J(H)':>8} {'J(B)':>8}"
        header += f" {'ΔJ(SV)':>8} {'p':>6} {'ΔJ(J)':>8} {'p':>6}"
        print(header)
        for lr in results:
            row = (
                f"  {lr.kappa:>4.0f}"
                f"  {lr.heston.sigma_v:>10.4f}"
                f"  {lr.bates.sigma_v:>10.4f}"
                f"  {lr.bates.sigma_J:>10.4f}"
                f"  {lr.constant_vol.j_stat:>8.3f}"
                f"  {lr.heston.j_stat:>8.3f}"
                f"  {lr.bates.j_stat:>8.3f}"
                f"  {lr.test_sv.diff_j:>8.3f}"
                f"  {lr.test_sv.p_value:>6.3f}"
                f"  {lr.test_jumps.diff_j:>8.3f}"
                f"  {lr.test_jumps.p_value:>6.3f}"
            )
            print(row)

    def _save_results_csv(self, results: list[LadderResult]) -> None:
        rows = []
        for lr in results:
            for res in [lr.constant_vol, lr.heston, lr.bates, lr.merton]:
                rows.append({
                    "kappa": lr.kappa,
                    "model": res.model,
                    "sigma_v": res.sigma_v,
                    "se_sigma_v": res.se_sigma_v,
                    "sigma_J": res.sigma_J,
                    "se_sigma_J": res.se_sigma_J,
                    "theta": res.theta,
                    "lambda": res.lambda_,
                    "j_stat": res.j_stat,
                    "j_dof": res.j_dof,
                    "j_pvalue": res.j_pvalue,
                    "n_real": res.n_real,
                    "n_sim": res.n_sim,
                    "objective": res.objective_value,
                })
        df = pd.DataFrame(rows)
        out = config.DATA_DIR / "processed" / "smm_ladder_results.csv"
        df.to_csv(out, index=False, float_format="%.6f")
        print(f"\nLadder results saved: {out}")
