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

Each model is warm-started at the optimum of the model it nests, so
J_restricted ≥ J_full holds by construction; diff_j_test raises if it
doesn't (that is a fitting bug, not a result).

Main selection run (§4.4)
-------------------------
κ is FREE and shared across the ladder (it enters only where σ_v > 0, so
it is estimated by Heston/Bates and inert in ConstantVol/Merton).  Pinning
κ to a grid value pushed different models to differently-misspecified
spots and produced impossible orderings (Heston worse than ConstantVol).

Standard errors (§4.5)
-----------------------
avar(θ̂) = (G'WG)⁻¹  (simplified; uses bootstrap W ≈ efficient matrix)
G = ∂m_sim/∂θ by central finite differences.

Robustness (§4.8)
-----------------
Re-run the full ladder with κ fixed at each of {1, 5, 10} and report
stability of the selection verdicts.
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
from src.figstyle import apply_publication_style, save_figure
from src.smm.stylized_facts import StylizedFacts, _decompose_panel


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
    # The full model nests the restricted one and is warm-started at its
    # optimum, so J_restricted ≥ J_full must hold (up to float dust).  If it
    # doesn't, the fit is broken — refuse to print a p-value we'd retract.
    tol = 1e-6 * max(1.0, abs(restricted.j_stat))
    if diff < -tol:
        raise AssertionError(
            f"J_restricted < J_full: {restricted.model} J={restricted.j_stat:.6f} "
            f"vs {full.model} J={full.j_stat:.6f} (ΔJ={diff:.3e}). "
            f"The {full.model} optimizer failed to reach the {restricted.model} "
            f"optimum it nests — fitting bug, not a result. Check warm starts "
            f"and the shared W."
        )
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
    kappa: float            # grid value (fixed-κ runs) or κ init (free-κ run)
    constant_vol: SMMResult
    heston: SMMResult
    bates: SMMResult
    merton: SMMResult
    test_sv: DiffJResult    # ConstantVol vs Heston — need for stochastic vol?
    test_jumps: DiffJResult # Heston vs Bates      — need for jumps?
    moment_table: pd.DataFrame
    free_kappa: bool = False


class NestedLadder:
    """Fit all models in the nested ladder and run selection tests.

    Parameters
    ----------
    calibrator : BatesSMM
        Pre-configured calibrator (controls n_sim, bootstrap, restarts).
    kappa_grid : list[float]
        Mean-reversion speeds for the §4.8 fixed-κ robustness grid.
    kappa_init : float
        Optimizer starting value for κ in the free-κ selection run.
    rho : float
        Correlation parameter (fixed at 0 per baseline spec).
    figures_dir : Path | None
        Where to save output figures.
    """

    def __init__(
        self,
        calibrator: BatesSMM | None = None,
        kappa_grid: list[float] | None = None,
        kappa_init: float = 5.0,
        rho: float = 0.0,
        figures_dir: Path | None = None,
    ):
        self.calibrator = calibrator or BatesSMM()
        self.kappa_grid = kappa_grid or [1.0, 5.0, 10.0]
        self.kappa_init = kappa_init
        self.rho = rho
        self.figures_dir = (
            figures_dir or (config.DATA_DIR / "processed" / "figures")
        )
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def run_selection(
        self,
        panel: pd.DataFrame,
        verbose: bool = True,
        save_as: "str | None" = "smm_ladder_results_main.csv",
        moment_table_as: "str | None" = "smm_moment_table_main.csv",
    ) -> LadderResult:
        """§4.4 MAIN selection run — κ free and shared across the ladder.

        This is the fit the selection tests are read from.  The fixed-κ
        grid lives in run() and is §4.8 robustness only.

        save_as / moment_table_as may be None so that callers running the
        ladder repeatedly (e.g. the §4.8 sampling sweep) do not overwrite
        the headline §4.4 artifacts.
        """
        cache = self.calibrator.prepare(panel)

        if verbose:
            print(f"\n=== §4.4 Nested Ladder — selection run (κ free) ===")
            print(f"  n_real={cache['n_real']}, n_sim={cache['n_sim']}, "
                  f"θ={cache['theta']:.4f}, λ={cache['lambda_']:.4f}")

        lr = self._fit_ladder(
            cache, kappa=self.kappa_init, free_kappa=True, verbose=verbose
        )
        self._print_selection_summary(lr)
        if save_as:
            self._save_results_csv([lr], filename=save_as)
        if moment_table_as:
            self._save_moment_table(lr, filename=moment_table_as)
        return lr

    def run(
        self,
        panel: pd.DataFrame,
        verbose: bool = True,
    ) -> list[LadderResult]:
        """§4.8 robustness: re-run the ladder with κ fixed at each grid value.

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
            print(f"\n=== §4.8 Nested Ladder — fixed-κ robustness grid ===")
            print(f"  n_real={cache['n_real']}, n_sim={cache['n_sim']}, "
                  f"θ={cache['theta']:.4f}, λ={cache['lambda_']:.4f}")

        results: list[LadderResult] = []
        for kappa in self.kappa_grid:
            if verbose:
                print(f"\n--- κ = {kappa} ---")
            lr = self._fit_ladder(cache, kappa, free_kappa=False,
                                  verbose=verbose)
            results.append(lr)

        self._print_robustness_table(results)
        self._save_results_csv(results)
        return results

    def run_family_split(
        self,
        panel: pd.DataFrame,
        verbose: bool = True,
    ) -> dict[str, LadderResult]:
        """§4.8 robustness: run the (κ-free) ladder separately per contract
        family.

        Kalshi tickers split by prefix (KXFED / KXCPI).  Polymarket
        condition IDs (0x…) are classified via the identity mapping in
        data/exports/polymarket_contract_identities.csv.  Election markets
        were cut from the corpus (July 2026); with a single remaining
        family this reduces to the main run and is skipped upstream.
        """
        out: dict[str, LadderResult] = {}
        ids = panel["contract_id"].astype(str)

        if ids.str.startswith("0x").all():
            id_path = Path("data/exports/polymarket_contract_identities.csv")
            groups = pd.read_csv(id_path).set_index("conditionId")["group"]
            present = sorted(ids.map(groups).dropna().unique())
            splits = [
                (g.capitalize(), ids.map(groups).eq(g)) for g in present
            ]
        else:
            splits = [
                ("FOMC", ids.str.startswith("KXFED")),
                ("CPI", ids.str.startswith("KXCPI")),
            ]

        for series, mask in splits:
            sub = panel[mask]
            if len(sub) < 50:
                print(f"  Skipping {series}: only {len(sub)} rows")
                continue
            if verbose:
                print(f"\n--- {series} sub-panel ({len(sub)} rows) ---")
            cache = self.calibrator.prepare(sub)
            lr = self._fit_ladder(cache, kappa=self.kappa_init,
                                  free_kappa=True, verbose=verbose)
            out[series] = lr

        rows = []
        for series, lr in out.items():
            for res in [lr.constant_vol, lr.heston, lr.bates, lr.merton]:
                rows.append({
                    "family": series, "kappa": res.kappa, "model": res.model,
                    "sigma_v": res.sigma_v, "sigma_J": res.sigma_J,
                    "j_stat": res.j_stat, "j_dof": res.j_dof,
                    "j_pvalue": res.j_pvalue, "n_real": res.n_real,
                })
        df = pd.DataFrame(rows)
        out_path = config.DATA_DIR / "processed" / "smm_family_split.csv"
        df.to_csv(out_path, index=False, float_format="%.6f")
        print(f"\nFamily split results saved: {out_path}")
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

    @staticmethod
    def _project(model: str, free_kappa: bool,
                 sv: float, sJ: float, kap: float) -> np.ndarray:
        """Project a candidate point onto one model's free-parameter space.

        Ordering matches each fit's `free` tuple:
          Heston ("sigma_v"[, "kappa"]) — drops σ_J
          Merton ("sigma_J",)           — drops σ_v and κ (both inert)
          Bates  ("sigma_v", "sigma_J"[, "kappa"])
        """
        if model == "Heston":
            return np.array([sv] + ([kap] if free_kappa else []))
        if model == "Merton":
            return np.array([sJ])
        if model == "Bates":
            return np.array([sv, sJ] + ([kap] if free_kappa else []))
        raise ValueError(model)

    def _fit_ladder(
        self,
        cache: dict,
        kappa: float,
        free_kappa: bool,
        verbose: bool,
        max_passes: int = 3,
    ) -> LadderResult:
        rho = self.rho
        cal = self.calibrator

        cv = cal.fit_constant_vol(cache, kappa=kappa, rho=rho,
                                  verbose=verbose)

        # Warm-start each model at the optimum of the model it nests, so the
        # richer model cannot end at a worse objective — worst case it stays
        # at the warm start.  Vectors are ordered to match each fit's `free`
        # tuple.
        hes_warm = np.array([0.0, kappa]) if free_kappa else np.array([0.0])
        hes = cal.fit_heston(cache, kappa=kappa, rho=rho,
                             free_kappa=free_kappa, warm_starts=[hes_warm],
                             verbose=verbose)

        mer = cal.fit_merton(cache, kappa=kappa, rho=rho,
                             warm_starts=[np.array([0.0])], verbose=verbose)

        # Bates nests both Heston (σ_J=0) and Merton (σ_v=0) — start at both.
        # κ is inert at σ_v=0, so Heston's κ̂ serves for the Merton start too.
        kap_b = hes.kappa if free_kappa else kappa
        bat_warms = [
            np.array([hes.sigma_v, 0.0] + ([kap_b] if free_kappa else [])),
            np.array([0.0, mer.sigma_J] + ([kap_b] if free_kappa else [])),
        ]
        bat = cal.fit_bates(cache, kappa=kap_b, rho=rho,
                            free_kappa=free_kappa, warm_starts=bat_warms,
                            verbose=verbose)

        # ------------------------------------------------------------------
        # Cross-warm-start polish.
        #
        # Warm-starting only downward (rich model at the restricted model's
        # optimum) guarantees J_restricted >= J_full, but it does NOT
        # guarantee that the RESTRICTED model reached its own optimum.  That
        # gap is not cosmetic: if Bates lands on a point with sigma_J = 0
        # that Heston's own search missed, the difference-in-J test credits
        # the whole improvement to the jump parameter when the jump
        # parameter is doing nothing.  Observed in practice on the
        # revision-time panel — Heston obj 117.12 against a Heston-FEASIBLE
        # Bates point at 73.16, reported as "jumps selected" with
        # sigma_J = 0.
        #
        # So: re-fit every model from the projection of every other model's
        # optimum onto its own parameter space, and repeat until nothing
        # improves.  Each re-fit can only lower the objective (its previous
        # optimum is always among the starts), so this terminates and never
        # degrades a fit.
        # ------------------------------------------------------------------
        for _ in range(max_passes - 1):
            improved = False

            for name in ("Heston", "Merton", "Bates"):
                current = {"Heston": hes, "Merton": mer, "Bates": bat}[name]
                # Recomputed per model, not per pass: Bates is polished last
                # and must see the improvement Heston just made in this same
                # pass.  Otherwise a pass can end with the restricted model
                # strictly better than the model that nests it, which
                # diff_j_test (correctly) refuses to interpret.
                candidates = [(r.sigma_v, r.sigma_J, r.kappa)
                              for r in (hes, mer, bat)]
                fk = free_kappa and name != "Merton"
                warms = [
                    self._project(name, fk, current.sigma_v,
                                  current.sigma_J, current.kappa)
                ] + [
                    self._project(name, fk, sv, sJ, k)
                    for (sv, sJ, k) in candidates
                ]
                kap_use = current.kappa if fk else kappa
                fit_fn = {"Heston": cal.fit_heston,
                          "Merton": cal.fit_merton,
                          "Bates":  cal.fit_bates}[name]
                kwargs = dict(kappa=kap_use, rho=rho,
                              warm_starts=warms, verbose=False)
                if name != "Merton":
                    kwargs["free_kappa"] = fk
                cand = fit_fn(cache, **kwargs)

                tol = 1e-9 * max(1.0, abs(current.objective_value))
                if cand.objective_value < current.objective_value - tol:
                    if verbose:
                        print(f"  [polish] {name}: obj "
                              f"{current.objective_value:.6f} → "
                              f"{cand.objective_value:.6f}  "
                              f"(σ_v={cand.sigma_v:.4f}, σ_J={cand.sigma_J:.4f}"
                              + (f", κ={cand.kappa:.4f})" if fk else ")"))
                    improved = True
                    if name == "Heston":
                        hes = cand
                    elif name == "Merton":
                        mer = cand
                    else:
                        bat = cand

            if not improved:
                break

        # Standard errors for free-parameter models
        se_sv_h, _, se_k_h       = cal.standard_errors(hes, cache)
        se_sv_b, se_sJ_b, se_k_b = cal.standard_errors(bat, cache)
        _, se_sJ_m, _            = cal.standard_errors(mer, cache)
        hes.se_sigma_v  = se_sv_h
        hes.se_kappa    = se_k_h
        bat.se_sigma_v  = se_sv_b
        bat.se_sigma_J  = se_sJ_b
        bat.se_kappa    = se_k_b
        mer.se_sigma_J  = se_sJ_m

        test_sv    = diff_j_test(cv, hes)   # need stochastic vol?
        test_jumps = diff_j_test(hes, bat)  # need jumps?

        if verbose:
            tag = "κ free" if free_kappa else f"κ={kappa}"
            print(f"\n  Difference-in-J tests ({tag}):")
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
            free_kappa=free_kappa,
        )

    @staticmethod
    def _print_selection_summary(lr: LadderResult) -> None:
        print("\n=== §4.4 Selection run summary (κ free) ===")
        print(f"  {'Model':<12} {'σ_v':>9} {'σ_J':>9} {'κ̂':>9} "
              f"{'J':>12} {'dof':>4} {'p':>7}")
        for res in [lr.constant_vol, lr.heston, lr.merton, lr.bates]:
            kap = f"{res.kappa:>9.4f}" if res.kappa_free else f"{'—':>9}"
            print(f"  {res.model:<12} {res.sigma_v:>9.4f} {res.sigma_J:>9.4f} "
                  f"{kap} {res.j_stat:>12.3f} {res.j_dof:>4d} "
                  f"{res.j_pvalue:>7.3f}")
        print(lr.test_sv)
        print(lr.test_jumps)

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

        apply_publication_style()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # Aggregational Gaussianity overlay — the informative panel.  The
        # empirical curve decays toward 0 (§4.2, diagnostic 3); a fitted
        # jump component predicts kurtosis that PERSISTS under aggregation.
        # Any visible gap here is evidence about the jump selection, not a
        # cosmetic defect: see §4.6.
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

        # Boundary scaling: empirical vs simulated slope.  The empirical
        # slope has to be COMPUTED from the real panel — it was previously
        # hard-coded to 0.0, which drew a comparison that was not one.
        from src.smm.stylized_facts import StylizedFacts as _SF, _predetermined_p
        p_pre = _predetermined_p(panel)
        bs_real = _SF()._boundary_scaling(inc_real, p_pre)
        bs_sim  = sim_facts["boundary_scaling"]
        ax2 = axes[1]
        ax2.bar(
            ["Empirical", f"Simulated ({result.model})"],
            [bs_real["slope"], bs_sim["slope"]],
            color=["steelblue", "salmon"],
        )
        ax2.axhline(0, color="black", linewidth=0.7)
        ax2.set_title("§4.7 Boundary scaling slope")
        ax2.set_ylabel("OLS slope on p(1−p)")
        for i, v in enumerate([bs_real["slope"], bs_sim["slope"]]):
            ax2.text(i, v, f"{v:.3f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=9)

        # The simulated panel is reconstructed by forward-integrating ΔX and
        # re-clipping, so the boundary and clustering curves are partly
        # circular; the aggregational panel is the one that carries evidence.
        print(f"  Boundary slope — empirical {bs_real['slope']:.4f} "
              f"vs simulated {bs_sim['slope']:.4f}")
        print("  Aggregational kurtosis (k: empirical / simulated):")
        for k in ks_r:
            if k in kag_sim:
                print(f"    k={k:<3d} {kag_real[k]:>8.2f} / {kag_sim[k]:>8.2f}")

        plt.tight_layout()
        out = self.figures_dir / f"validation_{result.model}.png"
        written = save_figure(out)
        print("  Validation figure: " + ", ".join(str(w) for w in written))

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

    @staticmethod
    def _save_moment_table(
        lr: LadderResult,
        filename: str = "smm_moment_table_main.csv",
    ) -> pd.DataFrame:
        """§4.5/§4.6 — export the five real moments beside the five simulated
        moments at each model's optimum.

        This is the exhibit §4.6 argues from: a reader can see directly which
        moments Heston misses and Bates hits, instead of taking it on trust.
        One row per moment; Target plus one column per model, with the signed
        relative miss alongside.
        """
        models = [lr.constant_vol, lr.heston, lr.merton, lr.bates]
        target = models[0].moments_real
        denom = np.maximum(np.abs(target), 1e-10)

        data: dict = {"Moment": list(MOMENT_LABELS), "Target": list(target)}
        for m in models:
            data[m.model] = list(m.moments_sim)
        for m in models:
            data[f"relmiss_{m.model}"] = list((m.moments_sim - target) / denom)

        df = pd.DataFrame(data)
        out = config.DATA_DIR / "processed" / filename
        df.to_csv(out, index=False, float_format="%.6f")
        print(f"Moment table (target vs achieved) saved: {out}")
        return df

    def _save_results_csv(
        self,
        results: list[LadderResult],
        filename: str = "smm_ladder_results.csv",
    ) -> None:
        rows = []
        for lr in results:
            for res in [lr.constant_vol, lr.heston, lr.bates, lr.merton]:
                rows.append({
                    "kappa": res.kappa,
                    "kappa_free": res.kappa_free,
                    "se_kappa": res.se_kappa,
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
        out = config.DATA_DIR / "processed" / filename
        df.to_csv(out, index=False, float_format="%.6f")
        print(f"\nLadder results saved: {out}")


# ---------------------------------------------------------------------------
# §4.6  Which moments do the selecting
# ---------------------------------------------------------------------------

def moment_selection_analysis(
    ladder_result: LadderResult,
    figures_dir: "Path | None" = None,
) -> pd.DataFrame:
    """§4.6 — Explicitly show which moments each model hits and misses.

    The load-bearing claim:
      ACF-of-squares moments (lag 1, lag 2) → what stochastic vol buys
      Tail (95th pct) + kurtosis            → what jumps buy

    If Heston fails on the tail moment and Bates fixes it, that is the
    economic argument for jumps.  This function makes that pattern explicit.

    Returns a wide DataFrame: moments as rows, models as columns.
    Saves a heatmap figure.
    """
    figures_dir = figures_dir or (config.DATA_DIR / "processed" / "figures")
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    models = [
        ladder_result.constant_vol,
        ladder_result.heston,
        ladder_result.bates,
        ladder_result.merton,
    ]
    target = models[0].moments_real

    data: dict[str, list] = {"Moment": MOMENT_LABELS, "Target": list(target)}
    for m in models:
        data[m.model] = list(m.moments_sim)

    df = pd.DataFrame(data)

    # Compute relative miss: (achieved - target) / |target|
    for m in models:
        col = f"miss_{m.model}"
        df[col] = (df[m.model] - df["Target"]) / df["Target"].abs().clip(lower=1e-10)

    print("\n=== §4.6 Which moments do the selecting ===")
    print("\nTarget vs achieved (absolute values):")
    print(df[["Moment", "Target"] + [m.model for m in models]].to_string(
        index=False, float_format="{:.4f}".format
    ))

    # Narrative
    print("\nMoment-level diagnosis:")
    acf_moments  = ["ACF(ΔX², lag=1)", "ACF(ΔX², lag=2)"]
    tail_moments = ["Kurt(ΔX)", "95th pct(ΔX)"]

    for row in df.itertuples():
        moment = row.Moment
        tgt    = row.Target
        hes    = getattr(row, "Heston", np.nan)
        bat    = getattr(row, "Bates",  np.nan)

        hes_miss = abs(hes - tgt) / max(abs(tgt), 1e-10)
        bat_miss = abs(bat - tgt) / max(abs(tgt), 1e-10)

        tag = ""
        if moment in acf_moments:
            tag = "← SV (Heston) should fix this"
        elif moment in tail_moments:
            tag = "← Jumps (Bates) should fix this"

        status = ""
        if hes_miss > 0.20 and bat_miss < hes_miss * 0.5:
            status = f"Heston fails ({hes_miss:.0%} off), Bates fixes ({bat_miss:.0%} off) ✓"
        elif hes_miss < 0.10:
            status = f"Heston already fits ({hes_miss:.0%} off)"
        else:
            status = f"Heston {hes_miss:.0%} off, Bates {bat_miss:.0%} off"

        print(f"  {moment:<28s}  {status}  {tag}")

    # Heatmap of relative misses.
    #
    # Colour encodes the ABSOLUTE miss, not the signed one.  With a signed
    # scale on RdYlGn_r a perfect fit (0.00) renders yellow while a total
    # miss of -1.00 renders dark green — i.e. the worst cell in the figure
    # looked like the best one, inverting the caption.  Magnitude drives
    # the colour; the signed value stays in the annotation so the direction
    # of each miss is still readable.
    try:
        import matplotlib.pyplot as plt

        miss_cols = [f"miss_{m.model}" for m in models]
        miss_data = df[miss_cols].values
        model_labels = [m.model for m in models]

        apply_publication_style()
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(np.abs(miss_data.T), aspect="auto",
                       cmap=plt.cm.RdYlGn_r, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(N_MOMENTS))
        ax.set_xticklabels(MOMENT_LABELS, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(model_labels)))
        ax.set_yticklabels(model_labels)
        plt.colorbar(im, ax=ax,
                     label="|relative miss|  (0 = on target, 1 = 100% off)")
        ax.set_title("§4.6 Moment fit by model  "
                     "(green = on target, red = misses; cells show signed miss)")

        for i in range(N_MOMENTS):
            for j, m in enumerate(models):
                val = miss_data[i, j]
                ax.text(i, j, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(val) > 0.6 else "black")

        plt.tight_layout()
        out = Path(figures_dir) / "moment_selection.png"
        written = save_figure(out)
        print("\n  Figure: " + ", ".join(str(w) for w in written))
    except Exception as e:
        print(f"  (Figure skipped: {e})")

    # The figure and the table must come from the same fit — export the
    # numbers behind the heatmap so §4.6 can cite them directly.
    out_csv = config.DATA_DIR / "processed" / "smm_moment_selection.csv"
    df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"  Table:  {out_csv}")

    return df


# ---------------------------------------------------------------------------
# §4.8  Robustness: truncation sensitivity and frequency sensitivity
# ---------------------------------------------------------------------------

def truncation_sensitivity(
    panel_base: "pd.DataFrame",
    calibrator: "BatesSMM",
    kappa: float = 5.0,
    clip_pairs: "list[tuple[float, float]] | None" = None,
    figures_dir: "Path | None" = None,
) -> pd.DataFrame:
    """§4.8 — Re-run Bates SMM at alternative clip bounds.

    Baseline is [0.02, 0.98].  The spec specifically requests a [0.01, 0.99]
    sensitivity to check whether jump evidence is a truncation artifact.

    Each clip pair gets a panel REBUILT FROM RAW, not re-derived from
    panel_base.  panel_base's rows are already first differences with the
    leading observation of each contract dropped, so re-differencing it lost
    one more increment per contract (4141 → 4102) and made the [0.02, 0.98]
    row a different sample from the §4.4 headline that carries the same
    label.  Rebuilding keeps every increment and keeps the baseline row
    comparable.  Each clip caches to its own parquet, so the §4.4 panel is
    never overwritten.

    The fit also goes through the same NestedLadder.run_selection path as
    §4.4 (warm starts + cross-model polish).  Calling fit_bates directly
    used grid starts only and settled in a worse corner of the same basin,
    which was the second reason the two tables disagreed.

    Returns a summary DataFrame comparing σ_v, σ_J, J, p across clip pairs.
    """
    import src.smm.panel as _panel_mod
    from src.smm.panel import CLIP_LO as _DEFAULT_LO, CLIP_HI as _DEFAULT_HI
    from src.smm.panel import SMMPanelBuilder as _Builder

    clip_pairs = clip_pairs or [
        (_DEFAULT_LO, _DEFAULT_HI),  # baseline [0.02, 0.98]
        (0.01, 0.99),                 # wider — spec's referee sensitivity
    ]
    figures_dir = figures_dir or (config.DATA_DIR / "processed" / "figures")

    tickers = list(pd.unique(panel_base["contract_id"]))
    saved_lo, saved_hi = _panel_mod.CLIP_LO, _panel_mod.CLIP_HI
    processed = config.DATA_DIR / "processed"

    rows = []
    try:
        for clip_lo, clip_hi in clip_pairs:
            label = f"[{clip_lo},{clip_hi}]"
            print(f"\n--- Truncation sensitivity {label} ---")

            # Rebuild from raw at this clip.  A dedicated cache path per clip
            # keeps the §4.4 baseline panel (smm_panel.parquet) untouched.
            _panel_mod.CLIP_LO, _panel_mod.CLIP_HI = clip_lo, clip_hi
            out_panel = processed / f"smm_panel_clip_{clip_lo}_{clip_hi}.parquet"

            class _ClipBuilder(_Builder):
                def panel_path(self):
                    return out_panel

            panel = _ClipBuilder(client=None).build(tickers=tickers, force=True)

            # Same ladder path as §4.4 so the baseline row is comparable.
            ladder = NestedLadder(calibrator=calibrator, kappa_init=kappa)
            res = ladder.run_selection(panel, verbose=False,
                                       save_as=None, moment_table_as=None).bates
            print(f"    σ_v={res.sigma_v:.4f}  σ_J={res.sigma_J:.4f}  "
                  f"κ̂={res.kappa:.6f}  J={res.j_stat:.3f}  n={res.n_real}")

            rows.append({
                "clip": label, "clip_lo": clip_lo, "clip_hi": clip_hi,
                "sigma_v": res.sigma_v, "sigma_J": res.sigma_J,
                "kappa": res.kappa,
                "theta": res.theta, "lambda": res.lambda_,
                "j_stat": res.j_stat, "j_pvalue": res.j_pvalue,
                "n_real": res.n_real,
            })
    finally:
        _panel_mod.CLIP_LO, _panel_mod.CLIP_HI = saved_lo, saved_hi

    df = pd.DataFrame(rows)
    print("\n=== §4.8 Truncation sensitivity ===")
    print(df.to_string(index=False, float_format="{:.4f}".format))

    out = config.DATA_DIR / "processed" / "smm_truncation_sensitivity.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"Saved: {out}")
    return df


def frequency_sensitivity(
    raw_dir: "Path | None" = None,
    calibrator: "BatesSMM | None" = None,
    kappa: float = 5.0,
    freqs: "list[str] | None" = None,
) -> pd.DataFrame:
    """§4.8 — Re-run panel and Bates SMM at alternative grid frequencies.

    Compares the daily (D) baseline against coarser grids (2D, 4D).
    Coarser grids have fewer forward-filled zero-increments but also
    less data — the sensitivity checks that results are stable.
    (The raw bars are daily, so grids finer than 1D would only add
    forward-filled artifacts.)

    This fits Bates only.  For the full ladder — and hence the selection
    verdicts — use sampling_sensitivity() below.
    """
    from src.smm.panel import SMMPanelBuilder

    raw_dir    = raw_dir or (config.DATA_DIR / "raw" / "polymarket")
    calibrator = calibrator or BatesSMM()
    freqs      = freqs or ["D", "2D", "4D"]

    rows = []
    for freq in freqs:
        print(f"\n--- Frequency sensitivity: {freq} ---")
        # Each (freq, sampling) writes to its own cache path, so this no
        # longer overwrites the baseline daily panel the §4.4 run reads.
        builder = SMMPanelBuilder(freq=freq)
        tickers = builder._load_catalog_tickers()
        if not tickers:
            print("  No tickers — run catalog step first.")
            continue
        panel = builder.build(tickers=tickers, force=True)
        cache = calibrator.prepare(panel)
        res   = calibrator.fit_bates(cache, kappa=kappa, free_kappa=True,
                                     verbose=True)
        rows.append({
            "freq": freq, "n_contracts": panel["contract_id"].nunique(),
            "n_real": res.n_real,
            "sigma_v": res.sigma_v, "sigma_J": res.sigma_J, "kappa": res.kappa,
            "j_stat": res.j_stat, "j_pvalue": res.j_pvalue,
        })

    df = pd.DataFrame(rows)
    print("\n=== §4.8 Frequency sensitivity ===")
    if not df.empty:
        print(df.to_string(index=False, float_format="{:.4f}".format))
        out = config.DATA_DIR / "processed" / "smm_frequency_sensitivity.csv"
        df.to_csv(out, index=False, float_format="%.6f")
        print(f"Saved: {out}")
    return df


# ---------------------------------------------------------------------------
# §4.8  Sampling-scheme sweep: the zero-increment test
# ---------------------------------------------------------------------------

SAMPLING_SPECS: list[tuple[str, str]] = [
    ("D",  "calendar"),   # baseline — the §4.4 panel
    ("2D", "calendar"),
    ("3D", "calendar"),
    ("4D", "calendar"),
    ("D",  "revision"),   # event time — zero-increment mass is 0 by construction
]


def _panel_descriptives(panel: pd.DataFrame) -> dict:
    """Zero mass, tail shape, and aggregational decay for one panel."""
    dx_raw = panel["delta_X"].dropna().values
    inc = _decompose_panel(panel)

    kurt_by_k: dict[int, float] = {}
    for k in (1, 2, 4, 8):
        pooled = []
        for dx in inc.values():
            nb = len(dx) // k
            if nb < 2:
                continue
            blocks = dx[: nb * k].reshape(nb, k).sum(axis=1)
            sd = blocks.std(ddof=1)
            if sd > 0:
                pooled.append((blocks - blocks.mean()) / sd)
        if pooled:
            kurt_by_k[k] = float(stats.kurtosis(np.concatenate(pooled),
                                                fisher=True))

    dt = panel["dt_steps"].dropna() if "dt_steps" in panel.columns else None
    return {
        "n_contracts": int(panel["contract_id"].nunique()),
        "n_increments": int(len(dx_raw)),
        "zero_frac": float(np.mean(dx_raw == 0.0)),
        "kurt_k1": kurt_by_k.get(1, np.nan),
        "kurt_k2": kurt_by_k.get(2, np.nan),
        "kurt_k4": kurt_by_k.get(4, np.nan),
        "kurt_k8": kurt_by_k.get(8, np.nan),
        "mean_dt_steps": float(dt.mean()) if dt is not None and len(dt) else 1.0,
        "p95_dt_steps": float(dt.quantile(0.95)) if dt is not None and len(dt) else 1.0,
    }


def sampling_sensitivity(
    calibrator: "BatesSMM | None" = None,
    kappa_init: float = 5.0,
    specs: "list[tuple[str, str]] | None" = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """§4.8 — Re-run the FULL nested ladder under each sampling scheme.

    Why this run exists
    -------------------
    The §4.4 panel is 36.6% exact zeros: the CLOB prices-history endpoint
    returns one daily bar per contract-day and stale days are forward-filled,
    so a third of the "increments" are not revisions.  A zero-inflated
    increment distribution deflates the variance moment and inflates
    kurtosis, which is exactly the signature the ladder reads as jumps.  So
    the §4.4 jump verdict and the §4.2 diagnostic-3 finding (kurtosis DECAYS
    under aggregation, i.e. no persistent jump component) may be in conflict
    for a purely mechanical reason.

    This is the test that resolves it.  Two ways to shrink the zero mass:

      * coarsen the calendar grid (2D/3D/4D) — keeps Δt constant, but
        drains the sample and only partly removes the zeros;
      * sample in revision (event) time — collapse repeated log-odds so
        every increment is a genuine revision.  Zero mass is 0 by
        construction and most of the sample survives, at the cost of a
        non-constant Δt (quantified by dt_steps).

    True trade-time sampling is not available: the pulled bars carry
    volume ≡ 0 and trade_count ≡ 0, so the archive has no trade clock.
    Revision time is the closest feasible approximation and is labelled as
    such rather than called trade time.

    Reading the output
    ------------------
    J levels are NOT comparable across rows — each panel gets its own
    bootstrap W, θ and λ.  What IS comparable is the selection verdict:
    the difference-in-J p-values for "need stochastic vol?" and "need
    jumps?".  If jumps stop being selected once the zeros are removed, the
    §4.4 finding was an artifact of the forward fill.

    Returns
    -------
    (summary, full) — one row per spec; one row per (spec, model).
    """
    from src.smm.panel import SMMPanelBuilder

    calibrator = calibrator or BatesSMM()
    specs = specs or SAMPLING_SPECS

    summary_rows: list[dict] = []
    full_rows: list[dict] = []

    for freq, sampling in specs:
        label = f"{freq}/{sampling}"
        print(f"\n{'='*66}\n--- Sampling sensitivity: {label} ---\n{'='*66}")

        builder = SMMPanelBuilder(freq=freq, sampling=sampling)
        tickers = builder._load_catalog_tickers()
        if not tickers:
            print("  No tickers — run the catalog step first.")
            continue
        panel = builder.build(tickers=tickers, force=True)

        desc = _panel_descriptives(panel)
        print(f"  {desc['n_contracts']} contracts, {desc['n_increments']} "
              f"increments, zero mass {desc['zero_frac']:.1%}, "
              f"kurt(k=1)={desc['kurt_k1']:.1f} → kurt(k=8)={desc['kurt_k8']:.1f}")

        ladder = NestedLadder(calibrator=calibrator, kappa_init=kappa_init)
        # save_as=None: these are robustness fits and must not overwrite the
        # headline §4.4 artifacts.
        lr = ladder.run_selection(panel, verbose=verbose,
                                  save_as=None, moment_table_as=None)

        row = {"freq": freq, "sampling": sampling, "spec": label}
        row.update(desc)
        row.update({
            "theta":          lr.bates.theta,
            "lambda":         lr.bates.lambda_,
            "sigma_v_heston": lr.heston.sigma_v,
            "sigma_v_bates":  lr.bates.sigma_v,
            "sigma_J_bates":  lr.bates.sigma_J,
            "sigma_J_merton": lr.merton.sigma_J,
            "kappa_bates":    lr.bates.kappa,
            "j_cv":           lr.constant_vol.j_stat,
            "j_heston":       lr.heston.j_stat,
            "j_merton":       lr.merton.j_stat,
            "j_bates":        lr.bates.j_stat,
            "diffj_sv":       lr.test_sv.diff_j,
            "p_sv":           lr.test_sv.p_value,
            "sv_selected":    lr.test_sv.reject_at_05,
            "diffj_jumps":    lr.test_jumps.diff_j,
            "p_jumps":        lr.test_jumps.p_value,
            "jumps_selected": lr.test_jumps.reject_at_05,
        })
        summary_rows.append(row)

        for res in (lr.constant_vol, lr.heston, lr.merton, lr.bates):
            full_rows.append({
                "spec": label, "freq": freq, "sampling": sampling,
                "model": res.model,
                "sigma_v": res.sigma_v, "se_sigma_v": res.se_sigma_v,
                "sigma_J": res.sigma_J, "se_sigma_J": res.se_sigma_J,
                "kappa": res.kappa, "se_kappa": res.se_kappa,
                "theta": res.theta, "lambda": res.lambda_,
                "j_stat": res.j_stat, "j_dof": res.j_dof,
                "j_pvalue": res.j_pvalue,
                "n_real": res.n_real, "n_sim": res.n_sim,
                "objective": res.objective_value,
            })

    summary = pd.DataFrame(summary_rows)
    full    = pd.DataFrame(full_rows)

    if not summary.empty:
        print(f"\n\n{'='*90}")
        print("=== §4.8 Sampling-scheme sweep — full ladder under each scheme ===")
        print(f"{'='*90}")
        print(f"{'spec':<14}{'n_c':>5}{'n_inc':>7}{'zero%':>8}"
              f"{'kurt k1':>9}{'kurt k8':>9}"
              f"{'sig_v(B)':>10}{'sig_J(B)':>10}"
              f"{'p(SV)':>8}{'p(jump)':>9}  verdict")
        for r in summary.itertuples():
            verdict = ("jumps selected" if r.jumps_selected
                       else "JUMPS NOT SELECTED")
            print(f"{r.spec:<14}{r.n_contracts:>5}{r.n_increments:>7}"
                  f"{r.zero_frac*100:>7.1f}%"
                  f"{r.kurt_k1:>9.1f}{r.kurt_k8:>9.1f}"
                  f"{r.sigma_v_bates:>10.4f}{r.sigma_J_bates:>10.4f}"
                  f"{r.p_sv:>8.3f}{r.p_jumps:>9.3f}  {verdict}")
        print("\n  J levels are not comparable across rows (each panel has its")
        print("  own bootstrap W, θ and λ); the selection verdicts are.")

        out_s = config.DATA_DIR / "processed" / "smm_sampling_sensitivity.csv"
        out_f = config.DATA_DIR / "processed" / "smm_sampling_ladder_full.csv"
        summary.to_csv(out_s, index=False, float_format="%.6f")
        full.to_csv(out_f, index=False, float_format="%.6f")
        print(f"\nSaved: {out_s}")
        print(f"Saved: {out_f}")

    return summary, full


def bucketing_analysis(
    panel: "pd.DataFrame",
    calibrator: "BatesSMM | None" = None,
    kappa: float = 5.0,
    n_buckets: int = 3,
) -> pd.DataFrame:
    """§4.8 — Cross-sectional bucketing: run ladder on terciles of contracts.

    Contracts are bucketed by number of increments (a proxy for trading
    activity / duration).  If σ_v and σ_J are stable across buckets, the
    estimates aren't driven by a handful of very active contracts.
    """
    calibrator = calibrator or BatesSMM()

    # Bucket by contract length
    lengths = (
        panel.groupby("contract_id")["delta_X"]
             .count()
             .rename("n_increments")
             .reset_index()
    )
    try:
        lengths["bucket"] = pd.qcut(
            lengths["n_increments"], q=n_buckets,
            # n_buckets is 3, so these are terciles: label them T, not Q.
            # "Q1-Q3" reads as quartiles and was mislabelled in §4.8.
            labels=[f"T{i+1}" for i in range(n_buckets)],
            duplicates="drop",
        )
    except ValueError:
        lengths["bucket"] = "T1"

    rows = []
    for bucket_label, bucket_tickers in lengths.groupby("bucket", observed=True)["contract_id"]:
        sub = panel[panel["contract_id"].isin(bucket_tickers)]
        n_c = sub["contract_id"].nunique()
        if n_c < 3:
            print(f"  Bucket {bucket_label}: only {n_c} contracts, skipping")
            continue

        print(f"\n--- Bucket {bucket_label} ({n_c} contracts) ---")
        cache = calibrator.prepare(sub)
        res   = calibrator.fit_bates(cache, kappa=kappa, free_kappa=True,
                                     verbose=True)

        avg_len = float(lengths.loc[
            lengths["contract_id"].isin(bucket_tickers), "n_increments"
        ].mean())
        rows.append({
            "bucket": str(bucket_label), "n_contracts": n_c,
            "avg_increments": avg_len,
            "sigma_v": res.sigma_v, "sigma_J": res.sigma_J, "kappa": res.kappa,
            "j_stat": res.j_stat, "j_pvalue": res.j_pvalue,
        })

    df = pd.DataFrame(rows)
    print("\n=== §4.8 Cross-sectional bucketing ===")
    if not df.empty:
        print(df.to_string(index=False, float_format="{:.4f}".format))
        out = config.DATA_DIR / "processed" / "smm_bucketing.csv"
        df.to_csv(out, index=False, float_format="%.6f")
        print(f"Saved: {out}")
    return df
