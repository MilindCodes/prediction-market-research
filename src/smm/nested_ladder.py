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
        """§4.8 robustness: run ladder separately per contract family.

        Kalshi tickers split by prefix (KXFED / KXCPI).  Polymarket condition
        IDs (0x…) are classified via the identity mapping in
        data/exports/polymarket_contract_identities.csv (fed vs election),
        since Polymarket carries no CPI markets.
        """
        out: dict[str, LadderResult] = {}
        ids = panel["contract_id"].astype(str)

        if ids.str.startswith("0x").all():
            id_path = Path("data/exports/polymarket_contract_identities.csv")
            groups = pd.read_csv(id_path).set_index("conditionId")["group"]
            splits = [
                ("Fed", ids.map(groups).eq("fed")),
                ("Election", ids.map(groups).eq("election")),
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

    # Heatmap of relative misses
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        miss_cols = [f"miss_{m.model}" for m in models]
        miss_data = df[miss_cols].values
        model_labels = [m.model for m in models]

        fig, ax = plt.subplots(figsize=(9, 4))
        cmap = plt.cm.RdYlGn_r
        im = ax.imshow(miss_data.T, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(N_MOMENTS))
        ax.set_xticklabels(MOMENT_LABELS, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(model_labels)))
        ax.set_yticklabels(model_labels)
        plt.colorbar(im, ax=ax, label="Relative miss (achieved−target)/|target|")
        ax.set_title("§4.6 Moment fit by model  (green=good, red=misses)")

        # Annotate with numbers
        for i in range(N_MOMENTS):
            for j, m in enumerate(models):
                val = miss_data[i, j]
                ax.text(i, j, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > 0.5 else "black")

        plt.tight_layout()
        out = Path(figures_dir) / "moment_selection.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"\n  Figure: {out}")
    except Exception as e:
        print(f"  (Figure skipped: {e})")

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

    Re-runs the full panel transformation (clip → logit → diff) for each
    clip pair, then re-runs the SMM calibration on each resulting panel.
    Returns a summary DataFrame comparing σ_v, σ_J, J, p across clip pairs.
    """
    import config as _cfg
    from src.smm.panel import CLIP_LO as _DEFAULT_LO, CLIP_HI as _DEFAULT_HI

    clip_pairs = clip_pairs or [
        (_DEFAULT_LO, _DEFAULT_HI),  # baseline [0.02, 0.98]
        (0.01, 0.99),                 # wider — spec's referee sensitivity
    ]
    figures_dir = figures_dir or (config.DATA_DIR / "processed" / "figures")

    rows = []
    for clip_lo, clip_hi in clip_pairs:
        label = f"[{clip_lo},{clip_hi}]"
        print(f"\n--- Truncation sensitivity {label} ---")

        # Re-apply clip/logit/diff to the base panel's raw p column.
        # p_raw is the unclipped price — clipping the already-clipped p
        # column would make every clip pair a no-op.
        panel = panel_base.copy()
        src_col = "p_raw" if "p_raw" in panel.columns else "p"
        p_re = np.clip(panel[src_col].values, clip_lo, clip_hi)
        X_re = np.log(p_re / (1.0 - p_re))

        # Recompute delta_X within each contract
        panel["p"] = p_re
        panel["X"] = X_re
        new_dx = []
        for cid, grp in panel.groupby("contract_id", sort=False):
            grp = grp.sort_values("t")
            X = grp["X"].values
            dx = np.empty(len(X))
            dx[0] = np.nan
            dx[1:] = X[1:] - X[:-1]
            new_dx.append(pd.Series(dx, index=grp.index))
        panel["delta_X"] = pd.concat(new_dx)
        panel = panel.dropna(subset=["delta_X"]).reset_index(drop=True)

        cache = calibrator.prepare(panel)
        res   = calibrator.fit_bates(cache, kappa=kappa, verbose=True)

        rows.append({
            "clip": label, "clip_lo": clip_lo, "clip_hi": clip_hi,
            "sigma_v": res.sigma_v, "sigma_J": res.sigma_J,
            "theta": res.theta, "lambda": res.lambda_,
            "j_stat": res.j_stat, "j_pvalue": res.j_pvalue,
            "n_real": res.n_real,
        })

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
    """§4.8 — Re-run panel and SMM at alternative grid frequencies.

    Compares the daily (D) baseline against coarser grids (2D, 4D).
    Coarser grids have fewer forward-filled zero-increments but also
    less data — the sensitivity checks that results are stable.
    (The raw bars are daily, so grids finer than 1D would only add
    forward-filled artifacts.)
    """
    from src.smm.panel import SMMPanelBuilder

    raw_dir    = raw_dir or (config.DATA_DIR / "raw" / "polymarket")
    calibrator = calibrator or BatesSMM()
    freqs      = freqs or ["D", "2D", "4D"]

    rows = []
    for freq in freqs:
        print(f"\n--- Frequency sensitivity: {freq} ---")
        builder = SMMPanelBuilder(freq=freq)
        tickers = builder._load_catalog_tickers()
        if not tickers:
            print("  No tickers — run catalog step first.")
            continue
        panel = builder.build(tickers=tickers, force=True)
        cache = calibrator.prepare(panel)
        res   = calibrator.fit_bates(cache, kappa=kappa, verbose=True)
        rows.append({
            "freq": freq, "n_contracts": panel["contract_id"].nunique(),
            "n_real": res.n_real,
            "sigma_v": res.sigma_v, "sigma_J": res.sigma_J,
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
            labels=[f"Q{i+1}" for i in range(n_buckets)],
            duplicates="drop",
        )
    except ValueError:
        lengths["bucket"] = "Q1"

    rows = []
    for bucket_label, bucket_tickers in lengths.groupby("bucket", observed=True)["contract_id"]:
        sub = panel[panel["contract_id"].isin(bucket_tickers)]
        n_c = sub["contract_id"].nunique()
        if n_c < 3:
            print(f"  Bucket {bucket_label}: only {n_c} contracts, skipping")
            continue

        print(f"\n--- Bucket {bucket_label} ({n_c} contracts) ---")
        cache = calibrator.prepare(sub)
        res   = calibrator.fit_bates(cache, kappa=kappa, verbose=True)

        avg_len = float(lengths.loc[
            lengths["contract_id"].isin(bucket_tickers), "n_increments"
        ].mean())
        rows.append({
            "bucket": str(bucket_label), "n_contracts": n_c,
            "avg_increments": avg_len,
            "sigma_v": res.sigma_v, "sigma_J": res.sigma_J,
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
