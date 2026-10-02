"""
§4.2  Stylized Facts

Five diagnostics on pooled log-odds increments.
Synthetic validation on two panels runs before touching real data.

Diagnostics
-----------
1. Fat tails   — excess kurtosis + Q-Q plot (standardised per-contract first)
2. Vol cluster — ACF(ΔX², lags 1-10) per contract, mean across contracts
                 + Ljung-Box fraction
3. Agg. Gauss  — kurtosis at k∈{1,2,4,8,16,32,64} non-overlapping steps
4. Boundary    — OLS: ΔX² ~ p(1-p) using predetermined p; cluster-robust SEs
5. Time effect — Var(last quarter) vs Var(first quarter); Levene + log-var-ratio t-test

Key methodological choices (easy to get wrong)
-----------------------------------------------
* Diagnostic 1: standardise within each contract BEFORE pooling, otherwise a
  variance mixture across contracts manufactures kurtosis that isn't there.
* Diagnostic 2: compute ACF per contract, then aggregate — a single
  concatenated series lets Ljung-Box pick up seams between contracts.
* Diagnostic 4: use p at the START of each increment (predetermined), not
  the contemporaneous p, which is mechanically endogenous to the same ΔX.
* Diagnostic 5: Levene's test (robust to non-normality found in diag. 1);
  paired log-variance-ratio t-test respects the panel structure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config
from src.figstyle import apply_publication_style, save_figure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _acf(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return np.nan
    c = np.corrcoef(x[:-lag], x[lag:])
    return float(c[0, 1]) if c.shape == (2, 2) else np.nan


def _ljung_box(x: np.ndarray, n_lags: int = 10) -> tuple[float, float]:
    """Manual Ljung-Box Q-stat (no statsmodels dependency)."""
    n = len(x)
    if n < n_lags + 2:
        return np.nan, np.nan
    acfs = np.array([_acf(x, k) for k in range(1, n_lags + 1)])
    ks = np.arange(1, n_lags + 1)
    q = n * (n + 2) * np.nansum(acfs**2 / (n - ks))
    p = float(1 - stats.chi2.cdf(q, df=n_lags))
    return float(q), p


def _decompose_panel(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return dict cid → delta_X array (raw, not demeaned)."""
    out: dict[str, np.ndarray] = {}
    for cid, grp in panel.groupby("contract_id", sort=False):
        dx = grp["delta_X"].dropna().values
        if len(dx) >= 5:
            out[str(cid)] = dx
    return out


def _predetermined_p(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return dict cid → p at the START of each increment (predetermined).

    §4.2 diagnostic 4 regresses ΔX² on p(1−p) and requires p to be
    predetermined: p[t] is a function of the same ΔX[t] being regressed, so
    using it builds in mechanical endogeneity.

    The panel arrives already differenced, with the leading row of each
    contract dropped, so every ΔX is non-null and the old `~isnan(delta_X)`
    filter selected everything — returning contemporaneous p[t] despite the
    name.  Shifting by one row within contract gives the intended p[t−1] and
    costs the first increment of each contract, which has no predecessor in
    the saved panel.
    """
    out: dict[str, np.ndarray] = {}
    for cid, grp in panel.groupby("contract_id", sort=False):
        grp = grp.sort_values("t")
        p_pre = grp["p"].values[:-1]          # p at the start of ΔX[1:]
        if len(p_pre) >= 5:
            out[str(cid)] = p_pre
    return out


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StylizedFacts:
    """Compute all §4.2 diagnostics and produce a summary figure.

    Parameters
    ----------
    figures_dir : Path | None
        Where to save output figures. Default: data/processed/figures/.
    tables_dir : Path | None
        Where to save the per-run results table. Default: data/processed/.
    """

    def __init__(
        self,
        figures_dir: Path | None = None,
        tables_dir: Path | None = None,
    ):
        self.figures_dir = (
            figures_dir or (config.DATA_DIR / "processed" / "figures")
        )
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir = tables_dir or (config.DATA_DIR / "processed")
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        panel: pd.DataFrame,
        tag: str = "real",
    ) -> dict[str, Any]:
        """Run all five diagnostics.

        Parameters
        ----------
        panel : pd.DataFrame
            Output of SMMPanelBuilder — columns: contract_id, t, p, X, delta_X
        tag : str
            Label for saved figure filename (e.g. "real", "synthetic_diffusion").

        Returns
        -------
        dict with one key per diagnostic.
        """
        inc = _decompose_panel(panel)
        p_pre = _predetermined_p(panel)
        n_contracts = len(inc)
        n_total = sum(len(v) for v in inc.values())

        print(f"\n=== §4.2 Stylized Facts [{tag}] "
              f"({n_contracts} contracts, {n_total} increments) ===")

        # Standardise within each contract then pool — diagnostic 1 requires this
        std_parts: list[np.ndarray] = []
        for dx in inc.values():
            mu, sd = dx.mean(), dx.std(ddof=1)
            if sd > 0:
                std_parts.append((dx - mu) / sd)
        pooled_std = np.concatenate(std_parts)

        results: dict[str, Any] = {
            "n_contracts": n_contracts,
            "n_increments": n_total,
        }
        results["fat_tails"]       = self._fat_tails(pooled_std)
        results["vol_clustering"]  = self._vol_clustering(inc)
        # Standardized: what §4.7's overlay and §4.8's sampling table quote.
        results["agg_gaussianity"] = self._agg_gaussianity(inc)
        # Unstandardized: the §4.2 diagnostic-3 ladder as published.
        results["agg_gaussianity_unstd"] = self._agg_gaussianity(
            inc, standardize=False
        )
        results["boundary_scaling"]= self._boundary_scaling(inc, p_pre)
        results["time_effect"]     = self._time_effect(inc)

        self._print_summary(results)
        self._plot(results, pooled_std, inc, p_pre, tag=tag)
        self._save_table(results, tag=tag)
        return results

    def validate_synthetic(self) -> dict[str, dict]:
        """Validate diagnostics on two synthetic panels before real data.

        Panel A — pure log-odds diffusion: should show Gaussian increments,
                  flat boundary slope, kurtosis decaying to zero under aggregation.
        Panel B — jump-diffusion: fat tails that persist under aggregation.

        The suite must separate these two panels; if it cannot, the diagnostics
        will not be informative on real Kalshi data.
        """
        print("\n=== §4.2 Synthetic Validation ===")

        rng = np.random.default_rng(42)
        n_contracts = 25
        n_steps = 500
        sigma_diff = 0.3

        def _build_panel(dx_all: np.ndarray, prefix: str) -> pd.DataFrame:
            rows: list[pd.DataFrame] = []
            for i, dx in enumerate(dx_all):
                X = np.cumsum(np.concatenate([[0.0], dx]))
                p = np.clip(1.0 / (1.0 + np.exp(-X)), 0.02, 0.98)
                X_c = np.log(p / (1.0 - p))
                dX = X_c[1:] - X_c[:-1]
                rows.append(pd.DataFrame({
                    "contract_id": f"{prefix}_{i:03d}",
                    "t": np.arange(len(dX)),
                    "p": p[:-1],
                    "X": X_c[:-1],
                    "delta_X": dX,
                }))
            return pd.concat(rows, ignore_index=True)

        # Panel A: pure diffusion
        dx_a = rng.normal(0, sigma_diff, size=(n_contracts, n_steps))
        panel_a = _build_panel(dx_a, "D")
        print("\n--- Panel A: pure diffusion ---")
        res_a = self.run(panel_a, tag="synthetic_diffusion")

        # Panel B: jump-diffusion (λ=5%, σ_J=1.5)
        jump_rate, jump_sd = 0.05, 1.5
        dx_b = rng.normal(0, sigma_diff * 0.7, size=(n_contracts, n_steps))
        has_jump = rng.uniform(size=(n_contracts, n_steps)) < jump_rate
        dx_b += has_jump * rng.normal(0, jump_sd, size=(n_contracts, n_steps))
        panel_b = _build_panel(dx_b, "J")
        print("\n--- Panel B: jump-diffusion ---")
        res_b = self.run(panel_b, tag="synthetic_jump")

        print("\n--- Separation check ---")
        kurt_a = res_a["fat_tails"]["excess_kurtosis"]
        kurt_b = res_b["fat_tails"]["excess_kurtosis"]
        print(f"  Kurtosis  — diffusion: {kurt_a:.2f}, jump: {kurt_b:.2f}")

        k8_a  = res_a["agg_gaussianity"]["kurtosis_by_k"].get(8,  np.nan)
        k8_b  = res_b["agg_gaussianity"]["kurtosis_by_k"].get(8,  np.nan)
        k64_a = res_a["agg_gaussianity"]["kurtosis_by_k"].get(64, np.nan)
        k64_b = res_b["agg_gaussianity"]["kurtosis_by_k"].get(64, np.nan)
        print(f"  Kurt@k=8  — diffusion: {k8_a:.2f}, jump: {k8_b:.2f}")
        print(f"  Kurt@k=64 — diffusion: {k64_a:.2f}, jump: {k64_b:.2f}")

        bs_a = res_a["boundary_scaling"]["slope"]
        bs_b = res_b["boundary_scaling"]["slope"]
        print(f"  Boundary slope — diffusion: {bs_a:.4f}, jump: {bs_b:.4f}")

        # Separation criteria:
        #   1. Overall kurtosis clearly higher for jump panel
        #   2. Kurtosis at intermediate scale (k=8) still elevated for jump
        # Note: at k=64 with λ=5% and ~500 steps, CLT pulls both toward zero —
        # the key test is whether kurtosis at k∈{4,8} remains elevated for jumps.
        kurt_sep  = kurt_b > kurt_a + 5.0
        decay_sep = (not np.isnan(k8_b)) and (k8_b > k8_a + 0.2)
        passed    = kurt_sep and decay_sep
        print(f"\n  Separation: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print("  WARNING: diagnostics cannot reliably distinguish diffusion "
                  "from jumps. Do not trust real-data results.")

        return {"diffusion": res_a, "jump_diffusion": res_b}

    # ------------------------------------------------------------------
    # Individual diagnostics
    # ------------------------------------------------------------------

    def _fat_tails(self, pooled_std: np.ndarray) -> dict[str, Any]:
        excess_kurt = float(stats.kurtosis(pooled_std, fisher=True))
        jb_stat, jb_p = stats.jarque_bera(pooled_std)
        return {
            "excess_kurtosis": excess_kurt,
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_p": float(jb_p),
            "n": int(len(pooled_std)),
        }

    def _vol_clustering(
        self, inc: dict[str, np.ndarray]
    ) -> dict[str, Any]:
        lags = [1, 2, 3, 5, 10]
        acf_by_lag: dict[int, list[float]] = {lag: [] for lag in lags}
        lb_reject = 0
        lb_total = 0

        for dx in inc.values():
            dx2 = dx ** 2
            for lag in lags:
                a = _acf(dx2, lag)
                if np.isfinite(a):
                    acf_by_lag[lag].append(a)
            if len(dx2) >= 15:
                _, lb_p = _ljung_box(dx2, n_lags=min(10, len(dx2) - 2))
                if np.isfinite(lb_p):
                    lb_total += 1
                    if lb_p < 0.05:
                        lb_reject += 1

        mean_acf = {
            lag: float(np.mean(vals)) if vals else np.nan
            for lag, vals in acf_by_lag.items()
        }
        return {
            "mean_acf_squared": mean_acf,
            "lb_reject_fraction": lb_reject / max(lb_total, 1),
            "lb_n_contracts": lb_total,
        }

    def _agg_gaussianity(
        self, inc: dict[str, np.ndarray], standardize: bool = True
    ) -> dict[str, Any]:
        """Excess kurtosis of k-step aggregated increments.

        `standardize` controls the normalisation, and the two answers are
        very different — roughly 18.0 against 54.5 at k=1 on the Fed panel:

        * True  — each contract's k-blocks are divided by their own SD
          before pooling.  This is the §4.2 diagnostic-1 normalisation, and
          it is what the §4.8 sampling table and the §4.7 validation overlay
          report (17.98 at daily calendar).
        * False — blocks are demeaned but not rescaled, so pooling across
          contracts of differing volatility leaves the scale mixture in.
          This is the §4.2 diagnostic-3 ladder as published
          (54.6 → 32.3 → 18.2 → 8.5 → 0.9 → −0.1) and the normalisation the
          SMM target moment uses.

        Both are legitimate; they must not be mixed inside one table.
        """
        ks = [1, 2, 4, 8, 16, 32, 64]
        kurtosis_by_k: dict[int, float] = {}
        n_by_k: dict[int, int] = {}

        for k in ks:
            pooled: list[np.ndarray] = []
            for dx in inc.values():
                n_blocks = len(dx) // k
                if n_blocks < 2:
                    continue
                blocks = dx[: n_blocks * k].reshape(n_blocks, k).sum(axis=1)
                if standardize:
                    sd = blocks.std(ddof=1)
                    if sd > 0:
                        pooled.append((blocks - blocks.mean()) / sd)
                else:
                    pooled.append(blocks - blocks.mean())
            if pooled:
                flat = np.concatenate(pooled)
                kurtosis_by_k[k] = float(stats.kurtosis(flat, fisher=True))
                n_by_k[k] = int(len(flat))

        return {"kurtosis_by_k": kurtosis_by_k, "n_by_k": n_by_k}

    def _boundary_scaling(
        self,
        inc: dict[str, np.ndarray],
        p_pre: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Regress ΔX² on p(1-p) using predetermined p; cluster-robust SEs."""
        dx2_list: list[np.ndarray] = []
        pp_list: list[np.ndarray] = []
        cid_list: list[np.ndarray] = []
        cid_labels: list[str] = list(inc.keys())

        for idx, cid in enumerate(cid_labels):
            dx = inc[cid]
            p = p_pre.get(cid)
            if p is None:
                continue
            # p is predetermined and therefore one shorter than dx: p[i-1]
            # is the start of increment dx[i].  Align from the END so the
            # first increment (the one with no predecessor) is the one
            # dropped, not the last.
            n = min(len(dx), len(p))
            dx_use = dx[len(dx) - n:]
            p_use = p[len(p) - n:]
            dx2_list.append(dx_use ** 2)
            pp_list.append(p_use * (1.0 - p_use))
            cid_list.append(np.full(n, idx, dtype=int))

        if not dx2_list:
            return {"slope": np.nan, "intercept": np.nan,
                    "se_slope_cluster": np.nan, "t_stat": np.nan,
                    "n_obs": 0, "n_contracts": 0}

        Y = np.concatenate(dx2_list)
        X_pp = np.concatenate(pp_list)
        C = np.concatenate(cid_list)
        X_mat = np.column_stack([np.ones(len(Y)), X_pp])

        try:
            beta, _, _, _ = np.linalg.lstsq(X_mat, Y, rcond=None)
            intercept, slope = float(beta[0]), float(beta[1])

            resid = Y - X_mat @ beta
            XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
            meat = np.zeros((2, 2))
            for cid_idx in np.unique(C):
                mask = C == cid_idx
                Xc = X_mat[mask]
                ec = resid[mask]
                score = Xc * ec[:, None]
                meat += score.T @ score

            cov_cr = XtX_inv @ meat @ XtX_inv
            se_slope = float(np.sqrt(max(cov_cr[1, 1], 0.0)))
            t_stat = slope / se_slope if se_slope > 0 else np.nan
        except np.linalg.LinAlgError:
            intercept = slope = se_slope = t_stat = np.nan

        return {
            "slope": slope,
            "intercept": intercept,
            "se_slope_cluster": se_slope,
            "t_stat": t_stat,
            "n_obs": int(len(Y)),
            "n_contracts": int(len(np.unique(C))),
        }

    def _time_effect(self, inc: dict[str, np.ndarray]) -> dict[str, Any]:
        """Compare variance in first vs last quarter of each contract."""
        first_pools: list[np.ndarray] = []
        last_pools: list[np.ndarray] = []
        log_ratios: list[float] = []

        for dx in inc.values():
            n = len(dx)
            if n < 20:
                continue
            q = n // 4
            first_q = dx[:q]
            last_q = dx[n - q:]
            v1 = float(np.var(first_q, ddof=1))
            v2 = float(np.var(last_q, ddof=1))
            if v1 > 0 and v2 > 0:
                first_pools.append(first_q)
                last_pools.append(last_q)
                log_ratios.append(np.log(v2 / v1))

        if not log_ratios:
            return {
                "levene_stat": np.nan, "levene_p": np.nan,
                "mean_log_var_ratio": np.nan,
                "log_var_ratio_tstat": np.nan, "log_var_ratio_p": np.nan,
                "n_contracts": 0,
            }

        lev_stat, lev_p = stats.levene(
            np.concatenate(first_pools),
            np.concatenate(last_pools),
        )
        lr = np.array(log_ratios)
        t_stat, t_p = stats.ttest_1samp(lr, 0.0)

        return {
            "levene_stat": float(lev_stat),
            "levene_p": float(lev_p),
            "mean_log_var_ratio": float(lr.mean()),
            "log_var_ratio_tstat": float(t_stat),
            "log_var_ratio_p": float(t_p),
            "n_contracts": int(len(log_ratios)),
        }

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_table(self, results: dict, tag: str) -> Path:
        """Write every diagnostic to stylized_facts_<tag>.csv.

        These numbers used to exist only in memory and stdout, so a value
        quoted in the draft had nothing to be checked against.  Long
        format, one row per statistic; nested dicts (ACF by lag, kurtosis
        by k) become `name[key]`.  %.6g rather than the usual %.6f so small
        p-values are not flattened to zero.
        """
        rows: list[dict[str, Any]] = []
        for diag, val in results.items():
            if not isinstance(val, dict):
                rows.append({"diagnostic": "panel", "statistic": diag,
                             "value": val})
                continue
            for stat, v in val.items():
                if isinstance(v, dict):
                    for key, vk in v.items():
                        rows.append({"diagnostic": diag,
                                     "statistic": f"{stat}[{key}]",
                                     "value": vk})
                else:
                    rows.append({"diagnostic": diag, "statistic": stat,
                                 "value": v})
        out = self.tables_dir / f"stylized_facts_{tag}.csv"
        pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g")
        print(f"  Table: {out}")
        return out

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def _print_summary(self, results: dict) -> None:
        ft = results["fat_tails"]
        vc = results["vol_clustering"]
        ag = results.get("agg_gaussianity_unstd") or results["agg_gaussianity"]
        bs = results["boundary_scaling"]
        te = results["time_effect"]

        print(f"  1. Fat tails: kurtosis = {ft['excess_kurtosis']:.2f}"
              f"  (JB p = {ft['jarque_bera_p']:.2e},"
              f" n = {ft['n']:,})")

        acf1 = vc["mean_acf_squared"].get(1, np.nan)
        print(f"  2. Vol cluster: mean ACF(ΔX²,lag=1) = {acf1:.3f},"
              f" LB reject = {vc['lb_reject_fraction']:.1%}"
              f" ({vc['lb_n_contracts']} contracts)")

        k1  = ag["kurtosis_by_k"].get(1, np.nan)
        k64 = ag["kurtosis_by_k"].get(64, np.nan)
        print(f"  3. Agg. Gauss: kurtosis k=1 → {k1:.2f}, k=64 → {k64:.2f}")

        print(f"  4. Boundary: slope = {bs['slope']:.4f}"
              f"  (cluster-SE = {bs['se_slope_cluster']:.4f},"
              f" t = {bs['t_stat']:.2f})")

        mean_lr = te["mean_log_var_ratio"]
        print(f"  5. Time effect: mean log(var_last/var_first) = {mean_lr:.3f},"
              f" Levene p = {te['levene_p']:.3f},"
              f" t-test p = {te['log_var_ratio_p']:.3f}"
              f" ({te['n_contracts']} contracts)")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot(
        self,
        results: dict,
        pooled_std: np.ndarray,
        inc: dict[str, np.ndarray],
        p_pre: dict[str, np.ndarray],
        tag: str = "real",
    ) -> None:
        apply_publication_style()
        fig = plt.figure(figsize=(15, 9))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

        # --- 1. Q-Q plot ---
        ax1 = fig.add_subplot(gs[0, 0])
        stats.probplot(pooled_std, dist="norm", plot=ax1)
        ax1.set_title("1. Q-Q (std. increments vs Normal)", fontsize=9)

        # --- 2. ACF of ΔX² ---
        ax2 = fig.add_subplot(gs[0, 1])
        lags_sorted = sorted(results["vol_clustering"]["mean_acf_squared"].keys())
        acfs = [results["vol_clustering"]["mean_acf_squared"][l] for l in lags_sorted]
        ax2.bar(lags_sorted, acfs, color="steelblue", width=0.6)
        ax2.axhline(0, color="black", linewidth=0.7)
        ax2.set_title("2. Mean ACF(ΔX²) across contracts", fontsize=9)
        ax2.set_xlabel("Lag")

        # --- 3. Aggregational Gaussianity ---
        ax3 = fig.add_subplot(gs[0, 2])
        ag = (results.get("agg_gaussianity_unstd")
              or results["agg_gaussianity"])["kurtosis_by_k"]
        ks = sorted(ag.keys())
        kurts = [ag[k] for k in ks]
        ax3.plot(ks, kurts, "o-", color="steelblue", markersize=5)
        ax3.axhline(0, color="red", linestyle="--", linewidth=0.8, label="Normal")
        ax3.set_title("3. Kurtosis vs aggregation scale k", fontsize=9)
        ax3.set_xlabel("k")
        ax3.set_xscale("log", base=2)
        ax3.set_ylabel("Excess kurtosis")
        ax3.legend(fontsize=7)

        # --- 4. Boundary scaling ---
        ax4 = fig.add_subplot(gs[1, 0])
        if p_pre:
            # Same predetermined-p alignment as _boundary_scaling: p is one
            # shorter than dx, so trim dx from the front per contract.
            dx2_parts, pp_parts = [], []
            for c in inc:
                pc = p_pre.get(c)
                if pc is None:
                    continue
                n = min(len(inc[c]), len(pc))
                dx2_parts.append(inc[c][len(inc[c]) - n:] ** 2)
                pp_parts.append(pc[len(pc) - n:] * (1 - pc[len(pc) - n:]))
            dx2_vals = np.concatenate(dx2_parts)
            pp_vals  = np.concatenate(pp_parts)
            n_plt = min(5000, len(dx2_vals))
            rng = np.random.default_rng(0)
            idx = rng.choice(len(dx2_vals), n_plt, replace=False)
            ax4.scatter(pp_vals[idx], dx2_vals[idx],
                        alpha=0.12, s=2, color="gray", rasterized=True)
            pp_grid = np.linspace(0, 0.25, 100)
            s = results["boundary_scaling"]["slope"]
            b = results["boundary_scaling"]["intercept"]
            ax4.plot(pp_grid, b + s * pp_grid, "r-", linewidth=1.5)
            ylim = np.nanpercentile(dx2_vals, 99)
            ax4.set_ylim(0, ylim)
        ax4.set_title(
            f"4. Boundary scaling  (slope={results['boundary_scaling']['slope']:.3f})",
            fontsize=9,
        )
        ax4.set_xlabel("p(1−p)")
        ax4.set_ylabel("ΔX²")

        # --- 5. Log variance ratio histogram ---
        ax5 = fig.add_subplot(gs[1, 1])
        lr_vals = []
        for dx in inc.values():
            n = len(dx)
            if n < 20:
                continue
            q = n // 4
            v1 = np.var(dx[:q], ddof=1)
            v2 = np.var(dx[n - q:], ddof=1)
            if v1 > 0 and v2 > 0:
                lr_vals.append(np.log(v2 / v1))
        if lr_vals:
            ax5.hist(lr_vals, bins=max(10, len(lr_vals) // 3),
                     color="steelblue", edgecolor="white", alpha=0.85)
            ax5.axvline(0, color="red", linestyle="--", linewidth=1)
        ax5.set_title("5. log(Var_last/Var_first) per contract", fontsize=9)
        ax5.set_xlabel("Log variance ratio")

        # Unused sixth panel
        fig.add_subplot(gs[1, 2]).set_visible(False)

        out = self.figures_dir / f"stylized_facts_{tag}.png"
        written = save_figure(out)
        print("  Figure: " + ", ".join(str(w) for w in written))
