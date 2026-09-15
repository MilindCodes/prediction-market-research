"""Mincer–Zarnowitz test of price unbiasedness on the Fed panel (§4, optional).

Regress the realised binary outcome on the contemporaneous YES price:

    y_i = alpha + beta * p_it + e_it,     H0: alpha = 0 and beta = 1

pooled over all (contract, time) observations, with contract-clustered
standard errors.  This is the formal unbiasedness test Bürgi, Deng and Whelan
apply to Kalshi; running it here answers the "your jump tail is really a
microstructure pricing bias" objection directly.  A favourite-longshot
distortion shows up as beta < 1 with alpha > 0.

OUTCOMES ARE INFERRED, NOT OBSERVED.  The repo stores no resolution field, so
y_i is taken from each contract's terminal price: > 0.95 -> 1, < 0.05 -> 0.
Five of the 39 contracts end between 0.05 and 0.95 (three sit at exactly
0.500 with the feed stopping after their FOMC meeting, i.e. stale rather than
genuinely uncertain) and are dropped.  Before this goes in the paper the
outcomes should come from Polymarket's resolution payouts instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config

RAW = config.DATA_DIR / "raw" / "polymarket"
PROCESSED = config.DATA_DIR / "processed"
EXPORTS = config.DATA_DIR / "exports"

HI, LO = 0.95, 0.05          # terminal-price thresholds for outcome inference


def _terminal_outcome(cid: str) -> float:
    """Infer the realised outcome from a contract's last observed price."""
    d = pd.read_parquet(RAW / f"{cid}.parquet")
    v = pd.to_numeric(d["close"], errors="coerce").dropna()
    if v.empty:
        return np.nan
    last = float(v.iloc[-1])
    if last > 1.5:                      # integer cents
        last /= 100.0
    if last >= HI:
        return 1.0
    if last <= LO:
        return 0.0
    return np.nan                       # ambiguous — excluded


def _cluster_ols(y: np.ndarray, X: np.ndarray, cid: np.ndarray):
    """OLS with one-way cluster-robust covariance. X includes the constant."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(cid):
        m = cid == c
        s = X[m] * resid[m][:, None]
        meat += s.T @ s
    G = len(np.unique(cid))
    n, k = X.shape
    adj = (G / (G - 1.0)) * ((n - 1.0) / (n - k))       # standard finite-G scale-up
    cov = XtX_inv @ meat @ XtX_inv * adj
    return beta, cov, G


def run(verbose: bool = True) -> dict:
    panel = pd.read_parquet(PROCESSED / "smm_panel.parquet")
    cids = list(pd.unique(panel.contract_id))

    outcomes = {c: _terminal_outcome(c) for c in cids}
    used = [c for c in cids if np.isfinite(outcomes[c])]
    dropped = [c for c in cids if not np.isfinite(outcomes[c])]

    sub = panel[panel.contract_id.isin(used)].copy()
    # p is the clipped price; p_raw is the traded price the forecast test wants
    price_col = "p_raw" if "p_raw" in sub.columns else "p"
    p = sub[price_col].astype(float).values
    y = sub.contract_id.map(outcomes).astype(float).values
    codes = pd.Categorical(sub.contract_id).codes

    X = np.column_stack([np.ones(len(p)), p])
    beta, cov, G = _cluster_ols(y, X, codes)
    se = np.sqrt(np.diag(cov))
    alpha, slope = beta
    se_a, se_b = se

    t_a = alpha / se_a                       # H0: alpha = 0
    t_b = (slope - 1.0) / se_b               # H0: beta  = 1
    df = G - 1
    p_a = 2 * stats.t.sf(abs(t_a), df)
    p_b = 2 * stats.t.sf(abs(t_b), df)

    # Joint Wald test of (alpha, beta) = (0, 1)
    d = np.array([alpha - 0.0, slope - 1.0])
    W = float(d @ np.linalg.inv(cov) @ d)
    p_joint = stats.chi2.sf(W, 2)
    # With only G clusters the chi-square reference is anti-conservative;
    # F(2, G-1) is the small-sample version and is what we report on.
    p_joint_F = stats.f.sf(W / 2.0, 2, G - 1)

    # Favourite–longshot calibration by price bucket
    edges = np.array([0, .05, .10, .20, .35, .50, .65, .80, .90, .95, 1.0])
    b = pd.DataFrame({"p": p, "y": y})
    b["bucket"] = pd.cut(b.p, edges, include_lowest=True)
    cal = b.groupby("bucket", observed=True).agg(
        n=("y", "size"), mean_price=("p", "mean"), freq=("y", "mean")).reset_index()
    cal["gap"] = cal.freq - cal.mean_price

    if verbose:
        print("\n=== Mincer–Zarnowitz: outcome on price (Fed panel) ===")
        print(f"  contracts used     : {len(used)} of {len(cids)}"
              f"   (dropped {len(dropped)}: ambiguous terminal price)")
        print(f"  observations       : {len(p):,}   clusters: {G}")
        print(f"  alpha  = {alpha:+.4f}  (cluster SE {se_a:.4f})  "
              f"t vs 0 = {t_a:+.3f}   p = {p_a:.4f}")
        print(f"  beta   = {slope:+.4f}  (cluster SE {se_b:.4f})  "
              f"t vs 1 = {t_b:+.3f}   p = {p_b:.4f}")
        print(f"  joint Wald (alpha=0, beta=1): W = {W:.3f}   "
              f"p = {p_joint:.4g} (chi2)   p = {p_joint_F:.4g} (F, small-G)")
        verdict = ("REJECTS unbiasedness" if p_joint_F < 0.05
                   else "does NOT reject unbiasedness")
        print(f"  -> {verdict} at the 5% level")
        print("\n  Calibration by price bucket (favourite–longshot):")
        print("    " + cal.to_string(index=False,
              float_format="{:.4f}".format).replace("\n", "\n    "))

    out = EXPORTS / "mincer_zarnowitz.csv"
    pd.DataFrame([{
        "n_contracts": len(used), "n_dropped": len(dropped), "n_obs": len(p),
        "alpha": alpha, "se_alpha": se_a, "t_alpha_vs0": t_a, "p_alpha": p_a,
        "beta": slope, "se_beta": se_b, "t_beta_vs1": t_b, "p_beta": p_b,
        "wald_joint": W, "p_joint_chi2": p_joint, "p_joint_F": p_joint_F,
    }]).to_csv(out, index=False, float_format="%.6f")
    cal.to_csv(EXPORTS / "mincer_zarnowitz_calibration.csv", index=False,
               float_format="%.6f")
    if verbose:
        print(f"\n  Saved: {out}")
        print(f"  Saved: {EXPORTS / 'mincer_zarnowitz_calibration.csv'}")

    return {"alpha": alpha, "beta": slope, "p_joint": p_joint_F,
            "n_contracts": len(used), "dropped": dropped, "calibration": cal}


if __name__ == "__main__":
    run()
