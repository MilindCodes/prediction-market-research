"""
DataCleaner for prediction market research datasets.

Three cleaning stages, each writing to data/clean/:
  1. clean_trades()      — raw OHLCV files (Polymarket daily / Kalshi hourly)
  2. clean_implied_vol() — implied vol parquet files
  3. clean_params()      — Heston/Bates calibration result quality flags

Original files in data/raw/ and data/processed/ are never modified.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config

# ── thresholds ────────────────────────────────────────────────────────────────

PRICE_STRICT_MIN: float = 1e-6      # at/below → terminal zero state, IV undefined
PRICE_STRICT_MAX: float = 1 - 1e-6  # at/above → terminal one state, IV undefined
SIGMA_CAP: float = 15.0             # clamp sigma_implied here instead of dropping
IV_MIN_OBS: int = 10                # discard a contract with fewer clean IV rows

# flag (don't drop) runs of identical close price longer than this many bars
STALE_RUN_MAX: int = 5

# Optimizer bounds — must stay in sync with HestonCalibrator.PARAM_BOUNDS
_HESTON_BOUNDS: dict[str, tuple[float, float]] = {
    "kappa": (0.01,   50.0),
    "theta": (1e-6,   10.0),
    "xi":    (0.01,    5.0),
    "rho":   (-0.999,  0.999),
    "v0":    (1e-6,    5.0),
}

# Additional bounds — must stay in sync with BatesCalibrator.BATES_BOUNDS
_BATES_EXTRA_BOUNDS: dict[str, tuple[float, float]] = {
    "lambda_j": (0.0,  100.0),
    "mu_j":     (-5.0,   5.0),
    "sigma_j":  (0.001,  3.0),
}

# Same 1% tolerance used by the calibrators' own boundary warnings
_BOUND_TOL: float = 0.01


# ── report dataclass ──────────────────────────────────────────────────────────

@dataclass
class CleaningReport:
    """Statistics from one cleaning stage."""
    stage: str
    files_in: int = 0
    files_out: int = 0
    rows_in: int = 0
    rows_out: int = 0
    files_dropped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out

    def print(self) -> None:
        pct = 100.0 * self.rows_dropped / max(self.rows_in, 1)
        print(f"--- {self.stage} ---")
        print(f"  Files:    {self.files_out:>4} / {self.files_in} survived")
        print(f"  Rows in:  {self.rows_in:>10,}")
        print(f"  Rows out: {self.rows_out:>10,}")
        print(f"  Dropped:  {self.rows_dropped:>10,}  ({pct:.1f}%)")
        if self.files_dropped:
            sample = self.files_dropped[:3]
            suffix = "..." if len(self.files_dropped) > 3 else ""
            print(f"  Discarded ({len(self.files_dropped)} contracts): {sample}{suffix}")
        for note in self.notes:
            print(f"  note: {note}")
        print()


# ── main cleaner ──────────────────────────────────────────────────────────────

class DataCleaner:
    """Clean prediction market datasets across three stages.

    Reads from data/raw/ and data/processed/, writes to data/clean/.
    Original files are never modified.

    Parameters
    ----------
    data_dir : Path
        Project root data directory (default: config.DATA_DIR).
    drop_boundary : bool
        Drop IV rows where price is near 0 or 1 (near_boundary == True).
        Default True — boundary observations produce extreme, unreliable IV.
    drop_jump_windows : bool
        Drop IV rows inside FOMC ±30-min windows. Default False — the
        is_jump_window flag is preserved in the output for callers to filter.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        drop_boundary: bool = True,
        drop_jump_windows: bool = False,
    ) -> None:
        self.data_dir = data_dir or config.DATA_DIR
        self.clean_dir = self.data_dir / "clean"
        self.drop_boundary = drop_boundary
        self.drop_jump_windows = drop_jump_windows

    # ── public API ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Run all three stages and print a full summary."""
        print("=== Data Cleaning Pipeline ===\n")

        for platform in ["polymarket", "kalshi"]:
            report = self.clean_trades(platform)
            if report is not None:
                report.print()

        self.clean_implied_vol().print()

        params_df = self.clean_params()
        if not params_df.empty:
            self._print_params_report(params_df)

        print(f"Clean data written to: {self.clean_dir}/")

    def clean_trades(self, platform: str) -> CleaningReport | None:
        """Clean raw OHLCV parquet files for one platform.

        Rules applied to each file:
          - Drop rows with NaN close price
          - Drop prices at or outside open interval (0, 1)
          - Drop duplicate timestamps, keeping the last observation per bar
          - Sort chronologically
          - Flag (do not drop) stale price runs > STALE_RUN_MAX consecutive bars

        Saves cleaned files to data/clean/{platform}/.

        Parameters
        ----------
        platform : str
            "polymarket" or "kalshi".

        Returns
        -------
        CleaningReport, or None if the raw directory does not exist yet.
        """
        raw_dir = self.data_dir / "raw" / platform
        if not raw_dir.exists():
            return None

        out_dir = self.clean_dir / platform
        out_dir.mkdir(parents=True, exist_ok=True)

        trade_files = [
            f for f in sorted(raw_dir.glob("*.parquet"))
            if f.stem not in ("catalog_full", "catalog_filtered")
        ]
        report = CleaningReport(stage=f"Trades ({platform})", files_in=len(trade_files))

        for path in trade_files:
            df = pd.read_parquet(path)
            report.rows_in += len(df)

            df = self._clean_ohlcv(df, path.stem, report)
            if df.empty:
                report.files_dropped.append(path.stem)
                continue

            df.to_parquet(out_dir / path.name, index=False)
            report.files_out += 1
            report.rows_out += len(df)

        return report

    def clean_implied_vol(self) -> CleaningReport:
        """Clean implied volatility parquet files.

        Rules per row:
          - Drop rows where sigma_implied is NaN
          - Drop rows where T <= 0 (at or after contract resolution)
          - Drop near-boundary rows if drop_boundary=True (price < 5% or > 95%)
          - Drop FOMC jump-window rows if drop_jump_windows=True
          - Clamp sigma_implied at SIGMA_CAP; recompute sigma_logit accordingly

        Discards the entire file if fewer than IV_MIN_OBS rows remain.

        Returns
        -------
        CleaningReport
        """
        iv_dir = self.data_dir / "processed" / "implied_vol"
        if not iv_dir.exists():
            return CleaningReport(
                stage="Implied Vol",
                notes=["data/processed/implied_vol/ not found — run backtest-implied-vol first"],
            )

        out_dir = self.clean_dir / "implied_vol"
        out_dir.mkdir(parents=True, exist_ok=True)

        iv_files = sorted(iv_dir.glob("*.parquet"))
        report = CleaningReport(stage="Implied Vol", files_in=len(iv_files))

        for path in iv_files:
            df = pd.read_parquet(path)
            report.rows_in += len(df)

            df = self._clean_iv_rows(df)
            if len(df) < IV_MIN_OBS:
                report.files_dropped.append(path.stem)
                continue

            df.to_parquet(out_dir / path.name, index=False)
            report.files_out += 1
            report.rows_out += len(df)

        return report

    def clean_params(self) -> pd.DataFrame:
        """Quality-flag all Heston and Bates calibration JSON results.

        For each contract flags:
          - converged == False
          - feller_satisfied == False
          - Any parameter within 1% of its optimization bound (degenerate fit
            where the solver ran out of room — common when kappa→50, theta→10,
            v0→1e-6, or lambda_j→100)
          - n_obs < IV_MIN_OBS

        h_quality_ok / b_quality_ok are True only when all four checks pass.

        Saves the flag table to data/clean/params_quality.csv.

        Returns
        -------
        pd.DataFrame
            One row per contract with parameters and quality flags.
            Empty DataFrame if no calibration results are found.
        """
        heston_dir = self.data_dir / "processed" / "heston_params"
        bates_dir = self.data_dir / "processed" / "bates_params"

        if not heston_dir or not heston_dir.exists():
            return pd.DataFrame()

        heston_files = sorted(heston_dir.glob("*.json"))
        if not heston_files:
            return pd.DataFrame()

        rows = []
        for h_path in heston_files:
            try:
                h = json.loads(h_path.read_text())
            except Exception:
                continue

            ticker = h_path.stem
            b: dict | None = None
            if bates_dir and bates_dir.exists():
                b_path = bates_dir / f"{ticker}.json"
                if b_path.exists():
                    try:
                        b = json.loads(b_path.read_text())
                    except Exception:
                        pass

            rows.append(self._build_quality_row(ticker, h, b))

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.clean_dir / "params_quality.csv", index=False)
        return df

    # ── private helpers ───────────────────────────────────────────────────────

    def _clean_ohlcv(
        self, df: pd.DataFrame, stem: str, report: CleaningReport
    ) -> pd.DataFrame:
        """Apply row-level cleaning rules to one OHLCV DataFrame."""
        if "close" not in df.columns or "timestamp" not in df.columns:
            report.notes.append(f"{stem}: missing 'close' or 'timestamp' — skipped")
            return pd.DataFrame()

        n_orig = len(df)

        # 1. NaN close
        df = df.dropna(subset=["close"])

        # 2. prices outside open (0, 1) — terminal/invalid states
        df = df[(df["close"] > PRICE_STRICT_MIN) & (df["close"] < PRICE_STRICT_MAX)]

        # 3. duplicate timestamps: keep last (most recent trade in the bar wins)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")

        # 4. chronological order
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 5. flag stale price runs without removing them
        if len(df) > 1:
            run_id = (df["close"] != df["close"].shift()).cumsum()
            run_lengths = run_id.map(run_id.value_counts())
            stale_mask = run_lengths > STALE_RUN_MAX
            if stale_mask.any():
                df = df.copy()
                df["stale_price_run"] = stale_mask
                report.notes.append(
                    f"{stem}: {int(stale_mask.sum())} rows flagged in stale runs "
                    f"(>{STALE_RUN_MAX} consecutive identical close)"
                )

        n_dropped = n_orig - len(df)
        if n_dropped > 0:
            report.notes.append(f"{stem}: {n_dropped} rows removed")

        return df

    def _clean_iv_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply row-level cleaning to one implied vol DataFrame."""
        # 1. NaN sigma — IV solver couldn't find a solution
        df = df.dropna(subset=["sigma_implied"])

        # 2. at or after resolution (T=0 IV is undefined)
        if "T" in df.columns:
            df = df[df["T"] > 0.0]

        # 3. near-boundary prices (IV is extreme and numerically unstable)
        if self.drop_boundary and "near_boundary" in df.columns:
            df = df[~df["near_boundary"]]

        # 4. FOMC jump windows
        if self.drop_jump_windows and "is_jump_window" in df.columns:
            df = df[~df["is_jump_window"]]

        if df.empty:
            return df.reset_index(drop=True)

        # 5. clamp sigma and recompute sigma_logit = sigma / (p*(1-p))
        df = df.copy()
        df["sigma_implied"] = df["sigma_implied"].clip(upper=SIGMA_CAP)

        if "price" in df.columns:
            p = df["price"].values
            sigma = df["sigma_implied"].values
            valid = (p > 0) & (p < 1)
            df["sigma_logit"] = np.where(valid, sigma / (p * (1.0 - p)), np.nan)

        return df.reset_index(drop=True)

    def _build_quality_row(
        self, ticker: str, h: dict, b: dict | None
    ) -> dict:
        """Build one quality-flag row for a contract."""
        row: dict = {
            "ticker": ticker,
            "platform": h.get("platform", "unknown"),
        }

        # ── Heston ────────────────────────────────────────────────────────────
        h_bound_flags: list[str] = []
        for param, (lo, hi) in _HESTON_BOUNDS.items():
            val = float(h.get(param, np.nan))
            tol = _BOUND_TOL * (hi - lo)
            if np.isfinite(val) and (val <= lo + tol or val >= hi - tol):
                h_bound_flags.append(param)
            row[f"h_{param}"] = val

        row["h_converged"]       = bool(h.get("converged", False))
        row["h_feller_ok"]       = bool(h.get("feller_satisfied", False))
        row["h_n_obs"]           = int(h.get("n_obs", 0))
        row["h_mse"]             = float(h.get("mse", np.nan))
        row["h_log_likelihood"]  = float(h.get("log_likelihood", np.nan))
        row["h_params_at_bound"] = ", ".join(h_bound_flags)
        row["h_quality_ok"]      = bool(
            row["h_converged"]
            and row["h_feller_ok"]
            and row["h_n_obs"] >= IV_MIN_OBS
            and not h_bound_flags
        )

        # ── Bates ─────────────────────────────────────────────────────────────
        if b is not None:
            b_bound_flags: list[str] = []
            for param, (lo, hi) in {**_HESTON_BOUNDS, **_BATES_EXTRA_BOUNDS}.items():
                val = float(b.get(param, np.nan))
                tol = _BOUND_TOL * (hi - lo)
                if np.isfinite(val) and (val <= lo + tol or val >= hi - tol):
                    b_bound_flags.append(param)

            row["b_converged"]            = bool(b.get("converged", False))
            row["b_feller_ok"]            = bool(b.get("feller_satisfied", False))
            row["b_n_obs"]                = int(b.get("n_obs", 0))
            row["b_mse"]                  = float(b.get("mse", np.nan))
            row["b_log_likelihood"]       = float(b.get("log_likelihood", np.nan))
            row["b_lambda_j"]             = float(b.get("lambda_j", np.nan))
            row["b_mu_j"]                 = float(b.get("mu_j", np.nan))
            row["b_sigma_j"]              = float(b.get("sigma_j", np.nan))
            row["b_p_jump"]               = float(b.get("p_value_jump_significance", np.nan))
            row["b_params_at_bound"]      = ", ".join(b_bound_flags)
            row["b_quality_ok"]           = bool(
                row["b_converged"]
                and row["b_feller_ok"]
                and row["b_n_obs"] >= IV_MIN_OBS
                and not b_bound_flags
            )
        else:
            for key in [
                "b_converged", "b_feller_ok", "b_n_obs", "b_mse",
                "b_log_likelihood", "b_lambda_j", "b_mu_j", "b_sigma_j",
                "b_p_jump", "b_params_at_bound", "b_quality_ok",
            ]:
                row[key] = None

        return row

    def _print_params_report(self, df: pd.DataFrame) -> None:
        n = len(df)
        print("--- Calibration Quality ---")
        print(f"  Heston  quality OK :  {int(df['h_quality_ok'].sum()):>4} / {n}")
        print(f"          converged  :  {int(df['h_converged'].sum()):>4} / {n}")
        print(f"          Feller OK  :  {int(df['h_feller_ok'].sum()):>4} / {n}")
        print(f"          at boundary:  {int((df['h_params_at_bound'] != '').sum()):>4} / {n}")

        b_mask = df["b_quality_ok"].notna()
        if b_mask.any():
            nb = int(b_mask.sum())
            b_sub = df.loc[b_mask]
            sig = int((b_sub["b_p_jump"] < 0.05).sum())
            print(f"  Bates   quality OK :  {int(b_sub['b_quality_ok'].sum()):>4} / {nb}")
            print(f"          converged  :  {int(b_sub['b_converged'].sum()):>4} / {nb}")
            print(f"          Feller OK  :  {int(b_sub['b_feller_ok'].sum()):>4} / {nb}")
            print(f"          at boundary:  {int((b_sub['b_params_at_bound'] != '').sum()):>4} / {nb}")
            print(f"          jumps p<.05:  {sig:>4} / {nb}")

        print(f"  Saved: {self.clean_dir / 'params_quality.csv'}")
        print()
