#!/usr/bin/env python3
"""
Pipeline runner for prediction market research.

Usage:
    python run_pipeline.py <step>

Run 'python run_pipeline.py help' to see all available steps.
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def check():
    """Verify that your environment is set up correctly."""
    print("Checking your setup...\n")

    ok = True

    import platform
    print(f"  Python version: {platform.python_version()}")
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        print("  PROBLEM: You need Python 3.10 or newer.")
        print("  You're probably not in the 'pmr' conda environment.")
        print("  Run:  conda activate pmr")
        ok = False
    else:
        print("  OK")

    print()
    for pkg in ["pandas", "numpy", "scipy", "requests", "pyarrow", "tqdm"]:
        try:
            __import__(pkg)
            print(f"  {pkg}: installed")
        except ImportError:
            print(f"  {pkg}: MISSING — run: pip install -r requirements.txt")
            ok = False

    print()
    import config
    if config.KALSHI_KEY_FILE:
        print(f"  Kalshi RSA key file: {config.KALSHI_KEY_FILE}")
        print(f"  Kalshi key ID: {config.KALSHI_KEY_ID or '(not set — also needed)'}")
    elif config.KALSHI_API_KEY:
        print(f"  Kalshi API key (Bearer): set ({config.KALSHI_API_KEY[:8]}...)")
    else:
        print("  Kalshi auth: not set — set KALSHI_KEY_FILE + KALSHI_KEY_ID")
        print("    (Download .pem from kalshi.com → Settings → API Keys)")
        print("    export KALSHI_KEY_FILE=/path/to/kalshi-api-key.pem")
        print("    export KALSHI_KEY_ID=<your-key-id>")

    if config.POLYMARKET_API_KEY:
        print(f"  Polymarket API key: set ({config.POLYMARKET_API_KEY[:8]}...)")
    else:
        print("  Polymarket API key: not set (OK — Gamma API is public, no key needed)")

    print()
    from pathlib import Path
    for d in ["data/raw/kalshi", "data/raw/polymarket",
              "data/processed/implied_vol", "data/processed/heston_params",
              "data/processed/bates_params", "data/processed/figures"]:
        p = Path(d)
        if p.exists():
            print(f"  {d}/ exists")
        else:
            p.mkdir(parents=True, exist_ok=True)
            print(f"  {d}/ created")

    print()
    if ok:
        print("Everything looks good. You're ready to run step 1.")
        print("  Next: python run_pipeline.py catalog-kalshi")
    else:
        print("Fix the issues above before continuing.")


def catalog_kalshi():
    """Pull the Kalshi market catalog for FOMC/CPI/economic events and apply filters.

    Targets FOMC rate decisions, CPI releases, PCE, GDP, and jobs reports.
    Political/election markets are excluded.

    Set your API credentials before running:
      export KALSHI_KEY_FILE=/path/to/kalshi-api-key.pem
      export KALSHI_KEY_ID=<your-key-id>
    """
    import config
    from src.kalshi.client import KalshiClient
    from src.kalshi.catalog import KalshiCatalog

    print(f"Targeting date range: {config.KALSHI_DATE_START} to {config.KALSHI_DATE_END}")
    print("Focus: FOMC, CPI, PCE, GDP, Jobs — political markets excluded\n")

    client = KalshiClient(
        api_key=config.KALSHI_API_KEY,
        key_id=config.KALSHI_KEY_ID,
        key_file=config.KALSHI_KEY_FILE,
    )
    catalog = KalshiCatalog(client)
    df = catalog.pull_catalog()

    print(f"\nDone. {len(df)} economic contracts passed keyword and duration filters.")
    print()
    print("Files created:")
    print("  data/raw/kalshi/catalog_full.parquet      (all pulled markets)")
    print("  data/raw/kalshi/catalog_filtered.parquet  (after applying filters)")
    print()
    print("Next step:")
    print("  python run_pipeline.py catalog-polymarket")


def catalog_polymarket():
    """Pull the full Polymarket market catalog, apply ALL filters, and save."""
    from src.polymarket.gamma_client import GammaClient
    from src.polymarket.catalog import PolymarketCatalog

    print("Connecting to Polymarket Gamma API (no API key needed)...")
    print("This may take several minutes — the API returns 20 markets per page.\n")

    client = GammaClient()
    catalog = PolymarketCatalog(client)
    df = catalog.pull_catalog()

    print(f"\nDone. {len(df)} contracts passed ALL filters (including liquidity).")
    print()
    print("Files created:")
    print("  data/raw/polymarket/catalog_full.parquet      (every resolved market)")
    print("  data/raw/polymarket/catalog_filtered.parquet   (after all filters)")
    print("  data/raw/polymarket/contracts_for_review.csv   (open in Excel to review)")
    print()
    print("NOTE: The Gamma API includes trade volume in every record, so the")
    print("liquidity filter already ran — no separate liquidity step needed.")
    print()
    print("Next step:")
    print("  python run_pipeline.py liquidity-kalshi")


def liquidity_kalshi():
    """Apply volume-based liquidity filter to Kalshi catalog."""
    import pandas as pd
    import config
    from src.kalshi.client import KalshiClient
    from src.kalshi.catalog import KalshiCatalog

    catalog_path = config.DATA_DIR / "raw" / "kalshi" / "catalog_filtered.parquet"
    if not catalog_path.exists():
        print("ERROR: No filtered catalog found. Run this first:")
        print("  python run_pipeline.py catalog-kalshi")
        return

    print("Loading filtered Kalshi catalog...")
    df = pd.read_parquet(catalog_path)
    print(f"  {len(df)} contracts to check.\n")

    print("Filtering by volume (volume_fp >= 500).")
    print("No per-contract API calls needed — volume is included from the event pull.\n")

    client = KalshiClient(
        api_key=config.KALSHI_API_KEY,
        key_id=config.KALSHI_KEY_ID,
        key_file=config.KALSHI_KEY_FILE,
    )
    catalog = KalshiCatalog(client)
    df = catalog.apply_liquidity_filter(df)

    print(f"\nDone. {len(df)} Kalshi contracts passed the liquidity filter.")
    print()
    print("Files created:")
    print("  data/raw/kalshi/contracts_for_review.csv  (open this in Excel to review)")
    print("  data/raw/kalshi/catalog_filtered.parquet   (updated)")
    print()
    print("=== PAUSE HERE ===")
    print("Open data/raw/kalshi/contracts_for_review.csv in Excel and review titles.")
    print()
    print("Then run:")
    print("  python run_pipeline.py trades-kalshi")
    print("  python run_pipeline.py trades-polymarket")


def liquidity_polymarket():
    """Polymarket liquidity is now handled in the catalog step (Gamma API)."""
    print("This step is no longer needed.")
    print()
    print("The Gamma API includes trade volume in every market record, so the")
    print("liquidity filter already ran during 'catalog-polymarket'.")
    print()
    print("Your review CSV is already at:")
    print("  data/raw/polymarket/contracts_for_review.csv")
    print()
    print("If you've already run 'liquidity-kalshi', your next steps are:")
    print()
    print("  1. Open both contracts_for_review.csv files in Excel")
    print("     (data/raw/kalshi/ and data/raw/polymarket/)")
    print("  2. Review the contract titles — remove any that:")
    print("     - Were misclassified by the keyword filter")
    print("     - Had UMA oracle disputes (Polymarket, check has_uma_dispute column)")
    print("     - Don't belong in your FED/FOMC or Political categories")
    print("  3. Then run:")
    print("     python run_pipeline.py trades-kalshi")
    print("     python run_pipeline.py trades-polymarket")


def trades_kalshi():
    """Download hourly trade data for each selected Kalshi contract."""
    import pandas as pd
    import config
    from src.kalshi.client import KalshiClient
    from src.kalshi.trades import KalshiTradesPuller

    catalog_path = config.DATA_DIR / "raw" / "kalshi" / "catalog_filtered.parquet"
    if not catalog_path.exists():
        print("ERROR: No filtered catalog found. Run the catalog and liquidity steps first.")
        return

    df = pd.read_parquet(catalog_path)
    tickers = df["ticker"].tolist()
    print(f"Pulling hourly trade data for {len(tickers)} Kalshi contracts...")
    print("Each contract gets its own file. Already-downloaded contracts are skipped.")
    print("This is the longest step — could take 30-60 minutes.\n")

    client = KalshiClient(
        api_key=config.KALSHI_API_KEY,
        key_id=config.KALSHI_KEY_ID,
        key_file=config.KALSHI_KEY_FILE,
    )
    puller = KalshiTradesPuller(client)
    puller.pull(tickers)

    print("Files created:")
    print("  data/raw/kalshi/<ticker>.parquet  (one file per contract, hourly OHLCV bars)")
    print()
    print("Next step:")
    print("  python run_pipeline.py trades-polymarket")


def _build_contracts_list(df: "pd.DataFrame", id_col: str) -> list[dict]:
    """Build the list of {condition_id, yes_token} dicts required by PolymarketTradesPuller.

    The CLOB /prices-history endpoint requires the YES-outcome token ID, not
    the conditionId. We extract it from the clobTokenIds column (first entry).
    Contracts with no parseable token are silently skipped.
    """
    import ast
    contracts = []
    for _, row in df.iterrows():
        cid = str(row[id_col])
        raw = row.get("clobTokenIds", "[]")
        try:
            token_ids = ast.literal_eval(raw) if isinstance(raw, str) else raw
            yes_token = str(token_ids[0]) if token_ids else None
        except (ValueError, SyntaxError, IndexError):
            yes_token = None
        if yes_token:
            contracts.append({"condition_id": cid, "yes_token": yes_token})
    return contracts


def trades_sample():
    """Download daily price history for a stratified 100-contract sample.

    Pulls ALL 43 FED/FOMC contracts (most relevant for jump modelling) plus
    the top 57 political contracts ranked by trade volume. Uses the public
    CLOB /prices-history endpoint (daily bars, no API key required).
    Takes ~5-10 minutes.
    """
    import pandas as pd
    import config
    from src.polymarket.client import PolymarketClient
    from src.polymarket.trades import PolymarketTradesPuller

    catalog_path = config.DATA_DIR / "raw" / "polymarket" / "catalog_filtered.parquet"
    if not catalog_path.exists():
        print("ERROR: No filtered catalog found. Run catalog-polymarket first.")
        return

    df = pd.read_parquet(catalog_path)

    id_col = None
    for col in ["conditionId", "condition_id", "id"]:
        if col in df.columns:
            id_col = col
            break

    if id_col is None:
        print("ERROR: No condition ID column found in the filtered catalog.")
        return

    # Stratified sample: all FED contracts + top political by volume
    fed = df[df["is_fed"] == True].copy()
    political = df[df["is_political"] == True].copy()

    n_political = max(0, 100 - len(fed))
    if "trade_count" in political.columns:
        political = political.nlargest(n_political, "trade_count")
    else:
        political = political.head(n_political)

    sample = pd.concat([fed, political], ignore_index=True)
    contracts = _build_contracts_list(sample, id_col)

    print(f"Stratified sample: {len(fed)} FED + {len(political)} political = {len(sample)} selected")
    print(f"  ({len(contracts)} have YES tokens — contracts without tokens are skipped)")
    print("Each contract gets its own file. Already-downloaded contracts are skipped.")
    print("Using CLOB /prices-history (public, daily bars, no API key needed).")
    print("Estimated time: 5-10 minutes.\n")

    api_key = config.POLYMARKET_API_KEY or ""
    client = PolymarketClient(api_key)
    puller = PolymarketTradesPuller(client)
    puller.pull(contracts)

    print("Files created:")
    print("  data/raw/polymarket/<conditionId>.parquet  (one file per contract, daily bars)")
    print()
    print("Note: 4 early-2022 FED contracts have no CLOB data (pre-CLOB era)")
    print("and will show as failures — this is expected.")
    print()
    print("Next step:")
    print("  python run_pipeline.py backtest-implied-vol")


def trades_polymarket():
    """Download daily price history for all 743 selected Polymarket contracts."""
    import pandas as pd
    import config
    from src.polymarket.client import PolymarketClient
    from src.polymarket.trades import PolymarketTradesPuller

    catalog_path = config.DATA_DIR / "raw" / "polymarket" / "catalog_filtered.parquet"
    if not catalog_path.exists():
        print("ERROR: No filtered catalog found. Run catalog-polymarket first.")
        return

    df = pd.read_parquet(catalog_path)

    id_col = None
    for col in ["conditionId", "condition_id", "id"]:
        if col in df.columns:
            id_col = col
            break

    if id_col is None:
        print("ERROR: No condition ID column found in the filtered catalog.")
        return

    contracts = _build_contracts_list(df, id_col)
    print(f"Pulling daily price history for {len(contracts)} Polymarket contracts...")
    print("Each contract gets its own file. Already-downloaded contracts are skipped.")
    print("Using CLOB /prices-history (public, daily bars, no API key needed).")
    print("Estimated time: 15-30 minutes.\n")

    api_key = config.POLYMARKET_API_KEY or ""
    client = PolymarketClient(api_key)
    puller = PolymarketTradesPuller(client)
    puller.pull(contracts)

    print("Files created:")
    print("  data/raw/polymarket/<conditionId>.parquet  (one file per contract, daily bars)")
    print()
    print("Dataset selection is complete. Your frozen datasets are in data/raw/.")
    print("Next phase is backtesting (implied vol, Heston, Bates).")


def backtest_implied_vol():
    """Extract Cash-or-Nothing implied volatility from all trade data."""
    import pandas as pd
    import config
    from src.models.implied_vol import ImpliedVolExtractor

    fomc_dates = [pd.Timestamp(d) for d in config.FOMC_DATES]
    processed = 0
    skipped = 0
    errors = 0

    for platform in ["kalshi", "polymarket"]:
        raw_dir = config.DATA_DIR / "raw" / platform
        catalog_path = raw_dir / "catalog_filtered.parquet"

        if not catalog_path.exists():
            print(f"  No catalog found for {platform} — skipping.")
            continue

        catalog = pd.read_parquet(catalog_path)

        # Determine the ID column and close time column
        if platform == "kalshi":
            id_col = "ticker"
            close_col = "close_time"
        else:
            id_col = None
            for col in ["conditionId", "condition_id", "id"]:
                if col in catalog.columns:
                    id_col = col
                    break
            close_col = "close_time"

        if id_col is None:
            print(f"  No ID column found for {platform} — skipping.")
            continue

        print(f"\n  Processing {platform} ({len(catalog)} contracts)...")

        for _, row in catalog.iterrows():
            ticker = str(row[id_col])
            trade_path = raw_dir / f"{ticker}.parquet"

            if not trade_path.exists():
                skipped += 1
                continue

            # Check if already processed
            out_path = config.DATA_DIR / "processed" / "implied_vol" / f"{ticker}.parquet"
            if out_path.exists():
                processed += 1
                continue

            try:
                _ct = pd.Timestamp(row[close_col])
                close_time = _ct.tz_localize("UTC") if _ct.tzinfo is None else _ct.tz_convert("UTC")
                df = pd.read_parquet(trade_path)

                extractor = ImpliedVolExtractor(
                    close_time=close_time,
                    fomc_dates=fomc_dates,
                )
                extractor.extract(df, ticker)
                processed += 1
            except Exception as e:
                print(f"    Error on {ticker}: {e}")
                errors += 1

    print(f"\n--- Implied Vol Extraction Summary ---")
    print(f"  Processed:  {processed}")
    print(f"  Skipped:    {skipped} (no trade data file)")
    print(f"  Errors:     {errors}")
    print(f"-------------------------------------\n")
    print("Files created:")
    print("  data/processed/implied_vol/<ticker>.parquet  (one per contract)")
    print()
    print("Next step:")
    print("  python run_pipeline.py backtest-bs")


def backtest_bs():
    """Calibrate the Black-Scholes Cash-or-Nothing model on all contracts.

    Finds the single constant sigma_BS per contract that best fits the
    implied volatility time series. Saves results to
    data/processed/bs_params/<ticker>.json — same format as Heston/Bates.
    """
    import pandas as pd
    import config
    from src.models.black_scholes import BSCalibrator

    calibrator = BSCalibrator()
    iv_dir = config.DATA_DIR / "processed" / "implied_vol"

    if not iv_dir.exists() or not any(iv_dir.glob("*.parquet")):
        print("ERROR: No implied vol data found. Run this first:")
        print("  python run_pipeline.py backtest-implied-vol")
        return

    # Build ticker -> platform lookup from catalogs
    platform_map = {}
    for platform in ["kalshi", "polymarket"]:
        cat_path = config.DATA_DIR / "raw" / platform / "catalog_filtered.parquet"
        if cat_path.exists():
            cat = pd.read_parquet(cat_path)
            if platform == "kalshi":
                for t in cat.get("ticker", []):
                    platform_map[str(t)] = "kalshi"
            else:
                for col in ["conditionId", "condition_id", "id"]:
                    if col in cat.columns:
                        for t in cat[col]:
                            platform_map[str(t)] = "polymarket"
                        break

    import numpy as np
    iv_files = sorted(iv_dir.glob("*.parquet"))
    print(f"Calibrating Black-Scholes model on {len(iv_files)} contracts...")
    print("Finding optimal constant sigma_BS per contract (k=1 parameter)\n")

    calibrated = 0
    errors = 0

    for iv_path in iv_files:
        ticker = iv_path.stem
        out_path = config.DATA_DIR / "processed" / "bs_params" / f"{ticker}.json"
        if out_path.exists():
            calibrated += 1
            continue

        try:
            df = pd.read_parquet(iv_path)
            sigma = df["sigma_implied"].values
            platform = platform_map.get(ticker, "unknown")
            result = calibrator.calibrate(sigma, ticker, platform)
            calibrated += 1
            print(f"  {ticker[:20]}...: sigma_BS={result.sigma_bs:.4f}, "
                  f"mse={result.mse:.6f}, n={result.n_obs}")
        except Exception as e:
            print(f"  Error on {ticker}: {e}")
            errors += 1

    print(f"\n--- Black-Scholes Calibration Summary ---")
    print(f"  Calibrated:  {calibrated}")
    print(f"  Errors:      {errors}")
    print(f"-----------------------------------------\n")
    print("Files created:")
    print("  data/processed/bs_params/<ticker>.json  (one per contract)")
    print()
    print("Next step:")
    print("  python run_pipeline.py backtest-heston")


def backtest_heston():
    """Calibrate the Heston stochastic volatility model on all contracts."""
    import numpy as np
    import pandas as pd
    import config
    from src.models.heston import HestonCalibrator

    calibrator = HestonCalibrator()
    iv_dir = config.DATA_DIR / "processed" / "implied_vol"

    if not iv_dir.exists() or not any(iv_dir.glob("*.parquet")):
        print("ERROR: No implied vol data found. Run this first:")
        print("  python run_pipeline.py backtest-implied-vol")
        return

    # Build a lookup of ticker -> platform from catalogs
    platform_map = {}
    for platform in ["kalshi", "polymarket"]:
        cat_path = config.DATA_DIR / "raw" / platform / "catalog_filtered.parquet"
        if cat_path.exists():
            cat = pd.read_parquet(cat_path)
            if platform == "kalshi":
                for t in cat.get("ticker", []):
                    platform_map[str(t)] = "kalshi"
            else:
                for col in ["conditionId", "condition_id", "id"]:
                    if col in cat.columns:
                        for t in cat[col]:
                            platform_map[str(t)] = "polymarket"
                        break

    iv_files = sorted(iv_dir.glob("*.parquet"))
    print(f"Calibrating Heston model on {len(iv_files)} contracts...")
    print(f"Using daily time step (dt = {config.DAILY_DT:.6e})\n")

    calibrated = 0
    skipped = 0
    errors = 0

    for iv_path in iv_files:
        ticker = iv_path.stem

        # Skip if already calibrated
        out_path = config.DATA_DIR / "processed" / "heston_params" / f"{ticker}.json"
        if out_path.exists():
            calibrated += 1
            continue

        try:
            df = pd.read_parquet(iv_path)
            sigma = df["sigma_implied"].values

            platform = platform_map.get(ticker, "unknown")
            result = calibrator.calibrate(
                sigma, ticker, platform, dt=config.DAILY_DT
            )

            if result.converged:
                calibrated += 1
                print(f"  {ticker}: kappa={result.kappa:.3f}, theta={result.theta:.4f}, "
                      f"xi={result.xi:.3f}, rho={result.rho:.3f} "
                      f"{'(Feller OK)' if result.feller_satisfied else '(Feller VIOLATED)'}")
            else:
                calibrated += 1
                print(f"  {ticker}: did not converge (still saved)")
        except Exception as e:
            print(f"  Error on {ticker}: {e}")
            errors += 1

    print(f"\n--- Heston Calibration Summary ---")
    print(f"  Calibrated:  {calibrated}")
    print(f"  Errors:      {errors}")
    print(f"----------------------------------\n")
    print("Files created:")
    print("  data/processed/heston_params/<ticker>.json  (one per contract)")
    print()
    print("Next step:")
    print("  python run_pipeline.py backtest-bates")


def backtest_bates():
    """Calibrate the Bates model (Heston + jumps) on all contracts."""
    import json
    import numpy as np
    import pandas as pd
    import config
    from src.models.heston import HestonResult
    from src.models.bates import BatesCalibrator

    calibrator = BatesCalibrator()
    iv_dir = config.DATA_DIR / "processed" / "implied_vol"
    heston_dir = config.DATA_DIR / "processed" / "heston_params"

    if not heston_dir.exists() or not any(heston_dir.glob("*.json")):
        print("ERROR: No Heston results found. Run this first:")
        print("  python run_pipeline.py backtest-heston")
        return

    # Build platform lookup
    platform_map = {}
    for platform in ["kalshi", "polymarket"]:
        cat_path = config.DATA_DIR / "raw" / platform / "catalog_filtered.parquet"
        if cat_path.exists():
            cat = pd.read_parquet(cat_path)
            if platform == "kalshi":
                for t in cat.get("ticker", []):
                    platform_map[str(t)] = "kalshi"
            else:
                for col in ["conditionId", "condition_id", "id"]:
                    if col in cat.columns:
                        for t in cat[col]:
                            platform_map[str(t)] = "polymarket"
                        break

    heston_files = sorted(heston_dir.glob("*.json"))
    print(f"Calibrating Bates model on {len(heston_files)} contracts...")
    print(f"Testing jump significance (chi-squared, 3 df)\n")

    calibrated = 0
    jumps_significant = 0
    errors = 0

    for h_path in heston_files:
        ticker = h_path.stem

        # Skip if already calibrated
        out_path = config.DATA_DIR / "processed" / "bates_params" / f"{ticker}.json"
        if out_path.exists():
            with open(out_path) as f:
                existing = json.load(f)
            calibrated += 1
            if existing.get("p_value_jump_significance", 1.0) < 0.05:
                jumps_significant += 1
            continue

        iv_path = iv_dir / f"{ticker}.parquet"
        if not iv_path.exists():
            continue

        try:
            df = pd.read_parquet(iv_path)
            sigma = df["sigma_implied"].values
            platform = platform_map.get(ticker, "unknown")

            # Load the Heston result for warm-starting
            with open(h_path) as f:
                h_data = json.load(f)
            heston_result = HestonResult(**h_data)

            result = calibrator.calibrate(
                sigma, ticker, platform,
                dt=config.DAILY_DT,
                heston_result=heston_result,
            )

            calibrated += 1
            sig = "YES" if result.p_value_jump_significance < 0.05 else "no"
            if result.p_value_jump_significance < 0.05:
                jumps_significant += 1
            print(f"  {ticker}: lambda={result.lambda_j:.2f}, mu_j={result.mu_j:.3f}, "
                  f"sigma_j={result.sigma_j:.3f}, jumps significant={sig} "
                  f"(p={result.p_value_jump_significance:.4f})")
        except Exception as e:
            print(f"  Error on {ticker}: {e}")
            errors += 1

    print(f"\n--- Bates Calibration Summary ---")
    print(f"  Calibrated:         {calibrated}")
    print(f"  Jumps significant:  {jumps_significant} / {calibrated} "
          f"({100*jumps_significant/max(calibrated,1):.0f}%)")
    print(f"  Errors:             {errors}")
    print(f"---------------------------------\n")
    print("Files created:")
    print("  data/processed/bates_params/<ticker>.json  (one per contract)")
    print()
    print("Next step:")
    print("  python run_pipeline.py compare")


def clean():
    """Clean raw and processed datasets and write results to data/clean/.

    Three stages:
      1. Trade OHLCV (Polymarket + Kalshi): drop NaN/out-of-range prices,
         duplicates; flag stale price runs.
      2. Implied vol: drop NaN sigma, T<=0, near-boundary, and extreme sigma
         rows; discard contracts with fewer than 10 clean observations.
      3. Calibration params: quality-flag Heston/Bates results for
         non-convergence, Feller violations, and boundary-stuck parameters.

    Original files in data/raw/ and data/processed/ are never modified.
    Clean data lands in data/clean/ mirroring the existing structure.
    """
    from src.cleaning import DataCleaner

    cleaner = DataCleaner()
    cleaner.run()

    print("Files created:")
    print("  data/clean/polymarket/        — cleaned Polymarket OHLCV parquets")
    print("  data/clean/kalshi/            — cleaned Kalshi OHLCV parquets")
    print("  data/clean/implied_vol/       — cleaned IV parquets")
    print("  data/clean/params_quality.csv — per-contract calibration quality flags")


def compare():
    """Run model comparison across all calibrated contracts."""
    from analysis.model_comparison import ModelComparison

    print("Comparing Heston vs Bates across all contracts...")
    print("Computing MSE, QLIKE, AIC, BIC, and likelihood ratio tests.\n")

    mc = ModelComparison()
    summary = mc.run()

    if summary.empty:
        print("No results to compare. Run the backtesting steps first.")
        return

    # Print key headline numbers
    print(f"\n--- Headline Results ---")
    if "mse_heston" in summary.columns and "mse_bates" in summary.columns:
        h_wins = (summary["mse_heston"] < summary["mse_bates"]).sum()
        b_wins = (summary["mse_bates"] < summary["mse_heston"]).sum()
        print(f"  MSE: Heston wins {h_wins}, Bates wins {b_wins}")

    if "p_value_jump" in summary.columns:
        sig = (summary["p_value_jump"] < 0.05).sum()
        print(f"  Jump significance (p<0.05): {sig} / {len(summary)} contracts")

    print()
    print("Files created:")
    print("  data/processed/model_comparison_summary.parquet")
    print()
    print("Next step:")
    print("  python run_pipeline.py plots")


def plots():
    """Generate all research figures."""
    import json
    import pandas as pd
    import config
    from analysis.plots import (
        plot_implied_vol_series,
        plot_heston_fit,
        plot_parameter_distribution,
        plot_model_comparison_table,
    )

    iv_dir = config.DATA_DIR / "processed" / "implied_vol"
    heston_dir = config.DATA_DIR / "processed" / "heston_params"

    # 1. IV series plots (first 5 contracts per platform as samples)
    print("Generating implied volatility time series plots...")
    iv_files = sorted(iv_dir.glob("*.parquet"))
    for iv_path in iv_files[:10]:
        try:
            plot_implied_vol_series(iv_path.stem)
        except Exception as e:
            print(f"  Skipped {iv_path.stem}: {e}")

    # 2. Heston fit plots (first 5 as samples)
    print("\nGenerating Heston model fit plots...")
    heston_files = sorted(heston_dir.glob("*.json"))
    for h_path in heston_files[:10]:
        try:
            plot_heston_fit(h_path.stem)
        except Exception as e:
            print(f"  Skipped {h_path.stem}: {e}")

    # 3. Parameter distributions
    print("\nGenerating parameter distribution histograms...")
    for param in ["kappa", "theta", "xi", "rho", "v0"]:
        try:
            plot_parameter_distribution(param)
        except Exception as e:
            print(f"  Skipped {param}: {e}")

    # 4. Model comparison heatmap
    print("\nGenerating model comparison heatmap...")
    summary_path = config.DATA_DIR / "processed" / "model_comparison_summary.parquet"
    if summary_path.exists():
        try:
            plot_model_comparison_table()
        except Exception as e:
            print(f"  Skipped heatmap: {e}")
    else:
        print("  No comparison summary found — run 'compare' first.")

    print("\nAll figures saved to:")
    print("  data/processed/figures/")
    print()
    print("Pipeline complete.")


def export():
    """Export all processed results to CSV files you can open in Excel."""
    import json
    import pandas as pd
    import config

    out_dir = config.DATA_DIR / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    exported = []

    # 1. Implied vol — one combined CSV across all contracts
    iv_dir = config.DATA_DIR / "processed" / "implied_vol"
    iv_files = sorted(iv_dir.glob("*.parquet")) if iv_dir.exists() else []
    if iv_files:
        chunks = []
        for f in iv_files:
            df = pd.read_parquet(f)
            df.insert(0, "ticker", f.stem)
            chunks.append(df)
        combined = pd.concat(chunks, ignore_index=True)
        # Make timestamps plain strings so Excel doesn't complain
        combined["timestamp"] = combined["timestamp"].astype(str)
        path = out_dir / "implied_vol_all.csv"
        combined.to_csv(path, index=False)
        print(f"  implied_vol_all.csv   — {len(combined):,} rows × {len(combined.columns)} cols")
        exported.append(path)
    else:
        print("  implied_vol: nothing yet (run backtest-implied-vol first)")

    # 2. Heston params — one row per contract
    heston_dir = config.DATA_DIR / "processed" / "heston_params"
    heston_files = sorted(heston_dir.glob("*.json")) if heston_dir.exists() else []
    if heston_files:
        rows = []
        for f in heston_files:
            with open(f) as fh:
                d = json.load(fh)
            d["ticker"] = f.stem
            rows.append(d)
        hdf = pd.DataFrame(rows)
        cols = ["ticker"] + [c for c in hdf.columns if c != "ticker"]
        path = out_dir / "heston_params.csv"
        hdf[cols].to_csv(path, index=False)
        print(f"  heston_params.csv     — {len(hdf)} contracts")
        exported.append(path)
    else:
        print("  heston_params: nothing yet (run backtest-heston first)")

    # 3. Bates params — one row per contract
    bates_dir = config.DATA_DIR / "processed" / "bates_params"
    bates_files = sorted(bates_dir.glob("*.json")) if bates_dir.exists() else []
    if bates_files:
        rows = []
        for f in bates_files:
            with open(f) as fh:
                d = json.load(fh)
            d["ticker"] = f.stem
            rows.append(d)
        bdf = pd.DataFrame(rows)
        cols = ["ticker"] + [c for c in bdf.columns if c != "ticker"]
        path = out_dir / "bates_params.csv"
        bdf[cols].to_csv(path, index=False)
        print(f"  bates_params.csv      — {len(bdf)} contracts")
        exported.append(path)
    else:
        print("  bates_params: nothing yet (run backtest-bates first)")

    # 4. Model comparison summary
    summary_path = config.DATA_DIR / "processed" / "model_comparison_summary.parquet"
    if summary_path.exists():
        sdf = pd.read_parquet(summary_path)
        path = out_dir / "model_comparison.csv"
        sdf.to_csv(path, index=False)
        print(f"  model_comparison.csv  — {len(sdf)} contracts")
        exported.append(path)
    else:
        print("  model_comparison: nothing yet (run compare first)")

    # 5. Raw trade data — one CSV per contract in exports/trades/
    trades_dir = config.DATA_DIR / "raw" / "polymarket"
    trade_files = [f for f in trades_dir.glob("*.parquet")
                   if f.stem not in ("catalog_full", "catalog_filtered")]
    if trade_files:
        trades_out = out_dir / "trades"
        trades_out.mkdir(exist_ok=True)
        for f in trade_files:
            df = pd.read_parquet(f)
            df["timestamp"] = df["timestamp"].astype(str)
            df.to_csv(trades_out / f"{f.stem}.csv", index=False)
        print(f"  trades/               — {len(trade_files)} individual CSVs")
        exported.append(trades_out)
    else:
        print("  trades: nothing yet (run trades-sample first)")

    print()
    if exported:
        print(f"All exports saved to:  data/exports/")
        print(f"Open that folder in Finder to access the files.")


STEPS = {
    "check": (check, "Verify your environment is set up correctly"),
    "catalog-kalshi": (catalog_kalshi, "Pull and filter the Kalshi market catalog"),
    "catalog-polymarket": (catalog_polymarket, "Pull and filter Polymarket (Gamma API, no key needed)"),
    "liquidity-kalshi": (liquidity_kalshi, "Apply trade-count filter to Kalshi"),
    "liquidity-polymarket": (liquidity_polymarket, "Already done in catalog-polymarket (info only)"),
    "trades-kalshi": (trades_kalshi, "Pull hourly trade data for Kalshi contracts"),
    "trades-sample": (trades_sample, "Pull trades for 100-contract sample (43 FED + 57 top political)"),
    "trades-polymarket": (trades_polymarket, "Pull hourly trade data for all 743 Polymarket contracts"),
    "backtest-implied-vol": (backtest_implied_vol, "Extract implied volatility from trade data (Layer 1)"),
    "backtest-bs": (backtest_bs, "Calibrate Black-Scholes Cash-or-Nothing baseline (Layer 2)"),
    "backtest-heston": (backtest_heston, "Calibrate Heston stochastic vol model (Layer 3)"),
    "backtest-bates": (backtest_bates, "Calibrate Bates model with jumps (Layer 3)"),
    "compare": (compare, "Compare Heston vs Bates across all contracts"),
    "plots": (plots, "Generate all research figures"),
    "export": (export, "Export all results to CSV (open in Excel)"),
    "clean": (clean, "Clean datasets and write quality flags to data/clean/"),
}


def print_help():
    print("Prediction Market Research — Full Pipeline Runner")
    print("=" * 55)
    print()
    print("Usage:  python run_pipeline.py <step>")
    print()
    print("PHASE 1: Data Collection")
    print("-" * 40)
    print(f"  1.  {'check':28s} Verify environment setup")
    print(f"  2.  {'catalog-kalshi':28s} Pull + filter Kalshi catalog")
    print(f"  3.  {'catalog-polymarket':28s} Pull + filter Polymarket (no key needed)")
    print(f"  4.  {'liquidity-kalshi':28s} Trade-count filter for Kalshi")
    print(f"      (Polymarket liquidity handled in step 3)")
    print(f"      --- PAUSE: review contracts_for_review.csv in Excel ---")
    print(f"  5.  {'trades-kalshi':28s} Pull hourly Kalshi trade data")
    print(f"  6a. {'trades-sample':28s} Pull 100-contract sample (FAST: 5-10 min, good for paper)")
    print(f"  6b. {'trades-polymarket':28s} Pull all 743 Polymarket contracts (SLOW: 30-60 min)")
    print()
    print("PHASE 2: Backtesting")
    print("-" * 40)
    print(f"  7.  {'backtest-implied-vol':28s} Extract IV from prices (Cash-or-Nothing formula)")
    print(f"  8.  {'backtest-bs':28s} Calibrate Black-Scholes baseline (k=1 param)")
    print(f"  9.  {'backtest-heston':28s} Calibrate Heston model (k=5 params)")
    print(f"  10. {'backtest-bates':28s} Calibrate Bates model (k=8 params: Heston + jumps)")
    print()
    print("PHASE 3: Results")
    print("-" * 40)
    print(f"  10. {'compare':28s} Compare Heston vs Bates (MSE, AIC, BIC, QLIKE)")
    print(f"  11. {'plots':28s} Generate all figures for the paper")
    print(f"  12. {'export':28s} Export everything to CSV (open in Excel)")
    print(f"  13. {'clean':28s} Clean datasets, flag degenerate fits → data/clean/")
    print()
    print("Start with:  python run_pipeline.py check")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("help", "--help", "-h"):
        print_help()
        sys.exit(0)

    step = sys.argv[1]
    if step not in STEPS:
        print(f"Unknown step: '{step}'")
        print(f"Run 'python run_pipeline.py help' to see available steps.")
        sys.exit(1)

    fn, desc = STEPS[step]
    print(f"=== {desc} ===\n")
    fn()
