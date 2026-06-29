#!/usr/bin/env python3
"""
Section 4 SMM pipeline — orchestrator.

Steps (run in order):
  panel          §4.1  Build the unified long panel (contract_id, t, p, X, delta_X)
  validate-syn   §4.2  Synthetic validation (must pass before real data)
  stylized       §4.2  Five stylized-fact diagnostics on the real panel
  calibrate      §4.3  Bates SMM calibration across κ grid
  ladder         §4.4  Nested model ladder + difference-in-J selection tests
  validate-model §4.7  Simulate selected model and overlay stylized facts
  split          §4.8  FOMC-vs-CPI robustness split

Usage
-----
  python run_smm.py panel
  python run_smm.py validate-syn
  python run_smm.py stylized
  python run_smm.py ladder
  python run_smm.py validate-model          # uses Bates as selected model by default

Environment variables (optional — only needed if pulling new data):
  KALSHI_KEY_ID   API key ID from the Kalshi dashboard
  KALSHI_KEY_FILE Path to the RSA .pem private key
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def _make_client():
    """Build a KalshiClient if credentials are set, else return None."""
    import config
    from src.kalshi.client import KalshiClient

    if config.KALSHI_KEY_FILE and config.KALSHI_KEY_ID:
        try:
            return KalshiClient(
                key_id=config.KALSHI_KEY_ID,
                key_file=config.KALSHI_KEY_FILE,
            )
        except Exception as e:
            print(f"  WARNING: Could not build KalshiClient: {e}")
    elif config.KALSHI_API_KEY:
        return KalshiClient(api_key=config.KALSHI_API_KEY)

    print("  No Kalshi credentials set — using existing raw files only.")
    return None


def step_panel(force: bool = False) -> None:
    """§4.1  Build the SMM panel."""
    print("\n=== Step: panel ===")
    from src.smm.panel import SMMPanelBuilder

    client = _make_client()
    builder = SMMPanelBuilder(client=client)
    panel = builder.build(force=force)
    print(f"\nPanel ready: {panel['contract_id'].nunique()} contracts, "
          f"{len(panel)} rows")
    print("  Columns:", list(panel.columns))
    print("  Preview:")
    print(panel.head(8).to_string(index=False))


def step_validate_synthetic() -> None:
    """§4.2  Validate diagnostics on synthetic panels before real data."""
    print("\n=== Step: validate-syn ===")
    from src.smm.stylized_facts import StylizedFacts

    sf = StylizedFacts()
    results = sf.validate_synthetic()
    diffusion_kurt = results["diffusion"]["fat_tails"]["excess_kurtosis"]
    jump_kurt      = results["jump_diffusion"]["fat_tails"]["excess_kurtosis"]
    print(f"\nSynthetic validation complete.")
    print(f"  Diffusion kurtosis: {diffusion_kurt:.2f}  "
          f"Jump kurtosis: {jump_kurt:.2f}")


def step_stylized(panel: "pd.DataFrame | None" = None) -> dict:
    """§4.2  Five stylized-fact diagnostics on real panel."""
    print("\n=== Step: stylized ===")
    import pandas as pd
    from src.smm.panel import SMMPanelBuilder
    from src.smm.stylized_facts import StylizedFacts

    if panel is None:
        panel_path = Path("data/processed/smm_panel.parquet")
        if not panel_path.exists():
            print("  Panel not found — running panel step first.")
            step_panel()
        panel = pd.read_parquet(panel_path)

    sf = StylizedFacts()
    return sf.run(panel, tag="real")


def step_calibrate(kappa: float = 5.0) -> None:
    """§4.3  Bates SMM calibration for a single κ (quick check)."""
    print(f"\n=== Step: calibrate (κ={kappa}) ===")
    import pandas as pd
    from src.smm.bates_smm import BatesSMM

    panel = pd.read_parquet("data/processed/smm_panel.parquet")
    cal = BatesSMM(n_sim_multiplier=20, n_bootstrap=300, n_restarts=3)
    cache = cal.prepare(panel)
    result = cal.fit_bates(cache, kappa=kappa, verbose=True)

    print(f"\nBates fit (κ={kappa}):")
    print(f"  σ_v = {result.sigma_v:.4f}  σ_J = {result.sigma_J:.4f}")
    print(f"  J   = {result.j_stat:.3f}  (χ²({result.j_dof}), p={result.j_pvalue:.3f})")
    print("\nMoment fit:")
    for label, real, sim in zip(
        result.moment_labels, result.moments_real, result.moments_sim
    ):
        print(f"  {label:<25s}  real={real:+.4f}  sim={sim:+.4f}  "
              f"diff={sim-real:+.4f}")


def step_ladder(
    kappa_grid: list[float] | None = None,
    n_sim_multiplier: int = 20,
    n_bootstrap: int = 400,
    n_restarts: int = 3,
) -> list:
    """§4.4  Full nested ladder across κ grid."""
    print("\n=== Step: ladder ===")
    import pandas as pd
    from src.smm.bates_smm import BatesSMM
    from src.smm.nested_ladder import NestedLadder

    panel = pd.read_parquet("data/processed/smm_panel.parquet")
    kappa_grid = kappa_grid or [1.0, 5.0, 10.0]

    cal = BatesSMM(
        n_sim_multiplier=n_sim_multiplier,
        n_bootstrap=n_bootstrap,
        n_restarts=n_restarts,
    )
    ladder = NestedLadder(calibrator=cal, kappa_grid=kappa_grid)
    results = ladder.run(panel, verbose=True)
    return results


def step_validate_model(model: str = "Bates") -> None:
    """§4.7  Simulate selected model and overlay stylized facts."""
    print(f"\n=== Step: validate-model ({model}) ===")
    import pandas as pd
    from src.smm.bates_smm import BatesSMM
    from src.smm.nested_ladder import NestedLadder

    panel = pd.read_parquet("data/processed/smm_panel.parquet")
    cal = BatesSMM(n_sim_multiplier=20, n_bootstrap=300, n_restarts=2)
    ladder = NestedLadder(calibrator=cal, kappa_grid=[5.0])
    results = ladder.run(panel, verbose=True)

    lr = results[0]
    selected = {
        "Bates":       lr.bates,
        "Heston":      lr.heston,
        "ConstantVol": lr.constant_vol,
        "Merton":      lr.merton,
    }.get(model, lr.bates)

    ladder.validate_loop(panel, selected)


def step_split() -> None:
    """§4.8  FOMC-vs-CPI robustness split."""
    print("\n=== Step: split (FOMC vs CPI) ===")
    import pandas as pd
    from src.smm.bates_smm import BatesSMM
    from src.smm.nested_ladder import NestedLadder

    panel = pd.read_parquet("data/processed/smm_panel.parquet")
    cal = BatesSMM(n_sim_multiplier=20, n_bootstrap=300, n_restarts=2)
    ladder = NestedLadder(calibrator=cal, kappa_grid=[5.0])
    results = ladder.run_fomc_cpi_split(panel, kappa=5.0, verbose=True)

    for name, lr in results.items():
        print(f"\n  {name}:  σ_v(B)={lr.bates.sigma_v:.4f}  "
              f"σ_J(B)={lr.bates.sigma_J:.4f}  "
              f"J(B)={lr.bates.j_stat:.3f}")


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

STEPS = {
    "panel":          ("§4.1 Build SMM panel",                         step_panel),
    "validate-syn":   ("§4.2 Synthetic validation",                    step_validate_synthetic),
    "stylized":       ("§4.2 Stylized facts on real data",             step_stylized),
    "calibrate":      ("§4.3 Bates SMM quick check (κ=5)",            lambda: step_calibrate(kappa=5.0)),
    "ladder":         ("§4.4 Full nested ladder",                      step_ladder),
    "validate-model": ("§4.7 Validation loop (simulate selected model)",step_validate_model),
    "split":          ("§4.8 FOMC-vs-CPI robustness split",            step_split),
}


def show_help() -> None:
    print(__doc__)
    print("Available steps:")
    for name, (desc, _) in STEPS.items():
        print(f"  {name:<20s}  {desc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] in ("help", "--help", "-h"):
        show_help()
        sys.exit(0)

    step_name = args[0]
    if step_name not in STEPS:
        print(f"Unknown step: {step_name}")
        show_help()
        sys.exit(1)

    _, fn = STEPS[step_name]
    fn()
