"""
LUT point query tool.

Plug in any combination of known variables and get the resulting
operating point. Useful for quickly checking what a device looks like
at a specific bias, or for sizing a transistor from a gm/Id target.

Two modes:

  1. BIAS MODE: specify L, Vgs, Vds, Vsb directly
     -> returns Id, gm, gds, gm/Id, gm/gds, ft, Vth, Vdsat

  2. DESIGN MODE: specify L, gm/Id target, Vds, Vsb
     -> finds the Vgs that hits that gm/Id, returns all quantities
     -> also returns W needed to hit a target gm or Id

Usage examples:

  # Bias mode -- what does this device look like at this bias?
  python query_lut.py --l 1.0 --vgs 1.0 --vds 0.9
  python query_lut.py --l 1.0 --vgs 1.0 --vds 0.9 --vsb 0.45 --path char_data/pmos_lut.npz

  # Design mode -- size a device for a gm/Id target
  python query_lut.py --l 1.0 --gmid 12 --vds 0.9
  python query_lut.py --l 1.0 --gmid 15 --vds 0.9 --gm 500e-6   # size for gm=500uS
  python query_lut.py --l 1.0 --gmid 15 --vds 0.9 --id 50e-6    # size for Id=50uA

  # Interactive mode -- no args, prompts you
  python query_lut.py
"""

from pathlib import Path
import argparse
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator


DEFAULT_NMOS = Path("./char_data/nmos_lut.npz")
DEFAULT_PMOS = Path("./char_data/pmos_lut.npz")


# ─────────────────────────── load & interpolate ──────────────────────────────

def load_lut(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No LUT at {path.resolve()}")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def detect_pmos(path: Path) -> bool:
    return "pmos" in path.stem.lower()


def nearest_index(arr, value):
    return int(np.argmin(np.abs(arr - value)))


def build_interpolators(lut: dict) -> dict:
    """
    Build a RegularGridInterpolator for each quantity over the 4D grid
    (L, Vgs, Vds, Vsb). Allows querying at non-grid points.
    """
    L   = lut["L"]
    Vgs = lut["Vgs"]
    Vds = lut["Vds"]
    Vsb = lut["Vsb"]
    points = (L, Vgs, Vds, Vsb)

    interps = {}
    for key in ("Id", "gm", "gds", "gmb", "cgg", "vth", "vdsat",
                "gm_id", "gm_gds", "ft"):
        if key not in lut:
            continue
        arr = lut[key]
        # Replace NaN with 0 for interpolation (NaN breaks the interpolator)
        clean = np.where(np.isfinite(arr), arr, 0.0)
        interps[key] = RegularGridInterpolator(
            points, clean,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
    return interps


def query_point(interps, L, Vgs, Vds, Vsb) -> dict:
    """Interpolate all quantities at a single (L, Vgs, Vds, Vsb) point."""
    pt = np.array([[L, Vgs, Vds, Vsb]])
    return {k: float(f(pt)[0]) for k, f in interps.items()}


# ─────────────────────────── find Vgs from gm/Id ─────────────────────────────

def find_vgs_for_gmid(lut, interps, L, gmid_target, Vds, Vsb,
                      is_pmos=False) -> float:
    """
    Find the Vgs (or Vsg for PMOS) that hits gmid_target at (L, Vds, Vsb).
    Uses the LUT grid directly (no interpolation needed for the search),
    then refines with interpolation.
    """
    Vgs_arr = lut["Vgs"]

    # Snap L, Vds, Vsb to nearest grid points for the coarse search
    L_arr   = lut["L"]
    Vds_arr = lut["Vds"]
    Vsb_arr = lut["Vsb"]
    iL  = nearest_index(L_arr,   L)
    iVd = nearest_index(Vds_arr, Vds)
    iVb = nearest_index(Vsb_arr, Vsb)

    gmid_col = lut["gm_id"][iL, :, iVd, iVb]
    Id_col   = lut["Id"][iL,   :, iVd, iVb]

    # Only consider bias points where Id > 1 nA (device is on)
    valid = np.isfinite(gmid_col) & (Id_col > 1e-9)
    if not valid.any():
        return np.nan

    gmid_valid = gmid_col[valid]
    vgs_valid  = Vgs_arr[valid]

    # gm/Id is monotonically decreasing with Vgs (mostly).
    # Find the crossing point.
    if gmid_target > gmid_valid.max() or gmid_target < gmid_valid.min():
        print(f"  Warning: gm/Id target {gmid_target:.1f} is outside "
              f"valid range [{gmid_valid.min():.1f}, "
              f"{gmid_valid.max():.1f}] at this bias.")
        return np.nan

    # Linear interpolation between the two bracketing points
    diffs = gmid_valid - gmid_target
    # Find where sign changes (crossing)
    crossings = np.where(np.diff(np.sign(diffs)))[0]
    if len(crossings) == 0:
        return float(vgs_valid[np.argmin(np.abs(diffs))])

    # Take the crossing closest to moderate inversion
    idx = crossings[len(crossings) // 2]
    vgs_lo, vgs_hi = vgs_valid[idx], vgs_valid[idx + 1]
    gm_lo,  gm_hi  = gmid_valid[idx], gmid_valid[idx + 1]
    # Linear interpolation
    frac = (gmid_target - gm_lo) / (gm_hi - gm_lo)
    return float(vgs_lo + frac * (vgs_hi - vgs_lo))


# ─────────────────────────── sizing helpers ──────────────────────────────────

def size_from_gm(q: dict, gm_target: float) -> float:
    """Return W [µm] needed to hit gm_target given the normalized point."""
    if q["gm"] <= 0:
        return np.nan
    return gm_target / q["gm"] * 1e6   # W ref = 10µm, so scale to µm


def size_from_id(q: dict, id_target: float, W_ref=10e-6) -> float:
    """Return W [µm] needed to hit id_target."""
    if q["Id"] <= 0:
        return np.nan
    return (id_target / q["Id"]) * W_ref * 1e6


# ─────────────────────────── print helpers ───────────────────────────────────

def print_divider():
    print("─" * 52)


def print_point(q: dict, L: float, Vgs: float, Vds: float, Vsb: float,
                is_pmos: bool, W_ref=10e-6):
    gs = "Vsg" if is_pmos else "Vgs"
    ds = "Vsd" if is_pmos else "Vds"
    dev = "PMOS" if is_pmos else "NMOS"

    print_divider()
    print(f"  {dev}  L = {L*1e6:.3f} µm   W_ref = {W_ref*1e6:.0f} µm")
    print(f"  {gs} = {Vgs:.3f} V    {ds} = {Vds:.3f} V    Vsb = {Vsb:.3f} V")
    print_divider()
    print(f"  {'Quantity':<18}  {'Value':>12}  {'Unit'}")
    print_divider()
    rows = [
        ("Id",        q.get("Id",    np.nan) * 1e6,  "µA"),
        ("Vth",       q.get("vth",   np.nan),         "V"),
        ("Vdsat",     q.get("vdsat", np.nan),         "V"),
        ("gm",        q.get("gm",    np.nan) * 1e6,  "µS"),
        ("gds",       q.get("gds",   np.nan) * 1e9,  "nS"),
        ("gmb",       q.get("gmb",   np.nan) * 1e6,  "µS"),
        ("gm/Id",     q.get("gm_id", np.nan),         "V⁻¹"),
        ("gm/gds",    q.get("gm_gds",np.nan),         "V/V"),
        ("ft",        q.get("ft",    np.nan) / 1e9,  "GHz"),
        ("Id/W",      q.get("Id",    np.nan) / W_ref * 1e3, "mA/m"),
    ]
    for name, val, unit in rows:
        if np.isfinite(val):
            print(f"  {name:<18}  {val:>12.4g}  {unit}")
        else:
            print(f"  {name:<18}  {'---':>12}")
    print_divider()


def print_sizing(q: dict, gm_target, id_target, W_ref=10e-6):
    if gm_target is not None or id_target is not None:
        print("  Sizing:")
        if gm_target is not None:
            W = size_from_gm(q, gm_target)
            id_at_W = q["Id"] * W * 1e-6 / W_ref if np.isfinite(W) else np.nan
            print(f"    For gm = {gm_target*1e6:.1f} µS  "
                  f"->  W = {W:.2f} µm  "
                  f"(Id = {id_at_W*1e6:.2f} µA)")
        if id_target is not None:
            W = size_from_id(q, id_target, W_ref)
            gm_at_W = q["gm"] * W * 1e-6 / W_ref if np.isfinite(W) else np.nan
            print(f"    For Id = {id_target*1e6:.1f} µA  "
                  f"->  W = {W:.2f} µm  "
                  f"(gm = {gm_at_W*1e6:.2f} µS)")
        print_divider()


# ─────────────────────────── interactive mode ────────────────────────────────

def interactive_mode(lut, interps, is_pmos, W_ref=10e-6):
    gs = "Vsg" if is_pmos else "Vgs"
    ds = "Vsd" if is_pmos else "Vds"
    dev = "PMOS" if is_pmos else "NMOS"

    L_arr   = lut["L"]
    Vds_arr = lut["Vds"]
    Vsb_arr = lut["Vsb"]

    print(f"\n{dev} LUT interactive query  (Ctrl-C to exit)")
    print(f"  L values:   {[f'{v*1e6:.2f}' for v in L_arr]} µm")
    print(f"  {ds} range:  [{Vds_arr.min():.2f}, {Vds_arr.max():.2f}] V")
    print(f"  Vsb values: {Vsb_arr.tolist()} V")
    print()

    while True:
        try:
            print("Enter query (leave blank for defaults shown in brackets):")
            L_input = input(f"  L [µm] (e.g. 1.0): ").strip()
            L = float(L_input) * 1e-6

            mode = input(
                f"  Mode: [b]ias (enter {gs}/Vds) or [d]esign (enter gm/Id)? "
            ).strip().lower()

            if mode.startswith("d"):
                gmid_str = input("  gm/Id target [V⁻¹]: ").strip()
                gmid_target = float(gmid_str)
                Vds_str = input(f"  {ds} [V] (default 0.9): ").strip()
                Vds = float(Vds_str) if Vds_str else 0.9
                Vsb_str = input("  Vsb [V] (default 0.0): ").strip()
                Vsb = float(Vsb_str) if Vsb_str else 0.0
                gm_str = input("  Target gm [µS] (optional, press enter to skip): ").strip()
                id_str = input("  Target Id [µA] (optional, press enter to skip): ").strip()
                gm_target = float(gm_str) * 1e-6 if gm_str else None
                id_target = float(id_str) * 1e-6 if id_str else None

                Vgs = find_vgs_for_gmid(lut, interps, L, gmid_target, Vds, Vsb, is_pmos)
                if np.isnan(Vgs):
                    print("  Could not find Vgs for that gm/Id target.")
                    print()
                    continue
                print(f"\n  Found {gs} = {Vgs:.4f} V for gm/Id = {gmid_target:.1f}")

            else:
                Vgs_str = input(f"  {gs} [V] (e.g. 0.9): ").strip()
                Vgs = float(Vgs_str)
                Vds_str = input(f"  {ds} [V] (default 0.9): ").strip()
                Vds = float(Vds_str) if Vds_str else 0.9
                Vsb_str = input("  Vsb [V] (default 0.0): ").strip()
                Vsb = float(Vsb_str) if Vsb_str else 0.0
                gm_target = None
                id_target = None

            q = query_point(interps, L, Vgs, Vds, Vsb)
            print()
            print_point(q, L, Vgs, Vds, Vsb, is_pmos, W_ref)
            print_sizing(q, gm_target, id_target, W_ref)
            print()

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except (ValueError, EOFError) as e:
            print(f"  Input error: {e}. Try again.\n")


# ─────────────────────────── main ────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Query a gm/Id LUT at specific operating points."
    )
    p.add_argument("--path", type=Path, default=None,
                   help="LUT file (auto-selects nmos_lut.npz if not given)")
    p.add_argument("--l",    type=float, default=None, metavar="UM",
                   help="Channel length [µm]")
    p.add_argument("--vgs",  type=float, default=None,
                   help="Vgs or Vsg [V] (bias mode)")
    p.add_argument("--gmid", type=float, default=None,
                   help="gm/Id target [V⁻¹] (design mode)")
    p.add_argument("--vds",  type=float, default=0.9,
                   help="Vds or Vsd [V] (default 0.9)")
    p.add_argument("--vsb",  type=float, default=0.0,
                   help="Vsb [V] (default 0.0)")
    p.add_argument("--gm",   type=float, default=None,
                   help="Target gm [S] for sizing, e.g. --gm 500e-6")
    p.add_argument("--id",   type=float, default=None, dest="id_target",
                   help="Target Id [A] for sizing, e.g. --id 50e-6")
    p.add_argument("--pmos", action="store_true",
                   help="Force PMOS mode (auto-detected from filename otherwise)")
    args = p.parse_args()

    # Resolve path
    if args.path is None:
        if args.pmos:
            args.path = DEFAULT_PMOS
        else:
            args.path = DEFAULT_NMOS

    is_pmos = args.pmos or detect_pmos(args.path)
    lut     = load_lut(args.path)
    W_ref   = float(lut["W"])

    print(f"Loaded {'PMOS' if is_pmos else 'NMOS'} LUT: {args.path}")
    print(f"Grid: L={lut['L'].size}  "
          f"Vgs={lut['Vgs'].size}  "
          f"Vds={lut['Vds'].size}  "
          f"Vsb={lut['Vsb'].size}")

    print("Building interpolators...", end=" ", flush=True)
    interps = build_interpolators(lut)
    print("done.")

    # No L specified -> interactive mode
    if args.l is None:
        interactive_mode(lut, interps, is_pmos, W_ref)
        return

    L = args.l * 1e-6

    # Design mode: find Vgs from gm/Id target
    if args.gmid is not None:
        Vgs = find_vgs_for_gmid(
            lut, interps, L, args.gmid, args.vds, args.vsb, is_pmos
        )
        if np.isnan(Vgs):
            print("Could not find Vgs for that gm/Id target.")
            sys.exit(1)
        gs = "Vsg" if is_pmos else "Vgs"
        print(f"\nFound {gs} = {Vgs:.4f} V for gm/Id = {args.gmid:.1f} V⁻¹")

    elif args.vgs is not None:
        Vgs = args.vgs
    else:
        print("Error: specify --vgs (bias mode) or --gmid (design mode).")
        sys.exit(1)

    q = query_point(interps, L, Vgs, args.vds, args.vsb)
    print()
    print_point(q, L, Vgs, args.vds, args.vsb, is_pmos, W_ref)
    print_sizing(q, args.gm, args.id_target, W_ref)


if __name__ == "__main__":
    main()
