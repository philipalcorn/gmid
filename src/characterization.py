"""
Combined NMOS + PMOS gm/Id 4D characterization sweep.

One LTSpice .op per bias point characterizes both devices simultaneously.

Testbench parameter names (from nmos_char.asc):
    NMOS (M1): Vg={Vg}, Vd={Vd}, source=0, body={Vb}
    PMOS (M2): V1={vdd} (source), Vg1={Vg_p}, Vd1={Vd_p}, Vb1={Vb_p}
               where Vg_p={VDD-Vg}, Vd_p={VDD-Vd}, Vb_p={VDD-Vb}
               (computed inside LTSpice via .param expressions)

Python sets: Vg, Vd, Vb, VDD, L
    Vg  : intrinsic Vgs (NMOS) = intrinsic Vsg (PMOS)
    Vd  : intrinsic Vds (NMOS) = intrinsic Vsd (PMOS)
    Vb  : set to -Vsb for NMOS body effect
          PMOS automatically gets Vb_p = VDD - Vb = VDD + Vsb (symmetric)
    VDD : supply rail, fixed at 1.8 V

Outputs:
    ./char_data/nmos_lut.npz
    ./char_data/pmos_lut.npz
"""

from pathlib import Path
import numpy as np

from callbacks import CallbackGetAllData
from parser import Parser
from spice import Spice
from helpers import Helpers
from runconfig import RunConfig


# ---------- Sweep configuration ----------
W_REF    = 10e-6
VDD      = 3.3 
NMOS_DEV = "M1"
PMOS_DEV = "M2"
_ROOT   = Path(__file__).parent.parent
DATA_DIR = _ROOT / "char_data"

L_VALUES = [0.35e-6, 0.7e-6, 1.0e-6, 2.0e-6]
VGS_MIN, VGS_MAX, VGS_STEP = 0.0, 5, 0.05         # [V]
VDS_MIN, VDS_MAX, VDS_STEP = 0.0, 5, 0.05          # [V]
#VGS_MIN, VGS_MAX, VGS_STEP = 0.0, 5, 0.5         # [V]
#VDS_MIN, VDS_MAX, VDS_STEP = 0.0, 5, 0.5          # [V]
VSB_VALUES = [0.0, 0.45, 0.9]                         # [V] logical Vsb


def build_config() -> RunConfig:
    cfg = RunConfig()
    cfg.asc_path  = _ROOT / "simulations" / "nmos_char.asc"
    cfg.out_path  = Path("./char_out")
    cfg.num_parallel_sims = 20
    cfg.debug_mode = False
    cfg.make_plots = False
    cfg.trace_names   = []
    cfg.meas_names    = []
    cfg.device_names  = [NMOS_DEV, PMOS_DEV]
    cfg.device_values = None
    if not cfg.asc_path.exists():
        raise FileNotFoundError(f"Testbench not found: {cfg.asc_path.resolve()}")
    return cfg


def run_sweep(cfg: RunConfig):
    Helpers.clean_directory(cfg.out_path)
    spice = Spice(
        cfg.exe_path,
        asc_path=cfg.asc_path,
        output_folder=cfg.out_path,
        callback_proc=CallbackGetAllData,
        parallel_sims=cfg.num_parallel_sims,
    )

    vgs_grid = np.arange(VGS_MIN, VGS_MAX + 1e-12, VGS_STEP)
    vds_grid = np.arange(VDS_MIN, VDS_MAX + 1e-12, VDS_STEP)

    # Fix VDD -- overrides the .param VDD 5 line in the schematic
    spice.set_parameter("VDD", f"{VDD:.6f}")

    id_to_bias = {}
    launch_index = 1
    total = len(VSB_VALUES) * len(L_VALUES) * len(vgs_grid) * len(vds_grid)
    print(f"Launching {total} sims (NMOS+PMOS per sim) "
          f"Vsb={len(VSB_VALUES)}, L={len(L_VALUES)}, "
          f"Vgs={len(vgs_grid)}, Vds={len(vds_grid)})")

    for vsb in VSB_VALUES:
        # Vb = -vsb sets NMOS body below source by vsb volts.
        # PMOS gets Vb_p = VDD - Vb = VDD + vsb, so PMOS body is
        # above its source by vsb volts -- symmetric body effect.
        spice.set_parameter("Vb", f"{-vsb:.6f}")
        for L in L_VALUES:
            spice.set_parameter("L", f"{L}")
            for vgs in vgs_grid:
                spice.set_parameter("Vg", f"{vgs:.6f}")
                for vds in vds_grid:
                    spice.set_parameter("Vd", f"{vds:.6f}")
                    spice.simulate()
                    id_to_bias[launch_index] = (
                        L, float(vgs), float(vds), float(vsb)
                    )
                    print(f"\rRunning simulation {launch_index}/{total}",
                          end="", flush=True)
                    launch_index += 1
    print()
    print("Waiting for completion...")
    spice.sim_runner.wait_completion()
    return spice, id_to_bias


def flat_params(pr, device_name: str) -> dict:
    for section, devs in pr.semi_ops.items():
        for dev_name, params in devs.items():
            if dev_name.lower() == device_name.lower():
                return params
    return {}


def find_key(params: dict, *candidates: str):
    lut = {k.lower(): k for k in params.keys()}
    for c in candidates:
        if c.lower() in lut:
            return lut[c.lower()]
    return None


def extract_device(parsed_results, id_to_bias,
                   device_name: str, verbose=True) -> dict:
    """Build 4D arrays (nL, nVgs, nVds, nVsb) for one device."""
    sample = {}
    for pr in parsed_results:
        sample = flat_params(pr, device_name)
        if sample:
            break
    if not sample:
        raise RuntimeError(f"Device '{device_name}' not found in any result.")

    if verbose:
        print(f"\nOp-point fields for {device_name}: {sorted(sample.keys())}")

    k_id    = find_key(sample, "Id", "Ids")
    k_gm    = find_key(sample, "Gm")
    k_gds   = find_key(sample, "Gds")
    k_gmb   = find_key(sample, "Gmb", "Gmbs")
    k_vth   = find_key(sample, "Vth")
    k_vdsat = find_key(sample, "Vdsat")
    k_dqg   = find_key(sample, "dQgdVgb")
    k_cgsov = find_key(sample, "Cgsov")
    k_cgdov = find_key(sample, "Cgdov")
    k_cgbov = find_key(sample, "Cgbov")

    missing = [n for n, k in [("Id", k_id), ("Gm", k_gm), ("Gds", k_gds)]
               if k is None]
    if missing:
        raise RuntimeError(f"{device_name}: required fields missing: {missing}")

    vgs_grid = np.arange(VGS_MIN, VGS_MAX + 1e-12, VGS_STEP)
    vds_grid = np.arange(VDS_MIN, VDS_MAX + 1e-12, VDS_STEP)
    nL, nVg, nVd, nVb = (len(L_VALUES), len(vgs_grid),
                          len(vds_grid), len(VSB_VALUES))
    shape = (nL, nVg, nVd, nVb)

    def empty(): return np.full(shape, np.nan)
    arrs = {k: empty() for k in
            ("Id", "gm", "gds", "gmb", "cgg", "vth", "vdsat")}

    L_idx   = {L: i for i, L in enumerate(L_VALUES)}
    Vsb_idx = {v: i for i, v in enumerate(VSB_VALUES)}
    def vgs_idx(v): return int(round((v - VGS_MIN) / VGS_STEP))
    def vds_idx(v): return int(round((v - VDS_MIN) / VDS_STEP))

    filled = 0
    for pr in parsed_results:
        bias = id_to_bias.get(pr.file_id)
        if bias is None:
            continue
        L, vgs, vds, vsb = bias
        iL  = L_idx[L]
        iVg = vgs_idx(vgs)
        iVd = vds_idx(vds)
        iVb = Vsb_idx[vsb]

        params = flat_params(pr, device_name)
        if not params:
            continue

        arrs["Id"][iL, iVg, iVd, iVb]  = abs(float(params[k_id]))
        arrs["gm"][iL, iVg, iVd, iVb]  = float(params[k_gm])
        arrs["gds"][iL, iVg, iVd, iVb] = float(params[k_gds])
        if k_gmb:
            arrs["gmb"][iL, iVg, iVd, iVb] = float(params[k_gmb])
        cgg = sum(float(params[kk]) for kk in
                  (k_dqg, k_cgsov, k_cgdov, k_cgbov) if kk)
        arrs["cgg"][iL, iVg, iVd, iVb] = cgg
        if k_vth:
            arrs["vth"][iL, iVg, iVd, iVb] = float(params[k_vth])
        if k_vdsat:
            arrs["vdsat"][iL, iVg, iVd, iVb] = float(params[k_vdsat])
        filled += 1

    print(f"{device_name}: filled {filled} / {nL*nVg*nVd*nVb} grid points")
    arrs.update({
        "L":   np.array(L_VALUES),
        "Vgs": vgs_grid,
        "Vds": vds_grid,
        "Vsb": np.array(VSB_VALUES),
        "W":   W_REF,
    })
    return arrs


def build_design_curves(lut: dict) -> dict:
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "gm_id":  lut["gm"] / lut["Id"],
            "id_w":   lut["Id"] / lut["W"],
            "gm_gds": lut["gm"] / lut["gds"],
            "ft":     lut["gm"] / (2 * np.pi * lut["cgg"]),
        }


def save_lut(lut: dict, curves: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **lut, **curves)
    print(f"Saved {path}")


def main():
    cfg = build_config()
    spice, id_to_bias = run_sweep(cfg)

    raw_results = list(spice.sim_runner)
    print(f"Collected {len(raw_results)} raw results")

    parser = Parser()
    parsed = list(parser.parse(
        raw_results,
        cfg.trace_names, cfg.meas_names,
        cfg.device_names, cfg.device_values,
    ))
    print(f"Parsed {len(parsed)} results")

    print("\n--- Extracting NMOS ---")
    nmos_lut = extract_device(parsed, id_to_bias, NMOS_DEV)
    save_lut(nmos_lut, build_design_curves(nmos_lut),
             DATA_DIR / "nmos_lut.npz")

    print("\n--- Extracting PMOS ---")
    pmos_lut = extract_device(parsed, id_to_bias, PMOS_DEV)
    save_lut(pmos_lut, build_design_curves(pmos_lut),
             DATA_DIR / "pmos_lut.npz")

    print("\nDone.")
    print("  python view_lut.py --path char_data/nmos_lut.npz")
    print("  python view_lut.py --path char_data/pmos_lut.npz")


if __name__ == "__main__":
    main()
