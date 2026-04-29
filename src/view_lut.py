"""
View NMOS or PMOS gm/Id 4D lookup tables.

Each plot opens in its own window. Device type (NMOS/PMOS) is detected
automatically from the filename.

Usage examples:
    python view_lut.py
    python view_lut.py --path char_data/pmos_lut.npz
    python view_lut.py --vds 0.3 --vsb 0.45
    python view_lut.py --l 0.35 1.0 2.0
    python view_lut.py --xlim 0 20 --ylim 0 60
    python view_lut.py --vgs 1.0
    python view_lut.py --show_all_vsb
    python view_lut.py --show_all_vds
    python view_lut.py --path char_data/pmos_lut.npz --vds 0.9 --l 0.7 2.0 --xlim 0 20

All arguments:
    --path        Path to .npz LUT file
    --vds         Vds/Vsd slice for design curves [V]  (default 0.9)
    --vsb         Vsb slice for design curves [V]       (default 0.0)
    --vov         Vov for slide 19 gm*ro plot [V]       (default 0.2)
    --vgs         Override Vgs/Vsg for slide 19 directly (overrides --vov)
    --l           Filter to specific L values in um, e.g. --l 0.35 1.0 2.0
    --xlim        gm/Id x-axis limits, e.g. --xlim 0 20
    --ylim        Id/W y-axis limits [µA/µm], e.g. --ylim 0 60
    --show_all_vsb  Overlay all Vsb values on design curves
    --show_all_vds  Overlay all Vds values on design curves
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


DEFAULT_PATH = Path(__file__).parent.parent / "char_data" / "nmos_lut.npz"
ID_FLOOR = 1e-9    # A


# ─────────────────────────── helpers ────────────────────────────────────────

def load_lut(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No LUT at {path.resolve()}. Run characterization.py first."
        )
    data = np.load(path)
    return {k: data[k] for k in data.files}


def detect_pmos(path: Path) -> bool:
    return "pmos" in path.stem.lower()


def nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(arr - value)))


def filter_l_indices(L_arr: np.ndarray, l_filter) -> list:
    """Return indices into L_arr matching l_filter (in metres)."""
    if l_filter is None:
        return list(range(len(L_arr)))
    indices = []
    for lum in l_filter:
        lm = lum * 1e-6
        idx = int(np.argmin(np.abs(L_arr - lm)))
        if idx not in indices:
            indices.append(idx)
    return sorted(indices)


def slice_2d(lut: dict, vds_target: float, vsb_target: float) -> dict:
    """Reduce 4D arrays to 2D (L, Vgs) at fixed Vds and Vsb."""
    Vds = lut["Vds"]
    Vsb = lut["Vsb"]
    iVd = nearest_index(Vds, vds_target)
    iVb = nearest_index(Vsb, vsb_target)
    out = {
        "L":   lut["L"],
        "Vgs": lut["Vgs"],
        "W":   float(lut["W"]),
        "Vds": float(Vds[iVd]),
        "Vsb": float(Vsb[iVb]),
    }
    for key in ("Id", "gm", "gds", "cgg", "vth", "vdsat",
                "gm_id", "id_w", "gm_gds", "ft"):
        if key in lut:
            out[key] = lut[key][:, :, iVd, iVb]
    return out


def make_label(Lv, extra=""):
    base = f"L={Lv*1e6:.2f} µm"
    return f"{base}  {extra}" if extra else base


def color_cycle(n):
    """Return n colours from a readable colormap."""
    cmap = cm.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


# ─────────────────────────── summary ────────────────────────────────────────

def print_summary(lut, vds_target, vsb_target, is_pmos, l_indices):
    L   = lut["L"]
    Vgs = lut["Vgs"]
    Vds = lut["Vds"]
    Vsb = lut["Vsb"]
    Id  = lut["Id"]

    dev      = "PMOS" if is_pmos else "NMOS"
    gs_label = "Vsg" if is_pmos else "Vgs"
    ds_label = "Vsd" if is_pmos else "Vds"

    print(f"\n========== {dev} LUT summary ==========")
    print(f"  Shape (L, {gs_label}, {ds_label}, Vsb): {Id.shape}")
    print(f"  L values ({len(L)}): {[f'{v*1e6:.2f} µm' for v in L]}")
    print(f"  Showing L: {[f'{L[i]*1e6:.2f} µm' for i in l_indices]}")
    print(f"  {gs_label}: {Vgs.size} pts  "
          f"[{Vgs.min():.3f}, {Vgs.max():.3f}] V  "
          f"step={Vgs[1]-Vgs[0]:.4f} V")
    print(f"  {ds_label}: {Vds.size} pts  "
          f"[{Vds.min():.3f}, {Vds.max():.3f}] V  "
          f"step={Vds[1]-Vds[0]:.4f} V")
    print(f"  Vsb: {Vsb.tolist()} V")
    print(f"  W ref: {float(lut['W'])*1e6:.1f} µm")
    print(f"  Valid Id: {np.isfinite(Id).mean()*100:.1f}%")
    iVd = nearest_index(Vds, vds_target)
    iVb = nearest_index(Vsb, vsb_target)
    print(f"  Default slice: {ds_label}={Vds[iVd]:.3f} V, "
          f"Vsb={Vsb[iVb]:.3f} V")

# ─────────────────────── design curves (3 windows) ───────────────────────────

def _get_slices(lut, args, is_pmos):
    """
    Return a list of (slice_dict, slice_label) to overlay.
    Priority:
      --vds_list   : overlay the specified Vds values (Vsb fixed)
      --show_all_vsb: one slice per Vsb value (Vds fixed)
      --show_all_vds: one slice per Vds value (Vsb fixed)
      default      : single slice at (--vds, --vsb)
    """
    Vds_arr = lut["Vds"]
    Vsb_arr = lut["Vsb"]
    ds = "Vsd" if is_pmos else "Vds"

    if args.vds_list is not None:
        # Snap each requested value to nearest grid point
        snapped = []
        for v in args.vds_list:
            idx = nearest_index(Vds_arr, v)
            snapped.append(float(Vds_arr[idx]))
        # Deduplicate while preserving order
        seen = set()
        unique = [v for v in snapped if not (v in seen or seen.add(v))]
        return [
            (slice_2d(lut, vds, args.vsb), f"{ds}={vds:.2f}V")
            for vds in unique
        ]
    if args.show_all_vsb:
        return [
            (slice_2d(lut, args.vds, vsb), f"Vsb={vsb:.2f}V")
            for vsb in Vsb_arr
        ]
    if args.show_all_vds:
        return [
            (slice_2d(lut, vds, args.vsb), f"{ds}={vds:.2f}V")
            for vds in Vds_arr
        ]
    label = f"{ds}={args.vds:.2f}V  Vsb={args.vsb:.2f}V"
    return [(slice_2d(lut, args.vds, args.vsb), label)]


def plot_current_density(lut, args, is_pmos, l_indices):
    """Window 1: Id/W vs gm/Id."""
    slices = _get_slices(lut, args, is_pmos)
    L = lut["L"]
    colors = color_cycle(len(l_indices))
    linestyles = ["-", "--", ":", "-."]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for si, (s, slice_label) in enumerate(slices):
        ls = linestyles[si % len(linestyles)]
        for ci, iL in enumerate(l_indices):
            Lv    = L[iL]
            Id    = s["Id"][iL]
            gm_id = s["gm_id"][iL]
            id_w  = s["id_w"][iL]   # A/m == µA/µm
            mask  = (np.isfinite(gm_id) & np.isfinite(id_w)
                     & (id_w > 0) & (Id > ID_FLOOR))
            extra = slice_label if len(slices) > 1 else ""
            ax.plot(gm_id[mask], id_w[mask],
                    color=colors[ci], ls=ls,
                    label=make_label(Lv, extra))

    ax.set_xlabel("gm/Id  [1/V]")
    ax.set_ylabel("Id/W  [µA/µm]")
    ax.set_title(f"{'PMOS' if is_pmos else 'NMOS'} — Current density")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    if args.xlim:
        ax.set_xlim(args.xlim)
    else:
        ax.set_xlim(0, 25)
    if args.ylim:
        ax.set_ylim(args.ylim)
    else:
        ax.set_ylim(0, 100)
    fig.tight_layout()


def plot_intrinsic_gain(lut, args, is_pmos, l_indices):
    """Window 2: gm/gds vs gm/Id."""
    slices = _get_slices(lut, args, is_pmos)
    L = lut["L"]
    colors = color_cycle(len(l_indices))
    linestyles = ["-", "--", ":", "-."]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for si, (s, slice_label) in enumerate(slices):
        ls = linestyles[si % len(linestyles)]
        for ci, iL in enumerate(l_indices):
            Lv    = L[iL]
            Id    = s["Id"][iL]
            gm_id = s["gm_id"][iL]
            gm_gds = s["gm_gds"][iL]
            mask  = (np.isfinite(gm_id) & np.isfinite(gm_gds)
                     & (Id > ID_FLOOR))
            extra = slice_label if len(slices) > 1 else ""
            ax.plot(gm_id[mask], gm_gds[mask],
                    color=colors[ci], ls=ls,
                    label=make_label(Lv, extra))

    ax.set_xlabel("gm/Id  [1/V]")
    ax.set_ylabel("gm/gds")
    ax.set_title(f"{'PMOS' if is_pmos else 'NMOS'} — Intrinsic gain")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    if args.xlim:
        ax.set_xlim(args.xlim)
    else:
        ax.set_xlim(0, 25)
    fig.tight_layout()


def plot_ft(lut, args, is_pmos, l_indices):
    """Window 3: ft vs gm/Id."""
    slices = _get_slices(lut, args, is_pmos)
    L = lut["L"]
    colors = color_cycle(len(l_indices))
    linestyles = ["-", "--", ":", "-."]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for si, (s, slice_label) in enumerate(slices):
        ls = linestyles[si % len(linestyles)]
        for ci, iL in enumerate(l_indices):
            Lv    = L[iL]
            Id    = s["Id"][iL]
            gm_id = s["gm_id"][iL]
            ft    = s["ft"][iL]
            mask  = (np.isfinite(gm_id) & np.isfinite(ft)
                     & (ft > 0) & (Id > ID_FLOOR))
            extra = slice_label if len(slices) > 1 else ""
            ax.plot(gm_id[mask], ft[mask] / 1e9,
                    color=colors[ci], ls=ls,
                    label=make_label(Lv, extra))

    ax.set_xlabel("gm/Id  [1/V]")
    ax.set_ylabel("ft  [GHz]")
    ax.set_title(f"{'PMOS' if is_pmos else 'NMOS'} — Transition frequency")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    if args.xlim:
        ax.set_xlim(args.xlim)
    else:
        ax.set_xlim(0, 25)
    fig.tight_layout()


# ─────────────────────────── slide 19 ───────────────────────────────────────

def plot_gmro_vs_vds(lut, args, is_pmos, l_indices):
    """Window 4: gm*ro vs Vds/Vsd — Tiwari slide 19."""
    L   = lut["L"]
    Vgs = lut["Vgs"]
    Vds = lut["Vds"]
    Vsb = lut["Vsb"]
    gm  = lut["gm"]
    gds = lut["gds"]
    vth = lut["vth"]

    ds_label = "Vsd  [V]" if is_pmos else "Vds  [V]"
    dev      = "PMOS" if is_pmos else "NMOS"

    iVb         = nearest_index(Vsb, args.vsb)
    iVd_for_vth = nearest_index(Vds, 0.9)

    colors = color_cycle(len(l_indices))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    with np.errstate(divide="ignore", invalid="ignore"):
        gm_ro = gm / gds

    vgs_override = args.vgs  # None if not specified

    for ci, iL in enumerate(l_indices):
        Lv = L[iL]

        if vgs_override is not None:
            iVg = nearest_index(Vgs, vgs_override)
            vgs_used = Vgs[iVg]
            bias_label = (f"V{'sg' if is_pmos else 'gs'}="
                          f"{vgs_used:.2f}V (manual)")
        else:
            vth_samples = vth[iL, :, iVd_for_vth, iVb]
            vth_typical = np.nanmedian(vth_samples)
            if is_pmos:
                target = abs(vth_typical) + args.vov
            else:
                target = vth_typical + args.vov
            iVg = nearest_index(Vgs, target)
            vgs_used = Vgs[iVg]
            bias_label = (f"V{'sg' if is_pmos else 'gs'}="
                          f"{vgs_used:.2f}V  "
                          f"(Vt~{vth_typical:.2f})")

        curve = gm_ro[iL, iVg, :, iVb]
        ax.plot(Vds, curve, color=colors[ci],
                label=f"L={Lv*1e6:.2f}µm  {bias_label}")

    ax.set_xlabel(ds_label)
    ax.set_ylabel("gm·ro  [V/V]")
    if vgs_override is not None:
        subtitle = (f"V{'sg' if is_pmos else 'gs'}={vgs_override:.2f}V  "
                    f"Vsb={Vsb[iVb]:.2f}V")
    else:
        subtitle = (f"Vov={args.vov*1000:.0f}mV  "
                    f"Vsb={Vsb[iVb]:.2f}V")
    ax.set_title(f"{dev} — Intrinsic gain vs V{'sd' if is_pmos else 'ds'}"
                 f"  ({subtitle})   Tiwari slide 19")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()


# ─────────────────────────── main ───────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="View NMOS/PMOS gm/Id LUT — each plot in its own window."
    )
    p.add_argument("--path", type=Path, default=DEFAULT_PATH,
                   help="Path to .npz LUT file")
    p.add_argument("--vds", type=float, default=0.9,
                   help="Vds/Vsd slice for design curves [V]")
    p.add_argument("--vds_list", type=float, nargs="+", default=None,
                   metavar="V",
                   help="Overlay specific Vds values, e.g. --vds_list 0.3 0.6 0.9 1.2")
    p.add_argument("--vsb", type=float, default=0.0,
                   help="Vsb slice [V]")
    p.add_argument("--vov", type=float, default=0.2,
                   help="Overdrive for slide 19 [V]")
    p.add_argument("--vgs", type=float, default=None,
                   help="Override Vgs/Vsg for slide 19 directly [V]")
    p.add_argument("--l", type=float, nargs="+", default=None,
                   metavar="UM",
                   help="Filter to specific L values in µm, e.g. --l 0.35 1.0")
    p.add_argument("--xlim", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="gm/Id x-axis limits, e.g. --xlim 0 20")
    p.add_argument("--ylim", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Id/W y-axis limits [µA/µm], e.g. --ylim 0 60")
    p.add_argument("--show_all_vsb", action="store_true",
                   help="Overlay curves at all Vsb values (dashed lines)")
    p.add_argument("--show_all_vds", action="store_true",
                   help="Overlay curves at all Vds values (dashed lines)")
    args = p.parse_args()

    is_pmos   = detect_pmos(args.path)
    lut       = load_lut(args.path)
    l_indices = filter_l_indices(lut["L"], args.l)

    print_summary(lut, args.vds, args.vsb, is_pmos, l_indices)

    plot_current_density(lut, args, is_pmos, l_indices)
    plot_intrinsic_gain (lut, args, is_pmos, l_indices)
    plot_ft             (lut, args, is_pmos, l_indices)
    plot_gmro_vs_vds    (lut, args, is_pmos, l_indices)

    plt.show()


if __name__ == "__main__":
    main()
