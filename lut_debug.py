"""Quick diagnostic dump of the 4D LUT to find why plots are empty."""

import numpy as np

d = np.load('./char_data/nmos_lut.npz')

print('Keys:', list(d.files))
print('Shape of Id:', d['Id'].shape)
print('L values:', d['L'])
print(f'Vgs range: {d["Vgs"].min():.3f} to {d["Vgs"].max():.3f}, n={d["Vgs"].size}')
print(f'Vds range: {d["Vds"].min():.3f} to {d["Vds"].max():.3f}, n={d["Vds"].size}')
print('Vsb values:', d['Vsb'])
print()

print(f'Valid Id fraction: {np.isfinite(d["Id"]).mean():.3f}')
print(f'Valid gm fraction: {np.isfinite(d["gm"]).mean():.3f}')
print(f'Valid gds fraction: {np.isfinite(d["gds"]).mean():.3f}')
print()

# Slice at L=0.7u (index 1), Vsb=0 (index 0)
Id_slice = d['Id'][1, :, :, 0]
print('Id slice shape (L=0.7u, Vsb=0):', Id_slice.shape)
print(f'  finite: {np.isfinite(Id_slice).sum()} / {Id_slice.size}')
if np.isfinite(Id_slice).any():
    print(f'  min/max of finite: {np.nanmin(Id_slice):.3e} / {np.nanmax(Id_slice):.3e}')
print()

# Pick a row: Vgs near 1.0 V
iVg = int(np.argmin(np.abs(d['Vgs'] - 1.0)))
vgs_val = d['Vgs'][iVg]
print(f'Id at L=0.7u, Vgs={vgs_val:.3f}, Vsb=0, across Vds:')
print(f'  First 10: {Id_slice[iVg, :10]}')
print(f'  Last 10:  {Id_slice[iVg, -10:]}')
print()

# Also check Vds=0.9 slice specifically (what view_lut picks by default)
iVd = int(np.argmin(np.abs(d['Vds'] - 0.9)))
vds_val = d['Vds'][iVd]
print(f'Vds[{iVd}] = {vds_val:.3f} V (what view_lut slices at)')
print(f'Id at that Vds, L=0.7u, Vsb=0, across Vgs:')
id_at_slice = d['Id'][1, :, iVd, 0]
print(f'  finite: {np.isfinite(id_at_slice).sum()} / {id_at_slice.size}')
print(f'  First 10 Vgs values: {d["Vgs"][:10]}')
print(f'  First 10 Id values:  {id_at_slice[:10]}')
print(f'  Last 10 Vgs values:  {d["Vgs"][-10:]}')
print(f'  Last 10 Id values:   {id_at_slice[-10:]}')
