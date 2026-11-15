# Ring weights of four samplings, computed with s2fft's own numpy quadrature functions.
# Run with the s2fft environment: ../../.venv/bin/python quadrature_data.py
# Writes quadrature_weights.npz, read by quadrature_plot.py.
import numpy as np
from s2fft.sampling import s2_samples as samples
from s2fft.utils import quadrature

L, NSIDE = 32, 16
out = {}
for s in ("gl", "dh", "cc"):
    w = np.asarray(quadrature.quad_weights(L, s))                # per pixel, one value per ring
    out[f"{s}_theta"] = samples.thetas(L, s)
    out[f"{s}_ring"] = w * samples.nphi_equiang(L, s)              # summed over the ring
w = np.asarray(quadrature.quad_weights(sampling="healpix", nside=NSIDE))
nphi = np.array([samples.nphi_ring(t, NSIDE) for t in range(samples.ntheta(sampling="healpix", nside=NSIDE))])
out["healpix_theta"] = samples.thetas(sampling="healpix", nside=NSIDE)
out["healpix_ring"] = w * nphi
for s in ("gl", "dh", "cc", "healpix"):
    print(s, out[f"{s}_theta"].shape, out[f"{s}_ring"].shape, "sum / 4pi =", out[f"{s}_ring"].sum() / (4 * np.pi))
np.savez("quadrature_weights.npz", **out)
