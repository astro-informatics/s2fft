# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import sys

sys.path.append("../")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
from jax.config import config
import pyssht as ssht
import s2fft as s2f

config.update("jax_enable_x64", True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compare All JAX methods to ground truth

# generate spherical harmonics (ground truth)
DEFAULT_SEED = 8966433580120847635
rn_gen = np.random.default_rng(DEFAULT_SEED)  # modify to use JAX random key approach?

### input params
L_in = 5  # input L, redefined to 2*nside if sampling is healpix
spin = 0
reality = True
L_lower = 2
nside_healpix = 2

# list jax implementations
list_jax_str = [
    "jax_vmap_double",
    "jax_vmap_scan",
    "jax_vmap_loop",
    "jax_map_double",
    "jax_map_scan",
]

# list of sampling approaches
list_sampling = ["mw", "mwss", "dh", "healpix"]

# Print inputs common to all sampling methods
print("-----------------------------------------")
print("Input params:")
print(
    f"L_in = {L_in}, spin = {spin}, reality = {reality}, L_lower = {L_lower}, nside_healpix={nside_healpix}"
)
print("-----------------------------------------")

# All JAX methods
for m_str in list_jax_str:

    for sampling in list_sampling:

        # Groundtruth and starting point f
        if sampling != "healpix":
            # Set nside to None if not healpix
            nside = None
            L = L_in

            # compute ground truth and starting point f
            flm_gt = s2f.utils.generate_flm(rn_gen, L, L_lower, spin, reality=reality)
            f = ssht.inverse(
                s2f.samples.flm_2d_to_1d(flm_gt, L),
                L,
                Method=sampling.upper(),
                Spin=spin,
                Reality=False,
            )

        else:
            # Set nside to healpix value and redefine L
            nside = nside_healpix
            L = 2 * nside  # L redefined to double nside

            # compute ground truth and starting point f
            flm_gt0 = s2f.utils.generate_flm(rn_gen, L, L_lower, spin, reality=reality)
            f = s2f.transform._inverse(
                flm_gt0, L, sampling=sampling, method="direct", nside=nside
            )

            # Compute numpy solution if sampling is healpix (used as GT)
            flm_sov_fft_vec = s2f.transform._forward(
                f,
                L,
                spin,
                sampling,
                method="sov_fft_vectorized",
                nside=nside,
                reality=reality,
                L_lower=L_lower,
            )

        # JAX implementation
        flm_sov_fft_vec_jax = s2f.transform._forward(
            f,
            L,
            spin,
            sampling,
            method=m_str,
            nside=nside,
            reality=reality,
            L_lower=L_lower,
        )

        # Compare to GT
        if sampling == "healpix":  # compare to numpy result rather than GT
            print(
                f"{m_str} vs numpy ({sampling}, L = {L}): {np.allclose(flm_sov_fft_vec, flm_sov_fft_vec_jax, atol=1e-14)}"
            )
        else:
            print(
                f"{m_str} vs GT ({sampling}, L = {L}): {np.allclose(flm_gt, flm_sov_fft_vec_jax, atol=1e-14)}"
            )

        # %timeit s2f.transform._forward(f, L, spin, sampling, method=m_str, nside=nside, reality=reality, L_lower=L_lower)
        # %timeit s2f.transform._forward(f, L, spin, sampling, method=m_str, nside=nside, reality=reality, L_lower=L_lower)
    print("-----------------------------------------")


# %%
