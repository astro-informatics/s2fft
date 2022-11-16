"""
Checking diff in compute_slice using Turok approach

from /Users/sofia/Documents_local/SAX project/s2fft/tests/test_wigner.py
"""

# %% imports
import pytest
import numpy as np
import jax.numpy as jnp
import s2fft.wigner as wigner
import s2fft.samples as samples
import pyssht as ssht

from jax.config import config

config.update("jax_enable_x64", True)

# L_to_test = [8, 16]
# spin_to_test = np.arange(-2, 2)
# sampling_schemes = ["mw", "mwss", "dh", "healpix"]

L = 8
sampling = "mw"
spin = 2

# %%
# Test all dl() terms up to L.
betas = samples.thetas(L, sampling, int(L / 2))

# Compute using SSHT.
for beta in betas:
    dl_array = ssht.generate_dl(beta, L)

    for el in range(L):
        if el >= np.abs(spin):

            #############################
            # Non-JAX
            # in terms of flip: opposite orientation to GT
            dl_turok = wigner.turok.compute_slice(
                beta, el, L, -spin
            )  # np.flip(wigner.turok.compute_slice(beta, el, L, -spin))

            # compare full array
            if not (
                np.allclose(
                    dl_turok[:],
                    np.flip(dl_array[el][L - 1 + spin][:]),  # flip GT!
                    atol=1e-10,
                    rtol=1e-12,
                )
            ):  # -----returns True for all el

                print("False")

            # compare slice?
            if not (
                np.allclose(
                    dl_turok[L - 1 - el : L - 1 + el + 1],
                    np.flip(
                        dl_array[el][L - 1 + spin][L - 1 - el : L - 1 + el + 1]
                    ),  # flip GT!
                    atol=1e-10,
                    rtol=1e-12,
                )
            ):

                print("False")

            ############################
            # JAX vs GT: only match for that slice

            # in terms of flip: same 'orientation' as GT (if GT is flipped, dl_turok_jax should too)
            dl_turok_jax = jnp.flip(wigner.turok_jax.compute_slice(beta, el, L, -spin))

            ### compare full array ---no match
            # if not(np.allclose(dl_turok_jax[:],
            #                     np.flip(dl_array[el][L - 1 + spin][:]), #flip GT!
            #                     atol=1e-10, rtol=1e-12)): # -----returns True for all el

            #     print('False')
            #     print('----')
            # if not(np.allclose(dl_turok_jax[:],
            #                     np.flip(dl_array[el][L - 1 - spin][:]), #np.flip(dl_array[el][L - 1  spin][:]),
            #                     atol=1e-10,rtol=1e-12)): #----False; they do not match

            #     print('False')

            ### compare specific slide (GT flipped)
            # if not(np.allclose(dl_turok_jax[L - 1 - el : L - 1 + el + 1],
            #                     np.flip(dl_array[el][L - 1 + spin][L - 1 - el : L - 1 + el + 1]), #flip GT!
            #                     atol=1e-10,rtol=1e-12)):

            #     print('False')
            # OJO with [L-1-spin works!]
            if not (
                np.allclose(
                    dl_turok_jax[L - 1 - el : L - 1 + el + 1],
                    np.flip(
                        dl_array[el][L - 1 - spin][L - 1 - el : L - 1 + el + 1]
                    ),  # flip GT!
                    atol=1e-10,
                    rtol=1e-12,
                )
            ):

                print("False")

            ##############################
            # Compare non-JAX and JAX ---they only match for that slice!
            if not (
                np.allclose(
                    dl_turok_jax[L - 1 - el : L - 1 + el + 1],
                    dl_turok[L - 1 - el : L - 1 + el + 1],
                    atol=1e-10,
                    rtol=1e-12,
                )
            ):  # they do match for that slice!

                print("False")

# %%
# Test all dl() terms up to L.
betas = samples.thetas(L, sampling, int(L / 2))

# Compute using SSHT.
for beta in betas:
    dl_array = ssht.generate_dl(beta, L)

    for el in range(L):
        if el >= np.abs(spin):

            #############################
            # Non-JAX vs GT
            # in terms of flip: opposite orientation to GT
            dl_turok = wigner.turok.compute_slice(
                beta, el, L, -spin
            )  # np.flip(wigner.turok.compute_slice(beta, el, L, -spin))

            # compare full array
            if not (
                np.allclose(
                    np.flip(dl_turok[:]),
                    dl_array[el][L - 1 + spin][:],
                    atol=1e-10,
                    rtol=1e-12,
                )
            ):  # -----returns True for all el

                print("False")

            # compare slice?
            if not (
                np.allclose(
                    np.flip(dl_turok[L - 1 - el : L - 1 + el + 1]),
                    dl_array[el][L - 1 + spin][L - 1 - el : L - 1 + el + 1],
                    atol=1e-10,
                    rtol=1e-12,
                )
            ):

                print("False")

            #####################################
            # JAX vs non-JAX
            # to match them I need to:
            # - flip one of them
            # - change sign to spin
            # then they will match in this specific slice: [L - 1 - el : L - 1 + el + 1]
            spin = -spin  # OJO change sign!!!!
            dl_turok_jax = wigner.turok_jax.compute_slice(beta, el, L, -spin)

            # compare full array ---FALSE
            # if not(np.allclose(dl_turok,
            #                     jnp.flip(dl_turok_jax),
            #                     atol=1e-10, rtol=1e-12)):

            #     print('False for full array')

            # compare slice?
            if not (
                np.allclose(
                    dl_turok[L - 1 - el : L - 1 + el + 1],
                    jnp.flip(dl_turok_jax[L - 1 - el : L - 1 + el + 1]),
                    atol=1e-10,
                    rtol=1e-12,
                )
            ):

                print("False")


# %%
import pytest
import numpy as np
import jax.numpy as jnp
import s2fft.wigner as wigner
import s2fft.samples as samples
import pyssht as ssht

from jax.config import config

config.update("jax_enable_x64", True)

L = 8
sampling = "mw"
spin = -2
# Test all dl() terms up to L.
betas = samples.thetas(L, sampling, int(L / 2))

# Compute using SSHT.
match_per_beta = dict()
for beta in betas:
    dl_array = ssht.generate_dl(beta, L)

    match_per_el = list()
    for el in range(L):
        if el >= np.abs(spin):

            # Non-JAX
            dl_turok = wigner.turok.compute_slice(
                beta, el, L, -spin
            ) 

            # JAX
            # to match the JAX and the non-JAX output I need to:
            # - flip one of them
            # - change the sign of the spin passed as input to 'turok_jax'
            # then they will match in this specific slice: [L - 1 - el : L - 1 + el + 1]
            spin = -spin
            dl_turok_jax = wigner.turok_jax.compute_slice(beta, el, L, -spin)

            # compare slice
            match_per_el.append(
                np.allclose(
                    dl_turok[L - 1 - el : L - 1 + el + 1],
                    jnp.flip(dl_turok_jax[L - 1 - el : L - 1 + el + 1]),
                    atol=1e-10,
                    rtol=1e-12,
                )
            )

    match_per_beta[beta] = np.all(match_per_el)

# Check if all true
print(np.all(list(match_per_beta.values())))

# [print(f"beta = {k:.2f}---{v}") for k, v in match_per_beta.items()] # all True
# %%
