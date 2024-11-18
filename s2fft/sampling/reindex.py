from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit


@partial(jit, static_argnums=(1))
def flm_1d_to_2d_fast(flm_1d: jnp.ndarray, L: int) -> jnp.ndarray:
    r"""
    Convert from 1D indexed harmnonic coefficients to 2D indexed coefficients (JAX).
    
    Note:
        Storage conventions for harmonic coefficients :math:`flm_{(\ell,m)}`, for 
        e.g. :math:`L = 3`, are as follows.

        .. math::

            \text{ 2D data format}:
                \begin{bmatrix}
                    0 & 0 & flm_{(0,0)} & 0 & 0 \\
                    0 & flm_{(1,-1)} & flm_{(1,0)} & flm_{(1,1)} & 0 \\
                    flm_{(2,-2)} & flm_{(2,-1)} & flm_{(2,0)} & flm_{(2,1)} & flm_{(2,2)}
                \end{bmatrix}
        
        .. math::

            \text{1D data format}:  [flm_{0,0}, flm_{1,-1}, flm_{1,0}, flm_{1,1}, \dots]

    Args:
        flm_1d (jnp.ndarray): 1D indexed harmonic coefficients.

        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: 2D indexed harmonic coefficients.

    """
    flm_2d = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    row_indices, col_indices = np.arange(L)[:, None], np.arange(2 * L - 1)[None, :]
    el_indices, m_indices = np.where(
        (row_indices <= col_indices)[::-1, :] & (row_indices <= col_indices)[::-1, ::-1]
    )
    return flm_2d.at[el_indices, m_indices].set(flm_1d)


@partial(jit, static_argnums=(1))
def flm_2d_to_1d_fast(flm_2d: jnp.ndarray, L: int) -> jnp.ndarray:
    r"""
    Convert from 2D indexed harmonic coefficients to 1D indexed coefficients (JAX).
    
    Note:
        Storage conventions for harmonic coefficients :math:`flm_{(\ell,m)}`, for 
        e.g. :math:`L = 3`, are as follows.

        .. math::

            \text{ 2D data format}:
                \begin{bmatrix}
                    0 & 0 & flm_{(0,0)} & 0 & 0 \\
                    0 & flm_{(1,-1)} & flm_{(1,0)} & flm_{(1,1)} & 0 \\
                    flm_{(2,-2)} & flm_{(2,-1)} & flm_{(2,0)} & flm_{(2,1)} & flm_{(2,2)}
                \end{bmatrix}
        
        .. math::

            \text{1D data format}:  [flm_{0,0}, flm_{1,-1}, flm_{1,0}, flm_{1,1}, \dots]

    Args:
        flm_2d (jnp.ndarray): 2D indexed harmonic coefficients.

        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: 1D indexed harmonic coefficients.

    """
    row_indices, col_indices = np.arange(L)[:, None], np.arange(2 * L - 1)[None, :]
    el_indices, m_indices = np.where(
        (row_indices <= col_indices)[::-1, :] & (row_indices <= col_indices)[::-1, ::-1]
    )
    return flm_2d[el_indices, m_indices]


@partial(jit, static_argnums=(1))
def flm_hp_to_2d_fast(flm_hp: jnp.ndarray, L: int) -> jnp.ndarray:
    r"""
    Converts from HEALPix (healpy) indexed harmonic coefficients to 2D indexed
    coefficients (JAX).
    
    Notes:
        HEALPix implicitly assumes conjugate symmetry and thus only stores positive `m` 
        coefficients. Here we unpack that into harmonic coefficients of an 
        explicitly real signal.

    Warning:
        Note that the harmonic band-limit `L` differs to the HEALPix `lmax` convention,
        where `L = lmax + 1`.

    Note:
        Storage conventions for harmonic coefficients :math:`f_{(\ell,m)}`, for 
        e.g. :math:`L = 3`, are as follows.

        .. math::
            \text{ 2D data format}:
                \begin{bmatrix}
                    0 & 0 & flm_{(0,0)} & 0 & 0 \\
                    0 & flm_{(1,-1)} & flm_{(1,0)} & flm_{(1,1)} & 0 \\
                    flm_{(2,-2)} & flm_{(2,-1)} & flm_{(2,0)} & flm_{(2,1)} & flm_{(2,2)}
                \end{bmatrix}
        
        .. math::

            \text{HEALPix}: [flm_{(0,0)}, \dots, flm_{(2,0)}, flm_{(1,1)}, \dots, flm_{(L-1,1)}, \dots]

    Note:
        Returns harmonic coefficients of an explicitly real signal.

    Args:
        flm_hp (jnp.ndarray): HEALPix indexed harmonic coefficients.

        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: 2D indexed harmonic coefficients.

    """
    flm_2d = jnp.zeros((L, 2 * L - 1), dtype=flm_hp.dtype)
    m_indices, el_indices = np.triu_indices(n=L + 1, m=L)
    flm_2d = flm_2d.at[el_indices, L - 1 + m_indices].set(flm_hp)
    flm_2d = flm_2d.at[el_indices, L - 1 - m_indices].set(
        (-1) ** m_indices * flm_hp.conj()
    )
    return flm_2d


@partial(jit, static_argnums=(1))
def flm_2d_to_hp_fast(flm_2d: jnp.ndarray, L: int) -> jnp.ndarray:
    r"""
    Converts from 2D indexed harmonic coefficients to HEALPix (healpy) indexed
    coefficients (JAX).
    
    Note:
        HEALPix implicitly assumes conjugate symmetry and thus only stores positive `m` 
        coefficients. So this function discards the negative `m` values. This process 
        is NOT invertible! See the `healpy api docs <https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.alm2map.html>`_ 
        for details on healpy indexing and lengths.

    Note:
        Storage conventions for harmonic coefficients :math:`f_{(\ell,m)}`, for 
        e.g. :math:`L = 3`, are as follows.

        .. math::
            \text{ 2D data format}:
                \begin{bmatrix}
                    0 & 0 & flm_{(0,0)} & 0 & 0 \\
                    0 & flm_{(1,-1)} & flm_{(1,0)} & flm_{(1,1)} & 0 \\
                    flm_{(2,-2)} & flm_{(2,-1)} & flm_{(2,0)} & flm_{(2,1)} & flm_{(2,2)}
                \end{bmatrix}
        
        .. math::

            \text{HEALPix}: [flm_{(0,0)}, \dots, flm_{(2,0)}, flm_{(1,1)}, \dots, flm_{(L-1,1)}, \dots]

    Warning:
        Returns harmonic coefficients of an explicitly real signal.

    Warning:
        Note that the harmonic band-limit `L` differs to the HEALPix `lmax` convention,
        where `L = lmax + 1`.

    Args:
        flm_2d (jnp.ndarray): 2D indexed harmonic coefficients.

        L (int): Harmonic band-limit.
        
    Returns:
        jnp.ndarray: HEALPix indexed harmonic coefficients.

    """
    m_indices, el_indices = np.triu_indices(n=L + 1, m=L)
    return flm_2d[el_indices, L - 1 + m_indices]
