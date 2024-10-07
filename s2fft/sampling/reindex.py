from functools import partial

import jax.numpy as jnp
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
    els = jnp.arange(L)
    offset = els**2 + els
    for el in range(L):
        m_array = jnp.arange(-el, el + 1)
        flm_2d = flm_2d.at[el, L - 1 + m_array].set(flm_1d[offset[el] + m_array])
    return flm_2d


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
    flm_1d = jnp.zeros(L**2, dtype=jnp.complex128)
    els = jnp.arange(L)
    offset = els**2 + els
    for el in range(L):
        m_array = jnp.arange(-el, el + 1)
        flm_1d = flm_1d.at[offset[el] + m_array].set(flm_2d[el, L - 1 + m_array])
    return flm_1d


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
    flm_2d = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)

    for el in range(L):
        flm_2d = flm_2d.at[el, L - 1].set(flm_hp[el])
        m_array = jnp.arange(1, el + 1)
        hp_idx = m_array * (2 * L - 1 - m_array) // 2 + el
        flm_2d = flm_2d.at[el, L - 1 + m_array].set(flm_hp[hp_idx])
        flm_2d = flm_2d.at[el, L - 1 - m_array].set(
            (-1) ** m_array * jnp.conj(flm_hp[hp_idx])
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
    flm_hp = jnp.zeros(int(L * (L + 1) / 2), dtype=jnp.complex128)

    for el in range(L):
        m_array = jnp.arange(el + 1)
        hp_idx = m_array * (2 * L - 1 - m_array) // 2 + el
        flm_hp = flm_hp.at[hp_idx].set(flm_2d[el, L - 1 + m_array])

    return flm_hp
