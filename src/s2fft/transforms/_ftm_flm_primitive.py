"""
Custom JAX primitives for the latitudinal Wigner-d recursion steps.

The forward (``ftm -> flm``) and inverse (``flm -> ftm``) latitudinal steps
are linear maps and are transposes of each other. Wrapping them as JAX
primitives with explicit transpose, JVP and batching rules avoids the
limitations of ``custom_vjp`` (no transpose rule — breaks under nested
transforms) and ``linear_call`` (no batching rule).

Each primitive accepts the following operands:

* ``data`` — the array we differentiate w.r.t. (``flm`` or ``ftm``).
* ``thetas`` — JAX array of polar sample positions (never batched).
* ``spin`` — 0-d integer JAX array. Must be a JAX operand (not a static
  kwarg) because the call sites in ``spherical.py`` invoke the primitive
  while ``spin`` is a JIT tracer.
* ``*precomps`` — five JAX arrays produced by
  :func:`generate_precomputes_jax`: ``(lrenorm, vsign, cpi, cp2, indices)``.
  May be omitted entirely (zero operands), in which case the underlying
  function regenerates them at trace time.

Static keyword params: ``L``, ``nside``, ``sampling``, ``reality``,
``spmd``, ``L_lower``.

The primitives are batch-aware: the batcher lifts every operand (other
than ``thetas``) to share one leading batch dimension, and the abstract
eval / lowering / transpose then peel off that batch dim via internal
``vmap``. This means the analytical adjoint (forward <-> inverse step) is
preserved under ``vmap`` — composed transforms like ``grad(vmap(...))``
take the analytical-transpose path rather than autodiffing through the
``log``/``exp`` recursions inside ``otf`` (which can produce ``NaN`` in
edge cases).
"""

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import mlir

from s2fft.sampling import s2_samples as samples
from s2fft.transforms import otf_recursions as otf
from s2fft.utils.jax_primitive import register_primitive

# Both ``flm`` and ``ftm`` are 2D in the unbatched case; anything past these
# trailing dims is treated as a batch dimension shared across all batched
# operands.
_DATA_NDIM = 2


# ---------------------------------------------------------------------------
# Helper functions for constructing primitives
# ---------------------------------------------------------------------------


def _apply_with_batching(fn_unbatched, data, spin, precomps):
    """
    Call ``fn_unbatched(data2d, spin0d, precomps_or_None)`` over leading
    batch dims. ``data`` is 2D + batch, ``spin`` is 0d + same batch, each
    precomp array also has the same number of leading batch dims as ``data``.
    """
    n_batch = data.ndim - _DATA_NDIM
    inner = fn_unbatched
    in_axes = (0, 0, (0,) * len(precomps))
    for _ in range(n_batch):
        inner = jax.vmap(inner, in_axes=in_axes)
    return inner(data, spin, precomps)


def _tuplify_precomps(precomps: list | None) -> tuple:
    return tuple(precomps) if precomps is not None else ()


def _untuplify_precomps(precomps: tuple) -> list | None:
    return list(precomps) if precomps else None


def _as_spin_operand(spin) -> jnp.ndarray:
    """
    Normalize ``spin`` to a 0-d int JAX array. Accepts Python ints,
    numpy scalars or existing JAX tracers / arrays.
    """
    return jnp.asarray(spin, dtype=jnp.int64)


def _lift_to_batch(arr, ax, batch_size):
    """
    Move axis ``ax`` to position 0, or broadcast to a leading batch dim
    of size ``batch_size`` if ``ax is None``.
    """
    if ax is None:
        return jnp.broadcast_to(arr, (batch_size,) + arr.shape)
    return jnp.moveaxis(arr, ax, 0)


def _batch_primitive(primitive, batched_args, batch_axes, **params):
    """
    Batch ftm_to_flm or flm_to_ftm primitive given batched_args with first
    argument corresponding to either ftm or flm array (fxm used as placeholder).
    """
    fxm, thetas, spin, *precomps = batched_args
    fxm_ax, thetas_ax, spin_ax, *precomps_ax = batch_axes
    if thetas_ax is not None:
        raise NotImplementedError(
            "vmap over `thetas` is not supported (it is determined by the "
            "static sampling configuration)."
        )
    # Identify batch size from the first batched operand.
    arrays_axes = [(fxm, fxm_ax), (spin, spin_ax)] + list(
        zip(precomps, precomps_ax, strict=False)
    )
    batch_size = next(
        (arr.shape[ax] for arr, ax in arrays_axes if ax is not None),
        None,
    )
    if batch_size is None:
        return primitive.bind(fxm, thetas, spin, *precomps, **params), None
    fxm_b = _lift_to_batch(fxm, fxm_ax, batch_size)
    spin_b = _lift_to_batch(spin, spin_ax, batch_size)
    precomps_b = tuple(
        _lift_to_batch(p, ax, batch_size)
        for p, ax in zip(precomps, precomps_ax, strict=False)
    )
    return primitive.bind(fxm_b, thetas, spin_b, *precomps_b, **params), 0


# ---------------------------------------------------------------------------
# flm_to_ftm primitive (inverse latitudinal step)
# ---------------------------------------------------------------------------


def _flm_to_ftm_abstract(
    flm, thetas, spin, *precomps, L, nside, sampling, reality, spmd, L_lower
):
    # As inverse_latitudinal_step_jax determines shape of first dimension of ftm value
    # returned by size (of last dimension if batched) of thetas argument instead of
    # the value returned by samples.ntheta / samples.ftm_shape(...)[0], to allow
    # for the upsampling performed for MW + MWSS schemes, we cannot use
    # samples.ftm_shape directly here as might be expected
    out_shape = flm.shape[:-_DATA_NDIM] + (
        thetas.shape[-1],
        samples.ftm_shape(L, sampling, nside)[1],
    )
    return ShapedArray(out_shape, flm.dtype)


def _flm_to_ftm_impl(flm, thetas, spin, *precomps, **params):
    def fn(d, s, p):
        return otf.inverse_latitudinal_step_jax(
            flm=d, beta=thetas, spin=s, precomps=_untuplify_precomps(p), **params
        )

    return _apply_with_batching(fn, flm, spin, precomps)


def _flm_to_ftm_jvp(primals, tangents, **params):
    flm, thetas, spin, *precomps = primals
    flm_t, *_ = tangents
    primal_out = _flm_to_ftm_primitive.bind(flm, thetas, spin, *precomps, **params)
    if isinstance(flm_t, jax.interpreters.ad.Zero):
        tangent_out = jax.interpreters.ad.Zero(primal_out.aval)
    else:
        tangent_out = _flm_to_ftm_primitive.bind(
            flm_t, thetas, spin, *precomps, **params
        )
    return primal_out, tangent_out


def _flm_to_ftm_transpose(cotangent, flm, thetas, spin, *precomps, **params):
    # The transpose of the flm_to_ftm primitive (applied to the initial flm argument) is
    # the ftm_to_flm primitive. We do not pass through the supplied (flm_to_ftm) precomps
    # so these are regenerated internally for the ftm_to_flm primitive.
    cot_flm = ftm_to_flm(cotangent, thetas, spin=spin, **params)
    return (cot_flm, None, None) + (None,) * len(precomps)


def _flm_to_ftm_batcher(batched_args, batch_axes, **params):
    return _batch_primitive(_flm_to_ftm_primitive, batched_args, batch_axes, **params)


_flm_to_ftm_primitive = register_primitive(
    "flm_to_ftm",
    multiple_results=False,
    abstract_evaluation=_flm_to_ftm_abstract,
    lowering_per_platform={
        None: mlir.lower_fun(_flm_to_ftm_impl, multiple_results=False),
    },
    batcher=_flm_to_ftm_batcher,
    jacobian_vector_product=_flm_to_ftm_jvp,
    transpose=_flm_to_ftm_transpose,
    # While primitive is linear in first argument, we require thetas & spin arguments to
    # compute linear map (transposed) as function of first argument. JAX assumes either:
    # (i) all primal arguments not required to compute transpose (when using deflinear),
    # (ii) primitive is differentiable wrt all primal arguments (when using deflinear2).
    # As we cannot differentiate wrt integer spin argument we manually define tranpose
    # rule return None for arguments other than first.
    is_linear=False,
)


# ---------------------------------------------------------------------------
# ftm_to_flm primitive (forward latitudinal step)
# ---------------------------------------------------------------------------


def _ftm_to_flm_abstract(
    ftm, thetas, spin, *precomps, L, nside, sampling, reality, spmd, L_lower
):
    out_shape = ftm.shape[:-_DATA_NDIM] + samples.flm_shape(L)
    return ShapedArray(out_shape, ftm.dtype)


def _ftm_to_flm_impl(ftm, thetas, spin, *precomps, **params):
    def fn(d, s, p):
        return otf.forward_latitudinal_step_jax(
            ftm_in=d, beta_in=thetas, spin=s, precomps=_untuplify_precomps(p), **params
        )

    return _apply_with_batching(fn, ftm, spin, precomps)


def _ftm_to_flm_jvp(primals, tangents, **params):
    ftm, thetas, spin, *precomps = primals
    ftm_t, *_ = tangents
    primal_out = _ftm_to_flm_primitive.bind(ftm, thetas, spin, *precomps, **params)
    if isinstance(ftm_t, jax.interpreters.ad.Zero):
        tangent_out = jax.interpreters.ad.Zero(primal_out.aval)
    else:
        tangent_out = _ftm_to_flm_primitive.bind(
            ftm_t, thetas, spin, *precomps, **params
        )
    return primal_out, tangent_out


def _ftm_to_flm_transpose(cotangent, ftm, thetas, spin, *precomps, **params):
    # The transpose of the ftm_to_flm primitive (applied to the initial ftm argument) is
    # the flm_to_ftm primitive. We do not pass through the supplied (ftm_to_flm) precomps
    # so these are regenerated internally for the flm_to_ftm primitive.
    cot_ftm = flm_to_ftm(cotangent, thetas, spin=spin, **params)
    return (cot_ftm, None, None) + (None,) * len(precomps)


def _ftm_to_flm_batcher(batched_args, batch_axes, **params):
    return _batch_primitive(_ftm_to_flm_primitive, batched_args, batch_axes, **params)


_ftm_to_flm_primitive = register_primitive(
    "ftm_to_flm",
    multiple_results=False,
    abstract_evaluation=_ftm_to_flm_abstract,
    lowering_per_platform={
        None: mlir.lower_fun(_ftm_to_flm_impl, multiple_results=False),
    },
    batcher=_ftm_to_flm_batcher,
    jacobian_vector_product=_ftm_to_flm_jvp,
    transpose=_ftm_to_flm_transpose,
    # Primitive is linear in first argument but we cannot use JAX deflinear machinery.
    # See note in comment in _flm_to_ftm_primitive definition for explanation.
    is_linear=False,
)


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------


def flm_to_ftm(
    flm: jnp.ndarray,
    thetas: jnp.ndarray,
    *,
    L: int,
    spin,
    nside: int | None,
    sampling: str,
    reality: bool,
    spmd: bool,
    L_lower: int,
    precomps: list | None = None,
) -> jnp.ndarray:
    """
    Inverse latitudinal step (``flm -> ftm``) via custom JAX primitive.

    If ``precomps`` is None the underlying function regenerates them at
    trace time; this matches the behaviour of
    :func:`otf.inverse_latitudinal_step_jax`.
    """
    return _flm_to_ftm_primitive.bind(
        flm,
        thetas,
        _as_spin_operand(spin),
        *_tuplify_precomps(precomps),
        L=L,
        nside=nside,
        sampling=sampling,
        reality=reality,
        spmd=spmd,
        L_lower=L_lower,
    )


def ftm_to_flm(
    ftm: jnp.ndarray,
    thetas: jnp.ndarray,
    *,
    L: int,
    spin,
    nside: int | None,
    sampling: str,
    reality: bool,
    spmd: bool,
    L_lower: int,
    precomps: list | None = None,
) -> jnp.ndarray:
    """
    Forward latitudinal step (``ftm -> flm``) via custom JAX primitive.

    If ``precomps`` is None the underlying function regenerates them at
    trace time; this matches the behaviour of
    :func:`otf.forward_latitudinal_step_jax`.
    """
    return _ftm_to_flm_primitive.bind(
        ftm,
        thetas,
        _as_spin_operand(spin),
        *_tuplify_precomps(precomps),
        L=L,
        nside=nside,
        sampling=sampling,
        reality=reality,
        spmd=spmd,
        L_lower=L_lower,
    )
