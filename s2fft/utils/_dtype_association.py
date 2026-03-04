# A function is possibly overkill, though people _could_ be using weird types.
# In which case, we'd need a function that extracts the "number" and "float/complex"
# bit from the dtype and does the appropriate operation.
_cmplx_dtype_from = {
    "complex128": "complex128",
    "float64": "complex128",
    "complex64": "complex64",
    "float32": "complex64",
}
