"""Numerical helpers shared across the library.

Apple's Accelerate BLAS leaves stale floating-point exception flags behind in some
matmul kernels, so NumPy reports divide-by-zero/overflow/invalid on products whose
results are entirely finite. Routing matrix products through matmul() silences those
spurious warnings while still failing loudly when a product genuinely degenerates.
"""

import numpy as np


def matmul(a: np.ndarray, b: np.ndarray, check: bool = True) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = a @ b
    if check and not np.isfinite(out).all():
        raise FloatingPointError(
            f"non-finite values in a {a.shape} x {b.shape} product; the inputs are likely corrupt"
        )
    return out
