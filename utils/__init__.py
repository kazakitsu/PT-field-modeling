# utils/__init__.py

__version__ = "0.1.0"

try:
    import jax.numpy
    use_jax = True
except ImportError:
    use_jax = False

if use_jax:
    from .coord_jax import (
        rfftn_kvec,
        rfftn_khat,
        rfftn_mu2,
        rfftn_disp,
        rfftn_nabla,
        rfftn_Gauss,
        rfftn_sij,
        rfftn_Gij,
        func_extend,
        func_reduce,
        func_reduce_hermite,
        low_pass_filter_fourier,
        high_pass_filter_fourier,
    )
    from .forward_model_jax import Base_Forward, LPT_Forward, orthogonalize
else:
    from .coord import (
        rfftn_kvec,
        rfftn_khat,
        rfftn_mu2,
        rfftn_disp,
        rfftn_nabla,
        rfftn_Gauss,
        rfftn_sij,
        rfftn_Gij,
        func_extend,
        func_reduce,
        func_reduce_hermite,
        low_pass_filter_fourier,
        high_pass_filter_fourier,
    )
    from .forward_model import Base_Forward, LPT_Forward, orthogonalize

__all__ = [
    "__version__",
    "rfftn_kvec",
    "rfftn_khat",
    "rfftn_mu2",
    "rfftn_disp",
    "rfftn_nabla",
    "rfftn_Gauss",
    "rfftn_sij",
    "rfftn_Gij",
    "func_extend",
    "func_reduce",
    "func_reduce_hermite",
    "low_pass_filter_fourier",
    "high_pass_filter_fourier",
    "Base_Forward",
    "LPT_Forward",
    "orthogonalize",
]
