# utils/__init__.py

__version__ = "0.2.1"

try:
    import jax.numpy
    use_jax = True
except ImportError:
    use_jax = False

if use_jax:
    from .coord_jax import (
        #rfftn_kvec,
        #rfftn_khat,
        #rfftn_mu2,
        #rfftn_disp,
        #rfftn_nabla,
        #rfftn_Gauss,
        #rfftn_sij,
        #rfftn_Gij,
        func_extend,
        func_reduce,
        func_reduce_hermite,
        low_pass_filter_fourier,
        high_pass_filter_fourier,
    )
    from .forward_model_jax import Base_Forward, LPT_Forward
    from .utils_jax import growth_D_f, orthogonalize, beta_polyfit, compute_corr_2d, check_max_rij, compute_pks_2d
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
    from .forward_model import Base_Forward, LPT_Forward

__all__ = [
    "__version__",
    #"rfftn_kvec",
    #"rfftn_khat",
    #"rfftn_mu2",
    #"rfftn_disp",
    #"rfftn_nabla",
    #"rfftn_Gauss",
    #"rfftn_sij",
    #"rfftn_Gij",
    "func_extend",
    "func_reduce",
    "func_reduce_hermite",
    "low_pass_filter_fourier",
    "high_pass_filter_fourier",
    "Base_Forward",
    "LPT_Forward",
    "orthogonalize",
    "beta_polyfit",
    "check_max_rij",
    "compute_corr_2d",
    "compute_pks_2d",
    "growth_D_f",
]
