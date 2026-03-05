# utils/__init__.py

__version__ = "0.4.1"

try:
    import jax.numpy
    use_jax = True
except ImportError:
    use_jax = False

if use_jax:
    from .coord_jax import (
        kaxes_1d,
        mu2_grid,
        apply_disp_k,
        apply_Gij_k,
        apply_grad_k,
        apply_gauss_k,
        apply_traceless,
        func_extend,
        func_reduce,
        func_reduce_hermite,
        func_reduce_to_cube,
        func_reduce_hermite_to_cube,
        func_reduce_hermite_to_cube_fixed,
        low_pass_filter_fourier,
        high_pass_filter_fourier,
    )
    from .forward_model_jax import Base_Forward, LPT_Forward, EPT_Forward
    from .utils_jax import growth_D_f, chi_of_z, orthogonalize, beta_polyfit, compute_corr_2d, check_max_rij, compute_pks_2d
else:  ### to be implemented
    from .coord import (
        kaxes_1d,
        mu2_grid,
        apply_disp_k,
        apply_Gij_k,
        apply_grad_k,
        apply_gauss_k,
        apply_traceless,
        func_extend,
        func_reduce,
        func_reduce_hermite,
        low_pass_filter_fourier,
        high_pass_filter_fourier,
    )
    from .forward_model import Base_Forward, LPT_Forward, EPT_Forward
    from .utils import growth_D_f, orthogonalize, beta_polyfit, compute_corr_2d, check_max_rij, compute_pks_2d

__all__ = [
    "__version__",
    "kaxes_1d",
    "mu2_grid",
    "apply_disp_k",
    "apply_Gij_k",
    "apply_grad_k",
    "apply_gauss_k",
    "apply_traceless",
    "func_extend",
    "func_reduce",
    "func_reduce_hermite",
    "func_reduce_to_cube",
    "func_reduce_hermite_to_cube",
    "func_reduce_hermite_to_cube_fixed",
    "low_pass_filter_fourier",
    "high_pass_filter_fourier",
    "Base_Forward",
    "LPT_Forward",
    "EPT_Forward",
    "orthogonalize",
    "beta_polyfit",
    "check_max_rij",
    "compute_corr_2d",
    "compute_pks_2d",
    "growth_D_f",
]
