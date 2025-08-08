import jax.numpy as jnp
import jax
from jax import jit, vmap
from functools import partial

@jit
def _E(a, omega_m0):
    """E(a) = H(a)/H0 for flat ΛCDM (radiation neglected)."""
    return jnp.sqrt(omega_m0 / a**3 + (1.0 - omega_m0))


@jit
def _integrand(a, omega_m0):
    """1 / [a^3 E(a)^3]"""
    return 1.0 / (a**3 * _E(a, omega_m0) ** 3)


def _D_raw(a, omega_m0, n_steps=2048, a_min=1e-4):
    """growth function D(a, O_M0), not normalized"""
    a_grid = jnp.linspace(a_min, a, n_steps)
    I = jnp.trapezoid(_integrand(a_grid, omega_m0), a_grid)
    return 2.5 * omega_m0 * _E(a, omega_m0) * I


@partial(jit, static_argnames=("n_steps",))
def growth_D_f(z, omega_m0, *, n_steps=2048, a_min=1e-4):
    """
    f = d ln D / d ln a from auto-diff

    Parameters
    ----------
    z : array_like
        Redshift(s).
    omega_m0 : float
        Omega_{m,0}
    n_steps : int
        Number of integration grid steps (O(10^3) or more is sufficient).
    a_min : float
        Lower limit of integration. It does not significantly affect the results,
        but it is recommended to set it around 1e-4.

    Returns
    -------
    D, f : jnp.ndarray, jnp.ndarray
        Normalized growth function D(z) and growth rate f(z).
    """
    # Define the scalar function D(a)
    D1 = _D_raw(1.0, omega_m0, n_steps, a_min)  # Normalization constant

    def D_of_a(a):
        return _D_raw(a, omega_m0, n_steps, a_min) / D1

    # Prepare ln D composed with exp (input is ln a)
    lnD = lambda ln_a: jnp.log(D_of_a(jnp.exp(ln_a)))

    # Vectorize
    a = 1.0 / (1.0 + jnp.atleast_1d(z))           # shape = (N,)
    ln_a = jnp.log(a)                              # shape = (N,)

    D_vals = vmap(D_of_a)(a)                       # D(a)
    f_vals = vmap(jax.grad(lnD))(ln_a)             # d ln D / d ln a

    return D_vals.reshape(jnp.shape(z)), f_vals.reshape(jnp.shape(z))