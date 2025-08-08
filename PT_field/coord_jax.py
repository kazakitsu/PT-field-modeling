# !/usr/bin/env python3

import jax.numpy as jnp
from jax import jit
from functools import partial

# --- k-vector Generation ---

@partial(jit, static_argnames=('shape', 'dtype'))
def rfftn_kvec(shape, boxsize, dtype=jnp.float64):
    """
    Generate wavevectors for `jax.numpy.fft.rfftn`.
    """
    spacing = boxsize / (2.*jnp.pi) / shape[-1]
    # Create 1D frequency arrays for each dimension.
    freqs = [jnp.fft.fftfreq(n, d=spacing) for n in shape[:-1]]
    freqs.append(jnp.fft.rfftfreq(shape[-1], d=spacing))

    # Use jnp.meshgrid to create the coordinate grid.
    kvec_grid = jnp.meshgrid(*freqs, indexing='ij')
    
    # Stack the coordinate arrays to get the final (D, N1, N2, ...) shape.
    kvec = jnp.stack(kvec_grid, axis=0)
    
    return kvec.astype(dtype)

# --- Basic Fourier Operators ---

@jit
def rfftn_khat(kvec):
    """
    Calculates the unit wavevectors khat = k / |k|.
    """
    kmag_sq = jnp.sum(kvec**2, axis=0)
    kmag = jnp.sqrt(kmag_sq)
    # Use jnp.where to handle the k=0 case safely.
    return jnp.where(kmag == 0.0, 0.0, kvec / kmag)

@jit
def rfftn_mu2(kvec):
    """
    Calculates the squared cosine of the angle with the z-axis, (kz/|k|)^2.
    """
    k2 = jnp.sum(kvec**2, axis=0)
    return jnp.where(k2 == 0.0, 0.0, kvec[2]**2 / k2)

@jit
def rfftn_disp(kvec):
    """Calculates the displacement operator 1j * k / |k|^2."""
    k2 = jnp.sum(kvec**2, axis=0)
    return jnp.where(k2 == 0.0, 0.0, 1j * kvec / k2)

@jit
def rfftn_nabla(kvec):
    """Calculates the gradient operator (nabla) in Fourier space, 1j * k."""
    return 1j * kvec

@jit
def rfftn_Gauss(kvec, R):
    """Applies a Gaussian window/filter in Fourier space."""
    k2 = jnp.sum(kvec**2, axis=0)
    return jnp.exp(-0.5 * k2 * (R**2))

# --- Tensor Operators (Tide and G2) ---

@jit
def rfftn_sij(kvec):
    """
    Calculates the 6 unique components of the tidal tensor sij = ki*kj/|k|^2 - (1/3)delta_ij.
    """
    k2 = jnp.sum(kvec**2, axis=0)
    inv_k2 = jnp.where(k2 == 0.0, 0.0, 1.0 / k2)
    k_x, k_y, k_z = kvec[0], kvec[1], kvec[2]

    # Calculate the 6 components of ki*kj/|k|^2
    s_xx = k_x * k_x * inv_k2
    s_xy = k_x * k_y * inv_k2
    s_xz = k_x * k_z * inv_k2
    s_yy = k_y * k_y * inv_k2
    s_yz = k_y * k_z * inv_k2
    s_zz = k_z * k_z * inv_k2

    # Subtract the trace term (1/3 * delta_ij) from the diagonal components.
    trace_term = (1.0 / 3.0) * (k2 != 0.0)
    s_xx -= trace_term
    s_yy -= trace_term
    s_zz -= trace_term

    return jnp.stack([s_xx, s_xy, s_xz, s_yy, s_yz, s_zz], axis=0)

@jit
def rfftn_Gij(kvec):
    """
    Calculates the 6 unique components of the tensor Gij = ki*kj/|k|^2.
    """
    k2 = jnp.sum(kvec**2, axis=0)
    inv_k2 = jnp.where(k2 == 0.0, 0.0, 1.0 / k2)
    k_x, k_y, k_z = kvec[0], kvec[1], kvec[2]

    # Calculate the 6 components directly.
    g_xx = k_x * k_x * inv_k2
    g_xy = k_x * k_y * inv_k2
    g_xz = k_x * k_z * inv_k2
    g_yy = k_y * k_y * inv_k2
    g_yz = k_y * k_z * inv_k2
    g_zz = k_z * k_z * inv_k2

    return jnp.stack([g_xx, g_xy, g_xz, g_yy, g_yz, g_zz], axis=0)

# --- Grid Manipulation (Extend, Reduce, Hermite) ---

@partial(jit, static_argnames=('ng_ext',))
def func_extend(ng_ext, array_3d):
    """Pads a 3D array into a larger grid with zeros (JAX version)."""
    ng = array_3d.shape[0]
    ng_half = ng // 2

    array_extended = jnp.zeros((ng_ext, ng_ext, ng_ext // 2 + 1), dtype=array_3d.dtype)

    # Define source and destination indices.
    src_pos_x, dst_pos_x = slice(0, ng_half), slice(0, ng_half)
    src_neg_x, dst_neg_x = slice(-ng_half, None), slice(-ng_half, None)
    idx_y = jnp.r_[0:ng_half, ng_ext - ng_half:ng_ext]
    idx_z = slice(0, ng // 2 + 1)

    # Use JAX's functional update syntax.
    array_extended = array_extended.at[dst_pos_x, idx_y, idx_z].set(array_3d[src_pos_x, :, :])
    array_extended = array_extended.at[dst_neg_x, idx_y, idx_z].set(array_3d[src_neg_x, :, :])
    
    return array_extended

@partial(jit, static_argnames=('ng_red',))
def func_reduce(ng_red, array_3d):
    """Reduces a Fourier-space array by truncating high frequencies (JAX version)."""
    ng = array_3d.shape[0]
    ng_red_half = ng_red // 2

    idx_xy = jnp.concatenate([
        jnp.arange(ng_red_half),
        jnp.arange(ng - ng_red_half, ng)
    ])

    idx_z = jnp.arange(ng_red_half + 1)

    # Slicing in JAX is efficient.
    array_reduced = array_3d[jnp.ix_(idx_xy, idx_xy, idx_z)]
    return array_reduced

@jit
def _enforce_hermite(array_k):
    """Enforces Hermitian symmetry on a complex array (JAX version)."""
    ng_k = array_k.shape[0]
    nk_half = ng_k // 2
    
    # Use a temporary variable for chained updates.
    array_out = array_k

    # Corner points must be real.
    array_out = array_out.at[0, 0, 0].set(array_out[0, 0, 0].real)
    array_out = array_out.at[nk_half, 0, 0].set(array_out[nk_half, 0, 0].real)
    array_out = array_out.at[0, nk_half, 0].set(array_out[0, nk_half, 0].real)
    array_out = array_out.at[nk_half, nk_half, 0].set(array_out[nk_half, nk_half, 0].real)
    
    array_out = array_out.at[0, 0, -1].set(array_out[0, 0, -1].real)
    array_out = array_out.at[nk_half, 0, -1].set(array_out[nk_half, 0, -1].real)
    array_out = array_out.at[0, nk_half, -1].set(array_out[0, nk_half, -1].real)
    array_out = array_out.at[nk_half, nk_half, -1].set(array_out[nk_half, nk_half, -1].real)

    # Enforce conjugate symmetry on axes.
    array_out = array_out.at[0, -1:nk_half:-1, 0].set(array_out[0, 1:nk_half, 0].conj())
    array_out = array_out.at[-1:nk_half:-1, 0, 0].set(array_out[1:nk_half, 0, 0].conj())
    array_out = array_out.at[0, -1:nk_half:-1, -1].set(array_out[0, 1:nk_half, -1].conj())
    array_out = array_out.at[-1:nk_half:-1, 0, -1].set(array_out[1:nk_half, 0, -1].conj())

    return array_out

@partial(jit, static_argnames=('ng_red',))
def func_reduce_hermite(ng_red, array_3d):
    """Reduces an array and enforces Hermitian symmetry."""
    array_reduced = func_reduce(ng_red, array_3d)
    return _enforce_hermite(array_reduced)

# --- Fourier Space Filters ---

@partial(jit, static_argnames=('cutoff',))
def _apply_cubic_low_pass_filter(cutoff, array_3d):
    ng = array_3d.shape[0]
    ng_filter_half = int(cutoff) // 2
    filtered_array = jnp.zeros_like(array_3d)
    idx_xy = jnp.concatenate([
        jnp.arange(ng_filter_half),
        jnp.arange(ng - ng_filter_half, ng)
    ])
    idx_z  = jnp.arange(ng_filter_half + 1) 
    low_freq_grid_idx = jnp.ix_(idx_xy, idx_xy, idx_z)
    return filtered_array.at[low_freq_grid_idx].set(array_3d[low_freq_grid_idx])

@jit
def _apply_spherical_low_pass_filter(k_cutoff, array_3d, kvec):
    k2 = jnp.sum(kvec**2, axis=0)
    mask = k2 <= (k_cutoff**2)
    return array_3d * mask

@partial(jit, static_argnames=('cutoff',))
def _apply_cubic_high_pass_filter(cutoff, array_3d):
    ng = array_3d.shape[0]
    ng_filter_half = int(cutoff) // 2
    idx_xy = jnp.concatenate([
        jnp.arange(ng_filter_half),
        jnp.arange(ng - ng_filter_half, ng)
    ])
    idx_z  = jnp.arange(ng_filter_half + 1)
    low_freq_grid_idx = jnp.ix_(idx_xy, idx_xy, idx_z)
    return array_3d.at[low_freq_grid_idx].set(0)

@jit
def _apply_spherical_high_pass_filter(k_cutoff, array_3d, kvec):
    k2 = jnp.sum(kvec**2, axis=0)
    mask = k2 > (k_cutoff**2)
    return array_3d * mask

def low_pass_filter_fourier(filter_type, cutoff_param, array_3d, kvec=None):
    """
    Applies a sharp low-pass filter in Fourier space.
    
    Args:
        filter_type (str): 'cubic' or 'spherical'.
        cutoff_param (float): Grid size for cubic, radius for spherical.
        array_3d (jnp.ndarray): Input array.
        kvec (jnp.ndarray, optional): Wavevector array, required for spherical filter.
    """
    if filter_type == 'cubic':
        return _apply_cubic_low_pass_filter(cutoff_param, array_3d)
    elif filter_type == 'spherical':
        if kvec is None:
            raise ValueError("kvec must be provided for the spherical filter.")
        return _apply_spherical_low_pass_filter(cutoff_param, array_3d, kvec)
    else:
        raise ValueError("filter_type must be 'cubic' or 'spherical'")

def high_pass_filter_fourier(filter_type, cutoff_param, array_3d, kvec=None):
    """
    Applies a sharp high-pass filter in Fourier space.

    Args:
        filter_type (str): 'cubic' or 'spherical'.
        cutoff_param (float): Grid size for cubic, radius for spherical.
        array_3d (jnp.ndarray): Input array.
        kvec (jnp.ndarray, optional): Wavevector array, required for spherical filter.
    """
    if filter_type == 'cubic':
        return _apply_cubic_high_pass_filter(cutoff_param, array_3d)
    elif filter_type == 'spherical':
        if kvec is None:
            raise ValueError("kvec must be provided for the spherical filter.")
        return _apply_spherical_high_pass_filter(cutoff_param, array_3d, kvec)
    else:
        raise ValueError("filter_type must be 'cubic' or 'spherical'")
