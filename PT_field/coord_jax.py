# !/usr/bin/env python3
import jax.numpy as jnp
from jax import jit
from functools import partial

# =========================
# 1D k-axes (preferred API)
# =========================

def kaxes_1d(ng: int, boxsize: float, *, dtype=jnp.float32):
    """Return 1D physical k-axes for x,y (fftfreq) and z (rfftfreq)."""
    dtype = jnp.dtype(dtype)
    boxsize = jnp.asarray(boxsize, dtype)
    fac = (2.0 * jnp.pi) / boxsize
    # Use sample spacing d = 1/ng so that fac * fftfreq gives physical k.
    d = 1.0 / jnp.asarray(ng, dtype)
    kx = fac * jnp.fft.fftfreq(ng, d=d)
    ky = kx
    kz = fac * jnp.fft.rfftfreq(ng, d=d)
    return kx.astype(dtype), ky.astype(dtype), kz.astype(dtype)

@jit
def mu2_grid(kx, ky, kz):
    """mu^2 = (kz^2 / k^2) on the rfftn grid from 1D squared axes."""
    rtype = jnp.result_type(kx, ky, kz)
    kx2, ky2, kz2 = (kx*kx).astype(rtype), (ky*ky).astype(rtype), (kz*kz).astype(rtype)
    k2 = (kx2[:, None, None] + ky2[None, :, None] + kz2[None, None, :]).astype(rtype)
    return jnp.where(k2 > 0, kz2[None, None, :] / k2, 0.0).astype(rtype)

@jit
def apply_disp_k(delta_k, kx, ky, kz):
    """(3, ...): (1j * k_i / k^2) * delta_k applied on the fly from 1D axes."""
    rtype = jnp.result_type(kx, ky, kz)
    kx2, ky2, kz2 = (kx*kx).astype(rtype), (ky*ky).astype(rtype), (kz*kz).astype(rtype)
    k2   = (kx2[:, None, None] + ky2[None, :, None] + kz2[None, None, :]).astype(rtype)
    invk2 = jnp.where(k2 > 0, 1.0 / k2, 0.0).astype(rtype)

    ctype = delta_k.dtype
    j = jnp.array(1j, dtype=ctype)
    vx = (j * kx[:, None, None] * invk2 * delta_k).astype(ctype)
    vy = (j * ky[None, :, None] * invk2 * delta_k).astype(ctype)
    vz = (j * kz[None, None, :] * invk2 * delta_k).astype(ctype)
    return jnp.stack([vx, vy, vz], axis=0)

@jit
def apply_Gij_k(delta_k, kx, ky, kz):
    """(6, ...): (k_i k_j / k^2) * delta_k (xx,xy,xz,yy,yz,zz)."""
    rtype = jnp.result_type(kx, ky, kz)
    kx2, ky2, kz2 = (kx*kx).astype(rtype), (ky*ky).astype(rtype), (kz*kz).astype(rtype)
    k2   = (kx2[:, None, None] + ky2[None, :, None] + kz2[None, None, :]).astype(rtype)
    invk2 = jnp.where(k2 > 0, 1.0 / k2, 0.0).astype(rtype)

    kx3, ky3, kz3 = kx[:, None, None], ky[None, :, None], kz[None, None, :]
    g_xx = (kx3 * kx3 * invk2 * delta_k)
    g_xy = (kx3 * ky3 * invk2 * delta_k)
    g_xz = (kx3 * kz3 * invk2 * delta_k)
    g_yy = (ky3 * ky3 * invk2 * delta_k)
    g_yz = (ky3 * kz3 * invk2 * delta_k)
    g_zz = (kz3 * kz3 * invk2 * delta_k)
    return jnp.stack([g_xx, g_xy, g_xz, g_yy, g_yz, g_zz], axis=0)

@jit
def apply_nabla_k(delta_k, kx, ky, kz):
    """(3, ...): gradient in k-space, 1j * k_i * delta_k (no 1/k^2)."""
    ctype = delta_k.dtype

    j = jnp.array(1j, dtype=ctype)
    vx = (j * kx[:, None, None] * delta_k).astype(ctype)
    vy = (j * ky[None, :, None] * delta_k).astype(ctype)
    vz = (j * kz[None, None, :] * delta_k).astype(ctype)
    return jnp.stack([vx, vy, vz], axis=0)

@jit
def apply_gauss_k(delta_k, R, kx, ky, kz):
    """Gaussian filter in k-space: exp(-0.5 k^2 R^2) * delta_k."""
    rtype = jnp.result_type(kx, ky, kz)
    ctype = delta_k.dtype
    
    kx2, ky2, kz2 = (kx*kx).astype(rtype), (ky*ky).astype(rtype), (kz*kz).astype(rtype)
    k2 = (kx2[:, None, None] + ky2[None, :, None] + kz2[None, None, :]).astype(rtype)
    W  = jnp.exp(-0.5 * k2 * (jnp.asarray(R, dtype=rtype)**2))
    return delta_k * W.astype(ctype)

# ==========================================
# 3D k-vector interfaces
# ==========================================

@partial(jit, static_argnames=('shape',))
def rfftn_kvec(shape, boxsize):
    """Build 3D kvec = (kx,ky,kz) on demand. Prefer 1D axes + apply_* APIs."""
    rtype = boxsize.dtype

    kx, ky, kz = kaxes_1d(shape[0], boxsize, dtype=rtype)
    return jnp.stack([kx[:, None, None],
                      ky[None, :, None],
                      kz[None, None, :]], axis=0).astype(rtype)

@jit
def rfftn_khat(kvec):
    """Unit vector khat = k / |k| with safe k=0 handling."""
    k2   = jnp.sum(kvec**2, axis=0)
    kmag = jnp.sqrt(k2)
    return jnp.where(kmag == 0.0, 0.0, kvec / kmag)

@jit
def rfftn_mu2(kvec):
    """mu^2 = (kz/|k|)^2 with safe k=0 handling."""
    k2 = jnp.sum(kvec**2, axis=0)
    return jnp.where(k2 == 0.0, 0.0, kvec[2]**2 / k2)

@jit
def rfftn_disp(kvec):
    """Displacement operator 1j * k / k^2 with safe k=0 handling."""
    k2 = jnp.sum(kvec**2, axis=0)
    return jnp.where(k2 == 0.0, 0.0, 1j * kvec / k2)

@jit
def rfftn_nabla(kvec):
    """Gradient operator in k-space: 1j * k."""
    return 1j * kvec

@jit
def rfftn_Gauss(kvec, R):
    """Gaussian window exp(-0.5 k^2 R^2)."""
    k2 = jnp.sum(kvec**2, axis=0)
    return jnp.exp(-0.5 * k2 * (jnp.asarray(R, dtype=k2.dtype)**2))

@jit
def rfftn_sij(kvec):
    """
    6 unique components of tidal tensor s_ij = k_i k_j / k^2 - (1/3) δ_ij.
    Returns stack [xx, xy, xz, yy, yz, zz].
    """
    k2 = jnp.sum(kvec**2, axis=0)
    invk2 = jnp.where(k2 == 0.0, 0.0, 1.0 / k2)
    kx, ky, kz = kvec[0], kvec[1], kvec[2]

    s_xx = kx * kx * invk2
    s_xy = kx * ky * invk2
    s_xz = kx * kz * invk2
    s_yy = ky * ky * invk2
    s_yz = ky * kz * invk2
    s_zz = kz * kz * invk2

    trace_term = (1.0 / 3.0) * (k2 != 0.0)
    s_xx = s_xx - trace_term
    s_yy = s_yy - trace_term
    s_zz = s_zz - trace_term

    return jnp.stack([s_xx, s_xy, s_xz, s_yy, s_yz, s_zz], axis=0)

@jit
def rfftn_Gij(kvec):
    """
    6 unique components of G_ij = k_i k_j / k^2 (xx, xy, xz, yy, yz, zz).
    """
    k2 = jnp.sum(kvec**2, axis=0)
    inv = jnp.where(k2 == 0.0, 0.0, 1.0 / k2)
    kx, ky, kz = kvec[0], kvec[1], kvec[2]
    return jnp.stack([kx*kx*inv, kx*ky*inv, kx*kz*inv, ky*ky*inv, ky*kz*inv, kz*kz*inv], axis=0)

# =======================================
# Grid manipulation & Hermitian symmetry
# =======================================

@partial(jit, static_argnames=('ng_ext',))
def func_extend(ng_ext, array_3d):
    """Zero-pad Fourier rfftn layout array to a larger ng_ext."""
    ng = array_3d.shape[0]
    ng_half = ng // 2
    out = jnp.zeros((ng_ext, ng_ext, ng_ext // 2 + 1), dtype=array_3d.dtype)

    src_pos_x, dst_pos_x = slice(0, ng_half), slice(0, ng_half)
    src_neg_x, dst_neg_x = slice(-ng_half, None), slice(-ng_half, None)
    idx_y = jnp.r_[0:ng_half, ng_ext - ng_half:ng_ext]
    idx_z = slice(0, ng // 2 + 1)

    out = out.at[dst_pos_x, idx_y, idx_z].set(array_3d[src_pos_x, :, :])
    out = out.at[dst_neg_x, idx_y, idx_z].set(array_3d[src_neg_x, :, :])
    return out

@partial(jit, static_argnames=('ng_red',))
def func_reduce(ng_red, array_3d):
    """Truncate high-k modes in rfftn layout to reduce grid size to ng_red."""
    ng = array_3d.shape[0]
    half = ng_red // 2
    idx_xy = jnp.concatenate([jnp.arange(half), jnp.arange(ng - half, ng)])
    idx_z  = jnp.arange(half + 1)
    return array_3d[jnp.ix_(idx_xy, idx_xy, idx_z)]

@jit
def _enforce_hermite(array_k):
    """Ensure corner/axis Hermitian constraints in rfftn layout."""
    ng = array_k.shape[0]
    half = ng // 2
    x = array_k
    # make specific corners real
    for zidx in (0, -1):
        x = x.at[0, 0, zidx].set(x[0, 0, zidx].real)
        x = x.at[half, 0, zidx].set(x[half, 0, zidx].real)
        x = x.at[0, half, zidx].set(x[0, half, zidx].real)
        x = x.at[half, half, zidx].set(x[half, half, zidx].real)
    # conjugate sym along axes (minimal fix; full symmetry relies on using rfftn/irfftn)
    x = x.at[0, -1:half:-1, 0].set(x[0, 1:half, 0].conj())
    x = x.at[-1:half:-1, 0, 0].set(x[1:half, 0, 0].conj())
    x = x.at[0, -1:half:-1, -1].set(x[0, 1:half, -1].conj())
    x = x.at[-1:half:-1, 0, -1].set(x[1:half, 0, -1].conj())
    return x

@partial(jit, static_argnames=('ng_red',))
def func_reduce_hermite(ng_red, array_3d):
    """Reduce then enforce Hermitian constraints."""
    return _enforce_hermite(func_reduce(ng_red, array_3d))

# ===========================
# Sharp low/high-pass filters
# ===========================

@partial(jit, static_argnames=('cutoff',))
def _apply_cubic_low_pass_filter(cutoff, array_3d):
    ng = array_3d.shape[0]
    half = int(cutoff) // 2
    out = jnp.zeros_like(array_3d)
    idx_xy = jnp.concatenate([jnp.arange(half), jnp.arange(ng - half, ng)])
    idx_z  = jnp.arange(half + 1)
    sl = jnp.ix_(idx_xy, idx_xy, idx_z)
    return out.at[sl].set(array_3d[sl])

@jit
def _apply_spherical_low_pass_filter(k_cutoff, array_3d, kvec):
    k2 = jnp.sum(kvec**2, axis=0)
    return array_3d * (k2 <= (k_cutoff**2))

@partial(jit, static_argnames=('cutoff',))
def _apply_cubic_high_pass_filter(cutoff, array_3d):
    ng = array_3d.shape[0]
    half = int(cutoff) // 2
    idx_xy = jnp.concatenate([jnp.arange(half), jnp.arange(ng - half, ng)])
    idx_z  = jnp.arange(half + 1)
    sl = jnp.ix_(idx_xy, idx_xy, idx_z)
    return array_3d.at[sl].set(0)

@jit
def _apply_spherical_high_pass_filter(k_cutoff, array_3d, kvec):
    k2 = jnp.sum(kvec**2, axis=0)
    return array_3d * (k2 > (k_cutoff**2))

def low_pass_filter_fourier(filter_type, cutoff_param, array_3d, kvec=None):
    """Apply sharp low-pass filter in Fourier space ('cubic' or 'spherical')."""
    if filter_type == 'cubic':
        return _apply_cubic_low_pass_filter(cutoff_param, array_3d)
    elif filter_type == 'spherical':
        if kvec is None:
            raise ValueError("kvec must be provided for the spherical filter.")
        return _apply_spherical_low_pass_filter(cutoff_param, array_3d, kvec)
    else:
        raise ValueError("filter_type must be 'cubic' or 'spherical'")

def high_pass_filter_fourier(filter_type, cutoff_param, array_3d, kvec=None):
    """Apply sharp high-pass filter in Fourier space ('cubic' or 'spherical')."""
    if filter_type == 'cubic':
        return _apply_cubic_high_pass_filter(cutoff_param, array_3d)
    elif filter_type == 'spherical':
        if kvec is None:
            raise ValueError("kvec must be provided for the spherical filter.")
        return _apply_spherical_high_pass_filter(cutoff_param, array_3d, kvec)
    else:
        raise ValueError("filter_type must be 'cubic' or 'spherical'")
