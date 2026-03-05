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
def apply_grad_k(delta_k, kx, ky, kz):
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

@jit
def apply_traceless(Gij_k):
    """Traceless part of G_ij: G_ij - (1/3) \delta_ij G_kk."""
    # G_kk = G_xx + G_yy + G_zz
    g_xx, g_xy, g_xz, g_yy, g_yz, g_zz = Gij_k
    g_trace = (g_xx + g_yy + g_zz) / 3.0

    # Subtract trace from diagonal components
    g_xx -= g_trace
    g_yy -= g_trace
    g_zz -= g_trace

    return jnp.stack([g_xx, g_xy, g_xz, g_yy, g_yz, g_zz], axis=0)

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

# ---------- extend: single-scatter version ----------
@partial(jit, static_argnames=('ng_ext',))
def func_extend(ng_ext, array_3d):
    """
    Zero-pad an rfftn-layout array to a larger grid (ng_ext) with one scatter.
    We gather the source "low-k" planes with explicit index arrays along x/y,
    then place them into the destination low-k slots.
    """
    ng = array_3d.shape[0]
    half = ng // 2

    # source indices (x,y): [0..half-1, ng-half..ng-1]  (length = ng)
    idx_src_xy = jnp.concatenate([jnp.arange(half), jnp.arange(ng - half, ng)])
    # dest indices (x,y):   [0..half-1, ng_ext-half..ng_ext-1] (length = ng)
    idx_dst_xy = jnp.concatenate([jnp.arange(half), jnp.arange(ng_ext - half, ng_ext)])

    # z: copy only the available half-spectrum 0..ng//2
    idx_src_z = jnp.arange(ng // 2 + 1)
    idx_dst_z = idx_src_z  # same length, placed at low-z

    # gather once, then scatter once
    src = array_3d[jnp.ix_(idx_src_xy, idx_src_xy, idx_src_z)]
    out = jnp.zeros((ng_ext, ng_ext, ng_ext // 2 + 1), dtype=array_3d.dtype)
    out = out.at[jnp.ix_(idx_dst_xy, idx_dst_xy, idx_dst_z)].set(src)
    return out

# ---------- reduce ----------
@partial(jit, static_argnames=('ng_red',))
def func_reduce(ng_red, array_3d):
    """
    Truncate high-k modes in rfftn layout to reduce grid size to ng_red.
    Kept band is the same as func_extend's low-k region.
    """
    ng = array_3d.shape[0]
    half = ng_red // 2
    idx_xy = jnp.concatenate([jnp.arange(half), jnp.arange(ng - half, ng)])
    idx_z  = jnp.arange(half + 1)
    return array_3d[jnp.ix_(idx_xy, idx_xy, idx_z)]

# ---------- enforce Hermitian: vectorized ----------
@jit
def _enforce_hermite(array_k):
    """
    Enforce corner/axis Hermitian constraints for rfftn layout:
      - make specific corners purely real on z=0 and z=Nyq (last z-plane),
      - fix a minimal set of 1D-conjugate pairs along axes.
    Notes
    -----
    - rfftn/irfftn normally maintains Hermitian symmetry; this is a safe fix
      after manual slicing/reduction.
    """
    x = array_k
    ng = x.shape[0]
    half = ng // 2
    lastz = x.shape[2] - 1  # Nyquist z-plane

    # -- make the four corners on z=0 and z=Nyq purely real --
    def realify_corners(zidx, x):
        ii = jnp.array([0,     half,  0,     half], dtype=jnp.int32)
        jj = jnp.array([0,     0,     half,  half], dtype=jnp.int32)
        kk = jnp.full_like(ii, zidx)
        vals = jnp.real(x[ii, jj, kk]).astype(x.dtype)  # zero imag
        return x.at[ii, jj, kk].set(vals)

    x = realify_corners(0, x)
    x = realify_corners(lastz, x)

    # -- 1D axis-conjugate fixes for a minimal set of lines --
    # y-axis line at x=0 on z=0 and z=Nyq:
    y = jnp.arange(1, half, dtype=jnp.int32)  # 1..half-1
    x = x.at[0, -y, 0].set(jnp.conj(x[0, y, 0]))
    x = x.at[0, -y, lastz].set(jnp.conj(x[0, y, lastz]))

    # x-axis line at y=0 on z=0 and z=Nyq:
    x = x.at[-y, 0, 0].set(jnp.conj(x[y, 0, 0]))
    x = x.at[-y, 0, lastz].set(jnp.conj(x[y, 0, lastz]))

    return x

@partial(jit, static_argnames=('ng_red',))
def func_reduce_hermite(ng_red, array_3d):
    """Reduce then enforce Hermitian constraints."""
    return _enforce_hermite(func_reduce(ng_red, array_3d))

# ---------- reduce to cube (keep original func_reduce untouched) ----------
@partial(jit, static_argnames=("ng_red",))
def func_reduce_to_cube(ng_red: int, array_3d: jnp.ndarray) -> jnp.ndarray:
    """
    Truncate high-k modes in rfftn layout and make a cubic grid in real space.

    Input
    -----
    array_3d: rfftn(real_field) with shape (Nx, Ny, Nz//2 + 1), where the rfft axis is z.

    Output
    ------
    rfftn layout for a cube real-space grid (ng_red, ng_red, ng_red):
      shape (ng_red, ng_red, ng_red//2 + 1)

    Requirements
    ------------
    - ng_red is even
    - ng_red <= Nx, ng_red <= Ny, and ng_red <= Nz (real-space Nz)
    """
    nx, ny, nz_r = array_3d.shape
    nz = (nz_r - 1) * 2  # assumes even real-space Nz

    assert (ng_red % 2) == 0, "ng_red must be even."
    assert ng_red <= nx and ng_red <= ny and ng_red <= nz, "ng_red must be <= Nx, Ny, Nz."

    half = ng_red // 2

    # x,y: full FFT ordering => keep [0..half-1] and [N-half..N-1]
    idx_x = jnp.concatenate([jnp.arange(half), jnp.arange(nx - half, nx)])
    idx_y = jnp.concatenate([jnp.arange(half), jnp.arange(ny - half, ny)])

    # z: rFFT ordering => keep [0..half] (inclusive) => half+1 = ng_red//2+1
    idx_z = jnp.arange(half + 1)

    return array_3d[jnp.ix_(idx_x, idx_y, idx_z)]


# ---------- enforce Hermitian for cube rfftn layout ----------
@jit
def _enforce_hermite_cube(array_k: jnp.ndarray) -> jnp.ndarray:
    """
    Enforce corner/axis Hermitian constraints for rfftn layout AFTER manual reduction.

    Expects shape (ng, ng, ng//2+1) with even ng.
    """
    x = array_k
    ng0, ng1, _ = x.shape
    assert ng0 == ng1, "Expected square x/y for cube layout."
    ng = ng0
    assert (ng % 2) == 0, "Expected even ng."

    half = ng // 2
    lastz = x.shape[2] - 1  # kz = ng/2 plane

    # Make four corners on z=0 and z=Nyq purely real
    def realify_corners(zidx, arr):
        ii = jnp.array([0, half, 0, half], dtype=jnp.int32)
        jj = jnp.array([0, 0, half, half], dtype=jnp.int32)
        kk = jnp.full_like(ii, zidx)
        vals = jnp.real(arr[ii, jj, kk]).astype(arr.dtype)
        return arr.at[ii, jj, kk].set(vals)

    x = realify_corners(0, x)
    x = realify_corners(lastz, x)

    # Minimal 1D conjugate fixes along axes on z=0 and z=Nyq
    y = jnp.arange(1, half, dtype=jnp.int32)  # 1..half-1

    # y-axis line at x=0
    x = x.at[0, -y, 0].set(jnp.conj(x[0, y, 0]))
    x = x.at[0, -y, lastz].set(jnp.conj(x[0, y, lastz]))

    # x-axis line at y=0
    x = x.at[-y, 0, 0].set(jnp.conj(x[y, 0, 0]))
    x = x.at[-y, 0, lastz].set(jnp.conj(x[y, 0, lastz]))

    return x

@partial(jit, static_argnames=("ng_red",))
def func_reduce_hermite_to_cube(ng_red: int, array_3d: jnp.ndarray) -> jnp.ndarray:
    """Reduce-to-cube then enforce Hermitian constraints."""
    return _enforce_hermite_cube(func_reduce_to_cube(ng_red, array_3d))



@jit
def _hermitize_xy_plane(plane_xy: jnp.ndarray) -> jnp.ndarray:
    """
    Enforce 2D Hermitian symmetry on a single (x, y) Fourier plane:
      P(-kx, -ky) = conj(P(kx, ky)) in FFT ordering.

    This is required for kz=0 and kz=Nyquist planes in rfftn layout.
    """
    ng0, ng1 = plane_xy.shape
    assert ng0 == ng1, "Expected a square plane."
    ng = ng0
    assert (ng % 2) == 0, "Expected even ng."

    # Index mapping for k -> -k in FFT ordering
    idx = jnp.arange(ng, dtype=jnp.int32)
    idx_neg = (-idx) % ng

    # P_neg(kx, ky) = P(-kx, -ky)
    plane_neg = plane_xy[jnp.ix_(idx_neg, idx_neg)]

    # Symmetrize: P <- (P + conj(P(-k))) / 2
    plane_sym = 0.5 * (plane_xy + jnp.conj(plane_neg))

    # Self conjugate points must be purely real: (0,0), (0,Nyq), (Nyq,0), (Nyq,Nyq)
    half = ng // 2
    ii = jnp.array([0, 0, half, half], dtype=jnp.int32)
    jj = jnp.array([0, half, 0, half], dtype=jnp.int32)
    vals = jnp.real(plane_sym[ii, jj]).astype(plane_sym.dtype)
    plane_sym = plane_sym.at[ii, jj].set(vals)

    return plane_sym


@jit
def _enforce_hermite_cube_fullplanes(array_k: jnp.ndarray) -> jnp.ndarray:
    """
    Enforce the necessary Hermitian constraints for rfftn layout on a cube:
      - Apply full 2D Hermitian symmetrization on kz=0 and kz=Nyquist planes.

    Expects shape (ng, ng, ng//2+1) with even ng.
    """
    x = array_k
    ng0, ng1, nz_r = x.shape
    assert ng0 == ng1, "Expected square x and y for cube layout."
    ng = ng0
    assert (ng % 2) == 0, "Expected even ng."
    assert nz_r == (ng // 2 + 1), "Expected rfftn z axis to match cube."

    # kz=0 plane
    x = x.at[:, :, 0].set(_hermitize_xy_plane(x[:, :, 0]))

    # kz=Nyquist plane in the reduced grid is the last stored plane
    lastz = nz_r - 1
    x = x.at[:, :, lastz].set(_hermitize_xy_plane(x[:, :, lastz]))

    return x


@jit
def _drop_nyquist_planes_cube(array_k: jnp.ndarray) -> jnp.ndarray:
    """
    Optionally remove exact Nyquist modes to avoid checkerboard like patterns.
    This corresponds to using a strict cutoff below Nyquist.
    """
    x = array_k
    ng = x.shape[0]
    half = ng // 2
    lastz = x.shape[2] - 1

    x = x.at[half, :, :].set(jnp.zeros_like(x[half, :, :]))   # kx = Nyquist
    x = x.at[:, half, :].set(jnp.zeros_like(x[:, half, :]))   # ky = Nyquist
    x = x.at[:, :, lastz].set(jnp.zeros_like(x[:, :, lastz])) # kz = Nyquist
    return x


@partial(jit, static_argnames=("ng_red", "drop_nyquist"))
def func_reduce_hermite_to_cube_fixed(
    ng_red: int,
    array_3d: jnp.ndarray,
    *,
    drop_nyquist: bool = False,
) -> jnp.ndarray:
    """
    Reduce rfftn spectrum to (ng_red, ng_red, ng_red//2+1) and enforce
    full plane Hermitian constraints needed by irfftn.
    """
    x = func_reduce_to_cube(ng_red, array_3d)
    x = _enforce_hermite_cube_fullplanes(x)
    x = jnp.where(drop_nyquist, _drop_nyquist_planes_cube(x), x)
    return x




# ===========================
# Sharp low/high-pass filters
# ===========================

@partial(jit, static_argnames=('cutoff',))
def _apply_cubic_low_pass_filter(cutoff, array_3d):
    """
    Cubic low-pass via 1D boolean masks and broadcasting.
    Keeps x/y indices in [0..half-1] or [ng-half..ng-1] and z in [0..half].
    Equivalent passband to the original slice-based implementation.
    """
    ng = array_3d.shape[0]
    half = int(cutoff) // 2  # e.g., cutoff = 2*ng//3 -> half = ng//3

    idx_xy = jnp.arange(ng)
    mx = (idx_xy < half) | (idx_xy >= (ng - half))      # (ng,)
    mz = jnp.arange(ng // 2 + 1) <= half                # (ng//2+1,)

    mask = (mx[:, None, None] & mx[None, :, None] & mz[None, None, :])
    return array_3d * mask.astype(array_3d.dtype)

@jit
def _apply_spherical_low_pass_filter(k_cutoff, array_3d, kx, ky, kz):
    k2 = (kx*kx)[:, None, None] + (ky*ky)[None, :, None] + (kz*kz)[None, None, :]
    mask = (k2 <= (jnp.asarray(k_cutoff, dtype=k2.dtype)**2))
    return array_3d * mask.astype(array_3d.dtype)

@partial(jit, static_argnames=('cutoff',))
def _apply_cubic_high_pass_filter(cutoff, array_3d):
    """
    Cubic high-pass as complement of the above low-pass.
    """
    ng = array_3d.shape[0]
    half = int(cutoff) // 2

    idx_xy = jnp.arange(ng)
    mx = (idx_xy < half) | (idx_xy >= (ng - half))      # kept by low-pass
    mz = jnp.arange(ng // 2 + 1) <= half

    # complement mask
    mask = ~(mx[:, None, None] & mx[None, :, None] & mz[None, None, :])
    return array_3d * mask.astype(array_3d.dtype)

@jit
def _apply_spherical_high_pass_filter(k_cutoff, array_3d, kx, ky, kz):
    k2 = (kx*kx)[:, None, None] + (ky*ky)[None, :, None] + (kz*kz)[None, None, :]
    mask = (k2 >= (jnp.asarray(k_cutoff, dtype=k2.dtype)**2))
    return array_3d * mask.astype(array_3d.dtype)

def low_pass_filter_fourier(filter_type, cutoff_param, array_3d, kx=None, ky=None, kz=None):
    """Apply sharp low-pass filter in Fourier space ('cubic' or 'spherical')."""
    if filter_type == 'cubic':
        return _apply_cubic_low_pass_filter(cutoff_param, array_3d)
    elif filter_type == 'spherical':
        if kx is None or ky is None or kz is None:
            raise ValueError("kx, ky and kz must be provided for the spherical filter.")
        return _apply_spherical_low_pass_filter(cutoff_param, array_3d, kx, ky, kz)
    else:
        raise ValueError("filter_type must be 'cubic' or 'spherical'")

def high_pass_filter_fourier(filter_type, cutoff_param, array_3d, kx=None, ky=None, kz=None):
    """Apply sharp high-pass filter in Fourier space ('cubic' or 'spherical')."""
    if filter_type == 'cubic':
        return _apply_cubic_high_pass_filter(cutoff_param, array_3d)
    elif filter_type == 'spherical':
        if kx is None or ky is None or kz is None:
            raise ValueError("kx, ky and kz must be provided for the spherical filter.")
        return _apply_spherical_high_pass_filter(cutoff_param, array_3d, kx, ky, kz)
    else:
        raise ValueError("filter_type must be 'cubic' or 'spherical'")
