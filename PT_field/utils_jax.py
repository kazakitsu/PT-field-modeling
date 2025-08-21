import jax.numpy as jnp
import jax
from jax import jit, vmap, lax, debug as jdebug
from functools import partial

import PT_field.coord_jax as coord

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

# -------------------- Orthogonalize & interpolation utilities --------------------

@partial(jit, static_argnames=('measure_pk','Nmin','jitter','dtype',
                               'warn_on_low_stat', 'warn_max_print'))
def orthogonalize(
    fields: jnp.ndarray,            # (n, ng, ng, ng//2+1)
    measure_pk,                     # Measure_Pk instance (self is static)
    boxsize: float,
    k_edges: jnp.ndarray,           # (Nk+1,)
    mu_edges: jnp.ndarray,          # (Nmu+1,)
    Nmin: int = 5,
    jitter: float = 1e-14,
    dtype=jnp.float32,
    *,
    warn_on_low_stat: bool = True,
    warn_max_print: int = 32,
):
    """
    Orthonormalize Fourier-space fields per (k,mu) bin using Cholesky on the correlation matrix.
    Nearest interpolation only; expand Mcoef(k,mu) to the grid using a precomputed flat index.

    Extra:
      - If any (k,mu)-bin has low statistics (N_modes <= Nmin), emit a warning via jax.debug.print,
        including bin indices (k_idx, mu_idx) and their (k_mean, mu_mean). The number of printed
        bins is limited by `warn_max_print`.
    """
    n_fields   = fields.shape[0]
    ng  = fields.shape[1]
    ctype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64

    if dtype != measure_pk.dtype:
        raise ValueError(f"measure_pk.dtype must match dtype={dtype}, got {measure_pk.dtype}.")

    # Cast once to the complex working dtype.
    fields = jnp.asarray(fields, dtype=ctype)

    # ---- 1) Vectorized P(k,mu): auto & cross --------------------------------
    # Prepare mu-bin edges for vectorized calls: (Nmu,)
    mu0 = mu_edges[:-1]
    mu1 = mu_edges[1:]

    # Helper: evaluate Measure_Pk over all mu-bins for a given (f1, f2)
    # Returns arrays shaped (Nk, Nmu) for [Pk*V, N_modes].
    def _pk_mu_all(f1, f2):
        # map over mu-bins -> (Nmu, Nk, 3)
        res = vmap(lambda a, b: measure_pk(f1, f2, ell=0, mu_min=a, mu_max=b))(mu0, mu1)
        # transpose to (Nk, Nmu) and extract columns
        PkV = jnp.swapaxes(res[..., 1], 0, 1)  # (Nk, Nmu)
        Nm  = jnp.swapaxes(res[..., 2], 0, 1)  # (Nk, Nmu)
        return PkV.astype(dtype), Nm.astype(dtype)

    # Autos
    P_auto_0, N_modes = _pk_mu_all(fields[0], None)
    Nk  = int(P_auto_0.shape[0])
    Nmu = int(P_auto_0.shape[1])

    P_auto  = jnp.zeros((Nk, Nmu, n_fields),    dtype=dtype)
    P_cross = jnp.zeros((Nk, Nmu, n_fields, n_fields), dtype=dtype)
    P_auto = P_auto.at[:, :, 0].set(P_auto_0)
    # Loop remaining autos; Python loop is cheap (vmap inside handles the heavy lifting)
    for i in range(1, n_fields):
        Pi, _ = _pk_mu_all(fields[i], None)
        P_auto = P_auto.at[:, :, i].set(Pi)

    # Cross (upper triangle): for each fixed j, vectorize over i<j
    for j in range(n_fields):
        if j == 0:
            continue
        # Stack P(i,j) for all i<j via vmap to reduce call overhead
        def _cross_with_j(fi):
            Pij, _ = _pk_mu_all(fi, fields[j])
            return Pij  # (Nk, Nmu)
        P_ij_stack = vmap(_cross_with_j)(fields[:j])                 # (j, Nk, Nmu)
        P_ij_stack = jnp.swapaxes(P_ij_stack, 0, 1)                  # (Nk, j, Nmu)
        P_ij_stack = jnp.swapaxes(P_ij_stack, 1, 2)                  # (Nk, Nmu, j)

        # Fill upper triangle (i<j, j)
        P_cross = P_cross.at[:, :, :j, j].set(P_ij_stack)          # (Nk, Nmu, j)
        # Mirror to (j, i<j). No transpose is needed: shape matches  (Nk, Nmu, j).
        P_cross = P_cross.at[:, :, j, :j].set(P_ij_stack)          # (Nk, Nmu, j)

    # Put autos on the diagonal
    diag = jnp.arange(n_fields)
    P_cross = P_cross.at[..., diag, diag].set(P_auto)

    # ---- 2) Correlation matrix per (k,mu) & Cholesky -------------------------
    low_stat = N_modes <= Nmin

    # Optional warning on low-stat bins (printed once inside JIT)
    if warn_on_low_stat:
        def _do_warn(_):
            n_bad = jnp.sum(low_stat)
            k_mean = jnp.asarray(measure_pk.k_mean, dtype=dtype)  # (Nk,)
            mu_means = vmap(lambda a, b: measure_pk.compute_mu_mean(a, b))(
                mu_edges[:-1], mu_edges[1:]
            ).astype(dtype)                                       # (Nmu,)

            kk, mm = jnp.nonzero(low_stat, size=Nk * Nmu, fill_value=-1)
            jdebug.print(
                "[orthogonalize] WARNING: {n_bad} low-stat bins (N_modes <= Nmin={Nmin}).",
                n_bad=n_bad, Nmin=Nmin
            )

            MAXP = int(warn_max_print)  # static bound for JIT
            def body(i, _carry):
                def yes(_):
                    k_i = kk[i]; m_i = mm[i]
                    jdebug.print(
                        "  (k_idx={ki}, mu_idx={mi})  k_mean={k:.6g}, mu_mean={m:.6g}",
                        ki=k_i, mi=m_i, k=k_mean[k_i], m=mu_means[m_i]
                    )
                    return None
                def no(_):
                    return None
                return lax.cond(i < n_bad, yes, no, None)
            _ = lax.fori_loop(0, MAXP, body, None)

            def _trunc_msg(_):
                jdebug.print(
                    "  ... truncated: total {n_bad} bins, showing first {maxp}.",
                    n_bad=n_bad, maxp=MAXP
                )
                return None
            _ = lax.cond(n_bad > MAXP, _trunc_msg, lambda _: None, None)
            return None

        _ = lax.cond(jnp.any(low_stat), _do_warn, lambda _: None, None)

    # Corr = P_cross / sqrt(P_auto_i * P_auto_j)
    denom = jnp.sqrt(jnp.clip(P_auto[..., None] * P_auto[..., None, :], a_min=0.0))
    Corr  = jnp.where(denom > 0, P_cross / denom, 0.0)
    Corr  = Corr.at[..., diag, diag].set(1.0)
    Corr  = _nearest_fill_generic(Corr, low_stat, Nk, Nmu)  # fill low-stat bins

    # Cholesky per (Nk,Nmu) with a small jitter on the diagonal for stability
    eye = jnp.eye(n_fields, dtype=dtype)

    # vmap over flattened (Nk*Nmu) bins to keep compile size reasonable
    Corr_flat = Corr.reshape(Nk * Nmu, n_fields, n_fields)

    def _chol_inv_lower(A):
        # Compute L = chol(A + jitter*I) and solve L * X = I to get L^{-1}
        L = jnp.linalg.cholesky(A + jitter * eye)
        Linv = lax.linalg.triangular_solve(L, eye, left_side=True, lower=True)
        return Linv
    
    Linv = vmap(_chol_inv_lower)(Corr_flat).reshape(Nk, Nmu, n_fields, n_fields)  # (Nk,Nmu,n_fields,n_fields)

    # Build Mcoef = (Linv / diag(Linv)) * sqrt(P_auto_i / P_auto_j)
    diagL = jnp.diagonal(Linv, axis1=2, axis2=3)                            # (Nk,Nmu,n_fields)
    ratio = jnp.sqrt(jnp.where(P_auto[..., None, :] > 0,
                               P_auto[..., None] / jnp.maximum(P_auto[..., None, :], 1e-30),
                               0.0))                                         # (Nk,Nmu,n_fields,n_fields)
    Mcoef = (Linv / diagL[..., :, None]) * ratio                             # (Nk,Nmu,n_fields,n_fields)
    Mcoef = _nearest_fill_generic(Mcoef, low_stat, Nk, Nmu)                  # fill low-stat bins

    # ---- 3) Nearest expand & sequential update ----
    nearest_idx, _, _ = _build_nearest_idx_for_grid(ng, boxsize, k_edges, mu_edges, dtype=dtype)

    def _expand_nearest(table_NkNmu):
        # table_NkNmu: (Nk, Nmu) -> grid via nearest_idx
        flat = table_NkNmu.reshape(Nk * Nmu)
        return jnp.take(flat, nearest_idx, axis=0)  # (ng, ng, ng//2+1)

    # Sequential Gram-Schmidt-like update using (k,mu)-dependent mixing
    ortho = list(fields)
    for j in range(1, n_fields):
        f = ortho[j]
        for i in range(j):
            Mgrid = _expand_nearest(Mcoef[:, :, j, i])  # (...grid...)
            f       = f + Mgrid * fields[i]
        ortho[j] = f

    return jnp.array(ortho)

def _build_nearest_idx_for_grid(ng: int,
                                boxsize: float,
                                k_edges: jnp.ndarray,
                                mu_edges: jnp.ndarray,
                                dtype=jnp.float32) -> tuple[jnp.ndarray, int, int]:
    """Return (nearest_idx, Nk, Nmu) for the (ng,ng,ng//2+1) rfft grid."""
    kx, ky, kz = coord.kaxes_1d(ng, boxsize, dtype=dtype)
    kx2, ky2, kz2 = kx * kx, ky * ky, kz * kz
    k2 = kx2[:, None, None] + ky2[None, :, None] + kz2[None, None, :]

    # bin in k using k^2 (avoid sqrt), same for mu using mu^2
    k_edges2  = (k_edges * k_edges).astype(dtype)
    mu_edges2 = (mu_edges * mu_edges).astype(dtype)
    Nk  = int(k_edges.shape[0] - 1)
    Nmu = int(mu_edges.shape[0] - 1)

    kbin  = jnp.clip(jnp.searchsorted(k_edges2,  k2,  side="right") - 1, 0, Nk - 1)
    mu2   = jnp.where(k2 > 0, kz2[None, None, :] / k2, 0.0).astype(dtype)
    mubin = jnp.clip(jnp.searchsorted(mu_edges2, mu2, side="right") - 1, 0, Nmu - 1)

    nearest_idx = (kbin * Nmu + mubin).astype(jnp.int32)  # shape = (ng,ng,ng//2+1)
    return nearest_idx, Nk, Nmu

def _nearest_fill_generic(data2d, mask, Nk, Nmu):
    flat  = data2d.reshape((Nk * Nmu,) + data2d.shape[2:])
    mflat = mask.ravel()
    good = jnp.nonzero(~mflat, size=Nk * Nmu, fill_value=-1)[0]
    bad  = jnp.nonzero( mflat, size=Nk * Nmu, fill_value=-1)[0]
    gk, gm = good // Nmu, good % Nmu
    def choose(b):
        kb, mb = b // Nmu, b % Nmu
        d2 = (gk - kb)**2 + (gm - mb)**2
        d2 = jnp.where(good >= 0, d2, 1e12)
        return good[jnp.argmin(d2)]
    from jax import vmap
    repl = vmap(choose)(bad)
    repl = jnp.where(bad >= 0, repl, bad)
    flat_out = flat.at[bad].set(flat[repl])
    return flat_out.reshape(data2d.shape)


# -------------------- Polynomial fit --------------------

@partial(jit, static_argnames=('measure_pk','dtype'))
def fit_beta_poly_from_table(
    beta_tab: jnp.ndarray,           # (n_fields, Nk, Nmu)
    *,
    measure_pk,                      # Measure_Pk instance (self is static)
    mu_edges: jnp.ndarray,           # (Nmu+1,)
    poly_k_pows: jnp.ndarray,        # (Lk,) powers for k^p
    poly_mu2_pows: jnp.ndarray,      # (Lmu2,) powers for (mu^2)^q
    ridge: float = 0.0,              # optional L2 regularization (>=0)
    dtype=jnp.float32,
):
    """
    Fit beta_i(k, mu) ~= sum_{a,b} c_{i,a,b} * k^{p_a} * (mu^2)^{q_b} in least squares.

    Returns
    -------
    coeffs       : (n_fields, Lk, Lmu2)  polynomial coefficients c_{i,a,b}
    beta_fit     : (n_fields, Nk, Nmu)   reconstructed table from the fit

    Notes
    -----
    - The design matrix is shared across fields; we solve LS per field.
    - If `ridge>0`, ridge-regularized normal equations with Cholesky is used.
    - Constant beta is covered by poly_k_pows=[0], poly_mu2_pows=[0].    
    """
    n_fields, Nk, Nmu = beta_tab.shape
    dtype = jnp.dtype(dtype)

    # -- 1) Bin centers in k and mu (mu-mean per bin) -------------------------
    k_arr = jnp.asarray(measure_pk.k_mean, dtype=dtype)               # (Nk,)
    mu_lo = mu_edges[:-1]
    mu_hi = mu_edges[1:]
    mu_mean = vmap(lambda a, b: measure_pk.compute_mu_mean(a, b))(mu_lo, mu_hi)  # (Nmu,)
    mu2_mean = (mu_mean * mu_mean).astype(dtype)                      # (Nmu,)

    # -- 2) Build the design matrix Phi of size (Nk*Nmu) x (Lk*Lmu2) ----------
    poly_k_pows = jnp.asarray(poly_k_pows, dtype=jnp.int32)           # (Lk,)
    poly_mu2_pows = jnp.asarray(poly_mu2_pows, dtype=jnp.int32)       # (Lmu2,)

    # Basis along k and mu^2 axes
    K = (k_arr[:, None] ** poly_k_pows[None, :]).astype(dtype)        # (Nk, Lk)
    M = (mu2_mean[:, None] ** poly_mu2_pows[None, :]).astype(dtype)   # (Nmu, Lmu2)

    # Full features: Phi[i*m + j, a*Lmu2 + b] = K[i,a] * M[j,b]
    Phi = (K[:, None, :, None] * M[None, :, None, :]).reshape(Nk * Nmu, -1)  # (Nk*Nmu, Lk*Lmu2)

    # Targets stacked by field
    Y = beta_tab.reshape(n_fields, Nk * Nmu).astype(dtype)            # (n_fields, Nk*Nmu)

    # -- 3) Solve least squares for each field (shared Phi) -------------------
    if ridge == 0.0:
        # QR-based LS: c = solve(R, Q^T y)
        Q, R = jnp.linalg.qr(Phi, mode='reduced')                     # Q:(M,P), R:(P,P)
        QT = jnp.swapaxes(Q, -1, -2)                                  # (P, M)

        def solve_one(y):
            return jnp.linalg.solve(R, QT @ y)                         # (P,)
        Cvec = vmap(solve_one)(Y)                                      # (n_fields, P)
    else:
        # Ridge LS via normal equations: (Phi^T Phi + λI) c = Phi^T y
        lam = jnp.asarray(ridge, dtype=dtype)
        XtX = Phi.T @ Phi + lam * jnp.eye(Phi.shape[1], dtype=dtype)   # (P,P)
        L = jnp.linalg.cholesky(XtX)                                   # chol factor

        def solve_one(y):
            rhs = Phi.T @ y                                            # (P,)
            z   = lax.linalg.triangular_solve(L, rhs, left_side=True,  lower=True)
            return lax.linalg.triangular_solve(L.T, z, left_side=True, lower=False)
        Cvec = vmap(solve_one)(Y)                                      # (n_fields, P)

    # -- 4) Reshape coefficients and reconstruct beta table -------------------
    Lk = int(poly_k_pows.shape[0]); Lmu2 = int(poly_mu2_pows.shape[0])
    coeffs = Cvec.reshape(n_fields, Lk, Lmu2)                           # (n_fields, Lk, Lmu2)

    # Reconstruct beta on (Nk, Nmu) for sanity-check
    beta_fit = (Phi @ Cvec.T).T.reshape(n_fields, Nk, Nmu).astype(dtype)

    return coeffs, beta_fit


# -------------------- Utils for diagnostics --------------------

@partial(jit, static_argnames=('measure_pk',))
def compute_corr_2d(
    fields_k: jnp.ndarray,   # (n_fields, ng, ng, ng//2+1), complex
    mu_edges: jnp.ndarray,   # (Nmu+1,)
    *,
    measure_pk               # Measure_Pk instance (static to JIT)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute r_ij(k) within each mu-bin, evaluating only i<=j and mirroring.

    Returns
    -------
    k_arr : (Nk,)               # bin-averaged k (mu-independent)
    R_all : (Nmu, n, n, Nk)     # r_ij(k) for each mu-bin
    Notes
    -----
    - We evaluate only the upper triangle (i<=j) and symmetrize.
    - Diagonal r_ii(k) is set to exactly 1.
    - Uses lax.fori_loop over mu-bins to keep compile size modest.
    """
    n_fields = fields_k.shape[0]
    dtype = measure_pk.dtype

    # k-bin centers from the estimator (independent of mu)
    k_arr = jnp.asarray(measure_pk.k_mean, dtype=dtype)
    Nk = int(k_arr.shape[0])

    # Upper-triangular index set WITHOUT diagonal for off-diagonals (i<j)
    # (we'll set the diagonal to 1 explicitly below)
    ii, jj = jnp.triu_indices(n_fields, k=1)
    npairs = int(ii.shape[0])

    Nmu = int(mu_edges.shape[0] - 1)

    def pk_matrix_for_mu(mu_min, mu_max):
        """Build R(k) for a single mu-bin via upper-tri evaluation + symmetrization."""
        eps = jnp.asarray(1e-37, dtype=dtype)

        # 1) Autos P_ii(k) for i=0..n-1  (vectorized via lax.map over i)
        def auto_body(i):
            # measure_pk(a, None, ...) returns (Nk,3); take the P column
            out = measure_pk(fields_k[i], None, ell=0, mu_min=mu_min, mu_max=mu_max)
            return out[:, 1].astype(dtype)                      # (Nk,)
        Auto = lax.map(lambda i: auto_body(i), jnp.arange(n_fields, dtype=jnp.int32))  # (n, Nk)

        # 2) Off-diagonal P_ij(k) for i<j only (loop over 'pairs')
        def pair_body(pidx):
            i = ii[pidx]
            j = jj[pidx]
            out = measure_pk(fields_k[i], fields_k[j], ell=0, mu_min=mu_min, mu_max=mu_max)
            Pij = out[:, 1].astype(dtype)                       # (Nk,)
            denom = jnp.sqrt(Auto[i] * Auto[j]) + eps           # (Nk,)
            return Pij / denom                                  # (Nk,)
        # Stack r_ij(k) for i<j: shape (npairs, Nk)
        rij_pairs = lax.map(pair_body, jnp.arange(npairs, dtype=jnp.int32))

        # 3) Assemble full R(n,n,Nk): diag=1, off-diagonals filled & mirrored
        rij = jnp.zeros((n_fields, n_fields, Nk), dtype=dtype)
        # diag -> exactly 1 across all k (broadcast on last axis)
        eye = jnp.eye(n_fields, dtype=dtype)[..., None]                # (n, n, 1)
        rij = rij + eye
        # fill upper triangle (i<j)
        rij = rij.at[ii, jj, :].set(rij_pairs)                      # (npairs, Nk)
        # mirror to lower triangle
        rij = rij.at[jj, ii, :].set(rij_pairs)

        return rij                                                # (n, n, Nk)

    # 4) Loop over mu-bins with fori_loop
    def mu_body(m, acc):
        mu_min = mu_edges[m]
        mu_max = mu_edges[m + 1]
        rij_m = pk_matrix_for_mu(mu_min, mu_max)                   # (n, n, Nk)
        return acc.at[m].set(rij_m)

    rij0 = jnp.zeros((Nmu, n_fields, n_fields, Nk), dtype=dtype)
    rij_all = lax.fori_loop(0, Nmu, mu_body, rij0)                  # (Nmu, n, n, Nk)

    return k_arr, rij_all


def check_max_rij(rij: jnp.ndarray) -> None:
    """
    Pretty-print max_k |r_ij| for each mu-bin using the output of corr_by_mu(...).
    """
    Nmu, n, _, Nk = rij.shape
    for m in range(Nmu):
        print(m)
        Rm = rij[m]  # (n, n, Nk)
        for i in range(n):
            for j in range(i + 1, n):
                val = float(jnp.max(jnp.abs(Rm[i, j])))
                print(f"{i}{j}  max|r_ij| = {val:.3e}")


@partial(jit, static_argnames=('measure_pk',))
def compute_pks_2d(
    fields_k: jnp.ndarray,   # (n_fields, ng, ng, ng//2+1), complex
    mu_edges: jnp.ndarray,   # (Nmu+1,)
    *,
    measure_pk               # Measure_Pk instance (static)
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Measure auto P(k) for many fields across all mu-bins in one pass.

    Returns
    -------
    k_arr   : (Nk,)                 # bin-averaged k
    P_many  : (n, Nk, Nmu)          # auto power for each field and mu-bin
    N_modes : (Nk, Nmu)             # counts per (k,mu) (field-independent)
    Notes
    -----
    - Loops over mu-bins with fori_loop.
    - Inside each mu-bin, uses lax.map over field index.
    """
    n_fields = fields_k.shape[0]
    dtype = measure_pk.dtype
    k_arr = jnp.asarray(measure_pk.k_mean, dtype=dtype)
    Nk = int(k_arr.shape[0])
    Nmu = int(mu_edges.shape[0] - 1)

    def mu_body(m, carry):
        P_acc, N_acc = carry
        mu_min = mu_edges[m]
        mu_max = mu_edges[m + 1]

        # We also grab N_modes from the first field; it's field-independent.
        out0 = measure_pk(fields_k[0], None, ell=0, mu_min=mu_min, mu_max=mu_max)
        N_m  = out0[:, 2].astype(dtype)                         # (Nk,)

        def one_field(i):
            out = measure_pk(fields_k[i], None, ell=0, mu_min=mu_min, mu_max=mu_max)
            return out[:, 1].astype(dtype)                      # (Nk,)

        P_m = lax.map(one_field, jnp.arange(n_fields, dtype=jnp.int32))  # (n, Nk)
        P_acc = P_acc.at[:, :, m].set(P_m)
        N_acc = N_acc.at[:, m].set(N_m)
        return (P_acc, N_acc)

    P0 = jnp.zeros((n_fields, Nk, Nmu), dtype=dtype)
    N0 = jnp.zeros((Nk, Nmu), dtype=dtype)
    P_many, N_modes = lax.fori_loop(0, Nmu, mu_body, (P0, N0))
    return k_arr, P_many, N_modes
