#!/usr/bin/env python3
from __future__ import annotations
from typing import Tuple, Literal, Optional
import warnings

import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial

import PT_field.coord_jax as coord
import lss_utils.assign_util_jax as assign_util

# ------------------------------ Base class ------------------------------

class Base_Forward:
    r"""
    FFT helpers & k-axis; 
    Generate the linear field given the linear power spectrum;
    model-specific subclasses add physics.
    """

    def __init__(self, *, boxsize: float,
                 ng: Optional[int] = None,  # make ng optional
                 ng_L: int,
                 dtype=jnp.float32,
                 use_batched_fft: bool = True):
        self.boxsize: float = float(boxsize)
        self.ng:      Optional[int] = int(ng) if ng is not None else None
        self.ng_L:      int = int(ng_L)
        self.real_dtype = jnp.dtype(dtype)
        self.complex_dtype = jnp.complex64 if self.real_dtype == jnp.float32 else jnp.complex128
        self.use_batched_fft = bool(use_batched_fft)

        self.vol: float = self.boxsize**3

        # Heuristic: fused 3D path up to this ng; tiled beyond it.
        self._auto_threshold_ng: int = 192
        # (Optional) z-tiling thickness for the tiled path; using plane=1 keeps peak memory lowest.
        self._tile_z: int = 8

        # vmap'ed FFT helpers with forward normalization (unitary inverse)
        self.b_irfftn = jit(vmap(partial(jnp.fft.irfftn, norm='forward'), in_axes=0, out_axes=0))
        self.b_rfftn  = jit(vmap(partial(jnp.fft.rfftn,  norm='forward'), in_axes=0, out_axes=0))

        # Keep only 1D k-axes to avoid captured 3D constants
        self.kx,  self.ky,  self.kz  = coord.kaxes_1d(self.ng_L, self.boxsize, dtype=self.real_dtype)

    # -------- FFT helpers (normalized) -------------------------------
    def irfftn(self, array_k: jnp.ndarray) -> jnp.ndarray:
        return jnp.fft.irfftn(array_k, norm='forward').astype(self.real_dtype)

    def rfftn(self, array_r: jnp.ndarray) -> jnp.ndarray:
        return jnp.fft.rfftn(array_r.astype(self.real_dtype), norm='forward').astype(self.complex_dtype)

    def batched_irfftn(self, array_k: jnp.ndarray) -> jnp.ndarray:
        return self.b_irfftn(array_k).astype(self.real_dtype)

    def batched_rfftn(self, array_r: jnp.ndarray) -> jnp.ndarray:
        return self.b_rfftn(array_r.astype(self.real_dtype)).astype(self.complex_dtype)
    
    @partial(jit, static_argnames=('self',))
    def _irfftn_vec(self, array_k: jnp.ndarray) -> jnp.ndarray:
        r"""
        array_k: (m, ng, ng, ng//2+1) -> (m, ng, ng, ng) by IRFFT
        """
        if self.use_batched_fft:
            return self.b_irfftn(array_k).astype(self.real_dtype)
        else:
            outs = []
            for i in range(array_k.shape[0]):
                outs.append(self.irfftn(array_k[i]))
            return jnp.stack(outs, axis=0)
            
    @partial(jit, static_argnames=('self',))
    def _rfftn_vec(self, array_r: jnp.ndarray) -> jnp.ndarray:
        r"""
        array_r: (m, ng, ng, ng) -> (m, ng, ng, ng//2+1) by RFFT
        """
        if self.use_batched_fft:
            return self.b_rfftn(array_r).astype(self.complex_dtype)
        else:
            outs = []
            for i in range(array_r.shape[0]):
                outs.append(self.rfftn(array_r[i]))
            return jnp.stack(outs, axis=0)
        
    # ---------------- public knobs ----------------
    def set_ng(self, ng: int) -> None:
        r"""Set ng after construction, and invalidate grid caches."""
        ng = int(ng)
        if self.ng == ng:
            return
        self.ng = ng
        self._invalidate_small_k_caches()

    def _invalidate_small_k_caches(self) -> None:
        r"""Drop grid axes and k^2 caches; safe to call anytime."""
        for name in ("kx_", "ky_", "kz_", "_kxy2_", "_kz2_"):
            if hasattr(self, name):
                delattr(self, name)
        
    def _require_ng(self, caller: str) -> None:
        r"""Ensure ng is available when a grid path requires it."""
        if self.ng is None:
            raise RuntimeError(
                f"{caller}: 'ng' is required but not set. "
                f"Pass ng to Base_Forward(..., ng=...) or call .set_ng(ng) before using this method."
            )

    def _ensure_kaxes(self) -> None:
        r"""Build 1D-grid (kx_, ky_, kz_) once; safe to call multiple times."""
        self._require_ng("_ensure_kaxes_small")
        if not hasattr(self, "kx_"):
            kx_, ky_, kz_ = coord.kaxes_1d(self.ng, self.boxsize, dtype=self.real_dtype)
            self.kx_, self.ky_, self.kz_ = kx_, ky_, kz_

    def _ensure_k_caches(self) -> None:
        r"""Build grid k^2 caches once; safe to call many times."""
        self._ensure_kaxes()
        if not hasattr(self, "_kxy2_"):
            kx2 = (self.kx_ ** 2).astype(self.real_dtype)
            ky2 = (self.ky_ ** 2).astype(self.real_dtype)
            kz2 = (self.kz_ ** 2).astype(self.real_dtype)
            self._kxy2_ = (kx2[:, None] + ky2[None, :]).astype(self.real_dtype)  # (ng, ng)
            self._kz2_  = kz2  # (ng//2+1,)

    # ---------------- linear modes (public wrapper) ----------------
    def linear_modes(self, pk_lin: jnp.ndarray, gauss_3d: jnp.ndarray) -> jnp.ndarray:
        r"""
        Public, non-jitted wrapper:
          - check ng presence only when needed,
          - build caches,
          - call the jitted implementation.
        """
        self._ensure_k_caches()  # ensures ng is set and caches are ready
        return self._linear_modes(pk_lin, gauss_3d)
        
    @partial(jit, static_argnames=('self',))
    def _linear_modes(self, pk_lin: jnp.ndarray, gauss_3d: jnp.ndarray) -> jnp.ndarray:
        r"""
        Build linear modes in k-space. Auto-switch between:
          - fused 3D path (faster for small ng on GPUs),
          - z-sliced path (lower peak memory).
        Returns complex array of shape (ng, ng, ng//2+1).
        """
        pk_lin   = jnp.asarray(pk_lin,   dtype=self.real_dtype)     # (N, 2): k, P(k)
        gauss_3d = jnp.asarray(gauss_3d, dtype=self.complex_dtype)  # (ng, ng, ng//2+1)
        inv2V    = jnp.array(0.5 / self.vol, dtype=self.real_dtype)

        # Interp tables
        k_tab = pk_lin[:, 0]
        P_tab = pk_lin[:, 1]

        # Decide path at trace time (self is static)
        use_fused = (self.ng is not None) and (self.ng <= self._auto_threshold_ng)

        if use_fused:
            # ---- fused 3D path ----
            kx2 = (self.kx_ ** 2).astype(self.real_dtype)[:, None, None]
            ky2 = (self.ky_ ** 2).astype(self.real_dtype)[None, :, None]
            kz2 = (self.kz_ ** 2).astype(self.real_dtype)[None, None, :]

            kmag = jnp.sqrt(kx2 + ky2 + kz2).astype(self.real_dtype)
            Pk   = jnp.interp(kmag, k_tab, P_tab, left=0.0, right=0.0)
            Pk   = Pk.at[0, 0, 0].set(0.0)  # zero DC

            amp  = jnp.sqrt(Pk * inv2V).astype(self.real_dtype)
            return (amp * gauss_3d).astype(self.complex_dtype)
        else:
            # ---- z-sliced path ----
            out = jnp.zeros_like(gauss_3d, dtype=self.complex_dtype)

            def body(iz, acc):
                kmag_2d = jnp.sqrt(self._kxy2_ + self._kz2_[iz]).astype(self.real_dtype)
                P_2d    = jnp.interp(kmag_2d, k_tab, P_tab, left=0.0, right=0.0)
                P_2d    = jnp.where((self._kz2_[iz] == 0.0) & (self._kxy2_ == 0.0), 0.0, P_2d)
                amp_2d  = jnp.sqrt(P_2d * inv2V).astype(self.real_dtype)
                slab    = (amp_2d * gauss_3d[:, :, iz]).astype(self.complex_dtype)
                return acc.at[:, :, iz].set(slab)

            out = lax.fori_loop(0, self.kz_.shape[0], body, out)
            return out


class Beta_Combine_Mixin:
    r"""
    Shared helpers to combine basis fields in k-space and compute beta.
    Requires the subclass to define:
      - self.ng_E
      - self.kx2E, self.ky2E, self.kz2E
    """

    def _kmu_from_cache_or_build(self):
        r"""
        Returns:
          k2, mu2, mu4, kmag, mu (shape = (ng_E, ng_E, ng_E//2+1))
        """
        if hasattr(self, "_k2E_grid"):
            k2  = self._k2E_grid
            mu2 = self._mu2E_grid
            mu4 = self._mu4E_grid
        else:
            k2  = (self.kx2E[:, None, None] + self.ky2E[None, :, None] + self.kz2E[None, None, :]).astype(self.real_dtype)
            mu2 = jnp.where(k2 > 0, self.kz2E[None, None, :] / k2, 0.0).astype(self.real_dtype)
            mu4 = (mu2 * mu2).astype(self.real_dtype)

        kmag = jnp.sqrt(jnp.maximum(k2, 0.0)).astype(self.real_dtype)
        mu   = jnp.sqrt(mu2).astype(self.real_dtype)
        return k2, mu2, mu4, kmag, mu

    @partial(jit, static_argnames=('self', 'measure_pk'))
    def get_beta(
        self,
        true_field_k: jnp.ndarray,     # (..., ng, ng, ng//2+1), complex
        fields_k: jnp.ndarray,         # (n, ng, ng, ng//2+1), complex
        mu_edges: jnp.ndarray,         # (Nmu+1,)
        *,
        measure_pk,                    # Measure_Pk instance (static)
        eps: float = 0.0,
    ) -> jnp.ndarray:
        r"""
        Compute beta_i(k,mu) = P( true, field_i ) / P( field_i, field_i )
        for all i and all mu-bins in one go.

        Returns
        -------
        beta : (n_fields, Nk, Nmu)
        Notes
        -----
        - Uses a fori_loop over mu-bins to keep compile size moderate.
        - Within each mu-bin, uses lax.map over fields.
        - Adds 'eps' to denominator to avoid /0 in empty bins.
        """
        n_fields = fields_k.shape[0]
        dtype = measure_pk.dtype
        Nk = int(jnp.asarray(measure_pk.k_mean, dtype=dtype).shape[0])
        Nmu = int(mu_edges.shape[0] - 1)

        def mu_body(m, beta_acc):
            mu_min = mu_edges[m]
            mu_max = mu_edges[m + 1]

            # Compute P_auto_i(k) and P_cross_i(k) for all fields i
            def one_field(i):
                # cross(true, field_i)
                out_c  = measure_pk(true_field_k, fields_k[i], ell=0,
                                    mu_min=mu_min, mu_max=mu_max)
                Pci = out_c[:, 1].astype(dtype)                  # (Nk,)

                # auto(field_i)
                out_a  = measure_pk(fields_k[i], None, ell=0,
                                    mu_min=mu_min, mu_max=mu_max)
                Pai = out_a[:, 1].astype(dtype)                  # (Nk,)

                return Pci / (Pai + jnp.asarray(eps, dtype=dtype))

            beta_m = lax.map(one_field, jnp.arange(n_fields, dtype=jnp.int32))  # (n, Nk)
            return beta_acc.at[:, :, m].set(beta_m)

        beta0 = jnp.zeros((n_fields, Nk, Nmu), dtype=dtype)
        beta  = lax.fori_loop(0, Nmu, mu_body, beta0)
        return beta
        
    def _ensure_nearest_cache(self, k_edges: jnp.ndarray, mu_edges: jnp.ndarray):
        r"""
        Build (and cache) a nearest-neighbor interpolation map from grid points to (k, mu) bins.
        The map is reused as long as (k_edges, mu_edges) are unchanged.
        """
        # If edges are identical to the last call, just reuse the cached mapping.
        if (self._nearest_idx is not None
            and self._k_edges_cache is not None
            and self._mu_edges_cache is not None
            and jnp.array_equal(self._k_edges_cache, k_edges)
            and jnp.array_equal(self._mu_edges_cache, mu_edges)):
            return

        # Construct k^2 on the rfft grid to avoid a costly sqrt for k-binning.
        k2 = (self.kx2E[:, None, None] + self.ky2E[None, :, None] + self.kz2E[None, None, :])

        # Bin by k using k^2; edges are squared once (monotone -> binning is preserved).
        k_edges2 = (k_edges * k_edges).astype(self.real_dtype)
        Nk = int(k_edges.shape[0] - 1)
        kbin = jnp.clip(jnp.searchsorted(k_edges2, k2, side="right") - 1, 0, Nk - 1)

        # Bin by mu using mu^2 to avoid sqrt; again edges are squared once.
        mu2 = jnp.where(k2 > 0, self.kz2E[None, None, :] / k2, 0.0).astype(self.real_dtype)
        mu_edges2 = (mu_edges * mu_edges).astype(self.real_dtype)
        Nmu = int(mu_edges.shape[0] - 1)
        mubin = jnp.clip(jnp.searchsorted(mu_edges2, mu2, side="right") - 1, 0, Nmu - 1)

        # Flattened table index for (k,mu) -> Nk*Nmu
        self._nearest_idx = (kbin * Nmu + mubin).astype(jnp.int32)

        # Remember edges to detect changes next time
        self._k_edges_cache = jnp.asarray(k_edges, dtype=self.real_dtype)
        self._mu_edges_cache = jnp.asarray(mu_edges, dtype=self.real_dtype)

    @partial(jit, static_argnames=('self',))
    def _get_final_field_table(
        self,
        fields_k: jnp.ndarray,            # (n_fields, ng_E, ng_E, ng_E//2+1)
        beta_tab: jnp.ndarray,            # (n_fields, Nk, Nmu)
    ) -> jnp.ndarray:
        r"""
        Build delta_g(k) = sum_i beta_i[bin(k,mu)] * O_i(k,mu).
        Nearest-bin lookup version for beta given on (k,mu) table.
        """
        num_fields = fields_k.shape[0]
        NkNmu = beta_tab.shape[1] * beta_tab.shape[2]
        beta_flat = beta_tab.reshape(num_fields, NkNmu).astype(self.real_dtype)

        acc0 = jnp.zeros_like(fields_k[0])  # complex accumulator

        # Use fori_loop to avoid (carry, x) confusion and keep memory peak low.
        def loop(i, acc):
            # gather coefficients for the whole grid and axpy
            coeff = jnp.take(beta_flat[i], self._nearest_idx, axis=0)  # real, shape=grid
            return acc + coeff * fields_k[i]                            # complex

        acc = lax.fori_loop(0, num_fields, loop, acc0)
        return acc
    
    @partial(jit, static_argnames=('self',))
    def _get_final_field_const(
        self,
        fields_k: jnp.ndarray,        # (n_fields, ng, ng, ng//2+1), complex
        beta_const: jnp.ndarray,      # (n_fields,), real
    ) -> jnp.ndarray:
        r"""
        Constant coefficients: delta_g = sum_i beta[i] * O_i(k, mu).
        """
        n = fields_k.shape[0]
        beta_const = jnp.asarray(beta_const, dtype=self.real_dtype)

        acc0 = jnp.zeros_like(fields_k[0])

        def loop(i, acc):
            return acc + beta_const[i] * fields_k[i]
        return lax.fori_loop(0, n, loop, acc0)
    
    @partial(jit, static_argnames=('self',))
    def _get_final_field_const_kmu(
        self,
        fields_k: jnp.ndarray,           # (n_fields, ng, ng, ng//2+1), complex
        base_consts: jnp.ndarray,        # (n_fields,), real (others' constants)
        c_idx: int,                   # which field uses k–mu polynomial
        c_coeffs: jnp.ndarray,        # (4,) real: [c0, c1, c2, c4]
    ) -> jnp.ndarray:
        r"""
        Use constants for all fields except one, which gets (c0 k^2 + c1 mu^2 + c2 k^2 mu^2 + c4 k^2 mu^4).
        """
        k2, mu2, mu4, _, _ = self._kmu_from_cache_or_build()
        c0, c1, c2, c4 = [jnp.asarray(c_coeffs[i], dtype=self.real_dtype) for i in range(4)]
        # beta[i] + c0*k^2 + c1*mu^2 + c2*k^2*mu^2 + c4*k^2*mu^4
        poly_grid = c0 * k2 + c1 * mu2 + c2 * k2 * mu2 + c4 * k2 * mu4

        n = fields_k.shape[0]
        base_consts = jnp.asarray(base_consts, dtype=self.real_dtype)
        acc0 = jnp.zeros_like(fields_k[0])

        def body(i, acc):
            coef_grid = jnp.where(i == c_idx, base_consts[i] + poly_grid, base_consts[i])
            return acc + coef_grid * fields_k[i]

        return lax.fori_loop(0, n, body, acc0)
    
    @partial(jit, static_argnames=('self',))
    def _get_final_field_poly_grid(
        self,
        fields_k: jnp.ndarray,           # (n_fields, ng, ng, ng//2+1), complex
        coeffs: jnp.ndarray,             # (n_fields, Lk, Lmu), real
        k_pows: jnp.ndarray,             # (Lk,), int (exponents for k)
        mu_pows: jnp.ndarray,            # (Lmu,), int (exponents for mu)
    ) -> jnp.ndarray:
        r"""
        Polynomial on the Cartesian product basis:
        beta_i(k,mu) = sum_{a,b} coeffs[i,a,b] * (k**k_pows[a]) * (mu**mu_pows[b])
        """
        # Build k^2 and mu on the Eulerian rfft grid
        _, _, _, kmag, mu = self._kmu_from_cache_or_build()

        # Precompute powers stacks
        k_pows = k_pows.astype(jnp.int32);  mu_pows = mu_pows.astype(jnp.int32)
        Lk = int(k_pows.shape[0]); Lm = int(mu_pows.shape[0])

        def build_pows(base, exps):
            out0 = jnp.zeros((exps.shape[0],) + base.shape, dtype=self.real_dtype)
            def body(i, acc):
                e = exps[i]
                val = jnp.where(e == 0, jnp.ones_like(base), base ** e)
                return acc.at[i].set(val)
            return lax.fori_loop(0, exps.shape[0], body, out0)

        Kpow = build_pows(kmag, k_pows)   # (Lk, grid)
        Mpow = build_pows(mu,   mu_pows)  # (Lmu, grid)

        n_fields = fields_k.shape[0]
        coeffs = coeffs.astype(self.real_dtype)
        acc0 = jnp.zeros_like(fields_k[0])

        def loop_field(i, acc):
            coef_i = coeffs[i]  # (Lk, Lmu)
            beta_grid0 = jnp.zeros_like(kmag)
            def loop_a(a, beta_acc):
                Ka = Kpow[a]
                def loop_b(b, s):
                    return s + coef_i[a, b] * Mpow[b]
                s_ab = lax.fori_loop(0, Lm, loop_b, jnp.zeros_like(kmag))
                return beta_acc + Ka * s_ab
            beta_i_grid = lax.fori_loop(0, Lk, loop_a, beta_grid0)
            return acc + beta_i_grid * fields_k[i]

        return lax.fori_loop(0, n_fields, loop_field, acc0)
    
    @partial(jit, static_argnames=('self',))
    def _get_final_field_poly_pair(
        self,
        fields_k: jnp.ndarray,           # (n_fields, ng, ng, ng//2+1), complex
        coeffs: jnp.ndarray,             # (n_fields, L), real (pairwise terms)
        k_pows: jnp.ndarray,             # (L,), int exponents for k (term-wise)
        mu_pows: jnp.ndarray,            # (L,), int exponents for mu (term-wise)
    ) -> jnp.ndarray:
        r"""
        Polynomial on a pairwise basis:
        beta_i(k,mu) = sum_{t=0}^{L-1} coeffs[i,t] * (k**k_pows[t]) * (mu**mu_pows[t])

        Notes
        -----
        - `k_pows` and `mu_pows` have the same length L and define each term.
        - This matches `beta_polyfit(..., pairwise=True)` output directly.
        """
        # Build kmag and mu on the Eulerian rfft grid
        _, _, _, kmag, mu = self._kmu_from_cache_or_build()

        k_pows = k_pows.astype(jnp.int32)
        mu_pows = mu_pows.astype(jnp.int32)
        L = int(k_pows.shape[0])

        # Precompute per-term powers to avoid recomputing inside the field loop
        def build_terms(base, exps):
            out0 = jnp.zeros((exps.shape[0],) + base.shape, dtype=self.real_dtype)
            def body(t, acc):
                e = exps[t]
                val = jnp.where(e == 0, jnp.ones_like(base), base ** e)
                return acc.at[t].set(val)
            return lax.fori_loop(0, exps.shape[0], body, out0)

        Kt = build_terms(kmag, k_pows)   # (L, grid)
        Mt = build_terms(mu,   mu_pows)  # (L, grid)

        n_fields = fields_k.shape[0]
        coeffs = coeffs.astype(self.real_dtype)
        acc0 = jnp.zeros_like(fields_k[0])

        def loop_field(i, acc):
            beta_grid = jnp.zeros_like(kmag)
            def loop_term(t, b):
                return b + coeffs[i, t] * Kt[t] * Mt[t]
            beta_i = lax.fori_loop(0, L, loop_term, beta_grid)
            return acc + beta_i * fields_k[i]

        return lax.fori_loop(0, n_fields, loop_field, acc0)

    @partial(jit, static_argnames=('self',))
    def _get_final_field_poly_pair_perfield(
        self,
        fields_k: jnp.ndarray,            # (n_fields, ng, ng, ng//2+1), complex
        coeffs: jnp.ndarray,              # (n_fields, Lmax), real (padded)
        k_pows_2d: jnp.ndarray,           # (n_fields, Lmax), int
        mu_pows_2d: jnp.ndarray,          # (n_fields, Lmax), int
        term_mask: jnp.ndarray,           # (n_fields, Lmax), bool (True -> valid term)
    ) -> jnp.ndarray:
        r"""Pairwise basis where each field i has its own term list (padded to Lmax)."""
        _, _, _, kmag, mu = self._kmu_from_cache_or_build()

        n_fields, Lmax = coeffs.shape
        coeffs = coeffs.astype(self.real_dtype)
        k_pows_2d = k_pows_2d.astype(jnp.int32)
        mu_pows_2d = mu_pows_2d.astype(jnp.int32)
        term_mask_f = term_mask.astype(self.real_dtype)  # multiply as 0/1

        acc0 = jnp.zeros_like(fields_k[0])

        def loop_field(i, acc):
            beta_grid = jnp.zeros_like(kmag)
            def loop_term(t, b):
                m = term_mask_f[i, t]
                ek = k_pows_2d[i, t]
                em = mu_pows_2d[i, t]
                # if masked, contribute 0 without computing pow
                Kt = jnp.where(m > 0, jnp.where(ek == 0, jnp.ones_like(kmag), kmag ** ek), 0.0)
                Mt = jnp.where(m > 0, jnp.where(em == 0, jnp.ones_like(mu),   mu   ** em),  0.0)
                return b + coeffs[i, t] * Kt * Mt
            beta_i = lax.fori_loop(0, Lmax, loop_term, beta_grid)
            return acc + beta_i * fields_k[i]

        return lax.fori_loop(0, n_fields, loop_field, acc0)

    def get_final_field(
        self,
        fields_k: jnp.ndarray,        # (n_fields, ng, ng, ng//2+1)
        beta: jnp.ndarray,            # see beta_type below
        *,
        beta_type: Literal['table','const','poly'] = 'table',
        k_edges: jnp.ndarray | None = None,   # required for beta_type='table'
        mu_edges: jnp.ndarray | None = None,  # required for beta_type='table'
        poly_k_pows: jnp.ndarray | None = None,   # (Lk,), for beta_type='poly'
        poly_mu_pows: jnp.ndarray | None = None,  # (Lmu,), for beta_type='poly'
        term_mask: jnp.ndarray | None = None,     # for 'poly' 2D-per-field
    ) -> jnp.ndarray:
        r"""
        Build delta_g(k,mu) = sum_i beta_i(k,mu) * O_i(k,mu) in three modes:

        'table': beta is (n_fields, Nk, Nmu) and uses nearest-bin lookup.
        'const': beta is (n_fields,) and is constant per field.
        'poly' : polynomial in (k, mu).
                 - If beta.ndim == 3: grid basis (coeffs: n x Lk x Lmu).
                 - If beta.ndim == 2: pairwise basis (coeffs: n x L)  <-- pairwise=True
        """
        if fields_k.shape[1] != self.ng_E:
            raise ValueError(f"fields_k ng_E mismatch: got {fields_k.shape[1]}, expected {self.ng_E}")

        if beta_type == 'table':
            if k_edges is None or mu_edges is None:
                raise ValueError("beta_type='table' requires k_edges and mu_edges.")
            if beta.ndim != 3:
                raise ValueError("For beta_type='table', beta must have shape (n_fields, Nk, Nmu).")
            self._ensure_nearest_cache(k_edges, mu_edges)
            return self._get_final_field_table(fields_k, beta)

        elif beta_type == 'const':
            beta = jnp.asarray(beta, dtype=self.real_dtype)
            n_fields = int(fields_k.shape[0])

            try:
                is_lpt = isinstance(self, LPT_Forward)
            except NameError:
                is_lpt = False
            c_idx = 1 if (is_lpt and n_fields >= 2) else 0

            if beta.ndim != 1:
                raise ValueError("For beta_type='const', beta must have shape (n_fields,).")
            
            if beta.size == n_fields:
                return self._get_final_field_const(fields_k, beta)
            
            if beta.size == n_fields + 4:
                # + [c0, c1, c2, c4]
                base_consts = beta[:n_fields]
                c_coeffs = beta[n_fields:n_fields+4]
                return self._get_final_field_const_kmu(fields_k, base_consts, c_idx, c_coeffs)
            
            raise ValueError(
                "For beta_type='const', pass beta.shape == (n_fields,) or (n_fields+4,) "
                "where the last 4 are [c0, c1, c2, c4]."
            )

        elif beta_type == 'poly':
            if poly_k_pows is None or poly_mu_pows is None:
                raise ValueError("For beta_type='poly', provide poly_k_pows and poly_mu_pows.")
            # grid-form (Cartesian product): beta is (n, Lk, Lmu) and pows are 1D
            if beta.ndim == 3:
                if poly_k_pows.ndim != 1 or poly_mu_pows.ndim != 1:
                    raise ValueError("Grid basis expects 1D poly_k_pows / poly_mu_pows.")
                return self._get_final_field_poly_grid(fields_k, beta, poly_k_pows, poly_mu_pows)
            # pairwise-form (global terms): beta is (n, L) and pows are 1D
            if beta.ndim == 2 and poly_k_pows.ndim == 1 and poly_mu_pows.ndim == 1:
                return self._get_final_field_poly_pair(fields_k, beta, poly_k_pows, poly_mu_pows)
            # pairwise-form (per-field terms): beta is (n, Lmax) and pows are 2D
            if beta.ndim == 2 and poly_k_pows.ndim == 2 and poly_mu_pows.ndim == 2:
                if poly_k_pows.shape != beta.shape or poly_mu_pows.shape != beta.shape:
                    raise ValueError("Per-field pairwise expects shapes: beta, poly_k_pows, poly_mu_pows all (n,Lmax).")
                if term_mask is None:
                    # by default, use exponent >= 0 as 'valid'
                    term_mask = (poly_k_pows >= 0) & (poly_mu_pows >= 0)
                return self._get_final_field_poly_pair_perfield(fields_k, beta, poly_k_pows, poly_mu_pows, term_mask)
            raise ValueError("Unsupported 'poly' shapes combination.")
        else:
            raise ValueError(f"Unknown beta_type='{beta_type}'")
        


# ------------------------------ LPT (1LPT) ------------------------------

class LPT_Forward(Base_Forward, Beta_Combine_Mixin):
    r"""
    Compute shifted fields using LPT displacement.
    
    
    Assignment strategy:
      - assign_mode="auto" (default):
          use `vmap` if (ng_E <= vmap_ng_threshold and n_fields <= vmap_fields_threshold)
          otherwise use Python-for sequential assignment.
      - assign_mode="for":  always sequential assignment (lowest peak memory).
      - assign_mode="vmap": always vmap'ed assignment (may be faster for small problems).

    FFT+deconvolution:
      - Always uses `Mesh_Assignment.fft_deconvolve_batched` to batch FFT across fields.
    """

    def __init__(
        self,
        *,
        boxsize: float,
        ng: Optional[int] = None,
        ng_L: int,
        ng_E: int,
        mas_cfg: Tuple[int, bool],
        rsd: bool = False,
        lya: bool = False,
        lpt_order: int = 1,
        bias_order: int = 2,
        dtype=jnp.float32,
        max_scatter_indices: int = 200_000_000,
        use_batched_fft: bool = True,
        # assignment switching knobs
        assign_mode: Literal["auto", "for", "vmap"] = "auto",
        vmap_ng_threshold: int = 256,
        vmap_fields_threshold: int = 4,
    ) -> None:
        super().__init__(boxsize=boxsize, ng=ng, ng_L=ng_L, dtype=dtype, use_batched_fft=use_batched_fft)

        # Eulerian grid & MAS
        self.ng_E = int(ng_E)
        self.window_order, self.interlace = mas_cfg
        self.mesh = assign_util.Mesh_Assignment(
            self.boxsize, self.ng_E, self.window_order,
            interlace=bool(self.interlace), normalize=True, 
            max_scatter_indices=max_scatter_indices,
            dtype=self.real_dtype,
        )

        self.rsd = bool(rsd)
        self.lya = bool(lya)
        self.lpt_order = int(lpt_order)
        self.bias_order = int(bias_order)

        # Assignment switching policy
        self.assign_mode: Literal["auto", "for", "vmap"] = assign_mode
        self.vmap_ng_threshold: int = int(vmap_ng_threshold)
        self.vmap_fields_threshold: int = int(vmap_fields_threshold)

        # Do not store 3D base positions; build on-the-fly
        self.cell_size = self.boxsize / self.ng_L

        self.kxE, self.kyE, self.kzE = coord.kaxes_1d(self.ng_E, self.boxsize, dtype=self.real_dtype)
        self.kx2E = self.kxE * self.kxE
        self.ky2E = self.kyE * self.kyE
        self.kz2E = self.kzE * self.kzE

        # Small cache for 'nearest' interpolation map
        self._nearest_idx = None       # int32 indices of shape (ng_E, ng_E, ng_E//2+1)
        self._k_edges_cache = None     # last-used k_edges
        self._mu_edges_cache = None    # last-used mu_edges

    # -------------------- LPT displacement --------------------
    @partial(jit, static_argnames=('self',))
    def lpt(self, delta_k_L: jnp.ndarray, growth_f: float = 0.0) -> jnp.ndarray:
        delta_k_L = delta_k_L.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        disp1_k = coord.apply_disp_k(delta_k_L, self.kx, self.ky, self.kz).astype(self.complex_dtype)
        disp1_r = self._irfftn_vec(disp1_k).astype(self.real_dtype)
        if self.rsd:
            gf = jnp.asarray(growth_f, dtype=self.real_dtype)
            disp1_r = disp1_r.at[2].add(disp1_r[2] * gf)
        return disp1_r  # (3, ng, ng, ng)

    # -------- scalar fields in position space -------
    @partial(jit, static_argnames=('self',))
    def _scalar_fields_r(self, delta_k: jnp.ndarray) -> jnp.ndarray:
        r"""
        Build scalar fields in position space: [1, delta, d^2, G2, (G2_zz), (LyA extras...)]
        Returns stacked array with leading field axis.
        """
        delta_k = delta_k.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        fields = [jnp.ones((delta_k.shape[0],) * 3, dtype=self.real_dtype)]

        if self.bias_order >= 1:
            delta_r = self.irfftn(delta_k)
            fields.append(delta_r)

        if self.bias_order >= 2:
            d2_r = delta_r**2 - jnp.mean(delta_r**2)

            Gij_k = coord.apply_Gij_k(delta_k, self.kx, self.ky, self.kz)  # (6, ...)
            Gij_r = self._irfftn_vec(Gij_k)  # real
            # 0: xx, 1: xy, 2: xz, 3: yy, 4: yz, 5: zz

            phi2 = (Gij_r[0] * Gij_r[3] + Gij_r[3] * Gij_r[5] + Gij_r[5] * Gij_r[0]
                    - Gij_r[1]**2 - Gij_r[2]**2 - Gij_r[4]**2)
            G2_r = -2.0 * phi2
            G2_r = G2_r - jnp.mean(G2_r)

            fields.extend([d2_r, G2_r])

            # RSD term G2_zz
            if self.rsd or self.lya:
                mu2 = coord.mu2_grid(self.kx, self.ky, self.kz)
                G2_zz_r = self.irfftn(self.rfftn(G2_r) * mu2)
                G2_zz_r = G2_zz_r - jnp.mean(G2_zz_r)
                fields.append(G2_zz_r)

            # Ly-A extras
            if self.lya:
                eta_r = Gij_r[5]  # equals delta * mu^2 in k space (Gzz)
                eta2_r = eta_r**2 - jnp.mean(eta_r**2)
                deta_r = delta_r * eta_r - jnp.mean(delta_r * eta_r)
                GG_zz_r = (Gij_r[2]**2 + Gij_r[4]**2 + Gij_r[5]**2)
                GG_zz_r = GG_zz_r - jnp.mean(GG_zz_r)

                KK_zz_r = GG_zz_r - (2.0/3.0) * deta_r + (1.0/9.0) * d2_r
                KK_zz_r = KK_zz_r - jnp.mean(KK_zz_r)
                fields.extend([eta2_r, deta_r, KK_zz_r])

        return jnp.array(fields)
    
    # -------- tensor fields in position space -------
    @partial(jit, static_argnames=('self',))
    def _tensor_fields_r(self, delta_k: jnp.ndarray) -> jnp.ndarray:
        r"""
        Build tensor fields in position space: [K_ij, dK_ij, G_ij, T_ij, ...]
        Returns
        -------
        (n_fields, 6, ng, ng, ng) with order (xx,xy,xz,yy,yz,zz) in the 6-axis.
        """
        delta_k = delta_k.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        
        K_ij_k = coord.apply_Gij_k(delta_k, self.kx, self.ky, self.kz)  # (6, ng_L, ng_L, ng_L//2+1)
        K_ij_k = coord.apply_traceless(K_ij_k)

        fields = []

        if self.bias_order >= 1:
            K_ij_r = self._irfftn_vec(K_ij_k)  # (6, ng_L, ng_L, ng_L)
            fields.append(K_ij_r)

        if self.bias_order >= 2:
            delta_r = self.irfftn(delta_k)

            dK_ij_r = K_ij_r * delta_r[None, :, :, :]  # (6, ng_L, ng_L, ng_L)
            dK_ij_r = dK_ij_r - jnp.mean(dK_ij_r, axis=(1, 2, 3), keepdims=True)

            #G_ij_r = self._irfftn_vec(G_ij_k)  # (6, ng_L, ng_L, ng_L)

            #KK_xx_r = G_ij_r[0]*G_ij_r[0] + G_ij_r[1]*G_ij_r[1] + G_ij_r[2]*G_ij_r[2]
            #KK_xy_r = G_ij_r[0]*G_ij_r[1] + G_ij_r[1]*G_ij_r[3] + G_ij_r[2]*G_ij_r[4]
            #KK_xz_r = G_ij_r[0]*G_ij_r[2] + G_ij_r[1]*G_ij_r[4] + G_ij_r[2]*G_ij_r[5]
            #KK_yy_r = G_ij_r[3]*G_ij_r[3] + G_ij_r[1]*G_ij_r[1] + G_ij_r[4]*G_ij_r[4]
            #KK_yz_r = G_ij_r[3]*G_ij_r[4] + G_ij_r[1]*G_ij_r[2] + G_ij_r[4]*G_ij_r[5]
            #KK_zz_r = G_ij_r[5]*G_ij_r[5] + G_ij_r[2]*G_ij_r[2] + G_ij_r[4]*G_ij_r[4]

            KK_xx_r = K_ij_r[0]*K_ij_r[0] + K_ij_r[1]*K_ij_r[1] + K_ij_r[2]*K_ij_r[2]
            KK_xy_r = K_ij_r[0]*K_ij_r[1] + K_ij_r[1]*K_ij_r[3] + K_ij_r[2]*K_ij_r[4]
            KK_xz_r = K_ij_r[0]*K_ij_r[2] + K_ij_r[1]*K_ij_r[4] + K_ij_r[2]*K_ij_r[5]
            KK_yy_r = K_ij_r[3]*K_ij_r[3] + K_ij_r[1]*K_ij_r[1] + K_ij_r[4]*K_ij_r[4]
            KK_yz_r = K_ij_r[3]*K_ij_r[4] + K_ij_r[1]*K_ij_r[2] + K_ij_r[4]*K_ij_r[5]
            KK_zz_r = K_ij_r[5]*K_ij_r[5] + K_ij_r[2]*K_ij_r[2] + K_ij_r[4]*K_ij_r[4]

            K2_r = K_ij_r[0]**2 + K_ij_r[3]**2 + K_ij_r[5]**2 + 2.0*(K_ij_r[1]**2 + K_ij_r[2]**2 + K_ij_r[4]**2)
            K2_over_3 = (K2_r / 3.0).astype(self.real_dtype)
            
            ### Traceless
            KK_xx_r = KK_xx_r - K2_over_3
            KK_yy_r = KK_yy_r - K2_over_3
            KK_zz_r = KK_zz_r - K2_over_3
            
            KK_ij_r = jnp.stack([KK_xx_r, KK_xy_r, KK_xz_r, KK_yy_r, KK_yz_r, KK_zz_r], axis=0)
            KK_ij_r = KK_ij_r - jnp.mean(KK_ij_r, axis=(1, 2, 3), keepdims=True)
            G_ij_r = KK_ij_r - (1./3.)*dK_ij_r

            T_r = (delta_r*delta_r - 1.5 * K2_r).astype(self.real_dtype)
            T_ij_k = coord.apply_Gij_k(self.rfftn(T_r), self.kx, self.ky, self.kz)  # (6, nx, ny, nzr)
            T_ij_k = coord.apply_traceless(T_ij_k)
            T_ij_r = self._irfftn_vec(T_ij_k)  # (6, ng_L, ng_L, ng_L)
            T_ij_r = T_ij_r - jnp.mean(T_ij_r, axis=(1, 2, 3), keepdims=True)

            fields.extend([dK_ij_r, G_ij_r, T_ij_r])

        return jnp.array(fields)

    # -------- linear & quadratic helper fields (API preserved) --------
    @partial(jit, static_argnames=('self',))
    def eta_r(self, delta_k_L):
        return self.irfftn(self.eta_k(delta_k_L))

    @partial(jit, static_argnames=('self',))
    def eta_k(self, delta_k_L):
        delta_k_L = delta_k_L.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        mu2 = coord.mu2_grid(self.kx, self.ky, self.kz)
        return delta_k_L * mu2

    @partial(jit, static_argnames=('self',))
    def d2_r(self, delta_k_L):
        delta_k_L = delta_k_L.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        delta_r = self.irfftn(delta_k_L)
        d2 = delta_r ** 2
        return d2 - jnp.mean(d2)

    @partial(jit, static_argnames=('self',))
    def d2_k(self, delta_k_L):
        return self.rfftn(self.d2_r(delta_k_L))

    @partial(jit, static_argnames=('self',))
    def G2_r(self, delta_k_L):
        delta_k_L = delta_k_L.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        Gij_k = coord.apply_Gij_k(delta_k_L, self.kx, self.ky, self.kz)
        Gij_r = self._irfftn_vec(Gij_k)
        phi2_r = (Gij_r[0]*Gij_r[3] + Gij_r[3]*Gij_r[5] + Gij_r[5]*Gij_r[0]
                  - Gij_r[1]**2 - Gij_r[2]**2 - Gij_r[4]**2)
        G2 = -2.0 * phi2_r
        return G2 - jnp.mean(G2)

    @partial(jit, static_argnames=('self',))
    def G2_k(self, delta_k_L):
        delta_k_L = delta_k_L.astype(self.complex_dtype)
        return self.rfftn(self.G2_r(delta_k_L))

    @partial(jit, static_argnames=('self',))
    def G2_zz_r(self, delta_k_L):
        delta_k_L = delta_k_L.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        return self.irfftn(self.G2_zz_k(delta_k_L))

    @partial(jit, static_argnames=('self',))
    def G2_zz_k(self, delta_k_L):
        delta_k_L = delta_k_L.astype(self.complex_dtype)
        return self.rfftn(self.G2_r(delta_k_L)) * coord.mu2_grid(self.kx, self.ky, self.kz)

    # LyA helpers
    @partial(jit, static_argnames=('self',))
    def eta2_r(self, delta_k_L):
        delta_k_L = delta_k_L.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        eta = self.eta_r(delta_k_L)
        eta2 = eta ** 2
        return eta2 - jnp.mean(eta2)

    @partial(jit, static_argnames=('self',))
    def eta2_k(self, delta_k_L):
        return self.rfftn(self.eta2_r(delta_k_L))

    @partial(jit, static_argnames=('self',))
    def deta_r(self, delta_k_L):
        delta_k_L = delta_k_L.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        delta_r = self.irfftn(delta_k_L)
        eta_r = self.eta_r(delta_k_L)
        cross = delta_r * eta_r
        return cross - jnp.mean(cross)

    @partial(jit, static_argnames=('self',))
    def deta_k(self, delta_k_L):
        return self.rfftn(self.deta_r(delta_k_L))

    @partial(jit, static_argnames=('self',))
    def GG_zz_r(self, delta_k_L):
        delta_k_L = delta_k_L.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        Gij_k = coord.apply_Gij_k(delta_k_L, self.kx, self.ky, self.kz)
        Gij_r = self._irfftn_vec(Gij_k)
        GG_zz = (Gij_r[2]**2 + Gij_r[4]**2 + Gij_r[5]**2)
        return GG_zz - jnp.mean(GG_zz)

    @partial(jit, static_argnames=('self',))
    def GG_zz_k(self, delta_k_L):
        return self.rfftn(self.GG_zz_r(delta_k_L))

    @partial(jit, static_argnames=('self',))
    def KK_zz_r(self, delta_k_L):
        delta_k_L = delta_k_L.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        GG_zz_r = self.GG_zz_r(delta_k_L)
        KK_zz_r = GG_zz_r - (2.0/3.0)*self.deta_r(delta_k_L) + (1.0/9.0)*self.d2_r(delta_k_L)
        return KK_zz_r - jnp.mean(KK_zz_r)

    @partial(jit, static_argnames=('self',))
    def KK_zz_k(self, delta_k_L):
        return self.rfftn(self.KK_zz_r(delta_k_L))
    
    # -------------------- assignment policy helpers --------------------
    def _get_assign_mode(self, n_fields: int) -> Literal["for", "vmap"]:
        """Pick assignment strategy (`for` or `vmap`) under `assign_mode` policy."""
        if self.assign_mode == "for":
            return "for"
        if self.assign_mode == "vmap":
            return "vmap"
        # auto: small grids & few fields -> vmap; otherwise for
        if (self.ng_E <= self.vmap_ng_threshold) and (n_fields <= self.vmap_fields_threshold):
            return "vmap"
        return "for"
        
    def _assign_fields_from_disp_to_grid(
        self,
        disp_r: jnp.ndarray,      # (3, ng_L, ng_L, ng_L)
        fields_r: jnp.ndarray,    # (m, ng_L, ng_L, ng_L) for scalar or (m, 6, ng_L, ng_L, ng_L) for tensors
        *,
        interlace: bool = False,
        normalize_mean: bool = True,
        field_type: Literal['scalar', 'tensor'] = 'scalar',
        neighbor_mode: str = 'auto',
        fuse_updates_threshold: int = 100_000_000,
    ) -> jnp.ndarray:
        r"""
        Assign many position-space fields to Eulerian grid using either `for` or `vmap`.

        Returns
        -------
        fields_E : (n_out, ng_E, ng_E, ng_E) real
            - If field_type == 'scalar': n_out = m
            - If field_type == 'tensor': n_out = m * 6  (order = (xx,xy,xz,yy,yz,zz) per field, concatenated)
        """
        # --- Flatten tensor components into the leading axis if needed ---
        if field_type == 'tensor':
            if fields_r.ndim != 5 or fields_r.shape[1] != 6:
                raise ValueError("For field_type='tensor', fields_r must have shape (m, 6, ng_L, ng_L, ng_L).")
            m, c, nx, ny, nz = fields_r.shape
            fields_flat_r = fields_r.reshape(m * c, nx, ny, nz)
        else:
            if fields_r.ndim != 4:
                raise ValueError("For field_type='scalar', fields_r must have shape (m, ng_L, ng_L, ng_L).")
            m, nx, ny, nz = fields_r.shape
            fields_flat_r = fields_r

        n_out = int(fields_flat_r.shape[0])
        mode = self._get_assign_mode(n_out)

        if mode == "vmap":
            fn = lambda w: self.mesh.assign_from_disp_to_grid(
                disp_r, w, interlace=interlace, normalize_mean=normalize_mean, 
                neighbor_mode=neighbor_mode, fuse_updates_threshold=fuse_updates_threshold,
            )
            # vmap over field axis
            fields = vmap(fn, in_axes=0, out_axes=0)(fields_flat_r)
        else:
            fields = []
            for i in range(n_out):
                w = fields_flat_r[i]
                gi = self.mesh.assign_from_disp_to_grid(
                    disp_r, w, interlace=interlace, normalize_mean=normalize_mean, 
                    neighbor_mode=neighbor_mode, fuse_updates_threshold=fuse_updates_threshold,
                )
                fields.append(gi)
            fields = jnp.stack(fields, axis=0)

        return fields.astype(self.real_dtype)

    def get_shifted_fields(
        self,
        delta_k: jnp.ndarray,
        *,
        growth_f: float = 0.0,
        mode: str = 'k_space',
        field_type: Literal['scalar', 'tensor'] = 'scalar',
        neighbor_mode: str = 'auto',
        fuse_updates_threshold: int=100_000_000,
    ) -> jnp.ndarray:
        r"""
        Build displacement (1LPT) and scalar real-space fields, assign them to Eulerian grid,
        and (optionally) FFT+deconvolve them in a single batched call.

        Returns
        -------
        If field_type == 'scalar':
            - mode == 'k_space' : (m, ng_E, ng_E, ng_E//2+1), complex
            - else              : (m, ng_E, ng_E, ng_E),       real
        If field_type == 'tensor' (components in order xx,xy,xz,yy,yz,zz for each field):
            - mode == 'k_space' : (m, 6, ng_E, ng_E, ng_E//2+1), complex
            - else              : (m, 6, ng_E, ng_E, ng_E),       real
        """
        delta_k = delta_k.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        delta_k_L = coord.func_extend(self.ng_L, delta_k)

        # lpt displacement
        disp_r_L = self.lpt(delta_k_L, growth_f=growth_f)  # (3, ng, ng, ng)

        # list of scalar fields in position space
        if field_type == 'scalar':
            fields_r_L = self._scalar_fields_r(delta_k_L)    # (m, ng_L, ng_L, ng_L)
            m = int(fields_r_L.shape[0])
        elif field_type == 'tensor':
            fields_r_L = self._tensor_fields_r(delta_k_L)    # (m, 6, ng_L, ng_L, ng_L)
            m = int(fields_r_L.shape[0])
        else:
            raise ValueError("field_type must be 'scalar' or 'tensor'.")

        # assign to Eularian grid (and FFT/deconv)
        if mode == 'k_space':
            # Build non-interlaced and (optionally) interlaced stacks
            fields_r_E  = self._assign_fields_from_disp_to_grid(disp_r_L, fields_r_L,
                                                               interlace=False, normalize_mean=True,
                                                               field_type=field_type,
                                                               neighbor_mode=neighbor_mode,
                                                               fuse_updates_threshold=fuse_updates_threshold)
            fields_r_Ei  = None
            if self.mesh.interlace:
                fields_r_Ei = self._assign_fields_from_disp_to_grid(disp_r_L, fields_r_L,
                                                                    interlace=True, normalize_mean=True,
                                                                    field_type=field_type,
                                                                    neighbor_mode=neighbor_mode,
                                                                    fuse_updates_threshold=fuse_updates_threshold)
            # Single batched FFT + deconvolution (provided by Mesh_Assignment)
            fields_k = self.mesh.fft_deconvolve_batched(fields_r_E, fields_r_Ei)
            if field_type == 'tensor':
                # Reshape back to (m, 6, ng_E, ng_E, ng_E//2+1)
                fields_k = fields_k.reshape(m, 6, *fields_k.shape[1:])
            return fields_k.astype(self.complex_dtype)
        else:
            # Real-space: return non-interlaced assigned grids
            fields_r_E = self._assign_fields_from_disp_to_grid(disp_r_L, fields_r_L,
                                                               interlace=False, normalize_mean=True,
                                                               field_type=field_type,
                                                               neighbor_mode=neighbor_mode,
                                                               fuse_updates_threshold=fuse_updates_threshold)
            if field_type == 'tensor':
                fields_r_E = fields_r_E.reshape(m, 6, *fields_r_E.shape[1:])
            return fields_r_E.astype(self.real_dtype)

# ------------------------------ EPT ------------------------------

class EPT_Forward(Base_Forward, Beta_Combine_Mixin):
    """
    Compute Eulerian PT fields.
    """

    def __init__(
        self,
        *,
        boxsize: float,
        ng: Optional[int] = None,
        ng_L: int,
        ng_E: int,
        rsd: bool = False,
        pt_order: int = 1,
        bias_order: int = 2,
        dtype=jnp.float32,
        use_batched_fft: bool = True,
    ) -> None:
        super().__init__(boxsize=boxsize, ng=ng, ng_L=ng_L, dtype=dtype, use_batched_fft=use_batched_fft)

        # Eulerian grid 
        self.ng_E = int(ng_E)

        self.rsd = bool(rsd)
        self.pt_order = int(pt_order)
        self.bias_order = int(bias_order)

        self.kxE, self.kyE, self.kzE = coord.kaxes_1d(self.ng_E, self.boxsize, dtype=self.real_dtype)
        self.kx2E = self.kxE * self.kxE
        self.ky2E = self.kyE * self.kyE
        self.kz2E = self.kzE * self.kzE

        # Small cache for 'nearest' interpolation map
        self._nearest_idx = None       # int32 indices of shape (ng_E, ng_E, ng_E//2+1)
        self._k_edges_cache = None     # last-used k_edges
        self._mu_edges_cache = None    # last-used mu_edges

    # -------- scalar fields in Fourier space -------
    @partial(jit, static_argnames=('self',))
    def _scalar_fields_k(self, delta_k: jnp.ndarray) -> jnp.ndarray:
        r"""
        Build scalar fields in Fourier space: [delta, d^2, G2, \Psi_i \partial_i \delta]
        Returns stacked array in k space with leading field axis.
        """
        delta_k = delta_k.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        fields = []

        if self.pt_order >= 1:
            fields.append(delta_k)  # delta_k is the first field

        need_quad = (self.pt_order >= 2) or (self.bias_order >= 2)
        if need_quad:
            delta_r = self.irfftn(delta_k)

            # d^2 and its k-space
            d2_r = delta_r * delta_r
            d2_r = d2_r - jnp.mean(d2_r)
            d2_k = self.rfftn(d2_r).astype(self.complex_dtype)

            # Gij and G2
            Gij_k = coord.apply_Gij_k(delta_k, self.kx, self.ky, self.kz)   # (6, k)
            Gij_r = self._irfftn_vec(Gij_k)                                 # (6, r)
            # indices: 0:xx, 1:xy, 2:xz, 3:yy, 4:yz, 5:zz
            phi2 = (Gij_r[0]*Gij_r[3] + Gij_r[3]*Gij_r[5] + Gij_r[5]*Gij_r[0]
                         - Gij_r[1]**2 - Gij_r[2]**2 - Gij_r[4]**2)
            G2_r = (-2.0 * phi2).astype(self.real_dtype)
            G2_r = G2_r - jnp.mean(G2_r)
            G2_k = self.rfftn(G2_r).astype(self.complex_dtype)
            
        if self.pt_order == 2:
            # 4) \Psi \nabla \delta term (both as real 3-vectors)
            disp1_k = coord.apply_disp_k(delta_k, self.kx, self.ky, self.kz)   # (3, k),
            grad1_k = coord.apply_nabla_k(delta_k, self.kx, self.ky, self.kz)  # (3, k), +i k * \delta

            disp1_r = self._irfftn_vec(disp1_k)  # (3, r)
            grad1_r = self._irfftn_vec(grad1_k)  # (3, r)

            shift2_r = jnp.einsum('iabc,iabc->abc', disp1_r, grad1_r)
            shift2_r = shift2_r - jnp.mean(shift2_r)

            F2_r = (d2_r - shift2_r + (2.0/7.0) * G2_r).astype(self.real_dtype)
            F2_k = self.rfftn(F2_r).astype(self.complex_dtype)
            fields[0] = (fields[0] + F2_k)  # delta_k += F2_k
        elif self.pt_order > 2:
            delta_m_r_ = self.gridspt(delta_k, pt_order=self.pt_order, ng=int(2.*self.ng_L/3))
            delta_m_r = jnp.sum(delta_m_r_, axis=0)  # sum over pt_order
            delta_m_k = self.rfftn(delta_m_r)  # (ng_L, ng_L, ng_L//2+1)
            delta_m_k = delta_m_k.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_m_k[0] = 0.0
            fields[0] = delta_m_k
        
        if self.bias_order >= 2:
            # Append quadratic bias operators once (reuse computed results)
            fields.extend([d2_k, G2_k])

        return jnp.array(fields)
    
    # -------- tensor fields in Fourier space -------
    @partial(jit, static_argnames=('self',))
    def _tensor_fields_k(self, delta_k: jnp.ndarray) -> jnp.ndarray:
        r"""
        Build tensor fields in Fourier space: [K_ij, dK_ij, G_ij, T_ij, ...]
        Returns
        -------
        (n_fields, 6, ng_L, ng_L, ng_L//2+1) with order (xx,xy,xz,yy,yz,zz) in the 6-axis.
        """
        delta_k = delta_k.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        
        K_ij_k = coord.apply_Gij_k(delta_k, self.kx, self.ky, self.kz)  # (6, ng_L, ng_L, ng_L//2+1)
        K_ij_k = coord.apply_traceless(K_ij_k)

        fields = []

        if self.bias_order >= 1:
            fields.append(K_ij_k)

        if self.bias_order >= 2:
            delta_r = self.irfftn(delta_k)
            K_ij_r = self._irfftn_vec(K_ij_k)  # (6, ng_L, ng_L, ng_L)

            dK_ij_r = K_ij_r * delta_r[None, :, :, :]  # (6, ng_L, ng_L, ng_L)
            dK_ij_r = dK_ij_r - jnp.mean(dK_ij_r, axis=(1, 2, 3), keepdims=True)
            dK_ij_k = self._rfftn_vec(dK_ij_r)  # (6, ng_L, ng_L, ng_L//2+1)
            dK_ij_k = dK_ij_k.at[:,0,0,0].set(0.0).astype(self.complex_dtype)

            KK_xx_r = K_ij_r[0]*K_ij_r[0] + K_ij_r[1]*K_ij_r[1] + K_ij_r[2]*K_ij_r[2]
            KK_xy_r = K_ij_r[0]*K_ij_r[1] + K_ij_r[1]*K_ij_r[3] + K_ij_r[2]*K_ij_r[4]
            KK_xz_r = K_ij_r[0]*K_ij_r[2] + K_ij_r[1]*K_ij_r[4] + K_ij_r[2]*K_ij_r[5]
            KK_yy_r = K_ij_r[3]*K_ij_r[3] + K_ij_r[1]*K_ij_r[1] + K_ij_r[4]*K_ij_r[4]
            KK_yz_r = K_ij_r[3]*K_ij_r[4] + K_ij_r[1]*K_ij_r[2] + K_ij_r[4]*K_ij_r[5]
            KK_zz_r = K_ij_r[5]*K_ij_r[5] + K_ij_r[2]*K_ij_r[2] + K_ij_r[4]*K_ij_r[4]

            K2_r = K_ij_r[0]**2 + K_ij_r[3]**2 + K_ij_r[5]**2 + 2.0*(K_ij_r[1]**2 + K_ij_r[2]**2 + K_ij_r[4]**2)
            K2_over_3 = (K2_r / 3.0).astype(self.real_dtype)
            
            ### Traceless
            KK_xx_r = KK_xx_r - K2_over_3
            KK_yy_r = KK_yy_r - K2_over_3
            KK_zz_r = KK_zz_r - K2_over_3
            
            KK_ij_r = jnp.stack([KK_xx_r, KK_xy_r, KK_xz_r, KK_yy_r, KK_yz_r, KK_zz_r], axis=0)
            KK_ij_r = KK_ij_r - jnp.mean(KK_ij_r, axis=(1, 2, 3), keepdims=True)
            G_ij_r = KK_ij_r - (1./3.)*dK_ij_r
            G_ij_k = self._rfftn_vec(G_ij_r)  # (6, ng_L, ng_L, ng_L//2+1)
            G_ij_k = G_ij_k.at[:,0,0,0].set(0.0).astype(self.complex_dtype)

            T_r = (delta_r*delta_r - 1.5 * K2_r).astype(self.real_dtype)
            T_ij_k = coord.apply_Gij_k(self.rfftn(T_r), self.kx, self.ky, self.kz)  # (6, ng_L, ng_L, ng_L//2+1)
            T_ij_k = coord.apply_traceless(T_ij_k)
            T_ij_k = T_ij_k.at[:,0,0,0].set(0.0).astype(self.complex_dtype)

            fields.extend([dK_ij_k, G_ij_k, T_ij_k])

        return jnp.array(fields)
    
    def get_fields(
        self,
        delta_k: jnp.ndarray,
        *,
        mode: str = 'k_space',
        field_type: Literal['scalar', 'tensor'] = 'scalar',
    ) -> jnp.ndarray:
        r"""
        Build scalar or tensor real-space fields, resize them to Eulerian grid,

        Returns
        -------
        If field_type == 'scalar':
            - mode == 'k_space' : (m, ng_E, ng_E, ng_E//2+1), complex
            - else              : (m, ng_E, ng_E, ng_E),       real
        If field_type == 'tensor' (components in order xx,xy,xz,yy,yz,zz for each field):
            - mode == 'k_space' : (m, 6, ng_E, ng_E, ng_E//2+1), complex
            - else              : (m, 6, ng_E, ng_E, ng_E),       real
        """
        delta_k = delta_k.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        delta_k_L = coord.func_extend(self.ng_L, delta_k)

        if field_type == 'scalar':
            fields_k_L = self._scalar_fields_k(delta_k_L)             # (m, ng_L, ng_L, ng_L//2+1)
            m = int(fields_k_L.shape[0])

            # resize to Eulerian grid in k-space
            if self.ng_E < self.ng_L:
                reduce_to_E = jit(vmap(partial(coord.func_reduce, self.ng_E), in_axes=0, out_axes=0))
                fields_k_E = reduce_to_E(fields_k_L)                   # (m, ng_E, ng_E, ng_E//2+1)
            elif self.ng_E > self.ng_L:
                extend_to_E = jit(vmap(partial(coord.func_extend, self.ng_E), in_axes=0, out_axes=0))
                fields_k_E = extend_to_E(fields_k_L)                   # (m, ng_E, ng_E, ng_E//2+1)
            else:
                fields_k_E = fields_k_L                                # (m, ng_E, ng_E, ng_E//2+1)

            if mode == 'k_space':
                return fields_k_E.astype(self.complex_dtype)
            else:
                fields_r_E = self._irfftn_vec(fields_k_E)              # (m, ng_E, ng_E, ng_E)
                return fields_r_E.astype(self.real_dtype)
            
        elif field_type == 'tensor':
            fields_k_L = self._tensor_fields_k(delta_k_L)              # (m, 6, ng_L, ng_L, ng_L//2+1)
            m = int(fields_k_L.shape[0])

            # resize to Eulerian grid in k-space (vmap over 6 then over m)
            if self.ng_E < self.ng_L:
                reduce_6 = vmap(partial(coord.func_reduce, self.ng_E), in_axes=0, out_axes=0)   # over 6
                reduce_to_E = jit(vmap(reduce_6, in_axes=0, out_axes=0))                        # over m
                fields_k_E = reduce_to_E(fields_k_L)                    # (m, 6, ng_E, ng_E, ng_E//2+1)
            elif self.ng_E > self.ng_L:
                extend_6 = vmap(partial(coord.func_extend, self.ng_E), in_axes=0, out_axes=0)   # over 6
                extend_to_E = jit(vmap(extend_6, in_axes=0, out_axes=0))                        # over m
                fields_k_E = extend_to_E(fields_k_L)                    # (m, 6, ng_E, ng_E, ng_E//2+1)
            else:
                fields_k_E = fields_k_L                                 # (m, 6, ng_E, ng_E, ng_E//2+1)

            if mode == 'k_space':
                return fields_k_E.astype(self.complex_dtype)
            else:
                # flatten (m,6,...) -> (m*6, ...) for batched IRFFT, then reshape back
                flat_k = fields_k_E.reshape(m * 6, self.ng_E, self.ng_E, self.ng_E // 2 + 1)
                flat_r = self._irfftn_vec(flat_k)                       # (m*6, ng_E, ng_E, ng_E)
                fields_r_E = flat_r.reshape(m, 6, self.ng_E, self.ng_E, self.ng_E)
                return fields_r_E.astype(self.real_dtype)

        else:
            raise ValueError("field_type must be 'scalar' or 'tensor'.")

    def gridspt(self, delta_k_L: jnp.ndarray, pt_order: Optional[int] = None, ng: Optional[int] = None) -> jnp.ndarray:
        r"""
        Eulerian SPT on the large grid (ng_L) with de-aliasing controlled by (ng, ng_L).

        Strategy:
          - Low-pass the linear field once so it matches the target passband implied by ng.
          - Build higher-order fields; after forming each order, low-pass again before storing.

        Inputs
        ------
        delta_k_L : (ng_L, ng_L, ng_L//2+1) complex
            Linear density on the large rfftn layout. If not on ng_L, it is promoted/reduced.
        pt_order : int or None
            Maximum order (defaults to self.pt_order).
        ng : int or None
            Passband controller. If None and self.ng is set, use self.ng.
            If neither is set, default to floor(2/3 * ng_L) and warn.

        Returns
        -------
        deltas_r : (pt_order, ng_L, ng_L, ng_L) real
            Real-space fields [delta_1, delta_2, ..., delta_pt] on the large grid.
        """

        if pt_order is None:
            pt_order = int(self.pt_order)

        # Resolve passband ng (controls the kept region in Fourier space).
        if ng is not None:
            ng_eff = int(ng)
        elif getattr(self, "ng", None) is not None:
            ng_eff = int(self.ng)
        else:
            ng_eff = int((2 * self.ng_L) // 3)
            warnings.warn(
                f"gridspt: 'ng' not provided. Defaulting to floor(2/3 * ng_L) = {ng_eff}.",
                RuntimeWarning,
            )

        if ng_eff <= 0 or ng_eff > self.ng_L:
            raise ValueError(f"gridspt: require 0 < ng <= ng_L, got ng={ng_eff}, ng_L={self.ng_L}.")

        # Bring input to ng_L if needed.
        if delta_k_L.shape[0] == self.ng_L:
            delta_k = jnp.asarray(delta_k_L, dtype=self.complex_dtype)
        elif delta_k_L.shape[0] < self.ng_L:
            delta_k = coord.func_extend(self.ng_L, jnp.asarray(delta_k_L, dtype=self.complex_dtype))
        else:
            delta_k = coord.func_reduce(self.ng_L, jnp.asarray(delta_k_L, dtype=self.complex_dtype))

        # Zero DC mode.
        delta_k = delta_k.at[0, 0, 0].set(0.0)

        # Build a cubic passband mask that matches "reduce to ng_eff".
        ngL = int(self.ng_L)
        half = int(ng_eff) // 2
        idx_xy = jnp.arange(ngL)
        mx = (idx_xy < half) | (idx_xy >= (ngL - half))          # (ng_L,)
        mz = jnp.arange(ngL // 2 + 1) <= half                    # (ng_L//2+1,)
        lpf_mask = (mx[:, None, None] & mx[None, :, None] & mz[None, None, :]).astype(self.real_dtype)
        lpf_mask_c = lpf_mask.astype(self.complex_dtype)

        # Initial low-pass once for the linear field.
        del1_k = (delta_k * lpf_mask_c).at[0, 0, 0].set(0.0)
        the1_k = del1_k  # EdS closure

        delk = [del1_k]                   # list of k-space fields per order
        thek = [the1_k]
        deltas_r = [self.irfftn(del1_k)]  # real-space outputs per order

        # Main recursion for orders >= 2.
        for n in range(2, pt_order + 1):
            Sd_r = jnp.zeros((ngL, ngL, ngL), dtype=self.real_dtype)
            St_r = jnp.zeros_like(Sd_r)

            half_n = n // 2
            for m in range(1, half_n + 1):
                nm = n - m
                Sd_mn, St_mn = self._spt_pair_contrib_bandlimited(
                    delk[m - 1], thek[m - 1],
                    delk[nm - 1], thek[nm - 1],
                )
                factor = 2.0 if (m < nm) else 1.0
                Sd_r = Sd_r + factor * Sd_mn
                St_r = St_r + factor * St_mn

            # Closed-form coefficients (Einstein-de Sitter).
            coef = 2.0 / ((2 * n + 3) * (n - 1))
            delta_n_r = coef * ((n + 0.5) * Sd_r + St_r)
            theta_n_r = coef * (1.5 * Sd_r + n * St_r)

            # Post low-pass so the next iteration sees band-limited inputs.
            delta_n_k = (self.rfftn(delta_n_r) * lpf_mask_c).at[0, 0, 0].set(0.0)
            theta_n_k = (self.rfftn(theta_n_r) * lpf_mask_c).at[0, 0, 0].set(0.0)

            delk.append(delta_n_k)
            thek.append(theta_n_k)
            deltas_r.append(self.irfftn(delta_n_k))

        return jnp.stack(deltas_r, axis=0).astype(self.real_dtype)

    @partial(jit, static_argnames=('self',))
    def _spt_pair_contrib_bandlimited(
        self,
        del_k_m: jnp.ndarray,   # k-space, already band-limited on ng_L
        the_k_m: jnp.ndarray,   # k-space, already band-limited
        del_k_nm: jnp.ndarray,  # k-space, already band-limited
        the_k_nm: jnp.ndarray,  # k-space, already band-limited
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        r"""
        Pairwise contribution on the large grid assuming all inputs are already band-limited.
        This lets us skip per-pair pre-filtering for better performance.

        S_delta = dot(grad(delta_m), u_{n-m}) + delta_m * theta_{n-m}
        S_theta = trace(du_m du_{n-m}) + dot(u_m, grad(theta_{n-m}))
        """
        # Vector operators in k-space.
        grad_d_m_k  = coord.apply_nabla_k(del_k_m,  self.kx, self.ky, self.kz)     # (3, k)
        u_nm_k      = -coord.apply_disp_k(the_k_nm, self.kx, self.ky, self.kz)     # (3, k)

        u_m_k       = -coord.apply_disp_k(the_k_m,  self.kx, self.ky, self.kz)     # (3, k)
        grad_t_nm_k = coord.apply_nabla_k(the_k_nm, self.kx, self.ky, self.kz)     # (3, k)

        # Batch IRFFT for vectors and scalars in one go.
        pack1_k = jnp.concatenate([
            grad_d_m_k,               # 0..2
            u_nm_k,                   # 3..5
            u_m_k,                    # 6..8
            grad_t_nm_k,              # 9..11
            del_k_m[None, ...],       # 12
            the_k_nm[None, ...],      # 13
        ], axis=0)  # (14, ...)

        pack1_r = self._irfftn_vec(pack1_k).astype(self.real_dtype)
        grad_d_m_r  = pack1_r[0:3]
        u_nm_r      = pack1_r[3:6]
        u_m_r       = pack1_r[6:9]
        grad_t_nm_r = pack1_r[9:12]
        d_m_r       = pack1_r[12]
        t_nm_r      = pack1_r[13]

        # S_delta
        S_delta = jnp.einsum('iabc,iabc->abc', grad_d_m_r, u_nm_r) + d_m_r * t_nm_r

        # Tensor operators: diagonal (xx,yy,zz) and off-diagonal (xy,xz,yz).
        Gm = coord.apply_Gij_k(the_k_m,  self.kx, self.ky, self.kz)  # (6, k)
        Gn = coord.apply_Gij_k(the_k_nm, self.kx, self.ky, self.kz)  # (6, k)

        # Diagonal components (0,3,5)
        du_m_diag_r  = self._irfftn_vec(-Gm[[0, 3, 5]]).astype(self.real_dtype)
        du_nm_diag_r = self._irfftn_vec(-Gn[[0, 3, 5]]).astype(self.real_dtype)
        trace_diag = (du_m_diag_r[0] * du_nm_diag_r[0] +
                      du_m_diag_r[1] * du_nm_diag_r[1] +
                      du_m_diag_r[2] * du_nm_diag_r[2])

        # Off-diagonal components (1,2,4) with factor 2
        du_m_off_r  = self._irfftn_vec(-Gm[[1, 2, 4]]).astype(self.real_dtype)
        du_nm_off_r = self._irfftn_vec(-Gn[[1, 2, 4]]).astype(self.real_dtype)
        trace_off = 2.0 * (du_m_off_r[0] * du_nm_off_r[0] +
                           du_m_off_r[1] * du_nm_off_r[1] +
                           du_m_off_r[2] * du_nm_off_r[2])

        S_theta = trace_diag + trace_off + jnp.einsum('iabc,iabc->abc', u_m_r, grad_t_nm_r)

        # Remove means for numerical hygiene.
        S_delta = (S_delta - jnp.mean(S_delta)).astype(self.real_dtype)
        S_theta = (S_theta - jnp.mean(S_theta)).astype(self.real_dtype)
        return S_delta, S_theta
