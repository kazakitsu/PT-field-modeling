#!/usr/bin/env python3
from __future__ import annotations
from typing import Tuple, Literal, Optional
import warnings

import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial

import PT_field.coord_jax as coord
import PT_field.utils_jax as utils_jax
import lss_utils.assign_util_jax as assign_util

# ------------------------------ Base class ------------------------------

class Base_Forward:
    r"""
    FFT helpers & k-axis; 
    Generate the linear field given the linear power spectrum;
    model-specific subclasses add physics.
    """

    def __init__(self, *, boxsize: float,
                 ng: int,
                 ng_pad: Optional[int] = None, 
                 ng_L: Optional[int] = None,
                 dtype=jnp.float32,
                 use_batched_fft: bool = True):
        self.boxsize: float = float(boxsize)
        self.ng:      int = int(ng)
        self.ng_pad:  int = int(ng_pad) if ng_pad is not None else self._get_ng_pad(self.ng)
        self.ng_L:    Optional[int] = int(ng_L) if ng_L is not None else int(ng)
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
        self.kx,  self.ky,  self.kz  = coord.kaxes_1d(self.ng, self.boxsize, dtype=self.real_dtype)
        #self.kx_L,  self.ky_L,  self.kz_L  = coord.kaxes_1d(self.ng_L, self.boxsize, dtype=self.real_dtype)

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
        
    def _get_ng_pad(self, ng:int, pad_factor:float=1.5) -> int:
        """Return working grid size for dealiased products."""
        ng_pad = int(jnp.ceil(ng * pad_factor))
        if ng_pad % 2 == 1:
            ng_pad += 1
        return ng_pad
    
    @partial(jit, static_argnames=('self',))
    def pad_fields(
            self,
            fields_k: jnp.ndarray,   # (n_fields, ng, ng, ng//2+1)
        ) -> jnp.ndarray:
        """Zero-pad a batch of rfftn-layout fields along axis=0"""
        fields_k = jnp.asarray(fields_k, dtype=self.complex_dtype)
        num_fields, ng, _, _ = fields_k.shape

        if ng == self.ng_pad:
            return fields_k
        
        out = jnp.zeros((num_fields, self.ng_pad, self.ng_pad, self.ng_pad//2+1), dtype=self.complex_dtype)

        def body(i, acc):
            slab = fields_k[i]
            padded = coord.func_extend(self.ng_pad, slab)
            return acc.at[i].set(padded)
        
        return lax.fori_loop(0, num_fields, body, out)
    
    @partial(jit, static_argnames=('self',))
    def unpad_fields(
            self,
            fields_k: jnp.ndarray,   # (n_fields, ng, ng, ng//2+1)
        ) -> jnp.ndarray:
        """unpad a batch of rfftn-layout fields along axis=0"""
        fields_k = jnp.asarray(fields_k, dtype=self.complex_dtype)
        num_fields, ng, _, _ = fields_k.shape

        if ng == self.ng:
            return fields_k
        
        out = jnp.zeros((num_fields, self.ng, self.ng, self.ng//2+1), dtype=self.complex_dtype)

        def body(i, acc):
            slab = fields_k[i]
            padded = coord.func_reduce_hermite(self.ng, slab)
            return acc.at[i].set(padded)
        
        return lax.fori_loop(0, num_fields, body, out)
    
    @partial(jit, static_argnames=("self",))
    def _safe_square(
        self,
        f_k: jnp.ndarray,  # (ng, ng, ng//2+1)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute f^2 with de-aliasing and return:
          - f2_r  : real-space field on the physical grid (ng^3)
          - f2_k : k-space field on the physical grid (ng^3)
        """
        f_k = f_k.astype(self.complex_dtype)

        # pad to ng_pad, go to real, square
        f_k_pad = self.pad_fields(f_k[None, ...])[0]
        f_r_pad = self.irfftn(f_k_pad)
        f2_r_pad = f_r_pad * f_r_pad

        # back to k, low-pass to ng
        f2_k_pad = self.rfftn(f2_r_pad)
        f2_k = self.unpad_fields(f2_k_pad[None, ...])[0]
        f2_r = self.irfftn(f2_k)

        return f2_r, f2_k
    
    @partial(jit, static_argnames=("self",))
    def _safe_product(
        self,
        f1_k: jnp.ndarray,  # (ng, ng, ng//2+1)
        f2_k: jnp.ndarray,  # (ng, ng, ng//2+1)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute f1 * f2 with de-aliasing and return:
          - f1_f2_r    : real-space field on the physical grid (ng^3)
          - f1_f2_k: k-space field on the physical grid (ng^3)
        """
        f1_k = f1_k.astype(self.complex_dtype)
        f2_k = f2_k.astype(self.complex_dtype)

        # pad both to ng_pad
        f_k_arr = jnp.stack([
            f1_k,
            f2_k,
        ], axis=0)  # (2, ng, ng, ng//2+1)
        f_k_arr_pad = self.pad_fields(f_k_arr)
        f_r_arr_pad = self._irfftn_vec(f_k_arr_pad)  # (2, ng_pad, ng_pad, ng_pad)
        f1_r_pad, f2_r_pad = f_r_arr_pad[0], f_r_arr_pad[1]

        f1_f2_r_pad = f1_r_pad * f2_r_pad

        # back to k, low-pass to ng
        f1_f2_k_pad = self.rfftn(f1_f2_r_pad)
        f1_f2_k = self.unpad_fields(f1_f2_k_pad[None, ...])[0]
        f1_f2_r = self.irfftn(f1_f2_k)

        return f1_f2_r, f1_f2_k
    
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
        for name in ("kx", "ky", "kz", "_kxy2", "_kz2"):
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
            kx, ky, kz = coord.kaxes_1d(self.ng, self.boxsize, dtype=self.real_dtype)
            self.kx, self.ky, self.kz = kx, ky, kz

    def _ensure_k_caches(self) -> None:
        r"""Build grid k^2 caches once; safe to call many times."""
        self._ensure_kaxes()
        if not hasattr(self, "_kxy2"):
            kx2 = (self.kx ** 2).astype(self.real_dtype)
            ky2 = (self.ky ** 2).astype(self.real_dtype)
            kz2 = (self.kz ** 2).astype(self.real_dtype)
            self._kxy2 = (kx2[:, None] + ky2[None, :]).astype(self.real_dtype)  # (ng, ng)
            self._kz2  = kz2  # (ng//2+1,)

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
            kx2 = (self.kx ** 2).astype(self.real_dtype)[:, None, None]
            ky2 = (self.ky ** 2).astype(self.real_dtype)[None, :, None]
            kz2 = (self.kz ** 2).astype(self.real_dtype)[None, None, :]

            kmag = jnp.sqrt(kx2 + ky2 + kz2).astype(self.real_dtype)
            Pk   = jnp.interp(kmag, k_tab, P_tab, left=0.0, right=0.0)
            Pk   = Pk.at[0, 0, 0].set(0.0)  # zero DC

            amp  = jnp.sqrt(Pk * inv2V).astype(self.real_dtype)
            return (amp * gauss_3d).astype(self.complex_dtype)
        else:
            # ---- z-sliced path ----
            out = jnp.zeros_like(gauss_3d, dtype=self.complex_dtype)

            def body(iz, acc):
                kmag_2d = jnp.sqrt(self._kxy2 + self._kz2[iz]).astype(self.real_dtype)
                P_2d    = jnp.interp(kmag_2d, k_tab, P_tab, left=0.0, right=0.0)
                P_2d    = jnp.where((self._kz2[iz] == 0.0) & (self._kxy2 == 0.0), 0.0, P_2d)
                amp_2d  = jnp.sqrt(P_2d * inv2V).astype(self.real_dtype)
                slab    = (amp_2d * gauss_3d[:, :, iz]).astype(self.complex_dtype)
                return acc.at[:, :, iz].set(slab)

            out = lax.fori_loop(0, self.kz.shape[0], body, out)
            return out
        

class Beta_Combine_Mixin:
    r"""
    Shared helpers to combine basis fields in k-space and compute beta.
    Requires the subclass to define:
      - self.ng_E
      - self.kx2_E, self.ky2_E, self.kz2_E
    """

    def _kmu_from_cache_or_build(self):
        r"""
        Returns:
          k2, mu2, mu4, kmag, mu (shape = (ng_E, ng_E, ng_E//2+1))
        """
        if hasattr(self, "_k2_E_grid"):
            k2  = self._k2E_grid
            mu2 = self._mu2E_grid
            mu4 = self._mu4E_grid
        else:
            k2  = (self.kx2_E[:, None, None] + self.ky2_E[None, :, None] + self.kz2_E[None, None, :]).astype(self.real_dtype)
            mu2 = jnp.where(k2 > 0, self.kz2_E[None, None, :] / k2, 0.0).astype(self.real_dtype)
            mu4 = (mu2 * mu2).astype(self.real_dtype)

        kmag = jnp.sqrt(jnp.maximum(k2, 0.0)).astype(self.real_dtype)
        mu   = jnp.sqrt(mu2).astype(self.real_dtype)
        return k2, mu2, mu4, kmag, mu

    @partial(jit, static_argnames=('self', 'measure_pk'))
    def get_beta(
        self,
        delta_g_k: jnp.ndarray,     # (..., ng, ng, ng//2+1), complex
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
                out_c  = measure_pk(delta_g_k, fields_k[i], ell=0,
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
    
    @partial(jit, static_argnames=('self', 'measure_pk'))
    def get_beta_lstsq(
        self,
        delta_g_k: jnp.ndarray,        # (..., ng, ng, ng//2+1), complex
        fields_k: jnp.ndarray,         # (n, ng, ng, ng//2+1), complex
        mu_edges: jnp.ndarray,         # (Nmu+1,)
        *,
        measure_pk,                    # Measure_Pk instance (static)
        eps: float = 0.0,              # ridge term added to Gram diagonal per (k,mu)
    ) -> jnp.ndarray:
        """
        Solve per-(k,mu) least squares:
          min_beta sum_{modes in bin} |delta_g - sum_i beta_i O_i|^2

        Normal equations per bin:
          G_ij = <O_i O_j>_{bin}
          b_i  = <delta_g O_i>_{bin}
          beta = (G + eps * I)^{-1} b

        Returns
        -------
        beta : (n_fields, Nk, Nmu)
        Notes
        -----
        - Uses fori_loop over mu-bins to keep compile size moderate.
        - Within each mu-bin:
            * b is computed by lax.map over fields
            * G is assembled with diagonal autos + upper-triangle crosses
            * solve is vmap over k-bins
        """
        n_fields = int(fields_k.shape[0])
        dtype = measure_pk.dtype
        Nk = int(jnp.asarray(measure_pk.k_mean, dtype=dtype).shape[0])
        Nmu = int(mu_edges.shape[0] - 1)

        eye = jnp.eye(n_fields, dtype=dtype)
        lam = jnp.asarray(eps, dtype=dtype)

        def mu_body(m, beta_acc):
            mu_min = mu_edges[m]
            mu_max = mu_edges[m + 1]

            # ---- b_i(k) = <delta_g O_i> for all i ----
            def one_b(i):
                out_c = measure_pk(
                    delta_g_k, fields_k[i],
                    ell=0, mu_min=mu_min, mu_max=mu_max
                )
                return out_c[:, 1].astype(dtype)  # (Nk,)

            b_m = lax.map(one_b, jnp.arange(n_fields, dtype=jnp.int32))  # (n, Nk)

            # ---- G_ij(k) = <O_i O_j> for all i,j ----
            # Store as (n, n, Nk) for convenient scatter updates, then moveaxis to (Nk, n, n).
            G_m = jnp.zeros((n_fields, n_fields, Nk), dtype=dtype)

            # Diagonal autos
            def one_auto(i):
                out_a = measure_pk(
                    fields_k[i],
                    ell=0, mu_min=mu_min, mu_max=mu_max
                )
                return out_a[:, 1].astype(dtype)  # (Nk,)

            auto_stack = lax.map(one_auto, jnp.arange(n_fields, dtype=jnp.int32))  # (n, Nk)
            idx = jnp.arange(n_fields)
            G_m = G_m.at[idx, idx, :].set(auto_stack)

            # Upper triangle crosses (symmetric fill)
            for j in range(1, n_fields):
                def one_cross(i):
                    out_x = measure_pk(
                        fields_k[i], fields_k[j],
                        ell=0, mu_min=mu_min, mu_max=mu_max
                    )
                    return out_x[:, 1].astype(dtype)  # (Nk,)

                cross_stack = lax.map(one_cross, jnp.arange(j, dtype=jnp.int32))  # (j, Nk)
                G_m = G_m.at[:j, j, :].set(cross_stack)
                G_m = G_m.at[j, :j, :].set(cross_stack)

            # ---- Solve per k: (G + lam I) beta = b ----
            Gk = jnp.moveaxis(G_m, -1, 0)  # (Nk, n, n)
            bk = jnp.swapaxes(b_m, 0, 1)   # (Nk, n)

            Gk = Gk + lam * eye[None, :, :]

            def solve_one(A, y):
                return jnp.linalg.solve(A, y)

            beta_k = vmap(solve_one, in_axes=(0, 0))(Gk, bk)  # (Nk, n)
            beta_m = jnp.swapaxes(beta_k, 0, 1)              # (n, Nk)

            return beta_acc.at[:, :, m].set(beta_m)

        beta0 = jnp.zeros((n_fields, Nk, Nmu), dtype=dtype)
        beta = lax.fori_loop(0, Nmu, mu_body, beta0)
        return beta
    
    @partial(jit, static_argnames=('self', 'measure_pk'))
    def _get_Rt_from_bases(
        self,
        fields_k: jnp.ndarray,          # (n, ng, ng, ng//2+1), complex
        fields_ortho_k: jnp.ndarray,    # (n, ng, ng, ng//2+1), complex
        mu_edges: jnp.ndarray,          # (Nmu+1,)
        *,
        measure_pk,                     # Measure_Pk instance (static)
        eps: float = 0.0,               # ridge on G diagonal
    ) -> jnp.ndarray:
        """
        Compute Rt(k,mu) = R(k,mu)^T from two bases O and O_ortho.

        Definitions per (k,mu) bin:
          G = <O O^T>
          H = <O_ortho O^T>
          H = R G  ->  R^T = (G^T)^{-1} H^T

        Returns
        -------
        Tt : (Nk, Nmu, n, n)
        """
        n_fields = int(fields_k.shape[0])
        dtype = measure_pk.dtype
        Nk = int(jnp.asarray(measure_pk.k_mean, dtype=dtype).shape[0])
        Nmu = int(mu_edges.shape[0] - 1)

        eye = jnp.eye(n_fields, dtype=dtype)
        lam = jnp.asarray(eps, dtype=dtype)

        def mu_body(m, Tt_acc):
            mu_min = mu_edges[m]
            mu_max = mu_edges[m + 1]

            # ---- G_ij(k) = <O_i O_j> ----
            # Store as (n, n, Nk) then moveaxis to (Nk, n, n)
            G_m = jnp.zeros((n_fields, n_fields, Nk), dtype=dtype)

            def one_auto(i):
                out_a = measure_pk(
                    fields_k[i], None,
                    ell=0, mu_min=mu_min, mu_max=mu_max
                )
                return out_a[:, 1].astype(dtype)  # (Nk,)

            auto_stack = lax.map(one_auto, jnp.arange(n_fields, dtype=jnp.int32))  # (n, Nk)
            idx = jnp.arange(n_fields)
            G_m = G_m.at[idx, idx, :].set(auto_stack)

            for j in range(1, n_fields):
                def one_cross(i):
                    out_x = measure_pk(
                        fields_k[i], fields_k[j],
                        ell=0, mu_min=mu_min, mu_max=mu_max
                    )
                    return out_x[:, 1].astype(dtype)  # (Nk,)

                cross_stack = lax.map(one_cross, jnp.arange(j, dtype=jnp.int32))  # (j, Nk)
                G_m = G_m.at[:j, j, :].set(cross_stack)
                G_m = G_m.at[j, :j, :].set(cross_stack)

            # ---- H_ij(k) = <O_ortho_i O_j> ----
            H_m = jnp.zeros((n_fields, n_fields, Nk), dtype=dtype)

            for j in range(n_fields):
                def one_col(i):
                    out_h = measure_pk(
                        fields_ortho_k[i], fields_k[j],
                        ell=0, mu_min=mu_min, mu_max=mu_max
                    )
                    return out_h[:, 1].astype(dtype)  # (Nk,)

                col_stack = lax.map(one_col, jnp.arange(n_fields, dtype=jnp.int32))  # (n, Nk)
                H_m = H_m.at[:, j, :].set(col_stack)

            # ---- Tt(k) = solve(G(k)^T + lam I, H(k)^T) for each k ----
            Gk = jnp.moveaxis(G_m, -1, 0)  # (Nk, n, n)
            Hk = jnp.moveaxis(H_m, -1, 0)  # (Nk, n, n)

            A = jnp.swapaxes(Gk, -1, -2) + lam * eye[None, :, :]  # (Nk, n, n)
            B = jnp.swapaxes(Hk, -1, -2)                          # (Nk, n, n)

            def solve_mat(Ak, Bk):
                return jnp.linalg.solve(Ak, Bk)  # (n, n)

            Rt_k = vmap(solve_mat, in_axes=(0, 0))(A, B)  # (Nk, n, n)

            return Tt_acc.at[:, m, :, :].set(Rt_k)

        Rt0 = jnp.zeros((Nk, Nmu, n_fields, n_fields), dtype=dtype)
        Rt = lax.fori_loop(0, Nmu, mu_body, Rt0)
        return Rt

    @partial(jit, static_argnames=('self', 'measure_pk'))
    def beta_from_beta_ortho(
        self,
        beta_ortho: jnp.ndarray,        # (n, Nk, Nmu)
        fields_k: jnp.ndarray,          # (n, ng, ng, ng//2+1)
        fields_ortho_k: jnp.ndarray,    # (n, ng, ng, ng//2+1)
        mu_edges: jnp.ndarray,          # (Nmu+1,)
        *,
        measure_pk,
        eps: float = 0.0,
    ) -> jnp.ndarray:
        """
        Convert beta_ortho -> beta using both bases.

          beta = R^T beta_ortho
          R^T = solve(G^T, H^T),  G=<O O^T>, H=<O_ortho O^T>
        """
        Rt = self._get_Rt_from_bases(
            fields_k=fields_k,
            fields_ortho_k=fields_ortho_k,
            mu_edges=mu_edges,
            measure_pk=measure_pk,
            eps=eps,
        )  # (Nk, Nmu, n, n)

        # beta(i,k,mu) = sum_j Tt(k,mu,i,j) * beta_ortho(j,k,mu)
        beta = jnp.einsum("kmij,jkm->ikm", Rt, beta_ortho)
        return beta

    @partial(jit, static_argnames=('self', 'measure_pk'))
    def beta_ortho_from_beta(
        self,
        beta: jnp.ndarray,              # (n, Nk, Nmu)
        fields_k: jnp.ndarray,          # (n, ng, ng, ng//2+1)
        fields_ortho_k: jnp.ndarray,    # (n, ng, ng, ng//2+1)
        mu_edges: jnp.ndarray,          # (Nmu+1,)
        *,
        measure_pk,
        eps: float = 0.0,
    ) -> jnp.ndarray:
        """
        Convert beta -> beta_ortho using both bases.

          beta = R^T beta_ortho  ->  beta_ortho = solve(R^T, beta)
          R^T = solve(G^T, H^T),  G=<O O^T>, H=<O_ortho O^T>
        """
        Rt = self._get_Rt_from_bases(
            fields_k=fields_k,
            fields_ortho_k=fields_ortho_k,
            mu_edges=mu_edges,
            measure_pk=measure_pk,
            eps=eps,
        )  # (Nk, Nmu, n, n)

        n = int(beta.shape[0])
        Nk = int(beta.shape[1])
        Nmu = int(beta.shape[2])

        Rtf = Rt.reshape(Nk * Nmu, n, n)
        bf  = jnp.moveaxis(beta, 0, -1).reshape(Nk * Nmu, n)  # (Nk*Nmu, n)

        def solve_vec(A, y):
            return jnp.linalg.solve(A, y)  # (n,)

        x_f = vmap(solve_vec, in_axes=(0, 0))(Rtf, bf)  # (Nk*Nmu, n)
        x = x_f.reshape(Nk, Nmu, n)                     # (Nk, Nmu, n)
        beta_ortho = jnp.moveaxis(x, -1, 0)             # (n, Nk, Nmu)
        return beta_ortho


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
        k2 = (self.kx2_E[:, None, None] + self.ky2_E[None, :, None] + self.kz2_E[None, None, :])

        # Bin by k using k^2; edges are squared once (monotone -> binning is preserved).
        k_edges2 = (k_edges * k_edges).astype(self.real_dtype)
        Nk = int(k_edges.shape[0] - 1)
        kbin = jnp.clip(jnp.searchsorted(k_edges2, k2, side="right") - 1, 0, Nk - 1)

        # Bin by mu using mu^2 to avoid sqrt; again edges are squared once.
        mu2 = jnp.where(k2 > 0, self.kz2_E[None, None, :] / k2, 0.0).astype(self.real_dtype)
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
        ng: int,
        ng_pad: Optional[int] = None,
        ng_L: int,
        ng_E: int,
        mas_cfg: Tuple[int, bool],
        rsd: bool = False,
        lya: bool = False,
        lya_full_fields: bool = False,
        lpt_order: int = 1,
        bias_order: int = 2,
        renormalize: bool = True,
        dtype=jnp.float32,
        max_scatter_indices: int = 200_000_000,
        use_batched_fft: bool = True,
        # assignment switching knobs
        assign_mode: Literal["auto", "for", "vmap"] = "auto",
        vmap_ng_threshold: int = 256,
        vmap_fields_threshold: int = 4,
    ) -> None:
        super().__init__(boxsize=boxsize, ng=ng, ng_pad=ng_pad, ng_L=ng_L, dtype=dtype, use_batched_fft=use_batched_fft)

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
        self.lya_full_fields = bool(lya_full_fields)
        self.lpt_order = int(lpt_order)
        self.bias_order = int(bias_order)
        self.renormalize = bool(renormalize)

        # Assignment switching policy
        self.assign_mode: Literal["auto", "for", "vmap"] = assign_mode
        self.vmap_ng_threshold: int = int(vmap_ng_threshold)
        self.vmap_fields_threshold: int = int(vmap_fields_threshold)

        # Do not store 3D base positions; build on-the-fly
        self.cell_size = self.boxsize / self.ng_L

        self.kx_L, self.ky_L, self.kz_L = coord.kaxes_1d(self.ng_L, self.boxsize, dtype=self.real_dtype)
        self.kx_pad, self.ky_pad, self.kz_pad = coord.kaxes_1d(self.ng_pad, self.boxsize, dtype=self.real_dtype)

        self.kx_E, self.ky_E, self.kz_E = coord.kaxes_1d(self.ng_E, self.boxsize, dtype=self.real_dtype)
        self.kx2_E = self.kx_E * self.kx_E
        self.ky2_E = self.ky_E * self.ky_E
        self.kz2_E = self.kz_E * self.kz_E

        # Small cache for 'nearest' interpolation map
        self._nearest_idx = None       # int32 indices of shape (ng_E, ng_E, ng_E//2+1)
        self._k_edges_cache = None     # last-used k_edges
        self._mu_edges_cache = None    # last-used mu_edges

    @partial(jit, static_argnames=('self',))
    def pad_fields_L(
            self,
            fields_k: jnp.ndarray,   # (n_fields, ng, ng, ng//2+1)
        ) -> jnp.ndarray:
        """Zero-pad a batch of rfftn-layout fields along axis=0"""
        fields_k = jnp.asarray(fields_k, dtype=self.complex_dtype)
        num_fields, ng, _, _ = fields_k.shape

        if ng == self.ng_L:
            return fields_k
        
        out = jnp.zeros((num_fields, self.ng_L, self.ng_L, self.ng_L//2+1), dtype=self.complex_dtype)

        def body(i, acc):
            slab = fields_k[i]
            padded = coord.func_extend(self.ng_L, slab)
            return acc.at[i].set(padded)
        
        return lax.fori_loop(0, num_fields, body, out)
    
    @partial(jit, static_argnames=('self',))
    def unpad_fields_L(
            self,
            fields_k: jnp.ndarray,   # (n_fields, ng, ng, ng//2+1)
        ) -> jnp.ndarray:
        """unpad a batch of rfftn-layout fields along axis=0"""
        fields_k = jnp.asarray(fields_k, dtype=self.complex_dtype)
        num_fields, ng, _, _ = fields_k.shape

        if ng == self.ng_L:
            return fields_k
        
        out = jnp.zeros((num_fields, self.ng_L, self.ng_L, self.ng_L//2+1), dtype=self.complex_dtype)

        def body(i, acc):
            slab = fields_k[i]
            padded = coord.func_reduce_hermite(self.ng_L, slab)
            return acc.at[i].set(padded)
        
        return lax.fori_loop(0, num_fields, body, out)

    @partial(jit, static_argnames=("self",))
    def _safe_square_L(
        self,
        f_k: jnp.ndarray,  # (ng, ng, ng//2+1)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute f^2 with de-aliasing and return:
          - f2_r_L  : real-space field on the Lagrangian grid (ng_L^3)
          - f2_k : k-space field on the physical grid (ng^3)
        """
        f_k = f_k.astype(self.complex_dtype)

        # pad to ng_pad, go to real, square
        f_k_pad = self.pad_fields(f_k[None, ...])[0]
        f_r_pad = self.irfftn(f_k_pad)
        f2_r_pad = f_r_pad * f_r_pad

        # back to k, low-pass to ng
        f2_k_pad = self.rfftn(f2_r_pad)
        f2_k = self.unpad_fields(f2_k_pad[None, ...])[0]

        # extend to ng_L and go back to real
        f2_k_L = self.pad_fields_L(f2_k[None, ...])[0]
        f2_r_L = self.irfftn(f2_k_L)

        return f2_r_L, f2_k
    
    @partial(jit, static_argnames=("self",))
    def _safe_product_L(
        self,
        f1_k: jnp.ndarray,  # (ng, ng, ng//2+1)
        f2_k: jnp.ndarray,  # (ng, ng, ng//2+1)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute f1 * f2 with de-aliasing and return:
          - f1_f2_r_L    : real-space field on the Lagrangian grid (ng_L^3)
          - f1_f2_k: k-space field on the physical grid (ng^3)
        """
        f1_k = f1_k.astype(self.complex_dtype)
        f2_k = f2_k.astype(self.complex_dtype)

        # pad both to ng_pad
        f_k_arr = jnp.stack([
            f1_k,
            f2_k,
        ], axis=0)  # (2, ng, ng, ng//2+1)
        f_k_arr_pad = self.pad_fields(f_k_arr)
        f_r_arr_pad = self._irfftn_vec(f_k_arr_pad)  # (2, ng_pad, ng_pad, ng_pad)
        f1_r_pad, f2_r_pad = f_r_arr_pad[0], f_r_arr_pad[1]

        f1_f2_r_pad = f1_r_pad * f2_r_pad

        # back to k, low-pass to ng
        f1_f2_k_pad = self.rfftn(f1_f2_r_pad)
        f1_f2_k = self.unpad_fields(f1_f2_k_pad[None, ...])[0]

        # extend to ng_L and go back to real
        f1_f2_k_L = self.pad_fields_L(f1_f2_k[None, ...])[0]
        f1_f2_r_L = self.irfftn(f1_f2_k_L)

        return f1_f2_r_L, f1_f2_k
    
    def _to_r_L_from_k(self, field_k: jnp.ndarray) -> jnp.ndarray:
        r"""
        Map a k-space field on the ng grid to r-space on the ng_L grid with padding.
        """
        if field_k.ndim == 3:
            field_k_pad = self.pad_fields(field_k[None,...])[0]
            field_k_L = self.pad_fields_L(field_k_pad[None,...])[0]
            field_r_L = self.irfftn(field_k_L)
        else:
            field_k_pad = self.pad_fields(field_k)
            field_k_L = self.pad_fields_L(field_k_pad)
            field_r_L = self._irfftn_vec(field_k_L)
        return field_r_L
            
    # -------------------- LPT displacement --------------------
    @partial(jit, static_argnames=('self',))
    def lpt(self, delta_k: jnp.ndarray, growth_f: float = 0.0) -> jnp.ndarray:
        delta_k = delta_k.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        disp_k = coord.apply_disp_k(delta_k, self.kx, self.ky, self.kz).astype(self.complex_dtype)
        disp_k_L = self.pad_fields_L(disp_k)
        disp_r_L = self._irfftn_vec(disp_k_L).astype(self.real_dtype)
        if self.rsd:
            gf = jnp.asarray(growth_f, dtype=self.real_dtype)
            disp_r_L = disp_r_L.at[2].add(disp_r_L[2] * gf)
        return disp_r_L  # (3, ng, ng, ng)

    # -------- scalar fields in position space -------
    @partial(jit, static_argnames=('self'))
    def _scalar_fields_r(self, delta_k: jnp.ndarray) -> jnp.ndarray:
        r"""
        Build scalar fields in position space: [1, delta, d^2, G2, (G2_zz), (LyA extras...)]
        Returns stacked array with leading field axis.
        """
        delta_k = delta_k.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        delta_k_pad = coord.func_extend(self.ng_pad, delta_k)
        fields = [jnp.ones((self.ng_L,) * 3, dtype=self.real_dtype)]

        if self.bias_order >= 1:
            delta_k_L = self.pad_fields_L(delta_k[None,...])[0]
            delta_r_L = self.irfftn(delta_k_L)
            fields.append(delta_r_L)

        if self.bias_order >= 2:
            delta_r_pad = self.irfftn(delta_k_pad)
            d2_r_pad = delta_r_pad**2
            d2_k_pad = self.rfftn(d2_r_pad)
            d2_k     = self.unpad_fields(d2_k_pad[None,...])[0]
            d2_k_L   = self.pad_fields_L(d2_k[None,...])[0]
            d2_r_L   = self.irfftn(d2_k_L)
            sigma2_L = jnp.mean(d2_r_L)
            d2_r_L = d2_r_L - jnp.mean(d2_r_L)

            Gij_k = coord.apply_Gij_k(delta_k, self.kx, self.ky, self.kz)  # (6, ...)
            Gij_k_pad = self.pad_fields(Gij_k)  # (6, ...)
            Gij_r_pad = self._irfftn_vec(Gij_k_pad)  # real
            # 0: xx, 1: xy, 2: xz, 3: yy, 4: yz, 5: zz

            phi2_pad = (Gij_r_pad[0] * Gij_r_pad[3] + Gij_r_pad[3] * Gij_r_pad[5] + Gij_r_pad[5] * Gij_r_pad[0]
                        - Gij_r_pad[1]**2 - Gij_r_pad[2]**2 - Gij_r_pad[4]**2)
            G2_r_pad = -2.0 * phi2_pad
            G2_k_pad = self.rfftn(G2_r_pad)
            G2_k     = self.unpad_fields(G2_k_pad[None,...])[0]
            G2_k_L   = self.pad_fields_L(G2_k[None,...])[0]
            G2_r_L   = self.irfftn(G2_k_L)
            G2_r_L = G2_r_L - jnp.mean(G2_r_L)

            fields.extend([d2_r_L, G2_r_L])

            # RSD term G2_zz
            if self.rsd or self.lya:
                mu2 = coord.mu2_grid(self.kx, self.ky, self.kz)
                G2_zz_k = mu2 * G2_k
                G2_zz_k_L = self.pad_fields_L(G2_zz_k[None,...])[0]
                G2_zz_r_L = self.irfftn(G2_zz_k_L)
                G2_zz_r_L = G2_zz_r_L - jnp.mean(G2_zz_r_L)
                fields.append(G2_zz_r_L)

            # Ly-A extras
            if self.lya:
                #if self.lya_full_fields:
                #    eta_k_L = self.pad_fields_L(Gij_k[5][None,...])[0] # equals delta * mu^2 in k space (Gzz)
                #    eta_r_L = self.irfftn(eta_k_L)
                #    eta_r_L = eta_r_L - jnp.mean(eta_r_L)

                eta2_r_pad = Gij_r_pad[5]**2
                eta2_k_pad = self.rfftn(eta2_r_pad)
                eta2_k = self.unpad_fields(eta2_k_pad[None,...])[0]
                eta2_k_L = self.pad_fields_L(eta2_k[None,...])[0]
                eta2_r_L = self.irfftn(eta2_k_L)
                eta2_r_L = eta2_r_L - jnp.mean(eta2_r_L)

                deta_r_pad = delta_r_pad * Gij_r_pad[5]
                deta_k_pad = self.rfftn(deta_r_pad)
                deta_k     = self.unpad_fields(deta_k_pad[None,...])[0]
                deta_k_L   = self.pad_fields_L(deta_k[None,...])[0]
                deta_r_L = self.irfftn(deta_k_L)
                deta_r_L = deta_r_L - jnp.mean(deta_r_L)

                GG_zz_r_pad = Gij_r_pad[2]**2 + Gij_r_pad[4]**2 + Gij_r_pad[5]**2
                GG_zz_k_pad = self.rfftn(GG_zz_r_pad)
                GG_zz_k = self.unpad_fields(GG_zz_k_pad[None,...])[0]
                GG_zz_k_L = self.pad_fields_L(GG_zz_k[None,...])[0]
                GG_zz_r_L = self.irfftn(GG_zz_k_L)
                GG_zz_r_L = GG_zz_r_L - jnp.mean(GG_zz_r_L)

                KK_zz_r_L = GG_zz_r_L - (2./3.) * deta_r_L + (1./9.) * d2_r_L
                KK_zz_r_L = KK_zz_r_L - jnp.mean(KK_zz_r_L)

                if self.lya_full_fields:
                    ### degenerate operators
                    pi_zz_r_L = GG_zz_r_L - 5./7 * G2_zz_r_L
                    ### for a moment 
                    d3_r_pad = delta_r_pad**3
                    d3_k_pad = self.rfftn(d3_r_pad)
                    d3_k     = self.unpad_fields(d3_k_pad[None,...])[0]
                    d3_k_L   = self.pad_fields_L(d3_k[None,...])[0]
                    d3_r_L   = self.irfftn(d3_k_L)
                    d3_r_L   = d3_r_L - jnp.mean(d3_r_L)
                    fields.extend([eta2_r_L, deta_r_L, KK_zz_r_L, 
                                   pi_zz_r_L,
                                   d3_r_L,])
                else:
                    fields.extend([eta2_r_L, deta_r_L, KK_zz_r_L])

        if self.bias_order >= 3:
            #d3_r_L, d3_k   = self._safe_product_L(delta_k, d2_k)
            d3_r_pad = delta_r_pad**3
            d3_k_pad = self.rfftn(d3_r_pad)
            d3_k     = self.unpad_fields(d3_k_pad[None,...])[0]
            d3_k_L   = self.pad_fields_L(d3_k[None,...])[0]
            d3_r_L   = self.irfftn(d3_k_L)

            # dG2
            #dG2_r_L, dG2_k = self._safe_product_L(delta_k, G2_k)
            dG2_r_pad = delta_r_pad * G2_r_pad
            dG2_k_pad = self.rfftn(dG2_r_pad)
            dG2_k     = self.unpad_fields(dG2_k_pad[None,...])[0]
            dG2_k_L   = self.pad_fields_L(dG2_k[None,...])[0]
            dG2_r_L   = self.irfftn(dG2_k_L)

            # G3
            # 0: xx, 1: xy, 2: xz, 3: yy, 4: yz, 5: zz
            ### - Det(Gij_r)
            '''
            Gxy2_r, Gxy2_k = self._safe_square(Gij_k[1])
            Gxz2_r, Gxz2_k = self._safe_square(Gij_k[2])
            Gyz2_r, Gyz2_k = self._safe_square(Gij_k[4])

            GxzGyz_r, GxzGyz_k = self._safe_product(Gij_k[2], Gij_k[4])
            GyyGzz_r, GyyGzz_k = self._safe_product(Gij_k[3], Gij_k[5])

            GxxGyz2_r_L, GxxGyz2_k = self._safe_product_L(Gij_k[0], Gyz2_k)
            GyyGxz2_r_L, GyyGxz2_k = self._safe_product_L(Gij_k[3], Gxz2_k)
            GzzGxy2_r_L, GzzGxy2_k = self._safe_product_L(Gij_k[5], Gxy2_k)
            GxyGxzGyz_r_L, GxyGxzGyz_k = self._safe_product_L(Gij_k[1], GxzGyz_k)
            GxxGyyGzz_r_L, GxxGyyGzz_k = self._safe_product_L(Gij_k[0], GyyGzz_k)

            phi3a_r_L = GxxGyz2_r_L + GyyGxz2_r_L + GzzGxy2_r_L - 2.0*GxyGxzGyz_r_L - GxxGyyGzz_r_L
            G3_r_L    = -3.0*phi3a_r_L
            '''
            phi3a_r_pad = Gij_r_pad[0]*Gij_r_pad[4]*Gij_r_pad[4] + Gij_r_pad[3]*Gij_r_pad[2]*Gij_r_pad[2] + Gij_r_pad[5]*Gij_r_pad[1]*Gij_r_pad[1] - 2.0*Gij_r_pad[1]*Gij_r_pad[2]*Gij_r_pad[4] - Gij_r_pad[0]*Gij_r_pad[3]*Gij_r_pad[5]
            phi3a_k_pad = self.rfftn(phi3a_r_pad)
            phi3a_k     = self.unpad_fields(phi3a_k_pad[None,...])[0]
            phi3a_k_L   = self.pad_fields_L(phi3a_k[None,...])[0]
            phi3a_r_L   = self.irfftn(phi3a_k_L)
            G3_r_L      = -3.0*phi3a_r_L

            # Gamma3
            G2ij_k  = coord.apply_Gij_k(G2_k, self.kx, self.ky, self.kz)  # (6, ...)
            G2ij_k_pad = self.pad_fields(G2ij_k)  # (6, ...)
            '''
            GxxG2yy_r_L, _ = self._safe_product_L(Gij_k[0], G2ij_k[3])
            GxxG2zz_r_L, _ = self._safe_product_L(Gij_k[0], G2ij_k[5])
            GyyG2xx_r_L, _ = self._safe_product_L(Gij_k[3], G2ij_k[0])
            GyyG2zz_r_L, _ = self._safe_product_L(Gij_k[3], G2ij_k[5])
            GzzG2xx_r_L, _ = self._safe_product_L(Gij_k[5], G2ij_k[0])
            GzzG2yy_r_L, _ = self._safe_product_L(Gij_k[5], G2ij_k[3])

            GxyG2xy_r_L, _ = self._safe_product_L(Gij_k[1], G2ij_k[1])
            GxzG2xz_r_L, _ = self._safe_product_L(Gij_k[2], G2ij_k[2])
            GyzG2yz_r_L, _ = self._safe_product_L(Gij_k[4], G2ij_k[4])

            phi3b_r_L = 0.5* (GxxG2yy_r_L + GxxG2zz_r_L + GyyG2xx_r_L + GyyG2zz_r_L + GzzG2xx_r_L + GzzG2yy_r_L) \
                          - (GxyG2xy_r_L + GxzG2xz_r_L + GyzG2yz_r_L)
            '''
            G2ij_r_pad  = self._irfftn_vec(G2ij_k_pad)
            phi3b_r_pad = 0.5*Gij_r_pad[0]*(G2ij_r_pad[3]+G2ij_r_pad[5]) + 0.5*Gij_r_pad[3]*(G2ij_r_pad[0]+G2ij_r_pad[5]) + 0.5*Gij_r_pad[5]*(G2ij_r_pad[0]+G2ij_r_pad[3]) - Gij_r_pad[1]*G2ij_r_pad[1] - Gij_r_pad[2]*G2ij_r_pad[2] - Gij_r_pad[4]*G2ij_r_pad[4]
            phi3b_k_pad = self.rfftn(phi3b_r_pad)
            phi3b_k     = self.unpad_fields(phi3b_k_pad[None,...])[0]
            phi3b_k_L   = self.pad_fields_L(phi3b_k[None,...])[0]
            phi3b_r_L   = self.irfftn(phi3b_k_L)
            ### multiplying -10./21. results in one of the third order potential in LPT
            ### Gamma3 = -8/7 \phi^(3b)
            Gamma3_r_L = -8./7.*phi3b_r_L

            # S3 = \psi_2 \cdot \nabla \delta_1
            grad_delta_k     = coord.apply_grad_k(delta_k, self.kx, self.ky, self.kz)  # (3, ...)
            grad_delta_k_pad = self.pad_fields(grad_delta_k) # (3, ...)
            grad_delta_r_pad = self._irfftn_vec(grad_delta_k_pad)
            disp2_k_pad = - (3./14.) * coord.apply_disp_k(G2_k_pad, self.kx_pad, self.ky_pad, self.kz_pad)  # (3, ...)
            disp2_r_pad = self._irfftn_vec(disp2_k_pad)
            S3_r_pad = jnp.einsum('i...,i...->...', disp2_r_pad, grad_delta_r_pad)
            S3_k_pad = self.rfftn(S3_r_pad)
            S3_k     = self.unpad_fields(S3_k_pad[None,...])[0]
            S3_k_L   = self.pad_fields_L(S3_k[None,...])[0]
            S3_r_L   = self.irfftn(S3_k_L)

            if self.renormalize:
                d3_r_L  = d3_r_L - 3.*sigma2_L*delta_r_L
                dG2_r_L = dG2_r_L + 4./3.*sigma2_L*delta_r_L

            fields.extend([d3_r_L - jnp.mean(d3_r_L), 
                           dG2_r_L - jnp.mean(dG2_r_L), 
                           G3_r_L - jnp.mean(G3_r_L), 
                           Gamma3_r_L - jnp.mean(Gamma3_r_L),
                           S3_r_L - jnp.mean(S3_r_L),])

        return jnp.array(fields)
    
    # -------- tensor fields in position space -------
    @partial(jit, static_argnames=('self',))
    def _tensor_fields_r(self, delta_k: jnp.ndarray) -> jnp.ndarray:
        r"""
        Build tensor fields in position space: [K_ij, dK_ij, H_ij, T_ij, ...]
        Returns
        -------
        (n_fields, 6, ng, ng, ng) with order (xx,xy,xz,yy,yz,zz) in the 6-axis.
        """
        delta_k = delta_k.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        
        fields = []

        if self.bias_order >= 1:
            delta_k_L = self.pad_fields_L(delta_k[None,...])[0]
            Pi1_ij_k = coord.apply_Gij_k(delta_k_L, self.kx_L, self.ky_L, self.kz_L)  # (6, ng_L, ng_L, ng_L//2+1)
            Pi1_ij_r = self._irfftn_vec(Pi1_ij_k)  # (6, ng_L, ng_L, ng_L)
            K_ij_r = coord.apply_traceless(Pi1_ij_r)
            fields.append(K_ij_r)

        if self.bias_order >= 2:
            delta_k_pad = coord.func_extend(self.ng_pad, delta_k)
            delta_r_pad = self.irfftn(delta_k_pad)

            Pi1_ij_k = coord.apply_Gij_k(delta_k, self.kx, self.ky, self.kz)  # (6, ng, ng, ng//2+1)
            Pi1_ij_k_pad = self.pad_fields(Pi1_ij_k)  # (6, ng_pad, ng_pad, ng_pad//2+1)
            Pi1_ij_r_pad = self._irfftn_vec(Pi1_ij_k_pad)  # (6, ng_pad, ng_pad, ng_pad)

            dPi1_ij_r_pad = Pi1_ij_r_pad * delta_r_pad[None, :, :, :]
            dPi1_ij_k_pad = self._rfftn_vec(dPi1_ij_r_pad)
            dPi1_ij_k = self.unpad_fields(dPi1_ij_k_pad)
            dPi1_ij_k_L = self.pad_fields_L(dPi1_ij_k)
            dPi1_ij_r = self._irfftn_vec(dPi1_ij_k_L)
            dPi1_ij_r = dPi1_ij_r - jnp.mean(dPi1_ij_r, axis=(1, 2, 3), keepdims=True)

            Pi1_sqrt_xx_r_pad = Pi1_ij_r_pad[0]*Pi1_ij_r_pad[0] + Pi1_ij_r_pad[1]*Pi1_ij_r_pad[1] + Pi1_ij_r_pad[2]*Pi1_ij_r_pad[2]
            Pi1_sqrt_xy_r_pad = Pi1_ij_r_pad[0]*Pi1_ij_r_pad[1] + Pi1_ij_r_pad[1]*Pi1_ij_r_pad[3] + Pi1_ij_r_pad[2]*Pi1_ij_r_pad[4]
            Pi1_sqrt_xz_r_pad = Pi1_ij_r_pad[0]*Pi1_ij_r_pad[2] + Pi1_ij_r_pad[1]*Pi1_ij_r_pad[4] + Pi1_ij_r_pad[2]*Pi1_ij_r_pad[5]
            Pi1_sqrt_yy_r_pad = Pi1_ij_r_pad[3]*Pi1_ij_r_pad[3] + Pi1_ij_r_pad[1]*Pi1_ij_r_pad[1] + Pi1_ij_r_pad[4]*Pi1_ij_r_pad[4]
            Pi1_sqrt_yz_r_pad = Pi1_ij_r_pad[3]*Pi1_ij_r_pad[4] + Pi1_ij_r_pad[1]*Pi1_ij_r_pad[2] + Pi1_ij_r_pad[4]*Pi1_ij_r_pad[5]
            Pi1_sqrt_zz_r_pad = Pi1_ij_r_pad[5]*Pi1_ij_r_pad[5] + Pi1_ij_r_pad[2]*Pi1_ij_r_pad[2] + Pi1_ij_r_pad[4]*Pi1_ij_r_pad[4]

            Pi1_sqrt_r_pad = jnp.stack([Pi1_sqrt_xx_r_pad,
                                        Pi1_sqrt_xy_r_pad,
                                        Pi1_sqrt_xz_r_pad,
                                        Pi1_sqrt_yy_r_pad,
                                        Pi1_sqrt_yz_r_pad,
                                        Pi1_sqrt_zz_r_pad])
            
            H_ij_r_pad = Pi1_sqrt_r_pad - dPi1_ij_r_pad
            H_ij_k_pad = self._irfftn_vec(H_ij_r_pad)
            H_ij_k     = self.unpad_fields(H_ij_k_pad)
            H_ij_k_L   = self.pad_fields_L(H_ij_k)
            H_ij_r_L   = self._rfftn_vec(H_ij_k_L)
            H_ij_r_L   = coord.apply_traceless(H_ij_r_L)
            H_ij_r_L   = H_ij_r_L - jnp.mean(H_ij_r_L, axis=(1, 2, 3), keepdims=True)

            dPi1_ij_k   = self.unpad_fields(dPi1_ij_k_pad)
            dPi1_ij_k_L = self.pad_fields_L(dPi1_ij_k)
            dPi1_ij_r_L = self._irfftn_vec(dPi1_ij_k_L)
            dPi1_ij_r_L = coord.apply_traceless(dPi1_ij_r_L)
            dPi1_ij_r_L = dPi1_ij_r - jnp.mean(dPi1_ij_r, axis=(1, 2, 3), keepdims=True)

            G2_r_pad    = -2.0 * (Pi1_sqrt_xx_r_pad + Pi1_sqrt_yy_r_pad + Pi1_sqrt_zz_r_pad)
            G2_k_pad    = self.irfftn(G2_r_pad)
            G2_k        = self.unpad_fields(G2_k_pad)
            G2_k_L      = self.pad_fields_L(G2_k)
            T_ij_k_L    = -3./2.*coord.apply_Gij_k(G2_k_L, self.kx_L, self.ky_L, self.kz_L)
            T_ij_r_L    = self._irfftn_vec(T_ij_k_L)
            T_ij_r_L    = coord.apply_traceless(T_ij_r_L)
            T_ij_r_L    = T_ij_r_L - jnp.mean(T_ij_r_L, axis=(1, 2, 3), keepdims=True)

            fields.extend([dPi1_ij_r_L, H_ij_r_L, T_ij_r_L])

        if self.bias_order >= 3:
            ValueError("Third-order tensor fields not implemented yet.")

        return jnp.array(fields)
    
    @partial(jit, donate_argnums=(1,), static_argnames=('self',))
    def to_lya_fields(self, fields: jnp.ndarray, growth_f: float) -> jnp.ndarray:
        """
        Rewrite fields along axis=0 for LyA combination.
        The input buffer is donated to reduce peak memory if possible.
        """
        orig = fields

        fac = 3./7. * growth_f

        if self.lya_full_fields:
            out = (orig
                   .at[0].set(orig[1])                  # new[0] = shifted_d,
                   .at[1].set(orig[0] - fac * orig[4])  # new[1] = shifted_eta_new = shifted_1 - fac * shifted_G2_zz
                   .at[2].set(orig[2])                  # new[2] = shifted_d2
                   .at[3].set(orig[9])                  # new[3] = shifted_d3
                   .at[4].set(orig[3])                  # new[4] = shifted_G2
                   .at[5].set(orig[6])                  # new[5] = shifted_deta
                   .at[6].set(orig[5])                  # new[6] = shifted_eta2
                   .at[7].set(orig[7])                  # new[7] = shifted_KK_zz
                   .at[8].set(orig[8])                  # new[8] = shifted_pi_zz          
                )
        else:
            out = (orig
                   .at[0].set(orig[1])                  # new[0] = shifted_d,
                   .at[1].set(orig[0] - fac * orig[4])  # new[1] = shifted_eta_new = shifted_1 - fac * shifted_G2_zz
                   .at[2].set(orig[2])                  # new[2] = shifted_d2
                   .at[3].set(orig[3])                  # new[3] = shifted_G2
                   .at[4].set(orig[6])                  # new[4] = shifted_deta
                   .at[5].set(orig[5])                  # new[5] = shifted_eta2
                   .at[6].set(orig[7])                  # new[6] = shifted_KK_zz
                   )
        return out
    
    # -------- linear & quadratic helper fields (API preserved) --------
    @partial(jit, static_argnames=('self',))
    def eta_r(self, delta_k):
        return self.irfftn(self.eta_k(delta_k))

    @partial(jit, static_argnames=('self',))
    def eta_k(self, delta_k):
        delta_k = delta_k.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        mu2 = coord.mu2_grid(self.kx, self.ky, self.kz)
        return delta_k * mu2

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
        Gij_k = coord.apply_Gij_k(delta_k_L, self.kx_L, self.ky_L, self.kz_L)
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
        return self.rfftn(self.G2_r(delta_k_L)) * coord.mu2_grid(self.kx_L, self.ky_L, self.kz_L)

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
        Gij_k = coord.apply_Gij_k(delta_k_L, self.kx_L, self.ky_L, self.kz_L)
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
    
    @partial(jit, static_argnames=('self',))
    def Kij_k(self, delta_k_L):
        delta_k_L = delta_k_L.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        
        K_ij_k = coord.apply_Gij_k(delta_k_L, self.kx_L, self.ky_L, self.kz_L)  # (6, ng_L, ng_L, ng_L//2+1)
        K_ij_k = coord.apply_traceless(K_ij_k)

        return K_ij_k
    
    @partial(jit, static_argnames=('self',))
    def Kij_r(self, delta_k_L):
        K_ij_k = self.Kij_k(delta_k_L)
        K_ij_r = self._irfftn_vec(K_ij_k)  # (6, ng_L, ng_L, ng_L)
        return K_ij_r
    
    @partial(jit, static_argnames=('self',))
    def dKij_r(self, delta_k_L):
        delta_k_L = delta_k_L.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        K_ij_r = self.Kij_r(delta_k_L)

        delta_r = self.irfftn(delta_k_L)

        dK_ij_r = K_ij_r * delta_r[None, :, :, :]  # (6, ng_L, ng_L, ng_L)
        dK_ij_r = dK_ij_r - jnp.mean(dK_ij_r, axis=(1, 2, 3), keepdims=True)
        return dK_ij_r
    
    @partial(jit, static_argnames=('self',))
    def dKij_k(self, delta_k_L):
        return self._rfftn_vec(self.dKij_r(delta_k_L))
    
    @partial(jit, static_argnames=('self',))
    def Hij_r(self, delta_k_L):
        delta_k_L = delta_k_L.at[0,0,0].set(0.0).astype(self.complex_dtype)
        K_ij_r = self.Kij_r(delta_k_L)
        delta_r = self.irfftn(delta_k_L)
        dK_ij_r = K_ij_r * delta_r[None, :, :, :]  # (6, ng_L, ng_L, ng_L)
        dK_ij_r = dK_ij_r - jnp.mean(dK_ij_r, axis=(1, 2, 3), keepdims=True)

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
        H_ij_r = KK_ij_r - (1./3.)*dK_ij_r

        return H_ij_r
    
    @partial(jit, static_argnames=('self',))
    def Hij_k(self, delta_k_L):
        return self._rfftn_vec(self.Hij_r(delta_k_L))
    
    @partial(jit, static_argnames=('self',))
    def Tij_r(self, delta_k_L):
        delta_k_L = delta_k_L.at[0,0,0].set(0.0).astype(self.complex_dtype)
        K_ij_r = self.Kij_r(delta_k_L)
        delta_r = self.irfftn(delta_k_L)
        K2_r = K_ij_r[0]**2 + K_ij_r[3]**2 + K_ij_r[5]**2 + 2.0*(K_ij_r[1]**2 + K_ij_r[2]**2 + K_ij_r[4]**2)
        
        T_r = (delta_r*delta_r - 1.5 * K2_r).astype(self.real_dtype)
        T_ij_k = coord.apply_Gij_k(self.rfftn(T_r), self.kx_L, self.ky_L, self.kz_L)  # (6, nx, ny, nzr)
        T_ij_k = coord.apply_traceless(T_ij_k)
        T_ij_r = self._irfftn_vec(T_ij_k)  # (6, ng_L, ng_L, ng_L)
        T_ij_r = T_ij_r - jnp.mean(T_ij_r, axis=(1, 2, 3), keepdims=True)

        return T_ij_r
    
    @partial(jit, static_argnames=('self',))
    def Tij_k(self, delta_k_L):
        return self._rfftn_vec(self.Tij_r(delta_k_L))

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

        # lpt displacement
        disp_r_L = self.lpt(delta_k, growth_f=growth_f)  # (3, ng, ng, ng)

        # list of scalar fields in position space
        if field_type == 'scalar':
            fields_r_L = self._scalar_fields_r(delta_k)    # (m, ng_L, ng_L, ng_L)
            m = int(fields_r_L.shape[0])
        elif field_type == 'tensor':
            fields_r_L = self._tensor_fields_r(delta_k)    # (m, 6, ng_L, ng_L, ng_L)
            m = int(fields_r_L.shape[0])
        else:
            raise ValueError("field_type must be 'scalar' or 'tensor'.")

        # 2lpt correction
        if self.lpt_order >= 2:
            if field_type != 'scalar':
                raise NotImplementedError("2LPT correction only implemented for field_type='scalar'.")
            if self.bias_order < 2:
                G2_k = self.G2_k(delta_k)
            else:
                G2_k_L = self.rfftn(fields_r_L[3])
                G2_k
            disp_r_L = disp_r_L - (3./14.) * self.lpt(G2_k_L, growth_f=growth_f).astype(self.real_dtype)

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
        
    def get_shifted_fields_lightcone_fast(
        self,
        delta_k: jnp.ndarray,
        *,
        D_ic: float,      # D(z_ic)
        chi_edges: jnp.ndarray, # (Ns+1,) chi(z_edges)
        D_mid: jnp.ndarray,     # (Ns,)   D(z_mid)
        beta_mid: jnp.ndarray,   # (num_fields, Ns) bias factors at z_mid for each field (only constant bias for each field)
        observer_pos: Optional[jnp.ndarray] = None,  # (3,) in [0, boxsize)
        growth_powers: Optional[jnp.ndarray] = None,     # (num_fields,)
        f_mid: Optional[jnp.ndarray] = None,     # (Ns,)   growth rate at z_mid for RSD
        mode: str = 'k_space',
        field_type: Literal['scalar', 'tensor'] = 'scalar',
        neighbor_mode: str = 'auto',
        fuse_updates_threshold: int=100_000_000,
    ) -> jnp.ndarray:
        r"""
        Shell-based light-cone construction.

        For each slice i:
          - uses Psi_ic and Lagrangian operators at z_ic
          - scales them by D_mid[i]/D_ic and growth_powers
          - keeps only cells with radius in [chi_edges[i], chi_edges[i+1])
          - assigns to Eulerian grid and accumulates.
        """
        delta_k = delta_k.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0

        # lpt displacement, w/o RSD for lightcone construction
        disp_r_L = self.lpt(delta_k, growth_f=0.0)  # (3, ng, ng, ng)

        # list of scalar fields in position space
        if field_type == 'scalar':
            fields_r_L = self._scalar_fields_r(delta_k)    # (m, ng_L, ng_L, ng_L)
            m = int(fields_r_L.shape[0])
        elif field_type == 'tensor':
            fields_r_L = self._tensor_fields_r(delta_k)    # (m, 6, ng_L, ng_L, ng_L)
            m = int(fields_r_L.shape[0])
        else:
            raise ValueError("field_type must be 'scalar' or 'tensor'.")

        # 2lpt correction
        if self.lpt_order >= 2:
            raise NotImplementedError("2LPT correction is not implemented for lightcone.")
        
        # observer position and Lagrangain coordinates
        dq = self.boxsize / self.ng_L
        q1d = (jnp.arange(self.ng_L, dtype=self.real_dtype)) * dq

        if observer_pos is None:
            observer_pos = jnp.array([0., 0., 0.], dtype=self.real_dtype)
        else:
            observer_pos = jnp.asarray(observer_pos, dtype=self.real_dtype)

        qx_rel = (q1d - observer_pos[0])[:, None, None]
        qy_rel = (q1d - observer_pos[1])[None, :, None]
        qz_rel = (q1d - observer_pos[2])[None, None, :]

        # growth powers (D(z)/D(z_ic))^n for each field
        if field_type == 'scalar':
            if growth_powers is None:
                growth_powers = [0, 1, 2, 2,] ### for [1, d, d2, G2]
            growth_powers = jnp.asarray(growth_powers, dtype=self.real_dtype)
            if growth_powers.shape[0] != m:
                raise ValueError("growth_powers length must match # scalar fields.")
        elif field_type == "tensor":
            if growth_powers is None:
                growth_powers = [1, 2, 2, 2,] ### for [Kij, dKij, Hij, Tij]
            growth_powers = jnp.asarray(growth_powers, dtype=self.real_dtype)
            if growth_powers.shape[0] != m:
                raise ValueError("growth_powers length must match # tensor fields.")

        Ns = int(D_mid.shape[0])   ### Number of z-slices

        W   = jnp.zeros((self.ng_L, self.ng_L, self.ng_L), dtype=self.real_dtype)  # eventually D_ratio(z_slice(q)) or 0
        if self.rsd:
            W_f = jnp.zeros((self.ng_L, self.ng_L, self.ng_L), dtype=self.real_dtype)

        # loop over slices (Python for; you can turn into lax.fori_loop if Ns is small/static) ----
        for i in range(Ns):
            chi2_low = chi_edges[i] ** 2
            chi2_high = chi_edges[i+1] ** 2
            D_ratio = D_mid[i] / D_ic

            # displacement at slice i
            disp_r_L_i = disp_r_L * D_ratio   # (3, ng_L, ng_L, ng_L)
            disp_x, disp_y, disp_z = disp_r_L_i

            # relative positions in the periodic box
            #dx = (qx_rel + disp_x) - self.boxsize * jnp.round((qx_rel + disp_x) / self.boxsize)
            #dy = (qy_rel + disp_y) - self.boxsize * jnp.round((qy_rel + disp_y) / self.boxsize)
            #dz = (qz_rel + disp_z) - self.boxsize * jnp.round((qz_rel + disp_z) / self.boxsize)

            dx = (qx_rel + disp_x)
            dy = (qy_rel + disp_y)
            dz = (qz_rel + disp_z)

            # distance at slice i
            r2_i = dx*dx + dy*dy + dz*dz  # (ng_L, ng_L, ng_L)

            # mask: which Lagrangian particles are in this shell?
            in_shell = (r2_i >= chi2_low) & (r2_i < chi2_high)
            in_shell_f = in_shell.astype(self.real_dtype)  ### zero for out-of-shell particles

            W = W + D_ratio * in_shell_f # since each cell belongs to at most one slice, W(q)=D_ratio_j

            # rescale bias operators
            for a in range(m):
                p_a = growth_powers[a]
                beta_ai = beta_mid[a, i]

                #  O_a_final = O_a_ic * D_ratio^p_a * beta_ai
                factor_ai = (D_ratio ** p_a) * beta_ai
                # multiply by factor_ai only inside shell; keep as-is outside
                # mult = 1 outside, factor_ai inside
                fac = (1.0 - in_shell_f) + factor_ai * in_shell_f
                fields_r_L = fields_r_L.at[a].set(fields_r_L[a] * fac)
            if self.rsd:
                W_f = W_f + f_mid[i] * in_shell_f

        mask_any = (W > 0.0).astype(self.real_dtype)  # 0 or 1

        # final displacement
        disp_r_L = disp_r_L * W[None, :, :, :]  # (3, ng_L, ng_L, ng_L)
        if self.rsd:
            disp_r_L = disp_r_L.at[2].add(disp_r_L[2] * W_f)  # (3, ng_L, ng_L, ng_L)

        for a in range(m):
            fields_r_L = fields_r_L.at[a].set(fields_r_L[a] * mask_any)

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

    def get_shifted_fields_lightcone_bruteforce(
        self,
        delta_k: jnp.ndarray,
        *,
        D_ic: float,                       # D(z_ic)
        chi_edges: jnp.ndarray,            # (Ns+1,)
        D_mid: jnp.ndarray,                # (Ns,)
        observer_pos: Optional[jnp.ndarray] = None,  # (3,)
        growth_powers: jnp.ndarray,        # (n_fields0,)
        # Optional RSD (fixed LOS axis); used only in assignment displacement
        f_mid: Optional[jnp.ndarray] = None,         # (Ns,)
        rsd_axis: int = 2,
        # Ortho / beta(k,mu,z)
        measure_pk=None,
        k_edges: Optional[jnp.ndarray] = None,       # (Nk+1,)
        mu_edges: Optional[jnp.ndarray] = None,      # (Nmu+1,)
        beta_kmu_mid: jnp.ndarray = None,            # (Ns, n_fields_eff, Nk, Nmu)
        Nmin: int = 5,
        jitter: float = 0.0,
        dtype=jnp.float32,
        warn_on_low_stat: bool = True,
        # Assignment controls
        field_type: Literal["scalar", "tensor"] = "scalar",
        neighbor_mode: str = "auto",
        fuse_updates_threshold: int = 100_000_000,
    ) -> jnp.ndarray:
        """
        Brute-force lightcone (Eulerian masking after orthogonalization):

          For each slice s:
            1) Assign FULL-BOX Eulerian fields at z_s (Zel'dovich displacement; RSD optional)
            2) FFT+deconvolve and orthogonalize FULL-BOX fields in k-space
            3) Multiply orthogonalized fields by beta(k,mu,z_s) (piecewise-constant per (k,mu) bin)
               and sum over fields -> slice_k (single k-space field)
            4) iFFT -> slice_r, apply Eulerian shell mask in real space, rFFT back
            5) Accumulate into final_k

        This reuses nearest_idx from utils_jax._build_nearest_idx_for_grid both for Mcoef expansion
        inside orthogonalize (if used) and for expanding beta tables to the Fourier grid.

        Notes:
          - Shell selection is done in Eulerian grid (cell centers), not by particle membership in Lagrangian space.
          - This will smear the shell boundary at roughly the assignment stencil scale.
        """
        # --- sanity checks ---
        if field_type != "scalar":
            raise NotImplementedError("This brute-force implementation currently supports field_type='scalar' only.")
        if measure_pk is None or k_edges is None or mu_edges is None:
            raise ValueError("measure_pk, k_edges, mu_edges are required.")
        if beta_kmu_mid is None:
            raise ValueError("beta_kmu_mid is required.")
        if (self.rsd is True) and (f_mid is None):
            raise ValueError("f_mid is required when self.rsd=True.")

        # --- cast / normalize inputs ---
        delta_k = jnp.asarray(delta_k, dtype=self.complex_dtype)
        delta_k = delta_k.at[0, 0, 0].set(0.0)

        chi_edges = jnp.asarray(chi_edges, dtype=self.real_dtype)
        D_mid = jnp.asarray(D_mid, dtype=self.real_dtype)
        Ns = int(D_mid.shape[0])
        if chi_edges.shape[0] != Ns + 1:
            raise ValueError("chi_edges must have shape (Ns+1,)")

        k_edges = jnp.asarray(k_edges, dtype=self.real_dtype)
        mu_edges = jnp.asarray(mu_edges, dtype=self.real_dtype)
        Nk = int(k_edges.shape[0] - 1)
        Nmu = int(mu_edges.shape[0] - 1)

        # --- base Lagrangian displacement/operators at IC ---
        disp_r_L0 = self.lpt(delta_k, growth_f=0.0)  # (3, ng_L, ng_L, ng_L)
        fields_r_L0 = self._scalar_fields_r(delta_k)  # (n_fields0, ng_L, ng_L, ng_L)
        n_fields0 = int(fields_r_L0.shape[0])

        growth_powers = jnp.asarray(growth_powers, dtype=self.real_dtype)
        if growth_powers.shape[0] != n_fields0:
            raise ValueError(f"growth_powers must have length {n_fields0}, got {growth_powers.shape[0]}")

        # If you remap LyA operators and then drop the last field, define the effective count here.
        lya_reduce = (self.lya is True) and (self.rsd is True)
        n_fields_eff = n_fields0 - 1 if lya_reduce else n_fields0

        beta_kmu_mid = jnp.asarray(beta_kmu_mid, dtype=jnp.asarray(0.0, dtype=dtype).dtype)
        if beta_kmu_mid.shape[:4] != (Ns, n_fields_eff, Nk, Nmu):
            raise ValueError(f"beta_kmu_mid must have shape (Ns, n_fields_eff, Nk, Nmu) = {(Ns, n_fields_eff, Nk, Nmu)}")

        # --- observer position and Eulerian radius grid (cell centers) ---
        if observer_pos is None:
            observer_pos = jnp.array([0.0, 0.0, 0.0], dtype=self.real_dtype)
        else:
            observer_pos = jnp.asarray(observer_pos, dtype=self.real_dtype)

        ng_E = int(self.ng_E)
        dx_E = self.boxsize / ng_E
        # Cell-center convention: here we use x = i*dx (match your earlier convention).
        x1d = (jnp.arange(ng_E, dtype=self.real_dtype)) * dx_E
        x_rel = (x1d - observer_pos[0])[:, None, None]
        y_rel = (x1d - observer_pos[1])[None, :, None]
        z_rel = (x1d - observer_pos[2])[None, None, :]
        r2_grid = x_rel * x_rel + y_rel * y_rel + z_rel * z_rel  # (ng_E, ng_E, ng_E)

        # --- build nearest_idx once and reuse it for beta expansion ---
        # nearest_idx maps (kbin, mubin) -> flat index (kbin*Nmu + mubin) for each Fourier grid point.
        nearest_idx, Nk_chk, Nmu_chk = utils_jax._build_nearest_idx_for_grid(
            ng_E, self.boxsize, k_edges, mu_edges, dtype=self.real_dtype
        )
        if (Nk_chk != Nk) or (Nmu_chk != Nmu):
            raise ValueError("Inconsistent Nk/Nmu returned by _build_nearest_idx_for_grid.")

        # Helper: expand a (Nk,Nmu) table onto the rfft grid via nearest_idx
        def _expand_table_to_grid(table_NkNmu: jnp.ndarray) -> jnp.ndarray:
            flat = table_NkNmu.reshape(Nk * Nmu)
            return jnp.take(flat, nearest_idx, axis=0)  # (ng_E, ng_E, ng_E//2+1)

        # --- accumulator in k-space ---
        final_k = jnp.zeros((ng_E, ng_E, ng_E // 2 + 1), dtype=self.complex_dtype)

        for s in range(Ns):
            chi2_low = chi_edges[s] ** 2
            chi2_high = chi_edges[s + 1] ** 2

            # Eulerian shell mask on the grid (no displacement, no RSD)
            in_shell = (r2_grid >= chi2_low) & (r2_grid < chi2_high)
            in_shell_f = in_shell.astype(self.real_dtype)

            # Growth scaling for this slice
            D_ratio = jnp.asarray(D_mid[s] / D_ic, dtype=self.real_dtype)
            scale = jnp.power(D_ratio, growth_powers).astype(self.real_dtype)  # (n_fields0,)

            # Build slice displacement (Zel'dovich), apply RSD only in assignment
            disp_r_L = disp_r_L0 * D_ratio
            disp_r_L_assign = disp_r_L
            if self.rsd is True:
                gf = jnp.asarray(f_mid[s], dtype=self.real_dtype)
                disp_r_L_assign = disp_r_L_assign.at[rsd_axis].add(disp_r_L_assign[rsd_axis] * gf)

            # Full-box Lagrangian operators for this slice (no mask here)
            fields_r_L_full = fields_r_L0 * scale[:, None, None, None]  # (n_fields0, ng_L, ng_L, ng_L)

            # Optional LyA operator remapping (must match how beta_kmu_mid was defined)
            if lya_reduce:
                gf = jnp.asarray(f_mid[s], dtype=self.real_dtype)
                fields_r_L_full = self.to_lya_fields(fields_r_L_full, gf)[:-1]  # (n_fields_eff, ...)

            # Assign FULL-BOX -> Eulerian grid (real space)
            fields_r_E = self._assign_fields_from_disp_to_grid(
                disp_r_L_assign,
                fields_r_L_full,
                interlace=False,
                normalize_mean=True,
                field_type="scalar",
                neighbor_mode=neighbor_mode,
                fuse_updates_threshold=fuse_updates_threshold,
            )
            fields_r_Ei = None
            if self.mesh.interlace:
                fields_r_Ei = self._assign_fields_from_disp_to_grid(
                    disp_r_L_assign,
                    fields_r_L_full,
                    interlace=True,
                    normalize_mean=True,
                    field_type="scalar",
                    neighbor_mode=neighbor_mode,
                    fuse_updates_threshold=fuse_updates_threshold,
                )

            # FFT + deconvolution (full box)
            fields_k_full = self.mesh.fft_deconvolve_batched(fields_r_E, fields_r_Ei).astype(self.complex_dtype)
            fields_k_full = fields_k_full.at[:, 0, 0, 0].set(0.0)

            # Orthonormalize FULL-BOX fields in k-space (no Mcoef returned)
            fields_k_ortho = utils_jax.orthogonalize(
                fields_k_full,
                measure_pk,
                self.boxsize,
                k_edges,
                mu_edges,
                Nmin=Nmin,
                jitter=jitter,
                dtype=dtype,
                warn_on_low_stat=warn_on_low_stat,
                return_Mcoef=False,
                return_nearest_idx=False,
            ).astype(self.complex_dtype)
            fields_k_ortho = fields_k_ortho.at[:, 0, 0, 0].set(0.0)

            # Apply beta(k,mu,z_s) to orthogonalized fields and sum in k-space
            beta_table_s = beta_kmu_mid[s]  # (n_fields_eff, Nk, Nmu)
            slice_k = jnp.zeros((ng_E, ng_E, ng_E // 2 + 1), dtype=self.complex_dtype)
            for j in range(n_fields_eff):
                beta_grid = _expand_table_to_grid(beta_table_s[j]).astype(self.real_dtype)
                slice_k = slice_k + (beta_grid * fields_k_ortho[j]).astype(self.complex_dtype)
            slice_k = slice_k.at[0, 0, 0].set(0.0)

            # Go to real space, apply Eulerian shell mask, go back to k-space
            slice_r = self.irfftn(slice_k).astype(self.real_dtype)  # consistent with your norm='forward'
            slice_r = (slice_r * in_shell_f).astype(self.real_dtype)

            slice_k_masked = self.rfftn(slice_r).astype(self.complex_dtype)
            slice_k_masked = slice_k_masked.at[0, 0, 0].set(0.0)

            final_k = final_k + slice_k_masked

        return final_k.astype(self.complex_dtype)

    def _get_shifted_fields_lightcone_bruteforce_(
        self,
        delta_k: jnp.ndarray,
        *,
        D_ic: float,                       # D(z_ic)
        chi_edges: jnp.ndarray,            # (Ns+1,)
        D_mid: jnp.ndarray,                # (Ns,)
        observer_pos: Optional[jnp.ndarray] = None,  # (3,)
        growth_powers: jnp.ndarray, # (n_fields,)
        # Optional RSD (fixed LOS axis); only used for FINAL displacement in assignment
        f_mid: Optional[jnp.ndarray] = None,         # (Ns,)
        rsd_axis: int = 2,
        # Ortho / beta(k,mu,z)
        measure_pk=None,
        k_edges: Optional[jnp.ndarray] = None,       # (Nk+1,)
        mu_edges: Optional[jnp.ndarray] = None,      # (Nmu+1,)
        beta_kmu_mid: jnp.ndarray,  # (Ns, n_fields, Nk, Nmu)
        Nmin: int = 5,
        jitter: float = 0.0,
        dtype=jnp.float32,
        # Assignment / output
        field_type: Literal["scalar", "tensor"] = "scalar",
        neighbor_mode: str = "auto",
        fuse_updates_threshold: int = 100_000_000,
        # Performance
        recompute_M_every: int = 1,  # compute Mcoef every N slices; reuse otherwise
        warn_on_low_stat: bool = True,
    ) -> jnp.ndarray:
        """
        Slice-by-slice brute-force lightcone:
          1) Build IC Lagrangian displacement and operators once.
          2) For each z-slice:
               - compute in-shell mask in Lagrangian space (no periodic wrap by default)
               - assign masked operators to Eulerian grid and FFT
               - (optionally) recompute Mcoef(k,mu) using full-box fields for that slice
               - apply Mcoef and beta_j(k,mu,z_slice) to the masked fields
               - sum over fields and accumulate into final_k
          3) Return final_k.

        Notes:
          - Operators are still scaled by (D_mid/D_ic)^{growth_powers} if growth_powers is provided.
          - beta_kmu_mid must be per-slice and per-field: (Ns, n_fields, Nk, Nmu).
        """
        if measure_pk is None or k_edges is None or mu_edges is None:
            raise ValueError("measure_pk, k_edges, mu_edges are required to compute Mcoef(k,mu).")
        if self.rsd is True and f_mid is None:
            raise ValueError("f_mid is required for RSD")

        delta_k = delta_k.at[0, 0, 0].set(0.0).astype(self.complex_dtype)

        # --- base Lagrangian displacement at IC (no RSD for selection) ---
        disp_r_L0 = self.lpt(delta_k, growth_f=0.0)  # (3, ng_L, ng_L, ng_L)

        # --- base Lagrangian operators at IC ---
        if field_type == "scalar":
            fields_r_L0 = self._scalar_fields_r(delta_k)    # (n_fields, ng_L, ng_L, ng_L)
        elif field_type == "tensor":
            fields_r_L0 = self._tensor_fields_r(delta_k)    # (n_fields, 6, ng_L, ng_L, ng_L)
        else:
            raise ValueError("field_type must be 'scalar' or 'tensor'.")

        n_fields = int(fields_r_L0.shape[0])
        if self.rsd is True:
            if self.bias_order == 2:
                n_fields -= 1
        Ns = int(D_mid.shape[0])

        # --- growth powers for operator evolution ---
        growth_powers = jnp.asarray(growth_powers, dtype=self.real_dtype)

        # --- beta(k,mu,z) ---
        Nk = int(k_edges.shape[0] - 1)
        Nmu = int(mu_edges.shape[0] - 1)
        beta_kmu_mid = jnp.asarray(beta_kmu_mid, dtype=dtype)
        if beta_kmu_mid.shape[:4] != (Ns, n_fields, Nk, Nmu):
            raise ValueError(f"beta_kmu_mid must have shape (Ns, n_fields, Nk, Nmu) = {(Ns,n_fields,Nk,Nmu)}")

        # --- observer position and Lagrangian coordinate axes ---
        if observer_pos is None:
            observer_pos = jnp.array([0.0, 0.0, 0.0], dtype=self.real_dtype)
        else:
            observer_pos = jnp.asarray(observer_pos, dtype=self.real_dtype)

        dq = self.boxsize / self.ng_L
        q1d = (jnp.arange(self.ng_L, dtype=self.real_dtype)) * dq
        qx_rel = (q1d - observer_pos[0])[:, None, None]
        qy_rel = (q1d - observer_pos[1])[None, :, None]
        qz_rel = (q1d - observer_pos[2])[None, None, :]

        # --- output accumulator in Fourier space ---
        final_k = None

        # --- cached Mcoef / nearest_idx ---
        Mcoef_cached = None
        nearest_idx_cached = None

        for s in range(Ns):
            chi2_low = chi_edges[s] ** 2
            chi2_high = chi_edges[s + 1] ** 2
            D_ratio = jnp.asarray(D_mid[s] / D_ic, dtype=self.real_dtype)

            # --- displacement for this slice (selection uses no RSD) ---
            disp_r_L = disp_r_L0 * D_ratio
            disp_x, disp_y, disp_z = disp_r_L

            # --- shell membership test (no periodic wrapping by default) ---
            dx = qx_rel + disp_x
            dy = qy_rel + disp_y
            dz = qz_rel + disp_z
            r2 = dx * dx + dy * dy + dz * dz
            in_shell = (r2 >= chi2_low) & (r2 < chi2_high)
            in_shell_f = in_shell.astype(self.real_dtype)  # (ng_L, ng_L, ng_L)

            # --- scale operators by growth power for this slice ---
            scale = jnp.power(D_ratio, growth_powers).astype(self.real_dtype)  # (n_fields,)

            if field_type == "scalar":
                # (n_fields, ng_L, ng_L, ng_L)
                fields_r_L_mask = (fields_r_L0 * scale[:, None, None, None]) * in_shell_f[None, :, :, :]
                if (self.lya is True) and (self.rsd is True):
                    gf = jnp.asarray(f_mid[s], dtype=self.real_dtype)
                    fields_r_L_mask = self.to_lya_fields(fields_r_L_mask, gf)[:-1]
            else:
                # (n_fields, 6, ng_L, ng_L, ng_L)
                fields_r_L_mask = (fields_r_L0 * scale[:, None, None, None, None]) * in_shell_f[None, None, :, :, :]

            # --- displacement for assignment (RSD only here, if requested) ---
            disp_r_L_assign = disp_r_L
            if self.rsd is True:
                gf = jnp.asarray(f_mid[s], dtype=self.real_dtype)
                disp_r_L_assign = disp_r_L_assign.at[rsd_axis].add(disp_r_L_assign[rsd_axis] * gf)

            # --- assign masked fields -> Eulerian grid ---
            fields_r_E = self._assign_fields_from_disp_to_grid(
                disp_r_L_assign,
                fields_r_L_mask,
                interlace=False,
                normalize_mean=True,
                field_type=field_type,
                neighbor_mode=neighbor_mode,
                fuse_updates_threshold=fuse_updates_threshold,
            )
            fields_r_Ei = None
            if self.mesh.interlace:
                fields_r_Ei = self._assign_fields_from_disp_to_grid(
                    disp_r_L_assign,
                    fields_r_L_mask,
                    interlace=True,
                    normalize_mean=True,
                    field_type=field_type,
                    neighbor_mode=neighbor_mode,
                    fuse_updates_threshold=fuse_updates_threshold,
                )

            # --- FFT + deconvolution ---
            fields_k_mask = self.mesh.fft_deconvolve_batched(fields_r_E, fields_r_Ei).astype(self.complex_dtype)

            # --- (re)compute Mcoef using FULL-BOX (no mask) fields ---
            if (Mcoef_cached is None) or (recompute_M_every > 0 and (s % recompute_M_every == 0)):
                if field_type == "scalar":
                    fields_r_L_full = fields_r_L0 * scale[:, None, None, None]
                    if (self.lya is True) and (self.rsd is True):
                        gf = jnp.asarray(f_mid[s], dtype=self.real_dtype)
                        fields_r_L_full = self.to_lya_fields(fields_r_L_full, gf)[:-1]
                else:
                    fields_r_L_full = fields_r_L0 * scale[:, None, None, None, None]

                fields_r_E_full = self._assign_fields_from_disp_to_grid(
                    disp_r_L_assign,
                    fields_r_L_full,
                    interlace=False,
                    normalize_mean=True,
                    field_type=field_type,
                    neighbor_mode=neighbor_mode,
                    fuse_updates_threshold=fuse_updates_threshold,
                )
                fields_r_Ei_full = None
                if self.mesh.interlace:
                    fields_r_Ei_full = self._assign_fields_from_disp_to_grid(
                        disp_r_L_assign,
                        fields_r_L_full,
                        interlace=True,
                        normalize_mean=True,
                        field_type=field_type,
                        neighbor_mode=neighbor_mode,
                        fuse_updates_threshold=fuse_updates_threshold,
                    )

                fields_k_full = self.mesh.fft_deconvolve_batched(fields_r_E_full, fields_r_Ei_full).astype(self.complex_dtype)

                # orthogonalize must be patched to return (Mcoef, nearest_idx) when return_Mcoef=True.
                Mcoef_cached, nearest_idx_cached = utils_jax.orthogonalize(
                    fields_k_full,
                    measure_pk,
                    self.boxsize,
                    k_edges,
                    mu_edges,
                    Nmin=Nmin,
                    jitter=jitter,
                    dtype=dtype,
                    warn_on_low_stat=warn_on_low_stat,
                    return_Mcoef=True,
                    return_nearest_idx=True,
                )

            # --- apply Mcoef and beta(k,mu,z_slice) ---
            beta_table_s = beta_kmu_mid[s]  # (n_fields, Nk, Nmu)
            fields_k_out = utils_jax.apply_Mcoef(
                fields_k_mask,
                Mcoef_cached,
                nearest_idx_cached,
                beta_table=beta_table_s,
                apply_beta=True,
            )

            # --- accumulate final field (sum over orthogonalized fields) ---
            slice_k = jnp.sum(fields_k_out, axis=0)
            final_k = slice_k if (final_k is None) else (final_k + slice_k)

        return final_k.astype(self.complex_dtype)
    
    def get_noise_field(
        self,
        white_noise_k: jnp.ndarray,
        *,
        k_edges: jnp.ndarray,                 # (Nk+1,)
        mu_edges: jnp.ndarray,                # (Nmu+1,)
        noise_kmu_mid: jnp.ndarray,           # (Nk, Nmu)
        observer_pos: Optional[jnp.ndarray] = None,   # (3,), used only if chi_range is not None
        chi_range: Optional[tuple] = None,    # (chi_low, chi_high), optional shell mask
        nearest_idx: Optional[jnp.ndarray] = None,    # optional cache for rfft grid lookup
        r2_grid: Optional[jnp.ndarray] = None,        # optional cache for Eulerian radius^2
        dtype=jnp.float32,
    ) -> jnp.ndarray:
        """
        Build a Gaussian noise field in k-space for a single snapshot from a white-noise GRF,
        matching a target Perr(k, mu) table on the rfft grid.

        If chi_range=(chi_low, chi_high) is provided, the field is transformed to real space,
        masked in the Eulerian shell [chi_low, chi_high), then transformed back to k-space.
        This optional path is used by get_noise_lightcone().

        Returns
        -------
        noise_k : (ng_E, ng_E, ng_E//2+1), complex
            Noise field in k-space (rfftn convention).
        """
        # --- basic setup ---
        white_noise_k = jnp.asarray(white_noise_k, dtype=self.complex_dtype)
        white_noise_k = white_noise_k.at[0, 0, 0].set(0.0)  # force DC=0

        ng_E = int(self.ng_E)
        inv2V = jnp.array(0.5 / self.vol, dtype=self.real_dtype)

        k_edges = jnp.asarray(k_edges, dtype=self.real_dtype)
        mu_edges = jnp.asarray(mu_edges, dtype=self.real_dtype)

        Nk = int(k_edges.shape[0] - 1)
        Nmu = int(mu_edges.shape[0] - 1)

        noise_kmu_mid = jnp.asarray(noise_kmu_mid, dtype=self.real_dtype)
        if noise_kmu_mid.shape != (Nk, Nmu):
            raise ValueError(f"noise_kmu_mid must have shape (Nk, Nmu) = {(Nk, Nmu)}")

        # --- build (or reuse) bin-containment lookup on the rfft grid ---
        if nearest_idx is None:
            nearest_idx, Nk_chk, Nmu_chk = utils_jax._build_nearest_idx_for_grid(
                ng_E, self.boxsize, k_edges, mu_edges, dtype=self.real_dtype
            )
            if (Nk_chk != Nk) or (Nmu_chk != Nmu):
                raise RuntimeError(
                    f"Internal bin mismatch: edges imply (Nk,Nmu)=({Nk},{Nmu}) "
                    f"but lookup returned ({Nk_chk},{Nmu_chk})."
                )
        else:
            nearest_idx = jnp.asarray(nearest_idx)

        # --- expand Perr(k,mu) to the full rfft grid ---
        Perr_grid = jnp.take(noise_kmu_mid.reshape(Nk * Nmu), nearest_idx, axis=0)
        Perr_grid = Perr_grid.at[0, 0, 0].set(0.0)  # ensure DC=0

        # --- scale white noise in k-space ---
        amp = jnp.sqrt(jnp.maximum(Perr_grid, 0.0) * inv2V).astype(self.real_dtype)
        noise_k = (amp * white_noise_k).astype(self.complex_dtype)
        noise_k = noise_k.at[0, 0, 0].set(0.0)

        # --- snapshot case: no shell mask ---
        if chi_range is None:
            return noise_k.astype(self.complex_dtype)

        # --- light-cone slice case: apply Eulerian shell mask in real space ---
        chi_low, chi_high = chi_range
        chi_low = jnp.asarray(chi_low, dtype=self.real_dtype)
        chi_high = jnp.asarray(chi_high, dtype=self.real_dtype)

        if r2_grid is None:
            if observer_pos is None:
                observer_pos = jnp.array([0.0, 0.0, 0.0], dtype=self.real_dtype)
            else:
                observer_pos = jnp.asarray(observer_pos, dtype=self.real_dtype)

            dx = self.boxsize / ng_E
            x1d = (jnp.arange(ng_E, dtype=self.real_dtype)) * dx
            x_rel = (x1d - observer_pos[0])[:, None, None]
            y_rel = (x1d - observer_pos[1])[None, :, None]
            z_rel = (x1d - observer_pos[2])[None, None, :]
            r2_grid = x_rel * x_rel + y_rel * y_rel + z_rel * z_rel
        else:
            r2_grid = jnp.asarray(r2_grid, dtype=self.real_dtype)

        noise_r = jnp.fft.irfftn(noise_k, s=(ng_E, ng_E, ng_E), norm="ortho").astype(self.real_dtype)

        chi2_low = chi_low * chi_low
        chi2_high = chi_high * chi_high
        in_shell = (r2_grid >= chi2_low) & (r2_grid < chi2_high)

        noise_r_masked = jnp.where(in_shell, noise_r, 0.0).astype(self.real_dtype)
        noise_k_masked = jnp.fft.rfftn(noise_r_masked, norm="ortho").astype(self.complex_dtype)
        noise_k_masked = noise_k_masked.at[0, 0, 0].set(0.0)

        return noise_k_masked.astype(self.complex_dtype)


    def get_noise_lightcone(
        self,
        white_noise_k: jnp.ndarray,
        *,
        chi_edges: jnp.ndarray,            # (Ns+1,)
        observer_pos: Optional[jnp.ndarray] = None,  # (3,)
        rsd_axis: int = 2,
        # Ortho / beta(k,mu,z)
        measure_pk=None,
        k_edges: Optional[jnp.ndarray] = None,       # (Nk+1,)
        mu_edges: Optional[jnp.ndarray] = None,      # (Nmu+1,)
        noise_kmu_mid: jnp.ndarray = None,           # (Ns, Nk, Nmu)
        dtype=jnp.float32,
    ) -> jnp.ndarray:
        """
        Build a light-cone noise field by summing shell-masked snapshot noise fields.

        Internally, this calls get_noise_field(...) for each chi-slice with a shell mask.
        """
        # --- sanity checks ---
        if k_edges is None or mu_edges is None:
            raise ValueError("k_edges, mu_edges are required.")
        if noise_kmu_mid is None:
            raise ValueError("noise_kmu_mid is required.")

        # NOTE:
        # rsd_axis and measure_pk are kept for API compatibility.
        # The current (k, mu) lookup helper is assumed to use the default LOS convention
        # used elsewhere in your code (effectively z-axis for mu = |kz|/|k|).

        white_noise_k = jnp.asarray(white_noise_k, dtype=self.complex_dtype)
        white_noise_k = white_noise_k.at[0, 0, 0].set(0.0)  # force DC=0

        ng_E = int(self.ng_E)
        k_edges = jnp.asarray(k_edges, dtype=self.real_dtype)
        mu_edges = jnp.asarray(mu_edges, dtype=self.real_dtype)

        # --- build and cache rfft-grid lookup once ---
        nearest_idx, Nk, Nmu = utils_jax._build_nearest_idx_for_grid(
            ng_E, self.boxsize, k_edges, mu_edges, dtype=self.real_dtype
        )

        # --- observer position and Eulerian radius^2 grid (cached once) ---
        if observer_pos is None:
            observer_pos = jnp.array([0.0, 0.0, 0.0], dtype=self.real_dtype)
        else:
            observer_pos = jnp.asarray(observer_pos, dtype=self.real_dtype)

        dx = self.boxsize / ng_E
        x1d = (jnp.arange(ng_E, dtype=self.real_dtype)) * dx
        x_rel = (x1d - observer_pos[0])[:, None, None]
        y_rel = (x1d - observer_pos[1])[None, :, None]
        z_rel = (x1d - observer_pos[2])[None, None, :]
        r2_grid = x_rel * x_rel + y_rel * y_rel + z_rel * z_rel  # (ng_E, ng_E, ng_E)

        # --- validate inputs ---
        chi_edges = jnp.asarray(chi_edges, dtype=self.real_dtype)
        Ns = int(chi_edges.shape[0] - 1)

        noise_kmu_mid = jnp.asarray(noise_kmu_mid, dtype=self.real_dtype)
        if noise_kmu_mid.shape[:3] != (Ns, Nk, Nmu):
            raise ValueError(f"noise_kmu_mid must have shape (Ns, Nk, Nmu) = {(Ns, Nk, Nmu)}")

        # --- accumulate shell-masked k-space fields ---
        final_k = jnp.zeros_like(white_noise_k, dtype=self.complex_dtype)

        for s in range(Ns):
            slice_k = self.get_noise_field(
                white_noise_k,
                k_edges=k_edges,
                mu_edges=mu_edges,
                noise_kmu_mid=noise_kmu_mid[s],     # (Nk, Nmu)
                observer_pos=observer_pos,
                chi_range=(chi_edges[s], chi_edges[s + 1]),
                nearest_idx=nearest_idx,            # reuse cached lookup
                r2_grid=r2_grid,                    # reuse cached radius grid
                dtype=dtype,
            )
            final_k = final_k + slice_k

        final_k = final_k.at[0, 0, 0].set(0.0)
        return final_k.astype(self.complex_dtype)



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

        self.kx_E, self.ky_E, self.kz_E = coord.kaxes_1d(self.ng_E, self.boxsize, dtype=self.real_dtype)
        self.kx2_E = self.kx_E * self.kx_E
        self.ky2_E = self.ky_E * self.ky_E
        self.kz2_E = self.kz_E * self.kz_E

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
            Gij_k = coord.apply_Gij_k(delta_k, self.kx_L, self.ky_L, self.kz_L)   # (6, k)
            Gij_r = self._irfftn_vec(Gij_k)                                 # (6, r)
            # indices: 0:xx, 1:xy, 2:xz, 3:yy, 4:yz, 5:zz
            phi2 = (Gij_r[0]*Gij_r[3] + Gij_r[3]*Gij_r[5] + Gij_r[5]*Gij_r[0]
                         - Gij_r[1]**2 - Gij_r[2]**2 - Gij_r[4]**2)
            G2_r = (-2.0 * phi2).astype(self.real_dtype)
            G2_r = G2_r - jnp.mean(G2_r)
            G2_k = self.rfftn(G2_r).astype(self.complex_dtype)
            
        if self.pt_order == 2:
            # 4) \Psi \nabla \delta term (both as real 3-vectors)
            disp1_k = coord.apply_disp_k(delta_k, self.kx_L, self.ky_L, self.kz_L)   # (3, k),
            grad1_k = coord.apply_nabla_k(delta_k, self.kx_L, self.ky_L, self.kz_L)  # (3, k), +i k * \delta

            disp1_r = self._irfftn_vec(disp1_k)  # (3, r)
            grad1_r = self._irfftn_vec(grad1_k)  # (3, r)

            shift2_r = jnp.einsum('iabc,iabc->abc', disp1_r, grad1_r)
            shift2_r = shift2_r - jnp.mean(shift2_r)

            F2_r = (d2_r - shift2_r + (2.0/7.0) * G2_r).astype(self.real_dtype)
            F2_k = self.rfftn(F2_r).astype(self.complex_dtype)
            fields[0] = (fields[0] + F2_k)  # delta_k += F2_k
            fields[0] = fields[0].at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        elif self.pt_order > 2:
            self._ensure_kaxes()
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
        
        K_ij_k = coord.apply_Gij_k(delta_k, self.kx_L, self.ky_L, self.kz_L)  # (6, ng_L, ng_L, ng_L//2+1)
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
            H_ij_r = KK_ij_r - (1./3.)*dK_ij_r
            H_ij_k = self._rfftn_vec(H_ij_r)  # (6, ng_L, ng_L, ng_L//2+1)
            H_ij_k = H_ij_k.at[:,0,0,0].set(0.0).astype(self.complex_dtype)

            T_r = (delta_r*delta_r - 1.5 * K2_r).astype(self.real_dtype)
            T_ij_k = coord.apply_Gij_k(self.rfftn(T_r), self.kx_L, self.ky_L, self.kz_L)  # (6, ng_L, ng_L, ng_L//2+1)
            T_ij_k = coord.apply_traceless(T_ij_k)
            T_ij_k = T_ij_k.at[:,0,0,0].set(0.0).astype(self.complex_dtype)

            fields.extend([dK_ij_k, H_ij_k, T_ij_k])

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

    def gridspt(
            self, 
            delta_k: jnp.ndarray, 
            pt_order: Optional[int] = None, 
            ng: Optional[int] = None,
            ) -> jnp.ndarray:
        r"""
        Eulerian SPT on the ng grid with de-aliasing controlled by (ng, ng_L).

        Strategy:
          - Low-pass the linear field once so it matches the target passband implied by ng.
          - Build higher-order fields; after forming each order, low-pass again before storing.

        Inputs
        ------
        delta_k : (ng, ng, ng//2+1) complex
            Linear density on the large rfftn layout.
        pt_order : int or None
            Maximum order (defaults to self.pt_order).
        ng : int or None
            Passband controller. If None and self.ng is set, use self.ng.
            If neither is set, default to floor(2/3 * ng_L) and warn.

        Returns
        -------
        deltas_r : (pt_order, ng, ng, ng) real
            Real-space fields [delta_1, delta_2, ..., delta_pt] on the ng grid.
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
        #ngL = int(self.ng_L)
        #half = int(ng_eff) // 2
        #idx_xy = jnp.arange(ngL)
        #mx = (idx_xy < half) | (idx_xy >= (ngL - half))          # (ng_L,)
        #mz = jnp.arange(ngL // 2 + 1) <= half                    # (ng_L//2+1,)
        #lpf_mask = (mx[:, None, None] & mx[None, :, None] & mz[None, None, :]).astype(self.real_dtype)
        #lpf_mask_c = lpf_mask.astype(self.complex_dtype)

        # Initial low-pass once for the linear field.
        #del1_k = (delta_k * lpf_mask_c).at[0, 0, 0].set(0.0)
        #the1_k = delta_k  # EdS closure

        delk = [delta_k]                   # list of k-space fields per order
        thek = [delta_k]
        deltas_r = [self.irfftn(delta_k)]  # real-space outputs per order

        # Main recursion for orders >= 2.
        for n in range(2, pt_order + 1):
            Sd_r = jnp.zeros((ng, ng, ng), dtype=self.real_dtype)
            St_r = jnp.zeros_like(Sd_r)

            for m in range(1, n):
                nm = n - m
                Sd_mn, St_mn = self._spt_pair_contrib_bandlimited(
                    delk[m - 1], thek[m - 1],
                    delk[nm - 1], thek[nm - 1],
                )
                Sd_r = Sd_r + Sd_mn
                St_r = St_r + St_mn

            # Closed-form coefficients (Einstein-de Sitter).
            coef = 2.0 / ((2 * n + 3) * (n - 1))
            delta_n_r = coef * ((n + 0.5) * Sd_r + St_r)
            theta_n_r = coef * (1.5 * Sd_r + n * St_r)

            # Post low-pass so the next iteration sees band-limited inputs.
            #delta_n_k = (self.rfftn(delta_n_r) * lpf_mask_c).at[0, 0, 0].set(0.0)
            #theta_n_k = (self.rfftn(theta_n_r) * lpf_mask_c).at[0, 0, 0].set(0.0)

            delta_n_k = self.rfftn(delta_n_r).at[0,0,0].set(0.0)
            theta_n_k = self.rfftn(theta_n_r).at[0,0,0].set(0.0)

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
        idx_diag = jnp.array([0, 3, 5], dtype=jnp.int32)
        idx_off  = jnp.array([1, 2, 4], dtype=jnp.int32)
        du_m_diag_r  = self._irfftn_vec(jnp.take(Gm, idx_diag, axis=0)).astype(self.real_dtype)
        du_nm_diag_r = self._irfftn_vec(jnp.take(Gn, idx_diag, axis=0)).astype(self.real_dtype)
        trace_diag = (du_m_diag_r[0] * du_nm_diag_r[0] +
                      du_m_diag_r[1] * du_nm_diag_r[1] +
                      du_m_diag_r[2] * du_nm_diag_r[2])

        # Off-diagonal components (1,2,4) with factor 2
        du_m_off_r  = self._irfftn_vec(jnp.take(Gm, idx_off, axis=0)).astype(self.real_dtype)
        du_nm_off_r = self._irfftn_vec(jnp.take(Gn, idx_off, axis=0)).astype(self.real_dtype)
        trace_off = 2.0 * (du_m_off_r[0] * du_nm_off_r[0] +
                           du_m_off_r[1] * du_nm_off_r[1] +
                           du_m_off_r[2] * du_nm_off_r[2])

        S_theta = trace_diag + trace_off + jnp.einsum('iabc,iabc->abc', u_m_r, grad_t_nm_r)

        # Remove means for numerical hygiene.
        #S_delta = (S_delta - jnp.mean(S_delta)).astype(self.real_dtype)
        #S_theta = (S_theta - jnp.mean(S_theta)).astype(self.real_dtype)
        return S_delta, S_theta
