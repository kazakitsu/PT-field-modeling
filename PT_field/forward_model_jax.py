#!/usr/bin/env python3
from __future__ import annotations
from typing import Tuple, Callable

import warnings, numpy as np
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial

import PT_field.coord_jax as coord
import lss_utils.assign_util_jax as assign_util
import lss_utils.power_util_jax as power_util

class Base_Forward:
    """Contain only FFT helpers; model-specific subclasses add physics."""

    def __init__(self, *, boxsize: float, ng_L: int):
        self.boxsize: float = float(boxsize)
        self.ng_L:      int = int(ng_L)
        self._norm_L: float = self.ng_L ** 3

        # static JIT batched FFT kernels
        self.b_irfftn = jit(vmap(jnp.fft.irfftn, in_axes=0, out_axes=0))
        self.b_rfftn  = jit(vmap(jnp.fft.rfftn,  in_axes=0, out_axes=0))

        self.kvec_L   = coord.rfftn_kvec((self.ng_L,)*3, self.boxsize)
        self.Gij_k_L  = coord.rfftn_Gij(self.kvec_L)

    # -------- FFT helpers (normalized) -------------------------------
    def irfftn(self, array_k: jnp.ndarray) -> jnp.ndarray:
        return jnp.fft.irfftn(array_k) * self._norm_L

    def rfftn(self, array_r: jnp.ndarray) -> jnp.ndarray:
        return jnp.fft.rfftn(array_r) / self._norm_L

    def batched_irfftn(self, array_k: jnp.ndarray) -> jnp.ndarray:
        return self.b_irfftn(array_k) * self._norm_L

    def batched_rfftn(self, array_r: jnp.ndarray) -> jnp.ndarray:
        return self.b_rfftn(array_r) / self._norm_L

# ---------------------------------------------------------------------
# LPT forward model (1LPT)  -------------------------------------------
# ---------------------------------------------------------------------

class LPT_Forward(Base_Forward):
    """Compute shifted fields using LPT displacement."""

    def __init__(self,
                 *,
                 boxsize: float,
                 ng_L: int,
                 ng_E: int,
                 mas_cfg: Tuple[int, bool],
                 rsd: bool = False,
                 lya: bool = False,
                 lpt_order: int = 1,
                 bias_order: int = 2,
                ) -> None:
        super().__init__(boxsize=boxsize, ng_L=ng_L)

        # Eulerian grid & MAS
        self.ng_E = int(ng_E)
        self.window_order, self.interlace = mas_cfg
        self.mesh = assign_util.Mesh_Assignment(self.boxsize, self.ng_E, 
                                                self.window_order,
                                                interlace=self.interlace)
        self.rsd = bool(rsd)
        self.lya = bool(lya)
        self.lpt_order = int(lpt_order)
        self.bias_order = int(bias_order)

        # Lagrangian grid positions
        cell_size = self.boxsize / self.ng_L
        self.pos_q_L = (jnp.indices((self.ng_L,) * 3, ) * cell_size)

        # Helper ternsors
        self.mu2_L    = coord.rfftn_mu2(self.kvec_L)
        self.disp_k_L = coord.rfftn_disp(self.kvec_L)

    # -------- LPT displacement ----------------------------------------
    @partial(jit, static_argnames=('self'))
    def lpt(self, delta_k_L: jnp.ndarray, growth_f: float=0.0) -> jnp.ndarray:
        """Return displaced positions"""
        disp1_k = self.disp_k_L * delta_k_L
        disp1_r = self.batched_irfftn(disp1_k)
        if self.rsd:  ### LOS: z-axis
            disp1_r = disp1_r.at[2].add(disp1_r[2] * growth_f)
        pos_x   = self.pos_q_L + disp1_r
        return pos_x
    
    # -------- scalar displacement ----------------------------------------
    @partial(jit, static_argnames=('self'))
    def _scalar_fields_r(
        self, delta_k: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute scalar fields in position space.
        """
        delta_r = self.irfftn(delta_k)
        out = [delta_r,]

        if self.bias_order >= 2:
            d2_r = delta_r**2 - jnp.mean(delta_r**2)

            Gij_r = self.batched_irfftn(self.Gij_k_L * delta_k)
            ### 0:xx, 1:xy, 2:xz, 3:yy, 4:yz, 5:zz
            phi2  = (
                Gij_r[0] * Gij_r[3] + Gij_r[3] * Gij_r[5] + Gij_r[5] * Gij_r[0]
                - Gij_r[1]**2 - Gij_r[2]**2 - Gij_r[4]**2
            )
            G2_r = -2.0 * phi2 - jnp.mean(-2.0 * phi2)

            out.extend([d2_r, G2_r])

        # RSD term: G2_zz
        if (self.rsd or self.lya) and self.bias_order >= 2:
            G2_zz_r = self.irfftn(self.rfftn(G2_r) * self.mu2_L)
            G2_zz_r = G2_zz_r - jnp.mean(G2_zz_r)
            out.append(G2_zz_r)

        # Ly-A terms
        if self.lya:
            #eta_r   = self.irfftn(delta_k * self.mu2_L)
            ### eta = delta * mu^2 = Gzz
            eta_r   = Gij_r[5]
            #out.append(eta_r)

            if self.bias_order >= 2:
                eta2_r  = eta_r**2 - jnp.mean(eta_r**2)

                deta_r  = delta_r * eta_r - jnp.mean(delta_r * eta_r)

                GG_zz_r = Gij_r[2]**2 + Gij_r[4]**2 + Gij_r[5]**2
                GG_zz_r = GG_zz_r - jnp.mean(GG_zz_r)

                KK_zz_r = GG_zz_r - 2./3.*deta_r + 1./9.*d2_r
                KK_zz_r = KK_zz_r - jnp.mean(KK_zz_r)

                out.extend([eta2_r, deta_r, KK_zz_r])

        return tuple(out)
    
    # -------- linear fields -------------------------------------------
    @partial(jit, static_argnames=('self'))
    def eta_r(self, delta_k_L):
        eta_k = self.eta_k(delta_k_L)
        return self.irfftn(eta_k)
    
    @partial(jit, static_argnames=('self'))
    def eta_k(self, delta_k_L):
        return delta_k_L * self.mu2_L

    # -------- quadratic fields ----------------------------------------
    @partial(jit, static_argnames=('self'))
    def d2_r(self, delta_k_L):
        delta_r = self.irfftn(delta_k_L)
        d2_r = delta_r ** 2
        return d2_r - jnp.mean(d2_r)
    
    @partial(jit, static_argnames=('self'))
    def d2_k(self, delta_k_L):
        d2_r = self.d2_r(delta_k_L)
        return self.rfftn(d2_r)

    @partial(jit, static_argnames=('self'))
    def G2_r(self, delta_k_L):
        Gij_r = self.batch_irfftn(self.Gij_k_L * delta_k_L)
        phi2_r = (Gij_r[0]*Gij_r[3] + Gij_r[3]*Gij_r[5] + Gij_r[5]*Gij_r[0]
                   - Gij_r[1]**2 - Gij_r[2]**2 - Gij_r[4]**2)
        G2_r  = -2.0 * phi2_r
        return G2_r - jnp.mean(G2_r)
    
    @partial(jit, static_argnames=('self'))
    def G2_k(self, delta_k_L):
        G2_r = self.G2_r(delta_k_L)
        return self.rfftn(G2_r)
    
    @partial(jit, static_argnames=('self'))
    def G2_zz_r(self, delta_k_L):
        G2_zz_k = self.G2_zz_k(delta_k_L)
        return self.irfftn(G2_zz_k)
    
    @partial(jit, static_argnames=('self'))
    def G2_zz_k(self, delta_k_L):
        G2_r = self.G2_r(delta_k_L)
        return self.rfftn(G2_r) * self.mu2_L
    
    # -------- for LyA ---------------------------------------------
    @partial(jit, static_argnames=('self'))
    def eta2_r(self, delta_k_L):
        eta_r = self.eta_r(delta_k_L)
        eta2_r = eta_r ** 2
        return eta2_r - jnp.mean(eta2_r)
    
    @partial(jit, static_argnames=('self'))
    def eta2_k(self, delta_k_L):
        eta2_r = self.eta2_r(delta_k_L)
        return self.rfftn(eta2_r)
    
    @partial(jit, static_argnames=('self'))
    def deta_r(self, delta_k_L):
        delta_r = self.irfftn(delta_k_L)
        eta_r = self.eta_r(delta_k_L)
        deta_r = delta_r * eta_r
        return deta_r - jnp.mean(deta_r)
    
    @partial(jit, static_argnames=('self'))
    def deta_k(self, delta_k_L):
        deta_r = self.deta_r(delta_k_L)
        return self.rfftn(deta_r)
    
    @partial(jit, static_argnames=('self'))
    def GG_zz_r(self, delta_k_L):
        Gij_r = self.batch_irfftn(self.Gij_k_L * delta_k_L)
        GG_zz_r = Gij_r[2]**2 + Gij_r[4]**2 + Gij_r[5]**2
        return GG_zz_r - jnp.mean(GG_zz_r)
    
    @partial(jit, static_argnames=('self'))
    def GG_zz_k(self, delta_k_L):
        GG_zz_r = self.GG_zz_r(delta_k_L)
        return self.rfftn(GG_zz_r)
    
    @partial(jit, static_argnames=('self'))
    def KK_zz_r(self, delta_k_L):
        GG_zz_r = self.GG_zz_r(delta_k_L)
        KK_zz_r = GG_zz_r - 2./3.*self.deta_r(delta_k_L) + 1./9.*self.d2_r(delta_k_L)
        return KK_zz_r - jnp.mean(KK_zz_r)
    
    @partial(jit, static_argnames=('self'))
    def KK_zz_k(self, delta_k_L):
        KK_zz_r = self.KK_zz_r(delta_k_L)
        return self.rfftn(KK_zz_r)
    
    # -------- shifted field generators --------------------------------
    def _stack_for_shift(self, scalars: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        """Build stacked array depending on options."""
        if self.lya:    # 1, d, d2, G2, G2_zz, deta, eta2, KK_zz
            ones = jnp.ones_like(scalars[0])
            return jnp.stack((ones,) + scalars, axis=0)
        if self.rsd:    # 1, d, d2, G2, G2_zz
            ones = jnp.ones_like(scalars[0])
            return jnp.stack((ones,) + scalars[:4], axis=0)
        else:           # d, d2, G2
            return jnp.stack(scalars[:3], axis=0)               

    def _apply_mesh(
        self,
        pos: jnp.ndarray,
        fields_r: jnp.ndarray,
        fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ) -> jnp.ndarray:
        return jnp.array(vmap(fn, in_axes=(None, 0))(pos, fields_r))
    
    @partial(jit, static_argnames=('self', 'mode'))
    def get_shifted_fields(
        self,
        delta_k: jnp.ndarray,
        *,
        growth_f: float = 0.0,
        mode: str = 'k_space',          # "k_space" or "r_space"
    ) -> jnp.ndarray:
        """Return shifted fields on Eulerian grid (k- or r-space)."""
        delta_k_L = coord.func_extend(self.ng_L, delta_k)
        pos_x   = self.lpt(delta_k_L, growth_f=growth_f)
        scalars = self._scalar_fields_r(delta_k_L)
        stacked = self._stack_for_shift(scalars)

        assign = (self.mesh.assign_fft if mode == 'k_space'
                  else self.mesh.assign_to_grid)
        return self._apply_mesh(pos_x, stacked, assign)


    @partial(jit, static_argnames=('self', 'interp_mode'))
    def get_final_field(
        self,
        fields_k: jnp.ndarray,            # shape (n_fields, ng, ng, ng//2+1)
        beta_tab: jnp.ndarray,            # shape (n_fields, Nk, Nmu)
        *,
        k_edges: jnp.ndarray,             # (Nk+1,)
        mu_edges: jnp.ndarray,            # (Nmu+1,)
        interp_mode: str = "nearest",
    ) -> jnp.ndarray:
        """
        Build delta_g(k) = sum_i beta_i(k,mu) O_i(k).

        Notes
        -----
        * Each beta_i(k, mu) is supplied on the (Nk, Nmu) grid; it is first
          interpolated to the Fourier grid via `Interpolator`.
        * `fields_k` must share the same rfftn grid layout.
        * Returns an rfftn-layout array with identical shape to every
          `fields_k[i]`.
        """
        # Create (k, \mu) arrays on the Fourier grid -----------
        ng_E = fields_k.shape[1]                      # grid size
        if ng_E != self.ng_E:
            raise ValueError(f"fields_k must have ng_E={self.ng_E}, got {ng_E}.")
        kmag = jnp.sqrt((self.mesh.kvec**2).sum(axis=0))
        mu   = jnp.abs(self.mesh.kvec[2]) / jnp.where(kmag > 0, kmag, 1.)

        # Build interpolator and map over first axis ---------
        interp = Interpolator(kmag, mu, k_edges, mu_edges,
                              mode=interp_mode)
        # beta_full has same shape as fields_k
        beta_full = vmap(interp)(beta_tab)          # (n_fields, grid)

        # Weighted sum over the field index -----------------
        delta_g_k = jnp.sum(beta_full * fields_k, axis=0)  # grid
        return delta_g_k


    
@partial(jit, static_argnames=('measure_pk','interp_mode','Nmin','jitter'))
def orthogonalize(
    fields: jnp.ndarray,            # shape (n, ng,ng,ng/2+1)
    measure_pk,                     # instance of Measure_Pk（static）
    boxsize: float,
    k_edges: jnp.ndarray,
    mu_edges: jnp.ndarray,
    interp_mode: str = 'nearest',
    Nmin: int = 5,
    jitter: float = 1e-14,
):
    n = fields.shape[0]
    Nk, Nmu = k_edges.size - 1, mu_edges.size - 1

    # ---------- P_auto, P_cross, N_modes -------------
    # P_auto[:,m,i], N_modes[:,m], P_cross[:,m,i,j]
    P_auto  = jnp.zeros((Nk, Nmu, n))
    P_cross = jnp.zeros((Nk, Nmu, n, n))
    N_modes = jnp.zeros((Nk, Nmu))

    for m in range(Nmu):
        mu0, mu1 = mu_edges[m], mu_edges[m+1]
        # auto
        for i in range(n):
            res = measure_pk(fields[i], None, ell=0, mu_min=mu0, mu_max=mu1)
            P_auto = P_auto.at[:, m, i].set(res[:,1])
            N_modes = N_modes.at[:, m].set(res[:,2])
        # cross
        for j in range(n):
            for i in range(j):
                res = measure_pk(fields[i], fields[j], ell=0, mu_min=mu0, mu_max=mu1)
                Pij = res[:,1]
                P_cross = P_cross.at[:, m, i, j].set(Pij)
                P_cross = P_cross.at[:, m, j, i].set(Pij)
    P_cross = P_cross.at[..., jnp.arange(n), jnp.arange(n)].set(P_auto)

    # ---------- check low stats --------------------------
    low_stat = N_modes <= Nmin

    # ---------- Corr -> interp -> Cholesky -> Mcoef -------------
    denom = jnp.sqrt(P_auto[..., None] * P_auto[..., None,:])
    Corr  = jnp.where(denom>0, P_cross/denom, 0.)
    Corr  = Corr.at[..., jnp.arange(n), jnp.arange(n)].set(1.)
    Corr  = _nearest_fill_generic(Corr, low_stat, Nk, Nmu)

    eye = jnp.eye(n)
    L   = vmap(lambda A: jnp.linalg.cholesky(A + jitter*eye))(
               Corr.reshape(-1, n, n))
    Linv = vmap(jnp.linalg.inv)(L).reshape(Nk, Nmu, n, n)

    diag  = jnp.diagonal(Linv, axis1=2, axis2=3)
    ratio = jnp.sqrt(P_auto[...,None] / P_auto[...,None,:])
    Mcoef = (Linv / diag[..., :, None]) * ratio
    Mcoef = _nearest_fill_generic(Mcoef, low_stat, Nk, Nmu)

    # ---------- generate k–mu grid -------------------------
    ng = fields.shape[1]
    kvec = power_util.rfftn_kvec((ng,)*3, boxsize)
    kmag = jnp.sqrt((kvec**2).sum(axis=0))
    mu   = jnp.abs(kvec[2]) / jnp.where(kmag>0, kmag, 1.)

    interp = Interpolator(kmag, mu, k_edges, mu_edges, mode=interp_mode)

    # Gram–Schmidt
    ortho = list(fields)
    for j in range(1, n):
        f = fields[j]
        for i in range(j):
            coef_ji = Mcoef[..., j, i]          # (Nk,Nmu)
            Mgrid   = interp(coef_ji)           # (...grid...)
            f       = f + Mgrid * fields[i]
        ortho[j] = f

    #return P_cross, Corr, Mcoef, ortho
    return jnp.array(ortho)


class Interpolator:
    def __init__(self, kmag, mu, k_edges, mu_edges, mode: str = 'nearest'):
        self.mode = mode
        Nk, Nmu = k_edges.size - 1, mu_edges.size - 1
        # bin indices
        kbin  = jnp.clip(jnp.searchsorted(k_edges,  kmag, side="right") - 1, 0, Nk-1)
        mubin = jnp.clip(jnp.searchsorted(mu_edges, mu,   side="right") - 1, 0, Nmu-1)
        if mode == 'nearest':
            self.idx = kbin * Nmu + mubin
        else:
            kbin1  = jnp.clip(kbin + 1, 0, Nk-1)
            mubin1 = jnp.clip(mubin + 1, 0, Nmu-1)
            k0, k1 = k_edges[kbin],  k_edges[kbin1]
            m0, m1 = mu_edges[mubin], mu_edges[mubin1]
            self.tk = jnp.where(k1>k0, (kmag - k0)/(k1-k0), 0.)
            self.tm = jnp.where(m1>m0, (mu   - m0)/(m1-m0), 0.)
            flat_idx = lambda kb, mb: kb * Nmu + mb
            self.i00 = flat_idx(kbin,  mubin)
            self.i10 = flat_idx(kbin1, mubin)
            self.i01 = flat_idx(kbin,  mubin1)
            self.i11 = flat_idx(kbin1, mubin1)

    def __call__(self, table):
        Nk, Nmu = table.shape[:2]
        flat = table.reshape(Nk * Nmu, *table.shape[2:])

        if self.mode == 'nearest':
            return flat[self.idx]
        
        M00 = flat[self.i00]; M10 = flat[self.i10]
        M01 = flat[self.i01]; M11 = flat[self.i11]
        tk, tm = self.tk, self.tm
        return ((1-tk)*(1-tm))*M00 \
             + ((    tk)*(1-tm))*M10 \
             + ((1-tk)*(    tm))*M01 \
             + ((    tk)*(    tm))*M11



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
    repl = vmap(choose)(bad)
    repl = jnp.where(bad >= 0, repl, bad)
    flat_out = flat.at[bad].set(flat[repl])
    return flat_out.reshape(data2d.shape)

