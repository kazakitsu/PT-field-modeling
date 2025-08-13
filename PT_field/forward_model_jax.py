#!/usr/bin/env python3
from __future__ import annotations
from typing import Tuple, Callable, Literal

import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial

import PT_field.coord_jax as coord
import lss_utils.assign_util_jax as assign_util

# ------------------------------ Base class ------------------------------

class Base_Forward:
    """Only FFT helpers & k-axis; model-specific subclasses add physics."""

    def __init__(self, *, boxsize: float, ng_L: int, 
                 dtype=jnp.float32,
                 use_batched_fft: bool = True):
        self.boxsize: float = float(boxsize)
        self.ng_L:      int = int(ng_L)
        self.real_dtype = jnp.dtype(dtype)
        self.complex_dtype = jnp.complex64 if self.real_dtype == jnp.float32 else jnp.complex128
        self.use_batched_fft = bool(use_batched_fft)

        # vmap'ed FFT helpers with forward normalization (unitary inverse)
        self.b_irfftn = jit(vmap(partial(jnp.fft.irfftn, norm='forward'), in_axes=0, out_axes=0))
        self.b_rfftn  = jit(vmap(partial(jnp.fft.rfftn,  norm='forward'), in_axes=0, out_axes=0))

        # Keep only 1D k-axes to avoid captured 3D constants
        self.kx, self.ky, self.kz = coord.kaxes_1d(self.ng_L, self.boxsize, dtype=self.real_dtype)

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
        """
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
        """
        array_r: (m, ng, ng, ng) -> (m, ng, ng, ng//2+1) by RFFT
        """
        if self.use_batched_fft:
            return self.b_rfftn(array_r).astype(self.complex_dtype)
        else:
            outs = []
            for i in range(array_r.shape[0]):
                outs.append(self.rfftn(array_r[i]))
            return jnp.stack(outs, axis=0)

# ------------------------------ LPT (1LPT) ------------------------------

class LPT_Forward(Base_Forward):
    """
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
        super().__init__(boxsize=boxsize, ng_L=ng_L, dtype=dtype, use_batched_fft=use_batched_fft)

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

    # -------- scalar fields in real space -------
    @partial(jit, static_argnames=('self',))
    def _scalar_fields_r(self, delta_k: jnp.ndarray) -> jnp.ndarray:
        """
        Build scalar fields in real space: [1, delta, d^2, G2, (G2_zz), (LyA extras...)]
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
    
    def _apply_mesh_from_disp(self, disp_r, fields_r, fn_from_disp):
        return jnp.array(vmap(fn_from_disp, in_axes=(None, 0))(disp_r, fields_r))
    
    def _assign_fields_from_disp_to_grid(
        self,
        disp_r: jnp.ndarray,      # (3, ng_L, ng_L, ng_L)
        fields_r: jnp.ndarray,    # (m, ng_L, ng_L, ng_L)
        *,
        interlace: bool = False,
        normalize_mean: bool = True,
    ) -> jnp.ndarray:
        """
        Assign many real-space fields to Eulerian grid using either `for` or `vmap`.
        Returns stacked real grids: (m, ng_E, ng_E, ng_E).
        """
        m = int(fields_r.shape[0])
        mode = self._get_assign_mode(m)

        if mode == "vmap":
            fn = lambda w: self.mesh.assign_from_disp_to_grid(
                disp_r, w, interlace=interlace, normalize_mean=normalize_mean
            )
            # vmap over field axis
            fields = vmap(fn, in_axes=0, out_axes=0)(fields_r)
        else:
            fields = []
            for i in range(m):
                w = fields_r[i]
                gi = self.mesh.assign_from_disp_to_grid(
                    disp_r, w, interlace=interlace, normalize_mean=normalize_mean
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
        neighbor_mode: str = 'auto',
        fuse_updates_threshold: int=100_000_000,
    ) -> jnp.ndarray:
        """
        Build displacement (1LPT) and scalar real-space fields, assign them to Eulerian grid,
        and (optionally) FFT+deconvolve them in a single batched call.

        Returns:
          - mode=='k_space' : (n_fields, ng_E, ng_E, ng_E//2+1) complex
          - else            : (n_fields, ng_E, ng_E, ng_E)       real
        """
        delta_k = delta_k.at[0,0,0].set(0.0).astype(self.complex_dtype)  # Ensure delta_k[0] = 0.0
        delta_k_L = coord.func_extend(self.ng_L, delta_k)

        # lpt displacement
        disp_r_L = self.lpt(delta_k_L, growth_f=growth_f)  # (3, ng, ng, ng)

        # list of scalar fields in position space
        fields_r_L = self._scalar_fields_r(delta_k_L)    # (m, ng_L, ng_L, ng_L)

        # assign to Eularian grid (and FFT/deconv)
        if mode == 'k_space':
            # Build non-interlaced and (optionally) interlaced stacks
            fields_E_r   = self._assign_fields_from_disp_to_grid(disp_r_L, fields_r_L,
                                                               interlace=False, normalize_mean=True,
                                                               neighbor_mode=neighbor_mode,
                                                               fuse_updates_threshold=fuse_updates_threshold)
            fields_E_ri  = None
            if self.mesh.interlace:
                fields_E_ri = self._assign_fields_from_disp_to_grid(disp_r_L, fields_r_L,
                                                                  interlace=True, normalize_mean=True,
                                                                  neighbor_mode=neighbor_mode,
                                                                  fuse_updates_threshold=fuse_updates_threshold)
            # Single batched FFT + deconvolution (provided by Mesh_Assignment)
            fields_k = self.mesh.fft_deconvolve_batched(fields_E_r, fields_E_ri)
            return fields_k.astype(self.complex_dtype)
        else:
            # Real-space: return non-interlaced assigned grids
            fields_E_r = self._assign_fields_from_disp_to_grid(disp_r_L, fields_r_L,
                                                             interlace=False, normalize_mean=True,
                                                             neighbor_mode=neighbor_mode,
                                                             fuse_updates_threshold=fuse_updates_threshold)
            return fields_E_r.astype(self.real_dtype)
        
    @partial(jit, static_argnames=('self', 'measure_pk'))
    def get_beta(
        self,
        true_field_k: jnp.ndarray,     # (..., ng, ng, ng//2+1), complex
        fields_k: jnp.ndarray,         # (n, ng, ng, ng//2+1), complex
        mu_edges: jnp.ndarray,         # (Nmu+1,)
        *,
        measure_pk,                    # Measure_Pk instance (static)
        eps: float = 1e-20,
    ) -> jnp.ndarray:
        """
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
        """
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
    def _get_final_field(
        self,
        fields_k: jnp.ndarray,            # (n_fields, ng_E, ng_E, ng_E//2+1)
        beta_tab: jnp.ndarray,            # (n_fields, Nk, Nmu)
    ) -> jnp.ndarray:
        """
        Build delta_g(k) = sum_i beta_i[bin(k,mu)] * O_i(k) using a precomputed nearest map.
        The accumulation streams across the field-axis to keep peak memory low.
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

    def get_final_field(
        self,
        fields_k: jnp.ndarray,            # (n_fields, ng, ng, ng//2+1)
        beta_tab: jnp.ndarray,            # (n_fields, Nk, Nmu)
        *,
        k_edges: jnp.ndarray,             # (Nk+1,)
        mu_edges: jnp.ndarray,            # (Nmu+1,)
    ) -> jnp.ndarray:
        """Build delta_g(k) = sum_i beta_i(k,mu) O_i(k) with on-the-fly (k,mu)."""
        ng_E = fields_k.shape[1]
        if ng_E != self.ng_E:
            raise ValueError(f"fields_k must have ng_E={self.ng_E}, got {ng_E}.")

        # Build/reuse the grid -> (k,mu) bin mapping (host-side; no JIT cost).
        self._ensure_nearest_cache(k_edges, mu_edges)
        # Run the jitted accumulation kernel.
        return self._get_final_field(fields_k, beta_tab)

# -------------------- Orthogonalize & interpolation utilities --------------------

from jax import jit, vmap, lax
import jax.numpy as jnp

@partial(jit, static_argnames=('measure_pk','interp_mode','Nmin','jitter','dtype'))
def orthogonalize(
    fields: jnp.ndarray,            # (n, ng, ng, ng//2+1)
    measure_pk,                     # Measure_Pk instance (self is static)
    boxsize: float,
    k_edges: jnp.ndarray,           # (Nk+1,)
    mu_edges: jnp.ndarray,          # (Nmu+1,)
    interp_mode: str = 'nearest',
    Nmin: int = 5,
    jitter: float = 1e-14,
    dtype=jnp.float32,
):
    """
    Orthonormalize Fourier-space fields per (k,mu) bin using Cholesky on the correlation matrix.

    Speed & memory optimizations:
      * Vectorize power-spectrum evaluation across mu-bins with `vmap` to reduce Python overhead.
      * Compute L^{-1} via a triangular solve (no dense matrix inverse).
      * Batch updates when writing the cross-power upper triangle.
    """
    n   = fields.shape[0]
    Nk  = k_edges.size - 1
    Nmu = mu_edges.size - 1
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

    # Allocate outputs
    P_auto  = jnp.zeros((Nk, Nmu, n),     dtype=dtype)
    P_cross = jnp.zeros((Nk, Nmu, n, n),  dtype=dtype)

    # Autos: evaluate for each field; grab N_modes from the first field (identical for all)
    P0, N_modes = _pk_mu_all(fields[0], None)
    P_auto = P_auto.at[:, :, 0].set(P0)
    # Loop remaining autos; Python loop is cheap (vmap inside handles the heavy lifting)
    for i in range(1, n):
        Pi, _ = _pk_mu_all(fields[i], None)
        P_auto = P_auto.at[:, :, i].set(Pi)

    # Cross (upper triangle): for each fixed j, vectorize over i<j
    for j in range(n):
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
    diag = jnp.arange(n)
    P_cross = P_cross.at[..., diag, diag].set(P_auto)

    # ---- 2) Correlation matrix per (k,mu) & Cholesky -------------------------
    low_stat = N_modes <= Nmin

    # Corr = P_cross / sqrt(P_auto_i * P_auto_j)
    denom = jnp.sqrt(jnp.clip(P_auto[..., None] * P_auto[..., None, :], a_min=0.0))
    Corr  = jnp.where(denom > 0, P_cross / denom, 0.0)
    Corr  = Corr.at[..., diag, diag].set(1.0)
    Corr  = _nearest_fill_generic(Corr, low_stat, Nk, Nmu)  # fill low-stat bins

    # Cholesky per (Nk,Nmu) with a small jitter on the diagonal for stability
    eye = jnp.eye(n, dtype=dtype)

    # vmap over flattened (Nk*Nmu) bins to keep compile size reasonable
    Corr_flat = Corr.reshape(Nk * Nmu, n, n)
    def _chol_inv_lower(A):
        # Compute L = chol(A + jitter*I) and solve L * X = I to get L^{-1}
        L = jnp.linalg.cholesky(A + jitter * eye)
        Linv = lax.linalg.triangular_solve(L, eye, left_side=True, lower=True)
        return Linv
    Linv = vmap(_chol_inv_lower)(Corr_flat).reshape(Nk, Nmu, n, n)  # (Nk,Nmu,n,n)

    # Build Mcoef = (Linv / diag(Linv)) * sqrt(P_auto_i / P_auto_j)
    diagL = jnp.diagonal(Linv, axis1=2, axis2=3)                            # (Nk,Nmu,n)
    ratio = jnp.sqrt(jnp.where(P_auto[..., None, :] > 0,
                               P_auto[..., None] / jnp.maximum(P_auto[..., None, :], 1e-30),
                               0.0))                                         # (Nk,Nmu,n,n)
    Mcoef = (Linv / diagL[..., :, None]) * ratio                             # (Nk,Nmu,n,n)
    Mcoef = _nearest_fill_generic(Mcoef, low_stat, Nk, Nmu)                  # fill low-stat bins

    # ---- 3) Expand Mcoef(k,mu) to grid and perform sequential orthogonalization ----
    ng = fields.shape[1]
    # Build (k,mu) once for the target grid of size (ng,ng,ng//2+1)
    kx, ky, kz = coord.kaxes_1d(ng, boxsize, dtype=dtype)
    kx2, ky2, kz2 = kx * kx, ky * ky, kz * kz
    k2   = kx2[:, None, None] + ky2[None, :, None] + kz2[None, None, :]
    kmag = jnp.sqrt(jnp.maximum(k2, 0.0))
    mu   = jnp.where(k2 > 0, jnp.sqrt(kz2[None, None, :] / k2), 0.0)

    # Construct the interpolator once per call; reused for all (j,i)
    interp = Interpolator(kmag, mu, k_edges, mu_edges, mode=interp_mode)

    # Sequential Gram-Schmidt-like update using (k,mu)-dependent mixing
    ortho = list(fields)
    for j in range(1, n):
        f = ortho[j]
        for i in range(j):
            coef_ji = Mcoef[:, :, j, i]     # (Nk, Nmu)
            Mgrid   = interp(coef_ji)       # (...grid...)
            f       = f + Mgrid * fields[i]
        ortho[j] = f

    return jnp.array(ortho)

class Interpolator:
    def __init__(self, kmag, mu, k_edges, mu_edges, mode: str = 'nearest'):
        self.mode = mode
        Nk, Nmu = k_edges.size - 1, mu_edges.size - 1
        kbin  = jnp.clip(jnp.searchsorted(k_edges,  kmag, side="right") - 1, 0, Nk-1)
        mubin = jnp.clip(jnp.searchsorted(mu_edges, mu,   side="right") - 1, 0, Nmu-1)
        if mode == 'nearest':
            self.idx = kbin * Nmu + mubin
        else:
            kbin1  = jnp.clip(kbin + 1, 0, Nk-1)
            mubin1 = jnp.clip(mubin + 1, 0, Nmu-1)
            k0, k1 = k_edges[kbin],  k_edges[kbin1]
            m0, m1 = mu_edges[mubin], mu_edges[mubin1]
            self.tk = jnp.where(k1 > k0, (kmag - k0) / (k1 - k0), 0.0)
            self.tm = jnp.where(m1 > m0, (mu   - m0) / (m1 - m0), 0.0)
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
        return ((1 - tk) * (1 - tm)) * M00 \
             + ((    tk) * (1 - tm)) * M10 \
             + ((1 - tk) * (    tm)) * M01 \
             + ((    tk) * (    tm)) * M11

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
