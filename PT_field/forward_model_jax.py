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
        
    def _assign_fields_from_disp_to_grid(
        self,
        disp_r: jnp.ndarray,      # (3, ng_L, ng_L, ng_L)
        fields_r: jnp.ndarray,    # (m, ng_L, ng_L, ng_L)
        *,
        interlace: bool = False,
        normalize_mean: bool = True,
        neighbor_mode: str = 'auto',
        fuse_updates_threshold: int = 100_000_000,
    ) -> jnp.ndarray:
        """
        Assign many real-space fields to Eulerian grid using either `for` or `vmap`.
        Returns stacked real grids: (m, ng_E, ng_E, ng_E).
        """
        m = int(fields_r.shape[0])
        mode = self._get_assign_mode(m)

        if mode == "vmap":
            fn = lambda w: self.mesh.assign_from_disp_to_grid(
                disp_r, w, interlace=interlace, normalize_mean=normalize_mean, 
                neighbor_mode=neighbor_mode, fuse_updates_threshold=fuse_updates_threshold,
            )
            # vmap over field axis
            fields = vmap(fn, in_axes=0, out_axes=0)(fields_r)
        else:
            fields = []
            for i in range(m):
                w = fields_r[i]
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
    def _get_final_field_table(
        self,
        fields_k: jnp.ndarray,            # (n_fields, ng_E, ng_E, ng_E//2+1)
        beta_tab: jnp.ndarray,            # (n_fields, Nk, Nmu)
    ) -> jnp.ndarray:
        """
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
        """
        Constant coefficients: delta_g = sum_i beta[i] * O_i(k, mu).
        """
        n = fields_k.shape[0]
        beta_const = jnp.asarray(beta_const, dtype=self.real_dtype)

        acc0 = jnp.zeros_like(fields_k[0])

        def loop(i, acc):
            return acc + beta_const[i] * fields_k[i]
        return lax.fori_loop(0, n, loop, acc0)
    
    @partial(jit, static_argnames=('self',))
    def _get_final_field_poly(
        self,
        fields_k: jnp.ndarray,           # (n_fields, ng, ng, ng//2+1), complex
        coeffs: jnp.ndarray,             # (n_fields, Lk, Lmu), real
        k_pows: jnp.ndarray,             # (Lk,), int (exponents for k)
        mu2_pows: jnp.ndarray,            # (Lmu,), int
    ) -> jnp.ndarray:
        """
        Polynomial coefficients on (k,mu):
            beta_i(k,mu) = sum_{a,b} coeffs[i,a,b] * (k**k_pows[a]) * (mu2**mu2_pows[b]),
        """
        # Build k^2 and mu on the Eulerian rfft grid
        k2  = (self.kx2E[:, None, None] + self.ky2E[None, :, None] + self.kz2E[None, None, :]).astype(self.real_dtype)
        mu2  = jnp.where(k2 > 0, self.kz2E[None, None, :] / k2, 0.0).astype(self.real_dtype)
        base_k = jnp.sqrt(jnp.maximum(k2, 0.0)).astype(self.real_dtype)

        # Precompute required powers for base_k and mu2:
        # stacks: Kpow[a] = base_k ** k_pows[a],  Mpow[b] = mu2 ** mu2_pows[b]
        Lk = k_pows.shape[0]
        Lm = mu2_pows.shape[0]

        def build_pows(base, exps):
            # Allocate (L, grid) and fill with base**exps[i]
            out0 = jnp.zeros((exps.shape[0],) + base.shape, dtype=self.real_dtype)
            def body(i, acc):
                e = exps[i]
                val = jnp.where(e == 0, jnp.ones_like(base), base ** e)
                return acc.at[i].set(val)
            return lax.fori_loop(0, exps.shape[0], body, out0)

        Kpow = build_pows(base_k, k_pows.astype(jnp.int32))   # (Lk, grid)
        Mpow = build_pows(mu2,    mu2_pows.astype(jnp.int32))  # (Lmu, grid)

        n_fields = fields_k.shape[0]
        coeffs = coeffs.astype(self.real_dtype)

        acc0 = jnp.zeros_like(fields_k[0])  # complex

        def loop_field(i, acc):
            # Build beta_i(k,mu) on the fly to avoid storing a full (grid) per field.
            # beta_i = sum_{a,b} coeffs[i,a,b] * Kpow[a] * Mpow[b]
            coef_i = coeffs[i]  # (Lk, Lmu)

            # Accumulate over a, b with two nested loops.
            beta_grid0 = jnp.zeros_like(base_k)  # real
            def loop_a(a, beta_acc):
                Ka = Kpow[a]  # grid
                def loop_b(b, beta_acc2):
                    Mb = Mpow[b]
                    return beta_acc2 + coef_i[a, b] * Ka * Mb
                beta_acc = lax.fori_loop(0, Lm, loop_b, beta_acc)
                return beta_acc
            beta_i_grid = lax.fori_loop(0, Lk, loop_a, beta_grid0)  # real, shape=grid

            return acc + beta_i_grid * fields_k[i]  # complex

        return lax.fori_loop(0, n_fields, loop_field, acc0)

    def get_final_field(
        self,
        fields_k: jnp.ndarray,        # (n_fields, ng, ng, ng//2+1)
        beta: jnp.ndarray,            # see beta_kind below
        *,
        beta_kind: Literal['table','const','poly'] = 'table',
        k_edges: jnp.ndarray | None = None,   # required for beta_kind='table'
        mu_edges: jnp.ndarray | None = None,  # required for beta_kind='table'
        poly_k_pows: jnp.ndarray | None = None,   # (Lk,), for beta_kind='poly'
        poly_mu2_pows: jnp.ndarray | None = None,  # (Lmu,), for beta_kind='poly'
    ) -> jnp.ndarray:
        """
        Build delta_g(k, mu) = sum_i beta_i(k,mu) * O_i(k,mu) with three beta modes:

        beta_kind='table':
            beta: (n_fields, Nk, Nmu), nearest-bin lookup using (k_edges, mu_edges).

        beta_kind='const':
            beta: (n_fields,), constant per-field weights.

        beta_kind='poly':
            beta: (n_fields, Lk, Lmu) coefficients for a polynomial in k and mu.
                  Provide poly_k_pows=(Lk,), poly_mu2_pows=(Lmu,).
                  If use_k2=True, k-term uses (k^2)^p instead of k^p.
        """
        # Sanity checks
        if beta_kind == 'table':
            if k_edges is None or mu_edges is None:
                raise ValueError("beta_kind='table' requires k_edges and mu_edges.")
            if beta.ndim != 3:
                raise ValueError("For beta_kind='table', beta must have shape (n_fields, Nk, Nmu).")

            # Build/reuse the grid -> (k,mu) bin mapping and run the table kernel.
            self._ensure_nearest_cache(k_edges, mu_edges)
            return self._get_final_field_table(fields_k, beta)

        elif beta_kind == 'const':
            if beta.ndim != 1:
                raise ValueError("For beta_kind='const', beta must have shape (n_fields,).")
            return self._get_final_field_const(fields_k, beta)

        elif beta_kind == 'poly':
            if beta.ndim != 3:
                raise ValueError("For beta_kind='poly', beta must have shape (n_fields, Lk, Lmu).")
            if poly_k_pows is None or poly_mu2_pows is None:
                raise ValueError("For beta_kind='poly', provide poly_k_pows and poly_mu_pows.")
            return self._get_final_field_poly(fields_k, beta, poly_k_pows, poly_mu2_pows)

        else:
            raise ValueError(f"Unknown beta_kind='{beta_kind}'")
