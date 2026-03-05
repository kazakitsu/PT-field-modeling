# PT-field-modeling
Perturbative forward modeling at the field level

---

## Overview

**PT_field** provides JAX implementations for field-level forward modeling based on Lagrangian perturbation theory, with optional redshift-space distortion (RSD) and LyA helper fields. 
It integrates with [`lss_utils`](https://github.com/kazakitsu/lss_utils) for mesh assignment and batched FFT+deconvolution, and with its power-spectrum estimator for binned operations in $(k,\mu)$.

Core ideas:

- Build the LPT displacement from an input linear density field $\delta_L(\mathbf{k})$.
- Construct the bias operators in position space
  - for scalar: $[1, \delta, \delta^2, {\mathcal{G}_2}, \cdots]$.
  - for rank-2 tensor: $[K_{ij}, \delta K_{ij}, H_{ij}, T_{ij}]$.
- Assign these fields to an Eulerian grid using particle–mesh style advection from the Lagrangian grid.
- Optionally transform to Fourier space and deconvolve the assignment window in one batched call.
- Work per $(k,\mu)$ bin to orthogonalize fields or to combine them into a final modeled field.

---

## Features

- Minimal dependencies: `numpy`, `jax`, and `lss_utils`

- **LPT-based forward model (`LPT_Forward`)**
  - LPT displacement in real space, with optional **RSD**: $\psi_z \rightarrow \psi_z (1 + f)$.
  - Scalar field builders up to **cubic** order: $1, \delta_1, \delta_1^2, {\mathcal{G}_2}, \delta_1^3, \delta{\mathcal{G}_2}, {\mathcal{G}_3}, {\Gamma_3}$ and $S_3 = {\boldsymbol\psi_2}\cdot\nabla \delta_1$
    - With `rsd=True`, also builds ${\mathcal{G}^{\parallel}_{2}}$.  
    - With `lya=True`, also builds $\eta$-related helpers: $\eta^2$, $\delta\eta$, $KK^{\parallel}$.
    - With `lya_full_fields=True`, also builds $\Pi^{[2]}^{\parallel}$ (and $\delta^3$).
  - Tensor field builders up to **quadratic** order: $K_{ij}, \delta K_{ij}, H_{ij}, T_{ij}$
    - `rsd` is not supported yet.

- **Orthogonalization and diagnostics**
  - `orthogonalize(...)`: per $(k,\mu)$ bin Cholesky-based orthonormalization of fields.
  - `compute_corr_2d(...)`: computes $r_{ij}(k)$ matrices in each $\mu$-bin.  
  - `compute_pks_2d(...)`: measures auto power $P_i(k)$ for many fields across $\mu$-bins. 
    -With `cross=True`, also computes cross-power spectra.

---

## Prerequisites

- Python >= 3.10
- NumPy >= 2.1
- JAX >= 0.4.3
- [`lss_utils`](https://github.com/kazakitsu/lss_utils) (installed automatically with pip)

---

## Installation

Clone and install the package:

```bash
git clone https://github.com/kazakitsu/PT-field-modeling.git
cd PT-field-modeling
pip install .
```

If you want to work on or modify the code locally:

```bash
git clone https://github.com/kazakitsu/PT-field-modeling.git
cd PT-field-modeling
pip install -e .
```
