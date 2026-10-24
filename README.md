# Jax-AOtools

Jax-AOtools provides JAX-based numerical tools for atmospheric turbulence,
optical propagation, two-dimensional scalar interpolation, and random rough
surface generation. Its numerical kernels use JAX arrays and just-in-time
compilation. Hardware support depends on the installed JAX backend.

## Features

- FFT-based atmospheric phase screens with nested sub-harmonic or
  outer-scale-adaptive low-frequency sub-harmonic compensation.
- One-step and two-step Fresnel propagation, angular-spectrum propagation,
  and focal-plane field calculation.
- Bicubic interpolation of two-dimensional scalar fields.
- Gaussian beam construction and random rough surface generation.

## Repository Structure

```text
src/
    jax_aotools.py     # Public imports
    Jax_Tools.py       # Phase screens, propagation, and simulation utilities
    Jax_Interp2D.py    # Scalar-field interpolation
pyproject.toml        # Package metadata and dependencies
setup.py             # Setuptools entry point
LICENSE
README.md
```

## Requirements

The package metadata specifies Python 3.7 or later. Compatible dependency
versions must also be available for your Python version and platform; this
minimum does not imply compatibility with the latest dependency releases.

Runtime dependencies, declared in `pyproject.toml`, are:

- `jax` and `jaxlib`
- `numpy`
- `scipy`
- `matplotlib`
- `aotools`

Building the package requires `setuptools>=64` and `wheel`. GPU execution
requires a compatible JAX installation for the target platform and accelerator.

## Installation

Clone the repository and install from its root directory:

```bash
git clone https://github.com/tzwstartfresh/jax_aotools.git
cd jax_aotools
python -m pip install .
```

For development, use an editable installation:

```bash
python -m pip install -e .
```

## Basic Usage

Generate a turbulence phase screen and apply it to a Gaussian beam before
propagation:

```python
import jax
import jax.numpy as jnp
from jax_aotools import (
    angularSpectrum_jax,
    auto_random,
    ft_sh_phase_screen_jax_outer_scale,
    gauss_beam,
)

phscrn_func = auto_random(ft_sh_phase_screen_jax_outer_scale)
n = 256
pixel_scale = 0.01  # Metres per pixel
wavelength = 532e-9  # Metres

phase = phscrn_func(
    r0=0.15, N=n, delta=pixel_scale, L0=30.0, l0=0.01
)
field = gauss_beam(n, pixel_scale, waist=0.2)
field = field * jnp.exp(1j * phase)
propagated = angularSpectrum_jax(
    field, wavelength, pixel_scale, pixel_scale, z=1000.0
)
intensity = jnp.abs(propagated) ** 2
intensity.block_until_ready()
print("Intensity shape:", intensity.shape)
print("Available devices:", jax.devices())
```

Lengths in this example are in metres and phase is in radians. `auto_random`
manages the PRNG key internally: create `phscrn_func` once and call it repeatedly
to generate new screens without passing a key. By default, initialization uses
a time-based seed. For reproducible sequences, use
`phscrn_func = auto_random(ft_sh_phase_screen_jax_outer_scale, seed=0)`.
Phase-screen generators support batched generation through `batch_size`.

The scalar interpolator is available as `from jax_aotools import jax_interp2d`.
Its constructor takes source coordinate vectors, sampled values, and a bicubic
coefficient conversion matrix (`cvt_ary`) supplied by the caller. Query
coordinates must remain within the source grid.

Implementation modules can also be imported directly from `Jax_Tools` and
`Jax_Interp2D`.

## Timing Notes

The first call to a JIT-compiled kernel includes compilation overhead. For
execution-time measurements, warm up the kernel with the same array shapes
and static arguments, then call `block_until_ready()` on the returned array
before stopping the timer. Runtime and memory use depend on the hardware,
array dimensions, batch size, precision, and JAX version.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
