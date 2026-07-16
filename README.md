# Jax-AOtools

Jax-AOtools is a JAX-based Python package for adaptive-optics-style optical
propagation, atmospheric phase-screen generation, interpolation, and related
simulation utilities.

The installable package lives under `src/jax_aotools/`. The `examples/` and
`validation/` directories are repository materials for demonstrations and
reproducibility; they are not part of the installed Python package.

## Layout

- `src/jax_aotools/`: installable Python package.
- `examples/ghost_imaging/`: long-range computational ghost imaging example.
- `examples/anisoplanatic_imaging/`: anisoplanatic imaging example.
- `examples/quantum_key_distribution/`: spatially encoded free-space QKD example.
- `functionality/`: functionality examples used by the manuscript.
- `validation/`: validation and comparison benchmarks.

Each example folder contains its script, required local input data, and a
`results/` directory with the generated output files.

## Dependencies

Jax-AOtools requires Python 3.7 or later. The core package depends on:

- `aotools`
- `jax`
- `jaxlib`
- `matplotlib`
- `numpy`
- `scipy`

The example scripts additionally use:

- `opencv-python`
- `numba`

These dependencies are declared in `pyproject.toml`. Installing the package
with `pip install -e .` installs the core dependencies, while
`pip install -e ".[examples]"` also installs the optional example
dependencies.

## Installation

For local development, install the package from the repository root:

```bash
pip install -e .
```

To install optional dependencies used by the examples:

```bash
pip install -e ".[examples]"
```

After installation, the core functions can be imported as:

```python
from jax_aotools import angularSpectrum_jax, ft_sh_phase_screen_jax_outer_scale
from jax_aotools import jax_interp2d, jax_interp_2dmtx
```

## Example Commands

Run from the repository root:

```bash
python examples/quantum_key_distribution/quantum_key_distribution_example.py
python examples/ghost_imaging/ghost_imaging_example.py
python examples/anisoplanatic_imaging/anisoplanatic_imaging_example.py
```

Functionality and validation scripts:

```bash
python functionality/functionality_examples.py
python validation/validation_examples.py
```
