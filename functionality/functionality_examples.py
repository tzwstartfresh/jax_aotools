"""Generate functionality examples for Jax-AOtools.

This script is self-contained. It generates the functionality example figures
and writes a single JSON summary.

Outputs are written to:
    results/functionality_results.json
    results/fig2a_atmospheric_turbulence.png
    results/fig2b_optical_kernels.png
    results/fig2c_scalar_interpolation.png
    results/fig2d_functional_interpolation.png
    results/fig2e_random_rough_surface.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
from Jax_Interp2D import jax_interp2d
from Jax_Interp2DMtx import jax_interp_2dmtx
from Jax_Tools import (
    angularSpectrum_jax,
    auto_random,
    coarse_surface,
    ft_phase_screen_jaxbase,
    ft_sh_phase_screen_jax_nested,
    ft_sh_phase_screen_jax_outer_scale,
    gauss_beam,
    lensAgainst_jax,
    oneStepFresnel_jax,
    twoStepFresnel_jax,
)

OUT_DIR = SCRIPT_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)


def relative_to_script(path: Path) -> str:
    """Return a stable project-relative path for JSON references."""
    try:
        return str(path.resolve().relative_to(SCRIPT_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def block(value):
    """Synchronize JAX arrays so timings include actual execution."""
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value


def timed(func, warmup: int = 1, repeat: int = 3):
    """Run a callable with warm-up and return mean runtime plus the last output."""
    for _ in range(warmup):
        block(func())

    samples = []
    last = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        last = block(func())
        samples.append(time.perf_counter() - t0)

    return float(np.mean(samples)), last


def bicubic_conversion_matrix() -> np.ndarray:
    """Build the 16x16 conversion matrix for bicubic cell coefficients."""
    rows = []
    rhs = []

    def coeff_index(i: int, j: int) -> int:
        # jax_interp2d reshapes coefficients as (y_power, x_power, y_cell, x_cell).
        return j * 4 + i

    def add_value(x: int, y: int, pos: int) -> None:
        row = np.zeros(16)
        for i in range(4):
            for j in range(4):
                row[coeff_index(i, j)] = (x**i) * (y**j)
        rows.append(row)
        rhs.append(pos)

    def add_dx(x: int, y: int, pos: int) -> None:
        row = np.zeros(16)
        for i in range(1, 4):
            for j in range(4):
                row[coeff_index(i, j)] = i * (x ** (i - 1)) * (y**j)
        rows.append(row)
        rhs.append(pos)

    def add_dy(x: int, y: int, pos: int) -> None:
        row = np.zeros(16)
        for i in range(4):
            for j in range(1, 4):
                row[coeff_index(i, j)] = j * (x**i) * (y ** (j - 1))
        rows.append(row)
        rhs.append(pos)

    def add_dxdy(x: int, y: int, pos: int) -> None:
        row = np.zeros(16)
        for i in range(1, 4):
            for j in range(1, 4):
                row[coeff_index(i, j)] = i * j * (x ** (i - 1)) * (y ** (j - 1))
        rows.append(row)
        rhs.append(pos)

    points = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for pos, (x, y) in enumerate(points):
        add_value(x, y, pos)
    for pos, (x, y) in enumerate(points, start=4):
        add_dx(x, y, pos)
    for pos, (x, y) in enumerate(points, start=8):
        add_dy(x, y, pos)
    for pos, (x, y) in enumerate(points, start=12):
        add_dxdy(x, y, pos)

    lhs = np.asarray(rows)
    inv = np.linalg.inv(lhs)
    cvt = np.zeros((16, 16))
    for equation_row, data_pos in enumerate(rhs):
        cvt[:, data_pos] = inv[:, equation_row]
    return cvt.astype(np.float32)


def save_image_grid(path: Path, panels: list[dict], cmap: str = "viridis", figsize=(9.0, 6.4)) -> None:
    """Save a compact grid of image panels with individual colorbars.

    Each panel may optionally provide ``vmin`` and ``vmax`` so related images
    can share the same display scale.
    """
    rows = int(np.ceil(len(panels) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, panel in zip(axes, panels):
        im = ax.imshow(
            panel["data"],
            cmap=panel.get("cmap", cmap),
            origin="lower",
            vmin=panel.get("vmin"),
            vmax=panel.get("vmax"),
        )
        ax.set_title(panel["title"], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(panels) :]:
        ax.axis("off")

    save_figure(fig, path)
    plt.close(fig)


def save_figure(fig, path: Path, dpi: int = 300) -> None:
    """Save a figure robustly on Windows by writing a temporary file first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + "_tmp" + path.suffix)
    if tmp_path.exists():
        tmp_path.unlink()
    fig.savefig(tmp_path, dpi=dpi, bbox_inches="tight")
    if path.exists():
        path.unlink()
    tmp_path.replace(path)


def normalized_intensity(field) -> np.ndarray:
    """Return intensity normalized by its maximum value for visualization."""
    intensity = np.asarray(jnp.abs(field) ** 2)
    return intensity / (float(intensity.max()) + 1e-30)


def complex_standard_normal(key, shape):
    """Generate a reusable complex Gaussian noise tensor with unit variance."""
    real_key, imag_key = jax.random.split(key)
    return (jax.random.normal(real_key, shape) + 1j * jax.random.normal(imag_key, shape)) / jnp.sqrt(2.0)


def remove_piston(screen):
    """Remove the spatial mean from every phase-screen frame."""
    return screen - screen.mean(axis=(-2, -1), keepdims=True)


def von_karman_phase_psd(fx, fy, r0: float, outer_scale: float, inner_scale: float):
    """Modified von Karman phase power spectral density."""
    f = jnp.sqrt(fx**2 + fy**2)
    fm = 5.92 / (2.0 * jnp.pi * inner_scale)
    f0 = 1.0 / outer_scale
    return 0.023 * r0 ** (-5.0 / 3.0) * jnp.exp(-((f / fm) ** 2)) / ((f**2 + f0**2) ** (11.0 / 6.0))


def fft_phase_screen_from_noise(noise, r0: float, n: int, delta: float, outer_scale: float, inner_scale: float):
    """Create FFT phase screens from a supplied complex noise tensor.

    Supplying the noise explicitly lets all comparison panels share exactly the
    same high-frequency random Fourier coefficients.
    """
    del_f = 1.0 / (n * delta)
    freq = (jnp.arange(n) - n // 2) * del_f
    fx, fy = jnp.meshgrid(freq, freq, indexing="xy")
    psd = von_karman_phase_psd(fx, fy, r0, outer_scale, inner_scale)
    psd = psd.at[n // 2, n // 2].set(0.0)

    cn = noise * jnp.sqrt(psd)[None, :, :] * del_f
    screen = jnp.fft.ifftshift(
        jnp.fft.ifft2(jnp.fft.fftshift(cn, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    ).real * (n * del_f) ** 2
    return remove_piston(screen)


def nested_subharmonic_modes(n: int, delta: float, sub_samp: int, sub_order: int):
    """Frequency samples for the AOtools-style nested sub-harmonic grid."""
    if sub_samp % 2 == 0:
        sub_samp += 1
    del_f = 1.0 / (n * delta)
    half = sub_samp // 2
    offsets = jnp.arange(-half, half + 1)

    fx_all, fy_all, spacing_all = [], [], []
    for order in range(1, sub_order + 1):
        dfi = del_f / (3.0**order)
        fx, fy = jnp.meshgrid(offsets * dfi, offsets * dfi, indexing="xy")
        mask = ~((fx == 0.0) & (fy == 0.0))
        fx_all.append(fx[mask])
        fy_all.append(fy[mask])
        spacing_all.append(jnp.full((int(mask.sum()),), dfi))

    return jnp.concatenate(fx_all), jnp.concatenate(fy_all), jnp.concatenate(spacing_all)


def outer_scale_subharmonic_modes(n: int, delta: float, outer_scale: float, sub_order: int):
    """Non-uniform low-frequency grid between 1/L0 and the FFT boundary."""
    del_f = 1.0 / (n * delta)
    f_min = max(1.0 / outer_scale, 1.0e-12)
    rho = max((3.0 * del_f / f_min) ** (1.0 / sub_order), 1.01)
    positive = f_min * (rho ** jnp.arange(sub_order))
    freq = jnp.concatenate([-positive[::-1], jnp.array([0.0]), positive])
    interval = jnp.abs(jnp.gradient(freq))

    fx, fy = jnp.meshgrid(freq, freq, indexing="xy")
    dfx, dfy = jnp.meshgrid(interval, interval, indexing="xy")
    radius = jnp.sqrt(fx**2 + fy**2)
    mask = (radius > 0.0) & (radius <= 3.0 * del_f)
    spacing = jnp.sqrt(dfx[mask] * dfy[mask])
    return fx[mask], fy[mask], spacing


def subharmonic_screen_from_noise(noise, fx_modes, fy_modes, mode_spacing, r0, n, delta, outer_scale, inner_scale):
    """Synthesize a low-frequency phase screen from supplied modal noise."""
    x = (jnp.arange(n) - n // 2) * delta
    xx, yy = jnp.meshgrid(x, x, indexing="xy")
    psd = von_karman_phase_psd(fx_modes, fy_modes, r0, outer_scale, inner_scale)
    mode_count = fx_modes.shape[0]
    cn = noise[:, :mode_count] * jnp.sqrt(psd)[None, :] * mode_spacing[None, :]
    phase = 2.0 * jnp.pi * (fx_modes[:, None, None] * xx[None, :, :] + fy_modes[:, None, None] * yy[None, :, :])
    screen = (cn[:, :, None, None] * jnp.exp(1j * phase)[None, :, :, :]).sum(axis=1).real
    return remove_piston(screen)


def phase_structure_function(screen_stack: np.ndarray, delta: float, max_fraction: float = 0.25):
    """Estimate an averaged 1D phase structure function.

    The structure function is computed from x- and y-direction spatial shifts:
        D_phi(r) = <[phi(x+r)-phi(x)]^2>.
    Results are averaged over all frames in ``screen_stack`` and over both axes.
    """
    max_shift = max(2, int(screen_stack.shape[-1] * max_fraction))
    lags = np.arange(1, max_shift + 1)
    values = []
    for lag in lags:
        dx = screen_stack[:, :, lag:] - screen_stack[:, :, :-lag]
        dy = screen_stack[:, lag:, :] - screen_stack[:, :-lag, :]
        values.append(0.5 * (np.mean(dx**2) + np.mean(dy**2)))
    return lags * delta, np.asarray(values)


def atmospheric_turbulence_example() -> dict:
    """Generate comparable FFT and sub-harmonic atmospheric phase-screen examples.

    Layout:
        first row  : three comparable phase-screen realizations
        second row : one centered phase-structure-function comparison plot
    """
    n = 256
    batch = 8
    r0 = 0.05
    delta = 1.0 / n
    inner_scale = 0.001
    outer_scale = 300.0

    high_noise = complex_standard_normal(jax.random.PRNGKey(20250605), (batch, n, n))
    nested_fx, nested_fy, nested_spacing = nested_subharmonic_modes(n, delta, sub_samp=5, sub_order=4)
    outer_fx, outer_fy, outer_spacing = outer_scale_subharmonic_modes(n, delta, outer_scale, sub_order=7)
    low_mode_count = max(int(nested_fx.shape[0]), int(outer_fx.shape[0]))
    low_noise = complex_standard_normal(jax.random.PRNGKey(20250606), (batch, low_mode_count))

    def make_fft():
        return fft_phase_screen_from_noise(high_noise, r0, n, delta, outer_scale, inner_scale)

    def make_nested_subharmonic():
        base = fft_phase_screen_from_noise(high_noise, r0, n, delta, outer_scale, inner_scale)
        low = subharmonic_screen_from_noise(
            low_noise, nested_fx, nested_fy, nested_spacing,
            r0, n, delta, outer_scale, inner_scale,
        )
        return remove_piston(base + low)

    def make_outer_scale_subharmonic():
        base = fft_phase_screen_from_noise(high_noise, r0, n, delta, outer_scale, inner_scale)
        low = subharmonic_screen_from_noise(
            low_noise, outer_fx, outer_fy, outer_spacing,
            r0, n, delta, outer_scale, inner_scale,
        )
        return remove_piston(base + low)

    t_fft, fft = timed(make_fft)
    t_sh1, sh1 = timed(make_nested_subharmonic)
    t_sh2, sh2 = timed(make_outer_scale_subharmonic)
    fft_np, sh1_np, sh2_np = np.asarray(fft), np.asarray(sh1), np.asarray(sh2)

    frame_fft = fft_np[0]
    frame_sh1 = sh1_np[0]
    frame_sh2 = sh2_np[0]
    shared_limit = float(max(np.abs(frame_fft).max(), np.abs(frame_sh1).max(), np.abs(frame_sh2).max()))

    rho_fft, d_fft = phase_structure_function(fft_np, delta)
    rho_sh1, d_sh1 = phase_structure_function(sh1_np, delta)
    rho_sh2, d_sh2 = phase_structure_function(sh2_np, delta)

    figure = OUT_DIR / 'fig2a_atmospheric_turbulence.png'
    fig = plt.figure(figsize=(12.0, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.9])
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax1 = fig.add_subplot(gs[0, 2:4])
    ax2 = fig.add_subplot(gs[0, 4:6])
    ax3 = fig.add_subplot(gs[1, 1:5])

    im0 = ax0.imshow(frame_fft, cmap='twilight', origin='lower', vmin=-shared_limit, vmax=shared_limit)
    ax0.set_title('(a) Shared FFT phase screen', fontsize=11)
    ax0.set_xticks([])
    ax0.set_yticks([])

    im1 = ax1.imshow(frame_sh1, cmap='twilight', origin='lower', vmin=-shared_limit, vmax=shared_limit)
    ax1.set_title('(b) Shared FFT + nested compensation', fontsize=11)
    ax1.set_xticks([])
    ax1.set_yticks([])

    im2 = ax2.imshow(frame_sh2, cmap='twilight', origin='lower', vmin=-shared_limit, vmax=shared_limit)
    ax2.set_title('(c) Shared FFT + outer-scale compensation', fontsize=11)
    ax2.set_xticks([])
    ax2.set_yticks([])

    cbar = fig.colorbar(im2, ax=[ax0, ax1, ax2], fraction=0.028, pad=0.02, shrink=0.96)
    cbar.set_label('Phase (rad)', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax3.plot(rho_fft, d_fft, linewidth=2.0, label='FFT only')
    ax3.plot(rho_sh1, d_sh1, linewidth=2.0, label='Nested compensation')
    ax3.plot(rho_sh2, d_sh2, linewidth=2.0, label='Outer-scale compensation')
    ax3.set_title('(d) Phase structure function comparison', fontsize=11)
    ax3.set_xlabel('Separation r (m)', fontsize=10)
    ax3.set_ylabel(r'$D_\phi(r)$ (rad$^2$)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend(frameon=False, fontsize=9, ncol=3, loc='upper left')
    ax3.tick_params(labelsize=9)

    save_figure(fig, figure)
    plt.close(fig)

    corr_fft_sh1 = float(np.corrcoef(frame_fft.ravel(), frame_sh1.ravel())[0, 1])
    corr_fft_sh2 = float(np.corrcoef(frame_fft.ravel(), frame_sh2.ravel())[0, 1])
    corr_sh1_sh2 = float(np.corrcoef(frame_sh1.ravel(), frame_sh2.ravel())[0, 1])

    return {
        'figure': relative_to_script(figure),
        'N': n,
        'batch_size': batch,
        'parameters': {'r0': r0, 'delta': delta, 'L0': outer_scale, 'l0': inner_scale},
        'timing_seconds': {
            'fft': t_fft,
            'subharmonic_method1': t_sh1,
            'subharmonic_method2': t_sh2,
        },
        'layout': {
            'top_row': ['shared FFT phase screen', 'shared FFT + nested compensation', 'shared FFT + outer-scale compensation'],
            'bottom_row': ['phase structure function comparison (centered)'],
        },
        'display_scales': {
            'shared_phase_vmin': -shared_limit,
            'shared_phase_vmax': shared_limit,
        },
        'random_control': {
            'high_frequency_noise': 'identical complex Fourier-noise tensor used by all three phase-screen panels',
            'low_frequency_noise_stream': 'same complex Gaussian low-frequency noise stream used by both compensation methods; frequency grids differ by design',
            'high_frequency_seed': 20250605,
            'low_frequency_seed': 20250606,
            'nested_low_frequency_modes': int(nested_fx.shape[0]),
            'outer_scale_low_frequency_modes': int(outer_fx.shape[0]),
        },
        'phase_statistics': {
            'fft_std': float(fft_np.std()),
            'method1_std': float(sh1_np.std()),
            'method2_std': float(sh2_np.std()),
            'method2_mean': float(sh2_np.mean()),
            'method1_minus_fft_std': float((sh1_np - fft_np).std()),
            'method2_minus_fft_std': float((sh2_np - fft_np).std()),
            'method2_minus_method1_std': float((sh2_np - sh1_np).std()),
            'correlation_fft_method1_first_frame': corr_fft_sh1,
            'correlation_fft_method2_first_frame': corr_fft_sh2,
            'correlation_method1_method2_first_frame': corr_sh1_sh2,
        },
        'structure_function': {
            'separation_m': rho_fft.tolist(),
            'fft_only': d_fft.tolist(),
            'nested_compensation': d_sh1.tolist(),
            'outer_scale_compensation': d_sh2.tolist(),
        },
        'comparison_note': 'The first-row phase-screen panels share the same display scale and the same high-frequency Fourier random coefficients. The centered second-row panel compares ensemble-averaged phase structure functions.',
    }


def propagation_kernels_example() -> dict:
    """Generate a cleaner two-row figure for the propagation kernels.

    Layout:
        first row  : input, one-step Fresnel, two-step Fresnel
        second row : angular-spectrum, lens focal-plane
    """
    n = 256
    batch = 4
    width = 0.01
    d1 = width / n
    d2 = d1
    wvl = 500e-9
    z = n * d1**2 / wvl
    focal_length = z

    # Slightly larger input radius for a fuller input spot while preserving
    # visible broadening after propagation.
    waist = 0.0004
    rayleigh_length = np.pi * waist**2 / wvl
    z_over_rayleigh = z / rayleigh_length
    expected_gaussian_radius_ratio = np.sqrt(1.0 + z_over_rayleigh**2)

    field = gauss_beam(n, d1, waist=waist)
    fields = jnp.repeat(field[None, :, :], batch, axis=0)

    def run_one_step():
        return oneStepFresnel_jax(fields, wvl, d1, z)

    def run_two_step():
        return twoStepFresnel_jax(fields, wvl, d1, d2, z)

    def run_angular():
        return angularSpectrum_jax(fields, wvl, d1, d2, z)

    def run_lens():
        return lensAgainst_jax(fields, wvl, d1, focal_length)

    t_one, out_one = timed(run_one_step)
    t_two, out_two = timed(run_two_step)
    t_ang, out_ang = timed(run_angular)
    t_lens, out_lens = timed(run_lens)

    input_intensity = normalized_intensity(fields[0])
    one_intensity = normalized_intensity(out_one[0])
    two_intensity = normalized_intensity(out_two[0])
    angular_intensity = normalized_intensity(out_ang[0])
    lens_intensity = normalized_intensity(out_lens[0])

    figure = OUT_DIR / 'fig2b_optical_kernels.png'
    fig = plt.figure(figsize=(11.6, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax1 = fig.add_subplot(gs[0, 2:4])
    ax2 = fig.add_subplot(gs[0, 4:6])
    ax3 = fig.add_subplot(gs[1, 1:3])
    ax4 = fig.add_subplot(gs[1, 3:5])

    panels = [
        (ax0, '(a) Input Gaussian', input_intensity),
        (ax1, '(b) One-step Fresnel', one_intensity),
        (ax2, '(c) Two-step Fresnel', two_intensity),
        (ax3, '(d) Angular spectrum', angular_intensity),
        (ax4, '(e) Lens focal plane', lens_intensity),
    ]
    last_im = None
    for ax, title, data in panels:
        last_im = ax.imshow(data, cmap='magma', origin='lower', vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(last_im, ax=[ax0, ax1, ax2, ax3, ax4], fraction=0.028, pad=0.02, shrink=0.94)
    cbar.set_label('Normalized intensity', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    save_figure(fig, figure)
    plt.close(fig)

    input_power = float(np.asarray(jnp.abs(fields[0]) ** 2).sum() * d1**2)
    one_power = float(np.asarray(jnp.abs(out_one[0]) ** 2).sum() * d1**2)
    two_power = float(np.asarray(jnp.abs(out_two[0]) ** 2).sum() * d2**2)
    angular_power = float(np.asarray(jnp.abs(out_ang[0]) ** 2).sum() * d2**2)
    lens_power = float(np.asarray(jnp.abs(out_lens[0]) ** 2).sum() * d1**2)

    one_step_output_spacing = wvl * z / (n * d1)
    lens_output_spacing = wvl * focal_length / (n * d1)
    spacing_mismatch = max(abs(one_step_output_spacing - d1), abs(d2 - d1), abs(lens_output_spacing - d1))

    two_minus_angular_l2 = float(
        np.linalg.norm(two_intensity - angular_intensity) / (np.linalg.norm(angular_intensity) + 1e-30)
    )
    one_minus_two_l2 = float(
        np.linalg.norm(one_intensity - two_intensity) / (np.linalg.norm(two_intensity) + 1e-30)
    )

    return {
        'figure': relative_to_script(figure),
        'N': n,
        'batch_size': batch,
        'parameters': {
            'wavelength': wvl,
            'width': width,
            'd1': d1,
            'd2': d2,
            'z': z,
            'focal_length': focal_length,
            'gaussian_waist': waist,
            'rayleigh_length': rayleigh_length,
            'z_over_rayleigh': z_over_rayleigh,
            'expected_gaussian_radius_ratio': expected_gaussian_radius_ratio,
        },
        'figure_design': {
            'top_row': ['input Gaussian intensity', 'one-step Fresnel', 'two-step Fresnel'],
            'bottom_row': ['angular-spectrum propagation', 'lens focal-plane intensity'],
            'shared_intensity_color_scale': 'all five panels use vmin=0 and vmax=1',
        },
        'sampling_checks': {
            'one_step_expected_output_spacing': one_step_output_spacing,
            'two_step_output_spacing': d2,
            'angular_spectrum_output_spacing': d2,
            'lens_expected_focal_plane_spacing': lens_output_spacing,
            'max_spacing_mismatch_from_d1': float(spacing_mismatch),
            'sampling_is_consistent': bool(spacing_mismatch < 1e-15),
        },
        'timing_seconds': {
            'one_step_fresnel': t_one,
            'two_step_fresnel': t_two,
            'angular_spectrum': t_ang,
            'lens_against': t_lens,
        },
        'power': {
            'input_integrated_power': input_power,
            'one_step_integrated_power': one_power,
            'two_step_integrated_power': two_power,
            'angular_spectrum_integrated_power': angular_power,
            'lens_integrated_power': lens_power,
            'relative_error_one_step': abs(one_power - input_power) / (abs(input_power) + 1e-30),
            'relative_error_two_step': abs(two_power - input_power) / (abs(input_power) + 1e-30),
            'relative_error_angular_spectrum': abs(angular_power - input_power) / (abs(input_power) + 1e-30),
            'relative_error_lens': abs(lens_power - input_power) / (abs(input_power) + 1e-30),
        },
        'intensity_difference': {
            'relative_l2_one_step_vs_two_step': one_minus_two_l2,
            'relative_l2_two_step_vs_angular': two_minus_angular_l2,
        },
        'validation_note': 'The figure has no global title. The first row shows the input and two Fresnel propagators. The second row shows angular-spectrum propagation and lens focal-plane propagation. All panels use the same normalized-intensity scale under matched sampling.',
    }


def interpolation_example() -> dict:
    """Generate a scalar bicubic interpolation example."""
    cvt = jnp.asarray(bicubic_conversion_matrix())
    n = 96
    m = 384
    xaxis = jnp.linspace(-1.0, 1.0, n) * 4 * jnp.pi
    yaxis = jnp.linspace(-1.0, 1.0, n) * 4 * jnp.pi

    def analytic(x, y):
        return jnp.cos(jnp.sqrt(x**2 + 0.25 * y**2))

    signal = analytic(xaxis[None, :], yaxis[:, None])
    interp_obj = jax_interp2d(xaxis, yaxis, signal, cvt, uniform=True)
    xi = jnp.linspace(-0.7, 0.7, m) * 4 * jnp.pi
    yi = jnp.linspace(-0.7, 0.7, m) * 4 * jnp.pi

    def run_interp():
        return interp_obj.interp_jit(xi, yi)

    t_interp, fit = timed(run_interp)
    truth = analytic(xi[None, :], yi[:, None])
    error = np.asarray(fit - truth)

    figure = OUT_DIR / "fig2c_scalar_interpolation.png"
    save_image_grid(
        figure,
        [
            {"data": np.asarray(signal), "title": "Input samples"},
            {"data": np.asarray(fit), "title": "JAX bicubic interpolation"},
            {"data": np.asarray(truth), "title": "Analytic function"},
            {"data": np.abs(error), "title": "Absolute error"},
        ],
        cmap="viridis",
    )

    return {
        "figure": relative_to_script(figure),
        "input_grid": [n, n],
        "output_grid": [m, m],
        "timing_seconds": {"scalar_bicubic": t_interp},
        "accuracy": {
            "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(np.asarray(truth))),
            "max_abs": float(np.max(np.abs(error))),
        },
    }


def functional_interpolation_example() -> dict:
    """Generate a matrix-valued bicubic interpolation example."""
    cvt = jnp.asarray(bicubic_conversion_matrix())
    guide_n = 24
    query_n = 80
    patch_n = 32

    xaxis = jnp.linspace(-1.0, 1.0, guide_n)
    yaxis = jnp.linspace(-1.0, 1.0, guide_n)
    u = jnp.linspace(-1.0, 1.0, patch_n)
    xx, yy = jnp.meshgrid(u, u, indexing="xy")
    cx = xaxis[None, :, None, None]
    cy = yaxis[:, None, None, None]
    sigma_x = 0.24 + 0.05 * (1 + cx)
    sigma_y = 0.20 + 0.05 * (1 - cy)
    signal = jnp.exp(
        -(
            (xx[None, None] - 0.25 * cx) ** 2 / sigma_x**2
            + (yy[None, None] + 0.25 * cy) ** 2 / sigma_y**2
        )
    )

    interp_obj = jax_interp_2dmtx(xaxis, yaxis, signal, cvt, uniform=True)
    xi = jnp.linspace(-0.8, 0.8, query_n)
    yi = jnp.linspace(-0.8, 0.8, query_n)

    def run_interp():
        return interp_obj.interp_jit(xi, yi)

    t_interp, fit = timed(run_interp)
    fit_np = np.asarray(fit)

    panels = []
    for y_idx, x_idx in [(10, 10), (10, 68), (40, 40), (68, 68)]:
        panels.append({"data": fit_np[y_idx, x_idx], "title": f"Interpolated PSF ({y_idx}, {x_idx})"})

    figure = OUT_DIR / "fig2d_functional_interpolation.png"
    save_image_grid(figure, panels, cmap="magma")

    return {
        "figure": relative_to_script(figure),
        "guide_grid": [guide_n, guide_n],
        "query_grid": [query_n, query_n],
        "functional_patch": [patch_n, patch_n],
        "timing_seconds": {"functional_bicubic": t_interp},
        "output_shape": list(fit_np.shape),
    }


def rough_surface_example() -> dict:
    """Generate batched random rough-surface examples."""
    key = jax.random.PRNGKey(20250607)
    n = 256
    batch = 8
    delta = 1.0 / n
    surface_std = 0.25e-6
    corr_len = 0.025

    def run_surface():
        surface, _ = coarse_surface(key, surface_std, corr_len, delta, n, batch_size=batch)
        return surface

    t_surface, surfaces = timed(run_surface)
    surfaces_np = np.asarray(surfaces)

    figure = OUT_DIR / "fig2e_random_rough_surface.png"
    save_image_grid(
        figure,
        [
            {"data": surfaces_np[0], "title": "Rough surface, realization 1"},
            {"data": surfaces_np[1], "title": "Rough surface, realization 2"},
            {"data": surfaces_np.mean(axis=0), "title": "Ensemble mean"},
            {"data": surfaces_np.std(axis=0), "title": "Ensemble standard deviation"},
        ],
        cmap="coolwarm",
    )

    return {
        "figure": relative_to_script(figure),
        "N": n,
        "batch_size": batch,
        "parameters": {"surface_std": surface_std, "correlation_length": corr_len, "delta": delta},
        "timing_seconds": {"rough_surface_batch": t_surface},
        "surface_statistics": {
            "mean": float(surfaces_np.mean()),
            "std": float(surfaces_np.std()),
            "min": float(surfaces_np.min()),
            "max": float(surfaces_np.max()),
        },
    }


def auto_random_example() -> dict:
    """Check that auto_random advances the PRNG state between calls."""
    n = 128
    batch = 3
    phs_func = auto_random(ft_sh_phase_screen_jax_outer_scale, seed=20250605)
    first = phs_func(0.05, n, 1 / n, 300.0, 0.001, batch_size=batch, sub_order=5, zoom_scale=0.25)
    second = phs_func(0.05, n, 1 / n, 300.0, 0.001, batch_size=batch, sub_order=5, zoom_scale=0.25)
    first_np = np.asarray(first)
    second_np = np.asarray(second)

    return {
        "N": n,
        "batch_size": batch,
        "first_call_shape": list(first_np.shape),
        "second_call_shape": list(second_np.shape),
        "mean_abs_difference_between_calls": float(np.mean(np.abs(first_np - second_np))),
        "first_call_std": float(first_np.std()),
        "second_call_std": float(second_np.std()),
    }


def main() -> None:
    results = {
        "environment": {
            "jax_version": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "atmospheric_turbulence": atmospheric_turbulence_example(),
        "propagation_kernels": propagation_kernels_example(),
        "scalar_interpolation": interpolation_example(),
        "functional_interpolation": functional_interpolation_example(),
        "rough_surface": rough_surface_example(),
        "auto_random": auto_random_example(),
    }

    output_json = OUT_DIR / "functionality_results.json"
    output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote functionality results to {relative_to_script(output_json)}")


if __name__ == "__main__":
    main()
