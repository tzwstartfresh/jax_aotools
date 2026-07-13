"""Consolidated validation and benchmark pipeline for Jax-AOtools.

The script runs one unified benchmark suite and writes two outputs:

    results/validation_results.json
    results/fig3_validation_comparison.png

The scalar interpolation, functional interpolation, and rough-surface tests use
only the more complete comparison implementations, so each module appears once
in the result JSON.

Usage:
    python validation_examples.py
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Callable

import aotools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

SCRIPT_DIR = Path(__file__).resolve().parent
from jax_aotools import (
    angularSpectrum_jax,
    coarse_surface,
    ft_phase_screen_jaxbase,
    ft_sh_phase_screen_jax_nested,
    ft_sh_phase_screen_jax_outer_scale,
    gauss_beam,
    jax_interp2d,
    jax_interp_2dmtx,
)

OUTDIR = SCRIPT_DIR / "results"
OUTDIR.mkdir(exist_ok=True)

RESULT_JSON = OUTDIR / "validation_results.json"
COMPARISON_FIG = OUTDIR / "fig3_validation_comparison.png"


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def block(value: Any) -> Any:
    """Block on JAX outputs so timings include actual execution time."""
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value


def time_call(func: Callable[[], Any], repeat: int = 5, warmup: int = 1) -> tuple[float, float, Any]:
    """Warm up a callable, time repeated calls, and return a trimmed mean/std."""
    for _ in range(warmup):
        block(func())

    timings: list[float] = []
    last = None
    for _ in range(repeat):
        start = time.perf_counter()
        last = block(func())
        timings.append(time.perf_counter() - start)

    timing_array = np.sort(np.asarray(timings, dtype=np.float64))
    if timing_array.size >= 5:
        timing_array = timing_array[1:-1]

    return float(np.mean(timing_array)), float(np.std(timing_array)), last


def rel_l2(a: Any, b: Any) -> float:
    """Return the relative L2 difference between two arrays."""
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    return float(np.linalg.norm(a_np - b_np) / (np.linalg.norm(b_np) + 1e-30))


def environment_info() -> dict[str, Any]:
    """Collect runtime information for reproducibility."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
    }


def bicubic_conversion_matrix() -> np.ndarray:
    """Build the 16x16 corner-constraint matrix used by bicubic interpolation."""
    rows = []
    rhs = []

    def coeff_index(i: int, j: int) -> int:
        # jax_interp2d reshapes coefficients as (y_power, x_power, y_cell, x_cell).
        return j * 4 + i

    def add_value(x: float, y: float, pos: int) -> None:
        row = np.zeros(16)
        for i in range(4):
            for j in range(4):
                row[coeff_index(i, j)] = (x**i) * (y**j)
        rows.append(row)
        rhs.append(pos)

    def add_dx(x: float, y: float, pos: int) -> None:
        row = np.zeros(16)
        for i in range(1, 4):
            for j in range(4):
                row[coeff_index(i, j)] = i * (x ** (i - 1)) * (y**j)
        rows.append(row)
        rhs.append(pos)

    def add_dy(x: float, y: float, pos: int) -> None:
        row = np.zeros(16)
        for i in range(4):
            for j in range(1, 4):
                row[coeff_index(i, j)] = j * (x**i) * (y ** (j - 1))
        rows.append(row)
        rhs.append(pos)

    def add_dxdy(x: float, y: float, pos: int) -> None:
        row = np.zeros(16)
        for i in range(1, 4):
            for j in range(1, 4):
                row[coeff_index(i, j)] = i * j * (x ** (i - 1)) * (y ** (j - 1))
        rows.append(row)
        rhs.append(pos)

    corner_points = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for pos, (x, y) in enumerate(corner_points):
        add_value(x, y, pos)
    for pos, (x, y) in enumerate(corner_points, start=4):
        add_dx(x, y, pos)
    for pos, (x, y) in enumerate(corner_points, start=8):
        add_dy(x, y, pos)
    for pos, (x, y) in enumerate(corner_points, start=12):
        add_dxdy(x, y, pos)

    lhs = np.asarray(rows)
    inv = np.linalg.inv(lhs)
    cvt = np.zeros((16, 16))
    for equation_row, data_pos in enumerate(rhs):
        cvt[:, data_pos] = inv[:, equation_row]

    return cvt.astype(np.float32)


# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------


def phase_screen_benchmark() -> dict[str, Any]:
    """Benchmark batched phase-screen generation against looped AOtools."""
    r0 = 0.05
    n = 256
    delta = 1.0 / n
    inner_scale = 0.001
    outer_scale = 300.0
    batch = 64
    key = jax.random.PRNGKey(20250605)

    def jax_fft():
        phs, _ = ft_phase_screen_jaxbase(key, r0, n, delta, outer_scale, inner_scale, batch_size=batch)
        return phs

    def jax_nested_subharmonic():
        phs, _ = ft_sh_phase_screen_jax_nested(
            key,
            r0,
            n,
            delta,
            outer_scale,
            inner_scale,
            batch_size=batch,
            sub_samp=5,
            sub_order=4,
            zoom_scale=0.25,
        )
        return phs

    def jax_outer_scale_subharmonic():
        phs, _ = ft_sh_phase_screen_jax_outer_scale(
            key,
            r0,
            n,
            delta,
            outer_scale,
            inner_scale,
            batch_size=batch,
            sub_order=7,
            zoom_scale=0.25,
        )
        return phs

    fft_mean, fft_std, _ = time_call(jax_fft, repeat=7, warmup=3)
    sh1_mean, sh1_std, _ = time_call(jax_nested_subharmonic, repeat=7, warmup=3)
    sh2_mean, sh2_std, sh2_phs = time_call(jax_outer_scale_subharmonic, repeat=7, warmup=3)

    def aotools_batch():
        screens = [
            aotools.turbulence.ft_sh_phase_screen(r0, n, delta, outer_scale, inner_scale, seed=i)
            for i in range(batch)
        ]
        return np.asarray(screens)

    aot_mean, aot_std, aot_phs = time_call(aotools_batch, repeat=3, warmup=1)
    sh2_np = np.asarray(sh2_phs)
    aot_np = np.asarray(aot_phs)

    return {
        "N": n,
        "batch_size": batch,
        "parameters": {"r0": r0, "delta": delta, "L0": outer_scale, "l0": inner_scale},
        "timing_seconds": {
            "jax_fft_batch_mean": fft_mean,
            "jax_fft_batch_std": fft_std,
            "jax_nested_subharmonic_batch_mean": sh1_mean,
            "jax_nested_subharmonic_batch_std": sh1_std,
            "jax_outer_scale_subharmonic_batch_mean": sh2_mean,
            "jax_outer_scale_subharmonic_batch_std": sh2_std,
            "aotools_subharmonic_loop_mean": aot_mean,
            "aotools_subharmonic_loop_std": aot_std,
        },
        "fps": {
            "jax_fft": batch / fft_mean,
            "jax_nested_subharmonic": batch / sh1_mean,
            "jax_outer_scale_subharmonic": batch / sh2_mean,
            "aotools_subharmonic": batch / aot_mean,
        },
        "speedup_vs_aotools": {
            "jax_fft": aot_mean / fft_mean,
            "jax_nested_subharmonic": aot_mean / sh1_mean,
            "jax_outer_scale_subharmonic": aot_mean / sh2_mean,
        },
        "statistics": {
            "jax_outer_scale_phase_mean": float(sh2_np.mean()),
            "jax_outer_scale_phase_std": float(sh2_np.std()),
            "aotools_phase_mean": float(aot_np.mean()),
            "aotools_phase_std": float(aot_np.std()),
        },
    }


def propagation_benchmark() -> dict[str, Any]:
    """Benchmark batched angular-spectrum propagation against looped AOtools."""
    n = 256
    batch = 64
    width = 1.0
    delta = width / n
    wavelength = 500e-9
    z = 50.0

    field = gauss_beam(n, delta, waist=0.18)
    fields = jnp.repeat(field[None, :, :], batch, axis=0)

    def jax_propagation():
        return angularSpectrum_jax(fields, wavelength, delta, delta, z)

    jax_mean, jax_std, jax_out = time_call(jax_propagation, repeat=11, warmup=3)
    field_np = np.asarray(field)

    def aotools_propagation_batch():
        propagated = [
            aotools.opticalpropagation.angularSpectrum(field_np, wavelength, delta, delta, z)
            for _ in range(batch)
        ]
        return np.asarray(propagated)

    aot_mean, aot_std, aot_out = time_call(aotools_propagation_batch, repeat=3, warmup=1)

    return {
        "N": n,
        "batch_size": batch,
        "parameters": {"wavelength": wavelength, "delta": delta, "z": z},
        "timing_seconds": {
            "jax_angular_spectrum_batch_mean": jax_mean,
            "jax_angular_spectrum_batch_std": jax_std,
            "aotools_angular_spectrum_loop_mean": aot_mean,
            "aotools_angular_spectrum_loop_std": aot_std,
        },
        "fps": {
            "jax_angular_spectrum": batch / jax_mean,
            "aotools_angular_spectrum": batch / aot_mean,
        },
        "speedup_vs_aotools": aot_mean / jax_mean,
        "relative_l2_first_frame": rel_l2(np.asarray(jax_out)[0], np.asarray(aot_out)[0]),
    }


def scalar_interpolation_benchmark() -> dict[str, Any]:
    """Benchmark large-grid scalar bicubic interpolation against SciPy."""
    cvt = jnp.asarray(bicubic_conversion_matrix())
    train_n = 384
    interp_n = 2048
    xaxis = jnp.linspace(-1.0, 1.0, train_n) * 5 * jnp.pi
    yaxis = jnp.linspace(-1.0, 1.0, train_n) * 5 * jnp.pi

    def base_func(x, y):
        return jnp.cos(jnp.sqrt(x**2 + 0.25 * y**2))

    signal = base_func(xaxis[None, :], yaxis[:, None])
    interp = jax_interp2d(xaxis, yaxis, signal, cvt, uniform=True)
    xi = jnp.linspace(-0.65, 0.65, interp_n) * 5 * jnp.pi
    yi = jnp.linspace(-0.65, 0.65, interp_n) * 5 * jnp.pi

    def jax_interpolation():
        return interp.interp_jit(xi, yi)

    jax_mean, jax_std, jax_fit = time_call(jax_interpolation, repeat=8, warmup=2)

    x_np = np.asarray(xaxis)
    y_np = np.asarray(yaxis)
    xi_np = np.asarray(xi)
    yi_np = np.asarray(yi)
    signal_np = np.asarray(signal)
    spline = RectBivariateSpline(y_np, x_np, signal_np, kx=3, ky=3)

    def scipy_interpolation():
        return spline(yi_np, xi_np)

    scipy_mean, scipy_std, scipy_fit = time_call(scipy_interpolation, repeat=5, warmup=1)

    truth = np.asarray(base_func(xi[None, :], yi[:, None]))
    jax_np = np.asarray(jax_fit)

    return {
        "input_grid": [train_n, train_n],
        "output_grid": [interp_n, interp_n],
        "timing_seconds": {
            "jax_bicubic_mean": jax_mean,
            "jax_bicubic_std": jax_std,
            "scipy_rect_bivariate_spline_mean": scipy_mean,
            "scipy_rect_bivariate_spline_std": scipy_std,
        },
        "fps": {
            "jax_bicubic": 1.0 / jax_mean,
            "scipy_rect_bivariate_spline": 1.0 / scipy_mean,
        },
        "speedup_vs_scipy": scipy_mean / jax_mean,
        "accuracy_vs_analytic": {
            "jax_relative_l2": rel_l2(jax_np, truth),
            "jax_max_abs": float(np.max(np.abs(jax_np - truth))),
            "scipy_relative_l2": rel_l2(scipy_fit, truth),
            "scipy_max_abs": float(np.max(np.abs(scipy_fit - truth))),
        },
    }


def functional_interpolation_benchmark() -> dict[str, Any]:
    """Benchmark matrix-valued bicubic interpolation against a SciPy baseline."""
    cvt = jnp.asarray(bicubic_conversion_matrix())
    grid_n = 48
    patch_n = 16
    interp_n = 192
    xaxis = jnp.linspace(-1.0, 1.0, grid_n)
    yaxis = jnp.linspace(-1.0, 1.0, grid_n)
    u = jnp.linspace(-1.0, 1.0, patch_n)
    xx, yy = jnp.meshgrid(u, u, indexing="xy")
    centers_x = xaxis[None, :, None, None]
    centers_y = yaxis[:, None, None, None]
    width = 0.26 + 0.04 * jnp.sin(jnp.pi * centers_x) * jnp.cos(jnp.pi * centers_y)
    signal = jnp.exp(-((xx[None, None] - centers_x) ** 2 + (yy[None, None] - centers_y) ** 2) / width**2)

    interp = jax_interp_2dmtx(xaxis, yaxis, signal, cvt, uniform=True)
    xi = jnp.linspace(-0.8, 0.8, interp_n)
    yi = jnp.linspace(-0.8, 0.8, interp_n)

    def jax_interpolation():
        return interp.interp_jit(xi, yi)

    jax_mean, jax_std, jax_out = time_call(jax_interpolation, repeat=9, warmup=3)

    x_np = np.asarray(xaxis)
    y_np = np.asarray(yaxis)
    xi_np = np.asarray(xi)
    yi_np = np.asarray(yi)
    xxq, yyq = np.meshgrid(xi_np, yi_np, indexing="xy")
    points = np.column_stack([yyq.ravel(), xxq.ravel()])
    rgi = RegularGridInterpolator((y_np, x_np), np.asarray(signal), method="linear", bounds_error=True)

    def scipy_interpolation():
        return rgi(points).reshape(interp_n, interp_n, patch_n, patch_n)

    scipy_mean, scipy_std, scipy_out = time_call(scipy_interpolation, repeat=5, warmup=1)

    return {
        "input_grid": [grid_n, grid_n],
        "output_grid": [interp_n, interp_n],
        "functional_patch": [patch_n, patch_n],
        "output_shape": list(np.asarray(jax_out).shape),
        "baseline": "scipy.interpolate.RegularGridInterpolator, linear, vector-valued values",
        "timing_seconds": {
            "jax_functional_bicubic_mean": jax_mean,
            "jax_functional_bicubic_std": jax_std,
            "scipy_regular_grid_linear_mean": scipy_mean,
            "scipy_regular_grid_linear_std": scipy_std,
        },
        "fps": {
            "jax_functional_bicubic": 1.0 / jax_mean,
            "scipy_regular_grid_linear": 1.0 / scipy_mean,
        },
        "speedup_vs_scipy": scipy_mean / jax_mean,
        "relative_l2_between_methods": rel_l2(np.asarray(jax_out), scipy_out),
    }


def numpy_coarse_surface(key_seed: int, std: float, corr_len: float, delta: float, n: int, batch_size: int) -> np.ndarray:
    """NumPy reference implementation of Gaussian-spectrum rough-surface synthesis."""
    rng = np.random.default_rng(key_seed)
    total_length = delta * n
    delta_freq = 1 / total_length
    total_freq = 1 / delta

    freq_axis = (np.arange(n) - n // 2) * delta_freq
    fx, fy = np.meshgrid(freq_axis, freq_axis, indexing="xy")
    spectrum = (std**2) * (corr_len**2 / 4 * np.pi) * np.exp(
        -(np.pi**2) * (corr_len**2) * (fx**2 + fy**2)
    )
    spectrum[n // 2, n // 2] = 0

    random_complex = (
        rng.normal(size=(batch_size, n, n)) / np.sqrt(2)
        + 1j * rng.normal(size=(batch_size, n, n)) / np.sqrt(2)
    )
    frequency_field = 2 * np.pi * total_length * np.sqrt(spectrum)[None, ...] * random_complex

    surface = np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(frequency_field, axes=(-1, -2)), axes=(-1, -2)),
        axes=(-1, -2),
    )
    return (surface * total_freq**2).real


def rough_surface_benchmark() -> dict[str, Any]:
    """Benchmark batched rough-surface synthesis against a NumPy FFT baseline."""
    n = 256
    batch = 64
    target_std = 0.25e-6
    corr_len = 0.025
    delta = 1.0 / n
    key = jax.random.PRNGKey(20250606)

    def jax_surface():
        out, _ = coarse_surface(key, target_std, corr_len, delta, n, batch_size=batch)
        return out

    jax_mean, jax_std, jax_out = time_call(jax_surface, repeat=10, warmup=2)

    def numpy_surface():
        return numpy_coarse_surface(20250606, target_std, corr_len, delta, n, batch)

    numpy_mean, numpy_std, numpy_out = time_call(numpy_surface, repeat=5, warmup=1)

    return {
        "N": n,
        "batch_size": batch,
        "parameters": {"target_std": target_std, "correlation_length": corr_len, "delta": delta},
        "baseline": "NumPy FFT implementation of the same Gaussian power-spectrum method",
        "timing_seconds": {
            "jax_coarse_surface_batch_mean": jax_mean,
            "jax_coarse_surface_batch_std": jax_std,
            "numpy_fft_surface_batch_mean": numpy_mean,
            "numpy_fft_surface_batch_std": numpy_std,
        },
        "fps": {
            "jax_coarse_surface": batch / jax_mean,
            "numpy_fft_surface": batch / numpy_mean,
        },
        "speedup_vs_numpy": numpy_mean / jax_mean,
        "statistics": {
            "jax_std": float(np.asarray(jax_out).std()),
            "numpy_std": float(np.asarray(numpy_out).std()),
        },
    }


# -----------------------------------------------------------------------------
# Output and plotting
# -----------------------------------------------------------------------------


def run_benchmarks() -> dict[str, Any]:
    """Run the complete benchmark suite and return one flat result dictionary."""
    results: dict[str, Any] = {"environment": environment_info()}
    benchmark_suite: list[tuple[str, str, Callable[[], dict[str, Any]]]] = [
        ("phase_screen", "phase-screen generation", phase_screen_benchmark),
        ("propagation", "angular-spectrum propagation", propagation_benchmark),
        ("scalar_interpolation", "large scalar bicubic interpolation", scalar_interpolation_benchmark),
        ("functional_interpolation", "matrix-valued bicubic interpolation", functional_interpolation_benchmark),
        ("rough_surface", "rough-surface synthesis", rough_surface_benchmark),
    ]

    for key, label, benchmark in benchmark_suite:
        print(f"Running benchmark: {label}...")
        results[key] = benchmark()

    return results


def write_json(path: Path, results: dict[str, Any]) -> None:
    """Write benchmark results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def make_speed_figure(results: dict[str, Any], output_path: Path = COMPARISON_FIG) -> Path:
    """Create the throughput comparison figure."""
    phase = results["phase_screen"]
    prop = results["propagation"]
    interp = results["scalar_interpolation"]
    func_interp = results["functional_interpolation"]
    surface = results["rough_surface"]

    labels = [
        "Phase screen",
        "Angular\nspectrum",
        "Bicubic\ninterp.",
        "Functional\ninterp.",
        "Rough\nsurface",
    ]
    jax_fps = [
        phase["fps"]["jax_outer_scale_subharmonic"],
        prop["fps"]["jax_angular_spectrum"],
        interp["fps"]["jax_bicubic"],
        func_interp["fps"]["jax_functional_bicubic"],
        surface["fps"]["jax_coarse_surface"],
    ]
    baseline_fps = [
        phase["fps"]["aotools_subharmonic"],
        prop["fps"]["aotools_angular_spectrum"],
        interp["fps"]["scipy_rect_bivariate_spline"],
        func_interp["fps"]["scipy_regular_grid_linear"],
        surface["fps"]["numpy_fft_surface"],
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=200)
    ax.bar(x - width / 2, jax_fps, width, label="Jax-AOtools")
    ax.bar(x + width / 2, baseline_fps, width, label="AOtools/SciPy/NumPy baseline")
    ax.set_yscale("log")
    ax.set_ylabel("Throughput (frames s$^{-1}$)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(4, max(jax_fps + baseline_fps) * 2.2)
    ax.grid(axis="y", which="both", linewidth=0.5, alpha=0.55)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, borderaxespad=0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")
    return output_path


def run_pipeline() -> None:
    """Run all benchmarks, write one JSON file, and generate one comparison figure."""
    results = run_benchmarks()
    write_json(RESULT_JSON, results)
    make_speed_figure(results)


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
