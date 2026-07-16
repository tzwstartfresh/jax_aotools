import argparse
import json
import time
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
from Jax_Interp2D import jax_interp2d
from Jax_Tools import auto_random, ft_sh_phase_screen_jax_outer_scale


def load_bicubic_conversion_matrix(path=None):
    if path is None:
        path = SCRIPT_DIR / "CVT_MTX.npy"
    matrix_path = Path(path)
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"{path} was not found. The anisoplanatic example expects the "
            "bicubic conversion matrix to be provided in the working directory."
        )
    return np.load(matrix_path).astype(np.float32)


@nb.njit
def accumulate_row(image_out, psf_row, image_row, psf_grid_num, row_index):
    for col_index in range(psf_row.shape[0]):
        image_out[
            row_index:row_index + psf_grid_num,
            col_index:col_index + psf_grid_num,
        ] += psf_row[col_index] * image_row[col_index]
    return image_out


def load_image(path, size):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file was not found or could not be read: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def build_phase_interpolators(params, cvt_ary):
    phs_alti = np.arange(params["phase_start_m"], params["object_distance_m"], params["phase_spacing_m"])
    phs_layer_num = phs_alti.shape[0]
    phs_diam = (
        params["lens_diameter_m"]
        + (params["object_diameter_m"] - params["lens_diameter_m"]) * phs_alti / params["object_distance_m"]
    ) * params["phase_extend_scale"]
    phs_grid = np.clip(np.round(phs_diam / params["phase_reference_spacing_m"]), 512, None).astype(np.int32)
    phs_scale = phs_diam / phs_grid
    norm_strengths = np.ones_like(phs_alti) / phs_layer_num
    r0_layers = (((params["r0_total_m"] ** (-5 / 3)) * norm_strengths) ** (-3.0 / 5.0))

    phase_screen = auto_random(ft_sh_phase_screen_jax_outer_scale, seed=params["seed"])
    interpolators = []
    for layer_index in range(phs_layer_num):
        phase = phase_screen(
            r0_layers[layer_index],
            int(phs_grid[layer_index]),
            delta=float(phs_scale[layer_index]),
            L0=params["outer_scale_m"],
            l0=params["inner_scale_m"],
            sub_order=params["sub_order"],
            zoom_scale=params["zoom_scale"],
        ).astype(np.float32)
        phs_axis = np.linspace(
            -phs_diam[layer_index] / 2,
            phs_diam[layer_index] / 2,
            int(phs_grid[layer_index]),
        ).astype(np.float32)
        interpolators.append(jax_interp2d(phs_axis, phs_axis, phase, cvt_ary=cvt_ary, uniform=True))
    return phs_alti.astype(np.float32), interpolators


def run_anisoplanatic_example(
        image_path=None,
        output_dir=None,
        object_pixels=128,
        psf_grid_num=33,
        r0_total_m=0.02,
        wavelength_m=500e-9,
        lens_diameter_m=0.1,
        object_distance_m=1100.0,
        object_diameter_m=0.8,
        lens_focus_m=1.0,
        outer_scale_m=300.0,
        inner_scale_m=0.001,
        phase_start_m=100.0,
        phase_spacing_m=100.0,
        phase_extend_scale=1.05,
        phase_reference_spacing_m=0.1,
        sub_order=7,
        zoom_scale=0.2,
        seed=20250605,
        cvt_matrix_path=None):
    if image_path is None:
        image_path = SCRIPT_DIR / "chart.png"
    if output_dir is None:
        output_dir = SCRIPT_DIR / "results"
    if cvt_matrix_path is None:
        cvt_matrix_path = SCRIPT_DIR / "CVT_MTX.npy"
    params = {
        "image_path": image_path,
        "output_dir": output_dir,
        "object_pixels": object_pixels,
        "psf_grid_num": psf_grid_num,
        "r0_total_m": r0_total_m,
        "wavelength_m": wavelength_m,
        "lens_diameter_m": lens_diameter_m,
        "object_distance_m": object_distance_m,
        "object_diameter_m": object_diameter_m,
        "lens_focus_m": lens_focus_m,
        "outer_scale_m": outer_scale_m,
        "inner_scale_m": inner_scale_m,
        "phase_start_m": phase_start_m,
        "phase_spacing_m": phase_spacing_m,
        "phase_extend_scale": phase_extend_scale,
        "phase_reference_spacing_m": phase_reference_spacing_m,
        "sub_order": sub_order,
        "zoom_scale": zoom_scale,
        "seed": seed,
        "cvt_matrix_path": cvt_matrix_path,
    }
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    image = load_image(image_path, object_pixels)
    cvt_ary = jnp.asarray(load_bicubic_conversion_matrix(cvt_matrix_path))
    phase_altitudes, phase_interpolators = build_phase_interpolators(params, cvt_ary)

    obj_scale = object_diameter_m / object_pixels
    image_distance_m = 1 / (1 / lens_focus_m - 1 / object_distance_m)
    image_scale = obj_scale * image_distance_m / object_distance_m
    psf_scale = wavelength_m * image_distance_m / lens_diameter_m
    lens_grid_num = max(8, int(psf_grid_num * image_scale / psf_scale))

    obj_axis = np.linspace(-object_diameter_m / 2, object_diameter_m / 2, object_pixels).astype(np.float32)
    lens_axis = np.linspace(-lens_diameter_m / 2, lens_diameter_m / 2, lens_grid_num).astype(np.float32)
    phase_axis = (
        (obj_axis[np.newaxis, :, np.newaxis] - lens_axis[np.newaxis, np.newaxis, :])
        * phase_altitudes[:, np.newaxis, np.newaxis]
        / object_distance_m
        + lens_axis[np.newaxis, np.newaxis, :]
    )

    degraded = np.zeros((object_pixels + psf_grid_num - 1, object_pixels + psf_grid_num - 1), dtype=np.float32)
    t0 = time.perf_counter()
    for row_index in range(object_pixels):
        if (row_index + 1) % max(1, object_pixels // 8) == 0:
            print(f"anisoplanatic row {row_index + 1}/{object_pixels}")

        phase_sum = jnp.zeros((lens_grid_num, lens_grid_num * object_pixels), dtype=jnp.float32)
        for layer_index, interp in enumerate(phase_interpolators):
            y_query = phase_axis[layer_index, row_index]
            x_query = phase_axis[layer_index].flatten()
            phase_sum = phase_sum + interp.interp_jit(x_query, y_query)

        pupil_phase = phase_sum.reshape(lens_grid_num, -1, lens_grid_num).transpose((1, 0, 2))
        psf = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(jnp.exp(-1j * pupil_phase), axes=(-1, -2)), axes=(-1, -2))) ** 2
        psf = jax.image.resize(psf, (psf.shape[0], psf_grid_num, psf_grid_num), method="cubic")
        psf = jnp.clip(psf, 0.0)
        psf = psf / (psf.sum(axis=(-1, -2))[:, None, None] + 1e-12)
        degraded = accumulate_row(degraded, np.asarray(psf, dtype=np.float32), image[row_index], psf_grid_num, row_index)

    pad = psf_grid_num // 2
    degraded = degraded[pad:pad + object_pixels, pad:pad + object_pixels]
    degraded = np.clip(degraded, 0.0, 1.0)
    elapsed = time.perf_counter() - t0

    np.save(output_path / "anisoplanatic_image.npy", degraded)
    np.save(output_path / "input_image.npy", image)
    with open(output_path / "anisoplanatic_metadata.json", "w", encoding="utf-8") as f:
        json.dump({**params, "runtime_seconds": elapsed, "phase_layers": int(len(phase_altitudes)), "lens_grid_num": int(lens_grid_num)}, f, indent=2)

    plt.figure(figsize=(10, 4), dpi=160)
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title("Input image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(degraded, cmap="gray", vmin=0, vmax=1)
    plt.title("Anisoplanatic image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path / "anisoplanatic_imaging_example.png", bbox_inches="tight")
    plt.show()

    print(f"anisoplanatic imaging example: {object_pixels}x{object_pixels}, {len(phase_altitudes)} layers, {elapsed:.3f} s")
    return degraded


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Jax-AOtools anisoplanatic imaging example.")
    parser.add_argument("--image", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--object-pixels", type=int, default=128)
    parser.add_argument("--psf-grid-num", type=int, default=33)
    parser.add_argument("--phase-spacing-m", type=float, default=100.0)
    parser.add_argument("--r0-total-m", type=float, default=0.02)
    parser.add_argument("--cvt-matrix", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_anisoplanatic_example(
        image_path=args.image,
        output_dir=args.output_dir,
        object_pixels=args.object_pixels,
        psf_grid_num=args.psf_grid_num,
        phase_spacing_m=args.phase_spacing_m,
        r0_total_m=args.r0_total_m,
        cvt_matrix_path=args.cvt_matrix,
    )
