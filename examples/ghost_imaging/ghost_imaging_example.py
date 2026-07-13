import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
from jax_aotools import (
    angularSpectrum_jax,
    auto_random,
    coarse_surface,
    ft_sh_phase_screen_jax_outer_scale,
    gauss_beam,
)


def load_object_mask(n, mask_size, source_path=None):
    if source_path is None:
        source_path = SCRIPT_DIR / "AIOFM.npy"
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(
            f"{source_path} was not found. Place the object mask file in the "
            "working directory or pass a different source_path."
        )
    target = jnp.asarray(np.load(source), dtype=jnp.float32)
    target = target / jnp.maximum(target.max(), 1.0)
    target = jax.image.resize(target, (mask_size, mask_size), method="cubic")
    target = jnp.clip(target, 0.0, 1.0)
    pad0 = (n - mask_size) // 2
    mask = jnp.zeros((n, n), dtype=jnp.float32)
    return mask.at[pad0:pad0 + mask_size, pad0:pad0 + mask_size].set(target)


def normalize_for_display(image):
    image = image - image.min()
    return image / (image.max() + 1e-12)


def run_ghost_imaging_example(
        n=512,
        width=1.0,
        wavelength=500e-9,
        r0_total=0.05,
        L0=300.0,
        l0=0.001,
        layer_count_each_side=10,
        layer_spacing_m=50.0,
        batch_size=100,
        repeat_num=100,
        mask_size=300,
        object_path=None,
        output_dir=None):
    if object_path is None:
        object_path = SCRIPT_DIR / "AIOFM.npy"
    if output_dir is None:
        output_dir = SCRIPT_DIR / "results"
    pixel_scale = width / n
    total_frames = batch_size * repeat_num
    layer_range = jnp.arange(2 * layer_count_each_side + 2) * layer_spacing_m
    layer_spacing = layer_range[1:] - layer_range[:-1]
    norm_strengths = jnp.ones(2 * layer_count_each_side) / (2 * layer_count_each_side)
    layer_r0 = ((r0_total ** (-5 / 3)) * norm_strengths) ** (-3.0 / 5.0)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    object_mask = load_object_mask(n, mask_size, object_path)
    phase_screen = auto_random(ft_sh_phase_screen_jax_outer_scale, seed=20250605)
    rough_surface = auto_random(coarse_surface, seed=20250606)
    input_field = gauss_beam(n, pixel_scale, waist=0.6)

    reference_sum = jnp.zeros((n, n), dtype=jnp.float32)
    correlation_sum = jnp.zeros((n, n), dtype=jnp.float32)
    bucket_sum = jnp.asarray(0.0, dtype=jnp.float32)

    t0 = time.perf_counter()
    for repeat_index in range(repeat_num):
        if (repeat_index + 1) % 10 == 0:
            print(
                "ghost-imaging batch %d/%d, accumulated frames %d"
                % (repeat_index + 1, repeat_num, (repeat_index + 1) * batch_size)
            )

        surface = rough_surface(0.5 * wavelength, 0.025, pixel_scale, n, batch_size=batch_size)
        field_img = input_field[None, ...] * jnp.exp(2j * jnp.pi * surface / wavelength)
        field_ref = input_field[None, ...] * jnp.exp(2j * jnp.pi * surface / wavelength)

        for layer_index in range(layer_count_each_side):
            phase = phase_screen(
                layer_r0[layer_index], n, pixel_scale, L0, l0,
                batch_size=batch_size, sub_order=7, zoom_scale=0.2
            )
            distance = layer_spacing[layer_index]
            field_img = angularSpectrum_jax(field_img, wavelength, pixel_scale, pixel_scale, distance)
            field_ref = angularSpectrum_jax(field_ref, wavelength, pixel_scale, pixel_scale, distance)
            field_img = field_img * jnp.exp(1j * phase)

        field_img = field_img * object_mask[None, ...]

        for layer_index in range(layer_count_each_side, 2 * layer_count_each_side):
            phase = phase_screen(
                layer_r0[layer_index], n, pixel_scale, L0, l0,
                batch_size=batch_size, sub_order=7, zoom_scale=0.2
            )
            distance = layer_spacing[layer_index]
            field_img = angularSpectrum_jax(field_img, wavelength, pixel_scale, pixel_scale, distance)
            field_ref = angularSpectrum_jax(field_ref, wavelength, pixel_scale, pixel_scale, distance)
            field_img = field_img * jnp.exp(1j * phase)

        field_img = angularSpectrum_jax(field_img, wavelength, pixel_scale, pixel_scale, layer_spacing[-1])
        field_ref = angularSpectrum_jax(field_ref, wavelength, pixel_scale, pixel_scale, layer_spacing[-1])

        image_intensity = jnp.abs(field_img) ** 2
        reference_intensity = jnp.abs(field_ref) ** 2
        bucket = image_intensity.sum(axis=(-1, -2))
        reference_sum = reference_sum + reference_intensity.sum(axis=0)
        correlation_sum = correlation_sum + (bucket[:, None, None] * reference_intensity).sum(axis=0)
        bucket_sum = bucket_sum + bucket.sum()

    reference_mean = reference_sum / total_frames
    bucket_mean = bucket_sum / total_frames
    ghost_image = correlation_sum / total_frames - bucket_mean * reference_mean
    if hasattr(ghost_image, "block_until_ready"):
        ghost_image.block_until_ready()
    elapsed = time.perf_counter() - t0

    print(
        "ghost-imaging simulation: %d frames, %.3f s, %.3f frames/s"
        % (total_frames, elapsed, total_frames / elapsed)
    )

    np.save(output_path / "ghost_image.npy", np.asarray(ghost_image))
    np.save(output_path / "object_mask.npy", np.asarray(object_mask))

    plt.figure(figsize=(10, 4), dpi=160)
    plt.subplot(1, 2, 1)
    plt.imshow(object_mask, cmap="gray")
    plt.title("Object mask")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(normalize_for_display(ghost_image), cmap="gray")
    plt.title("Ghost image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path / "ghost_imaging_example.png", bbox_inches="tight")
    plt.show()
    return ghost_image


if __name__ == "__main__":
    run_ghost_imaging_example()
