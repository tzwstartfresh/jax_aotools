import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_genlaguerre

SCRIPT_DIR = Path(__file__).resolve().parent
from Jax_Tools import angularSpectrum_jax, ft_sh_phase_screen_jax_outer_scale


@dataclass
class OAMQKDParams:
    grid_size: int = 128
    wavelength: float = 532e-9
    width_in: float = 0.30
    width_out: float = 0.30
    propagation_distance: float = 1100.0
    beam_waist: float = 0.03
    r0: float = 0.10
    outer_scale: float = 50.0
    inner_scale: float = 1e-10
    screen_count: int = 5
    batch_size: int = 32
    sub_order: int = 5
    zoom_scale: float = 0.2
    seed: int = 20250605
    output_dir: str = str(SCRIPT_DIR / "results")

    @property
    def pixel_scale_in(self):
        return self.width_in / self.grid_size

    @property
    def pixel_scale_out(self):
        return self.width_out / self.grid_size


def laguerre_gauss_field(p_rad, l_oam, params, width=None):
    """Generate a normalized Laguerre-Gaussian mode on a square grid."""
    width = params.width_in if width is None else width
    coords = np.linspace(-width / 2, width / 2, params.grid_size, dtype=np.float32)
    x, y = np.meshgrid(coords, coords, indexing="xy")
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    rho = 2 * r**2 / params.beam_waist**2
    radial = (np.sqrt(2) * r / params.beam_waist) ** abs(l_oam)
    radial *= eval_genlaguerre(p_rad, abs(l_oam), rho)
    envelope = np.exp(-r**2 / params.beam_waist**2)
    phase = np.exp(1j * l_oam * theta)
    field = radial * envelope * phase
    return field / np.sqrt((np.abs(field) ** 2).sum() + 1e-30)


def angular_laguerre_gauss_field(index, mode_list, params, width=None):
    """Generate the angular basis as a discrete Fourier transform of OAM modes."""
    ndim = len(mode_list)
    fields = np.asarray([laguerre_gauss_field(p, l, params, width=width) for p, l in mode_list])
    weights = np.asarray([np.exp(-1j * 2 * np.pi * index * l / ndim) for _, l in mode_list])
    field = np.einsum("kij,k->ij", fields, weights) / np.sqrt(ndim)
    return field / np.sqrt((np.abs(field) ** 2).sum() + 1e-30)


def vacuum_propagate(field, params):
    out = angularSpectrum_jax(
        jnp.asarray(field)[None, :, :],
        params.wavelength,
        params.pixel_scale_in,
        params.pixel_scale_out,
        params.propagation_distance,
    )[0]
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    out = np.asarray(out)
    return out / np.sqrt((np.abs(out) ** 2).sum() + 1e-30)


def projection_bases(mode_list, params):
    ndim = len(mode_list)
    lgb_input = np.asarray(
        [laguerre_gauss_field(p, l, params, width=params.width_out) for p, l in mode_list]
    )
    angular_input = np.asarray(
        [angular_laguerre_gauss_field(i, mode_list, params, width=params.width_out) for i in range(ndim)]
    )
    lgb_measure = np.asarray([vacuum_propagate(field, params) for field in lgb_input])
    angular_measure = np.asarray([vacuum_propagate(field, params) for field in angular_input])
    return lgb_input, angular_input, lgb_measure, angular_measure


def propagate_through_turbulence(input_field, params, key):
    """Propagate one input mode through batched phase screens without aperture masking."""
    field = jnp.repeat(jnp.asarray(input_field)[None, :, :], params.batch_size, axis=0)
    layer_spacing = params.propagation_distance / params.screen_count
    layer_strengths = jnp.ones(params.screen_count) / params.screen_count
    layer_r0 = (((params.r0 ** (-5 / 3)) * layer_strengths) ** (-3.0 / 5.0))
    for layer_index in range(params.screen_count):
        key, subkey = jax.random.split(key)
        phase, _ = ft_sh_phase_screen_jax_outer_scale(
            subkey,
            layer_r0[layer_index],
            params.grid_size,
            params.pixel_scale_in,
            params.outer_scale,
            params.inner_scale,
            batch_size=params.batch_size,
            sub_order=params.sub_order,
            zoom_scale=params.zoom_scale,
        )
        field = angularSpectrum_jax(
            field,
            params.wavelength,
            params.pixel_scale_in,
            params.pixel_scale_out,
            layer_spacing,
        )
        field = field * jnp.exp(1j * phase)
    return field, key


def channel_matrix(input_fields, measurement_basis, params, key):
    transmitted = []
    t0 = time.perf_counter()
    for field in input_fields:
        out, key = propagate_through_turbulence(field, params, key)
        transmitted.append(np.asarray(out))
    transmitted = np.asarray(transmitted)
    elapsed = time.perf_counter() - t0
    overlaps = np.einsum("mfij,kij->mfk", transmitted, measurement_basis.conj())
    probabilities = np.abs(overlaps) ** 2
    cmtx = probabilities.mean(axis=1)
    cmtx = (cmtx.T / (cmtx.sum(axis=1) + 1e-30)).T
    return cmtx, elapsed, key


def avg_qerr_rate_onebasis(cmtx):
    off_diag = cmtx[~np.eye(cmtx.shape[0], dtype=bool)].reshape(cmtx.shape[0], -1)
    return float(np.mean(np.sum(off_diag, axis=1)))


def avg_qerr_rate(lgb_cmtx, angular_cmtx):
    return 0.5 * (avg_qerr_rate_onebasis(lgb_cmtx) + avg_qerr_rate_onebasis(angular_cmtx))


def secret_key_rate(lgb_cmtx, angular_cmtx):
    qerr = avg_qerr_rate(lgb_cmtx, angular_cmtx)
    dim = len(lgb_cmtx)
    if qerr <= 0:
        return float(np.log2(dim))
    if qerr >= 1:
        return 0.0
    raw = np.log2(dim) + 2 * (
        qerr * np.log2(qerr / (dim - 1)) + (1 - qerr) * np.log2(1 - qerr)
    )
    return float(max(raw, 0.0))


def run_qkd_example(params=None, mode_list=None):
    params = OAMQKDParams() if params is None else params
    mode_list = np.asarray([[0, -1], [0, 0], [0, 1]], dtype=int) if mode_list is None else np.asarray(mode_list, dtype=int)
    output_dir = Path(params.output_dir)
    output_dir.mkdir(exist_ok=True)

    key = jax.random.PRNGKey(params.seed)
    lgb_input, angular_input, lgb_measure, angular_measure = projection_bases(mode_list, params)
    lgb_cmtx, lgb_time, key = channel_matrix(lgb_input, lgb_measure, params, key)
    angular_cmtx, angular_time, key = channel_matrix(angular_input, angular_measure, params, key)
    qber = avg_qerr_rate(lgb_cmtx, angular_cmtx)
    key_rate = secret_key_rate(lgb_cmtx, angular_cmtx)
    total_frames = params.batch_size * len(mode_list) * 2
    total_time = lgb_time + angular_time

    results = {
        "params": asdict(params),
        "mode_list": mode_list.tolist(),
        "lgb_channel_matrix": lgb_cmtx.tolist(),
        "angular_channel_matrix": angular_cmtx.tolist(),
        "lgb_qber": avg_qerr_rate_onebasis(lgb_cmtx),
        "angular_qber": avg_qerr_rate_onebasis(angular_cmtx),
        "average_qber": qber,
        "secret_key_rate_bits_per_photon": key_rate,
        "simulation_time_seconds": total_time,
        "effective_frames": total_frames,
    }
    (output_dir / "qkd_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.save(output_dir / "lgb_channel_matrix.npy", lgb_cmtx)
    np.save(output_dir / "angular_channel_matrix.npy", angular_cmtx)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), dpi=180)
    for ax, matrix, title in [
        (axes[0], lgb_cmtx, "OAM basis"),
        (axes[1], angular_cmtx, "Angular basis"),
    ]:
        im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("Detected mode")
        ax.set_ylabel("Transmitted mode")
        ax.set_xticks(range(len(mode_list)))
        ax.set_yticks(range(len(mode_list)))
        labels = [str(l) for _, l in mode_list]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"QBER={qber:.3f}, key rate={key_rate:.3f} bits/photon")
    fig.tight_layout()
    fig.savefig(output_dir / "qkd_channel_matrices.png", bbox_inches="tight")
    plt.show()

    print(
        "OAM-QKD example: %d modes, %d frames, %.3f s, QBER %.3f, key rate %.3f bits/photon"
        % (len(mode_list), total_frames, total_time, qber, key_rate)
    )
    return results


if __name__ == "__main__":
    run_qkd_example()
