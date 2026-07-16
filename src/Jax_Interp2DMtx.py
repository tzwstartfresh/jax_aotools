import jax
import jax.numpy as jnp


class jax_interp_2dmtx():
# Bicubic interpolation for two-dimensional matrix-valued functions.

    def __init__(self, xaxis, yaxis, signal, cvt_ary, uniform=True):
        # Convert inputs to JAX arrays and align the coordinate ordering.

        self.uniform = uniform
        xaxis, yaxis, signal, cvt_ary = [jnp.asarray(var) for var in [xaxis, yaxis, signal, cvt_ary]]
        self.xaxis, self.yaxis, self.signal = self.sort_align_func(xaxis, yaxis, signal)

        self.cvt_ary = cvt_ary
        self.coff_ary = self.get_interp_coff(self.xaxis, self.yaxis, self.signal, self.cvt_ary, self.gradient1d,
                                             self.uniform)
        self.interp_func_jit = jax.jit(self.interp_func)

    # Interpolate the matrix-valued data at the query coordinates xi and yi.
    def interp(self, xi, yi):
        # The query coordinates must lie within the source grid; extrapolation is not allowed.
        assert xi.min() >= self.xaxis.min() and xi.max() <= self.xaxis.max() and yi.min() >= self.yaxis.min() and yi.max() <= self.yaxis.max()
        return self.interp_func(xi, yi, self.xaxis, self.yaxis, self.coff_ary)

    def interp_jit(self, xi, yi):
        # The query coordinates must lie within the source grid; extrapolation is not allowed.
        assert xi.min() >= self.xaxis.min() and xi.max() <= self.xaxis.max() and yi.min() >= self.yaxis.min() and yi.max() <= self.yaxis.max()
        return self.interp_func_jit(xi, yi, self.xaxis, self.yaxis, self.coff_ary)

    @staticmethod  # Sort data by coordinate value.
    def sort_align_func(xaxis, yaxis, signal):
        # Sort xaxis and yaxis in ascending order and apply the same ordering to signal.
        if jnp.all(xaxis[1:] - xaxis[:-1]) <= 0:  # Reverse x if it is descending.
            xaxis = xaxis[::-1]
            signal = signal[:, ::-1, ...]
        if jnp.all(yaxis[1:] - yaxis[:-1]) <= 0:  # Reverse y if it is descending.
            yaxis = yaxis[::-1]
            signal = signal[::-1, ...]
        # If either coordinate vector is still unordered, sort explicitly.
        if jnp.any((xaxis[1:] - xaxis[:-1]) < 0) or jnp.any((yaxis[1:] - yaxis[:-1]) < 0):
            xaxis, yaxis, signal = xaxis.sort(), yaxis.sort(), signal[yaxis.argsort(), ...][:, xaxis.argsort(), ...]
        return xaxis, yaxis, signal

    @staticmethod  # Compute a one-dimensional gradient for uniform or non-uniform samples.
    def gradient1d(xaxis, signal, axis=1, uniform=True):
        # Estimate the physical gradient along one coordinate direction.
        xgap_ary = jnp.abs(xaxis[1:] - xaxis[:-1])
        # assert jnp.abs(xgap_ary).mean() > 0
        assert axis in [0, 1]
        # Reshape xaxis so that it broadcasts along the selected signal axis.
        xshape = [1] * (signal.ndim-2)
        xshape[axis] = -1
        xaxis_new = jnp.reshape(xaxis, xshape)

        grad = jnp.zeros_like(signal)
        if not uniform:  # Non-uniform coordinate spacing.
            if axis == 1:
                x0, xi, x1 = xaxis_new[:, :-2, None, None], xaxis_new[:, 1:-1, None, None], xaxis_new[:, 2:, None, None]
                q0_x, qi_x, q1_x = signal[:, :-2, ...], signal[:, 1:-1, ...], signal[:, 2:, ...]
                grad = grad.at[:, 1:-1, ...].set(((qi_x - q0_x) / (xi - x0)) * ((x0 + x1 - 2 * xi) / (x1 - xi)) + (
                            (q1_x - q0_x) * (xi - x0) / ((x1 - xi) * (x1 - x0))))
                grad = grad.at[:, 0, ...].set(
                    (signal[:, 1, ...] - signal[:, 0, ...]) / (xaxis_new[:, 1, None, None] - xaxis_new[:, 0, None, None]))
                grad = grad.at[:, -1, ...].set(
                    (signal[:, -1, ...] - signal[:, -2, ...]) / (xaxis_new[:, -1, None, None] - xaxis_new[:, -2, None, None]))
            else:
                x0, xi, x1 = xaxis_new[:-2, :, None, None], xaxis_new[1:-1, :, None, None], xaxis_new[2:, :, None, None]
                q0_x, qi_x, q1_x = signal[:-2, ...], signal[1:-1, ...], signal[2:, ...]
                grad = grad.at[1:-1, ...].set(((qi_x - q0_x) / (xi - x0)) * ((x0 + x1 - 2 * xi) / (x1 - xi)) + (
                            (q1_x - q0_x) * (xi - x0) / ((x1 - xi) * (x1 - x0))))
                grad = grad.at[0, ...].set(
                    (signal[1, ...] - signal[0, ...]) / (xaxis_new[1, :, None, None] - xaxis_new[0, :, None, None]))
                grad = grad.at[-1, ...].set(
                    (signal[-1, ...] - signal[-2, ...]) / (xaxis_new[-1, :, None, None] - xaxis_new[-2, :, None, None]))
        else:  # Uniform coordinate spacing.
            grad = jnp.gradient(signal, axis=axis) / jnp.gradient(xaxis_new[..., None, None], axis=axis)

        return grad

    @staticmethod  # Compute interpolation coefficients.
    def get_interp_coff(xaxis, yaxis, signal, cvt_ary, gradient1d, uniform=True):

        xaxis, yaxis, signal, cvt_ary = [jnp.asarray(var) for var in [xaxis, yaxis, signal, cvt_ary]]
        # Check that the x and y coordinates do not contain duplicate values.
        delta_xary, delta_yary = xaxis[1:] - xaxis[:-1], yaxis[1:] - yaxis[:-1]
        assert jnp.abs(delta_xary).min() > 0 and jnp.abs(delta_yary).min() > 0  # Duplicate coordinates are not allowed.

        # Compute physical gradients.
        grad_dx = gradient1d(xaxis, signal, axis=1, uniform=uniform)
        grad_dy = gradient1d(yaxis, signal, axis=0, uniform=uniform)
        grad_dxdy = gradient1d(yaxis, grad_dx, axis=0, uniform=uniform)

        data_ary = jnp.stack([
            signal[:-1, :-1], signal[:-1, 1:], signal[1:, :-1], signal[1:, 1:],
            grad_dx[:-1, :-1] * delta_xary[..., None, None], grad_dx[:-1, 1:] * delta_xary[..., None, None], grad_dx[1:, :-1] * delta_xary[..., None, None],
            grad_dx[1:, 1:] * delta_xary[..., None, None],
            grad_dy[:-1, :-1] * delta_yary[..., None, None], grad_dy[:-1, 1:] * delta_yary[..., None, None], grad_dy[1:, :-1] * delta_yary[..., None, None],
            grad_dy[1:, 1:] * delta_yary[..., None, None],
            grad_dxdy[:-1, :-1] * (delta_xary[..., None, None] * delta_yary[..., None, None]), grad_dxdy[:-1, 1:] * (delta_xary[..., None, None] * delta_yary[..., None, None]),
            grad_dxdy[1:, :-1] * (delta_xary[..., None, None] * delta_yary[..., None, None]), grad_dxdy[1:, 1:] * (delta_xary[..., None, None] * delta_yary[..., None, None])])

        coff_ary = jnp.einsum('ij,jklmn->iklmn', cvt_ary, data_ary).reshape(
            (4, 4, signal.shape[0] - 1, signal.shape[1] - 1, signal.shape[-2], signal.shape[-1]))  # Coefficient matrix.
        return coff_ary

    @staticmethod  # Interpolation kernel.
    # @jax.jit
    def interp_func(xi, yi, xaxis, yaxis, coff_ary):
        xi, yi, xaxis, yaxis, coff_ary = [jnp.asarray(var) for var in [xi, yi, xaxis, yaxis, coff_ary]]

        xindex, yindex = jnp.clip(jnp.searchsorted(xaxis, xi) - 1, 0, xaxis.shape[-1] - 1), jnp.clip(
            jnp.searchsorted(yaxis, yi) - 1, 0, yaxis.shape[0] - 1)

        x0_ary, x1_ary, y0_ary, y1_ary = xaxis[xindex], xaxis[xindex + 1], yaxis[yindex], yaxis[yindex + 1]
        coff_interp_ary = coff_ary[:, :, yindex, ...][..., xindex, :, :]
        xi_norm, yi_norm = (xi - x0_ary) / (x1_ary - x0_ary), (yi - y0_ary) / (y1_ary - y0_ary)
        power_ary = jnp.arange(4)[None, :]
        signal_interp = jnp.einsum('ij,kl,ljkimn->kimn', xi_norm[:, None] ** power_ary, yi_norm[:, None] ** power_ary,
                                   coff_interp_ary)
        return signal_interp
