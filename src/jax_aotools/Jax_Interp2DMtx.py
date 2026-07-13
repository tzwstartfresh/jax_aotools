import jax, time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial


class jax_interp_2dmtx():
# 用于对2维矩阵函数进行插值

    def __init__(self, xaxis, yaxis, signal, cvt_ary, uniform=True):
        # 原始数据从小到大排列

        self.uniform = uniform
        xaxis, yaxis, signal, cvt_ary = [jnp.asarray(var) for var in [xaxis, yaxis, signal, cvt_ary]]
        self.xaxis, self.yaxis, self.signal = self.sort_align_func(xaxis, yaxis, signal)

        self.cvt_ary = cvt_ary
        self.coff_ary = self.get_interp_coff(self.xaxis, self.yaxis, self.signal, self.cvt_ary, self.gradient1d,
                                             self.uniform)
        self.interp_func_jit = jax.jit(self.interp_func)

    # 调用其他方法对xi,yi进行插值
    def interp(self, xi, yi):
        # 断言xi,yi范围不能大于xaxis,yaxis,即不允许外插值
        assert xi.min() >= self.xaxis.min() and xi.max() <= self.xaxis.max() and yi.min() >= self.yaxis.min() and yi.max() <= self.yaxis.max()
        return self.interp_func(xi, yi, self.xaxis, self.yaxis, self.coff_ary)

    def interp_jit(self, xi, yi):
        # 断言xi,yi范围不能大于xaxis,yaxis,即不允许外插值
        assert xi.min() >= self.xaxis.min() and xi.max() <= self.xaxis.max() and yi.min() >= self.yaxis.min() and yi.max() <= self.yaxis.max()
        return self.interp_func_jit(xi, yi, self.xaxis, self.yaxis, self.coff_ary)

    @staticmethod  # 对数据按照大小排列
    def sort_align_func(xaxis, yaxis, signal):
        # 本函数作用是将xaxis,yaxis按从小到大排列,同时signal也需进行同样排列
        if jnp.all(xaxis[1:] - xaxis[:-1]) <= 0:  # x方向为倒序则排正
            xaxis = xaxis[::-1]
            signal = signal[:, ::-1, ...]
        if jnp.all(yaxis[1:] - yaxis[:-1]) <= 0:  # y方向为倒序则排正
            yaxis = yaxis[::-1]
            signal = signal[::-1, ...]
        # x,y方向若仍未排列整齐,则重排
        if jnp.any((xaxis[1:] - xaxis[:-1]) < 0) or jnp.any((yaxis[1:] - yaxis[:-1]) < 0):
            xaxis, yaxis, signal = xaxis.sort(), yaxis.sort(), signal[yaxis.argsort(), ...][:, xaxis.argsort(), ...]
        return xaxis, yaxis, signal

    @staticmethod  # 计算一维方向梯度,可计算等间距和非等间距的数据
    def gradient1d(xaxis, signal, axis=1, uniform=True):
        # 本函数可用于利用差分方法和二次函数方法求解一维的物理梯度
        xgap_ary = jnp.abs(xaxis[1:] - xaxis[:-1])
        # assert jnp.abs(xgap_ary).mean() > 0
        assert axis in [0, 1]
        # 将xaxis重写为xaxis_new,最后一维为xaxis数据
        xshape = [1] * (signal.ndim-2)
        xshape[axis] = -1
        xaxis_new = jnp.reshape(xaxis, xshape)

        grad = jnp.zeros_like(signal)
        if not uniform:  # 坐标是非等间隔情况
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
        else:  # 坐标是等间隔情况
            grad = jnp.gradient(signal, axis=axis) / jnp.gradient(xaxis_new[..., None, None], axis=axis)

        return grad

    @staticmethod  # 计算插值系数
    def get_interp_coff(xaxis, yaxis, signal, cvt_ary, gradient1d, uniform=True):

        xaxis, yaxis, signal, cvt_ary = [jnp.asarray(var) for var in [xaxis, yaxis, signal, cvt_ary]]
        # 对数据按照x,y方向大小排列,并断言x,y坐标中不存在重复值
        delta_xary, delta_yary = xaxis[1:] - xaxis[:-1], yaxis[1:] - yaxis[:-1]
        assert jnp.abs(delta_xary).min() > 0 and jnp.abs(delta_yary).min() > 0  # 断言x,y坐标中不存在重复值

        # 计算物理梯度
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
            (4, 4, signal.shape[0] - 1, signal.shape[1] - 1, signal.shape[-2], signal.shape[-1]))  # 求出系数矩阵
        return coff_ary

    @staticmethod  # 插值函数
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


if __name__ == '__main__':
    # 以下为测试代码,测试插值函数
    basic_func = lambda x, y: x ** jnp.arange(5)[None, :] + y ** jnp.arange(4)[:, None] / 4
    xaxis = jnp.linspace(-1, 1, 200) * 5 * jnp.pi
    yaxis = jnp.linspace(-1, 1, 200) * 5 * jnp.pi

    # xaxis = jnp.asarray(np.random.rand(1000)-0.5) *10* jnp.pi
    # yaxis = jnp.asarray(np.random.rand(1000)-0.5) *10* jnp.pi
    signal = basic_func(xaxis[None, :, None, None], yaxis[:, None, None, None])

    cvt_ary = jnp.asarray(np.load('CVT_MTX.npy'))
    interp_obj = jax_interp_2dmtx(xaxis, yaxis, signal, cvt_ary, False)

    xi = jnp.linspace(0, 0.5, 500) * jnp.pi
    yi = jnp.linspace(0, 0.5, 500) * jnp.pi

    fit_ary = interp_obj.interp_jit(xi, yi)
    real_ary = basic_func(xi[None, :, None, None], yi[:, None, None, None])


    num = 100
    t1 = time.time()
    for i in range(num):
        fit_ary = interp_obj.interp(xi, yi)
    t2 = time.time()
    print('Without Jit:500*500矩阵函数双立方插值,插值帧率%.3f帧' % (num / (t2 - t1)))
    plt.figure()
    plt.plot(xi, fit_ary[0, :, -1, -1])
    plt.plot(xi, real_ary[0, :, -1, -1])
    plt.show()

    num = 1000
    t1 = time.time()
    for i in range(1000):
        fit_ary = interp_obj.interp_jit(xi, yi)
    t2 = time.time()
    print('With Jit:500*500矩阵函数双立方插值,插值帧率%.3f帧' % (num / (t2 - t1)))
    plt.figure()
    plt.plot(xi, fit_ary[0, :, -1, -1])
    plt.plot(xi, real_ary[0, :, -1, -1])
    plt.show()