import jax,time
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import aotools
from scipy.special import assoc_laguerre

@partial(jax.jit, static_argnames=['N', 'batch_size'])
def ft_phase_screen_jaxbase(prngkey, r0, N, delta, L0, l0, batch_size=1, ):
    delta_freq = 1 / (N * delta)
    freq_max, freq_min = 5.92 / l0 / (2 * jnp.pi), 1 / L0  # High- and low-frequency cutoffs of the turbulence spectrum.
    fx = (jnp.arange(N) - N//2) * delta_freq
    freq_magn = jnp.sqrt(fx[None, :] ** 2 + fx[:, None] ** 2)  # Spatial-frequency radius.
    PSD_phi = ((0.023 * r0 ** (-5 / 3)) * jnp.exp(-1 * ((freq_magn / freq_max) ** 2)) / ((freq_magn ** 2 + freq_min ** 2) ** (11 / 6))) * (freq_magn > 0)  # Phase power spectrum.

    cn = (jax.random.normal(prngkey, (2, batch_size, N, N), dtype=jnp.float32) * jnp.asarray([1, 1j])[:, None, None, None]).sum(axis=(0)) * (jnp.sqrt(PSD_phi[None, ...]) * delta_freq)
    prngkey = jax.random.split(prngkey, 1)[0]

    phs_hi = jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(cn))) * N ** 2
    phs_hi -= phs_hi.mean(axis=(-1, -2))[:, None, None]
    return phs_hi.real.squeeze(), prngkey

@partial(jax.jit, static_argnames=['N', 'batch_size', 'sub_order', 'sub_samp', 'zoom_scale'])
def ft_sh_phase_screen_jax_nested(prngkey, r0, N, delta, L0, l0, batch_size=1, sub_samp=5, sub_order=4, zoom_scale=1 / 3):
    '''
    Generate phase screens with low-frequency sub-harmonic compensation.
    1. The batch_size argument enables one call to generate multiple screens.
    2. sub_order, sub_samp and zoom_scale control the number of sub-harmonic
       orders, frequency samples and low-frequency spatial down-sampling.
    3. The low-frequency component is synthesized on a reduced grid, resized
       with cubic interpolation and added to the high-frequency screen.
    :param r0: Fried parameter.
    :param N: Number of phase-screen samples along one axis.
    :param delta: Spatial sampling interval.
    :param L0: Turbulence outer scale.
    :param l0: Turbulence inner scale.
    :param batch_size: Number of phase screens generated in one call.
    :param sub_samp: Number of frequency samples per sub-harmonic axis.
    :param sub_order: Number of sub-harmonic orders.
    :param zoom_scale: Spatial scale factor for low-frequency synthesis.
    :return: Phase screen with sub-harmonic compensation.
    '''
    sub_samp = int(2 * (sub_samp // 2) + 1)
    sub_scale = sub_samp

    phs_hf, prngkey = ft_phase_screen_jaxbase(prngkey, r0, N, delta, L0, l0, batch_size=batch_size)

    delta_freq = 1 / (2 * N * delta)  # Frequencies below this boundary are compensated by sub-harmonics.
    freq_max, freq_min = 5.92 / l0 / (2 * jnp.pi), 1 / L0  # High- and low-frequency cutoffs of the turbulence spectrum.

    df_order = (delta_freq / (sub_scale ** jnp.arange(sub_order)))
    subfx = df_order[:, None, None] * jnp.linspace(-1, 1, sub_samp)[None, None, :]
    subfy = df_order[:, None, None] * jnp.linspace(-1, 1, sub_samp)[None, :, None]
    subfr = jnp.sqrt(subfx ** 2 + subfy ** 2)

    PSD_sub = ((0.023 * r0 ** (-5 / 3)) * jnp.exp(-1 * ((subfr / freq_max) ** 2)) / ((subfr ** 2 + freq_min ** 2) ** (11 / 6))) * (subfr > 0)  # Sub-harmonic power spectrum.
    cn = (((jax.random.normal(prngkey, (2, sub_order, sub_samp, sub_samp, batch_size)) * jnp.asarray([1, 1j])[:, None, None, None, None]).sum(axis=0))
          * jnp.sqrt(PSD_sub[:, :, :, None])
          * ((2 / (sub_samp - 1)) * df_order[:, None, None, None]))
    prngkey = jax.random.split(prngkey, 1)[0]
    # Dimension order: sub-harmonic order, frequency samples, batch size and spatial coordinates.
    sub_pixel = int(max(N * zoom_scale, 128))
    crood = jnp.linspace(-N / 2, N / 2, sub_pixel) * delta
    xaxis, yaxis = jnp.meshgrid(crood, crood, indexing='xy')
    phs_lf = (cn[..., None, None] * (jnp.exp(2j * jnp.pi * (subfx[..., None, None, None] * xaxis[None, None, None, None, ...] + subfy[..., None, None, None] * yaxis[None, None, None, None, ...])))).real.sum(axis=(0, 1, 2))
    phs_lf -= phs_lf.mean(axis=(-1, -2))[..., None, None]  # Remove the piston term from the low-frequency screen.
    phs_lf = jax.image.resize(phs_lf, (batch_size, N, N), method='cubic').squeeze()

    return phs_lf + phs_hf, prngkey

@partial(jax.jit, static_argnames=['N', 'batch_size', 'sub_order', 'zoom_scale'])
def ft_sh_phase_screen_jax_outer_scale(prngkey, r0, N, delta, L0, l0, batch_size=1, sub_order=7, zoom_scale=0.3):
    '''
    Generate phase screens with outer-scale-aware sub-harmonic compensation.
    1. The batch_size argument enables one call to generate multiple screens.
    2. The non-uniform low-frequency grid is determined from L0 and the FFT
       low-frequency boundary.
    3. The low-frequency component is synthesized on a reduced grid and then
       resized to the original resolution.
    :param r0: Fried parameter.
    :param N: Number of phase-screen samples along one axis.
    :param delta: Spatial sampling interval.
    :param L0: Turbulence outer scale.
    :param l0: Turbulence inner scale.
    :param batch_size: Number of phase screens generated in one call.
    :param sub_order: Number of sub-harmonic orders.
    :param zoom_scale: Spatial scale factor for low-frequency synthesis.
    :return: Phase screen with outer-scale-aware low-frequency compensation.
    '''
    sub_samp = int(2 * sub_order + 1)

    phs_hf, prngkey = ft_phase_screen_jaxbase(prngkey, r0, N, delta, L0, l0, batch_size=batch_size)

    delta_freq = 1 / (2 * N * delta)
    freq_max, freq_min = 5.92 / l0 / (2 * jnp.pi), 1 / L0  # High- and low-frequency cutoffs of the turbulence spectrum.

    # Determine the geometric scale factor.
    sub_scale = (3 * delta_freq / freq_min) ** (1 / sub_order)
    sub_scale = sub_scale * (sub_scale > 1) + 1.01 * (sub_scale < 1)

    sub_fl = delta_freq * (sub_scale / (3 * sub_scale - 2))
    sub_freq = jnp.concatenate([-1 * sub_fl / (sub_scale ** (jnp.arange(0, sub_order))), jnp.asarray([0]), (sub_fl / (sub_scale ** (jnp.arange(0, sub_order))))[::-1]])

    sub_df = jnp.gradient(sub_freq, axis=0)
    subfr = jnp.sqrt(sub_freq[:, None] ** 2 + sub_freq[None, :] ** 2)
    PSD_sub = ((0.023 * r0 ** (-5 / 3)) * jnp.exp(-1 * ((subfr / freq_max) ** 2)) / ((subfr ** 2 + freq_min ** 2) ** (11 / 6))) * (subfr > 0)  # Sub-harmonic power spectrum.

    # Dimension labels: i is complex component; j,k are frequency axes; l is batch size.
    cn = jnp.einsum('ijkl,i,jk,jk->jkl', jax.random.normal(prngkey, (2, sub_samp, sub_samp, batch_size)), jnp.asarray([1, 1j]), jnp.sqrt(PSD_sub), jnp.sqrt(sub_df[:, None] * sub_df[None, :]))
    prngkey = jax.random.split(prngkey, 1)[0]

    sub_pixel = int(max(N * zoom_scale, 128))
    crood = jnp.linspace(-N / 2, N / 2, sub_pixel) * delta
    phs_lf = jnp.einsum('ijk,ijlm->klm', cn, jnp.exp(2j * jnp.pi * ((sub_freq[None, :, None, None] * crood[None, None, None, :]) + (sub_freq[:, None, None, None] * crood[None, None, :, None])))).real

    phs_lf -= phs_lf.mean(axis=(-1, -2))[..., None, None]
    phs_lf = jax.image.resize(phs_lf, (batch_size, N, N), method='cubic').squeeze()

    return phs_lf + phs_hf, prngkey

# Backward-compatible aliases for earlier drafts and example scripts.
ft_sh_phase_screen_jaxbase1 = ft_sh_phase_screen_jax_nested
ft_sh_phase_screen_jaxbase2 = ft_sh_phase_screen_jax_outer_scale

@jax.jit
def oneStepFresnel_jax(Uin, wvl, d1, z):

    N = Uin.shape[-1]    # Assume a square grid.
    k = 2*jnp.pi/wvl  # Optical wavevector.
    # Source-plane coordinates.
    axis = jnp.arange(N)-N//2
    x1,y1 = jnp.meshgrid(axis*d1,axis*d1)
    # Observation-plane coordinates.
    d2 = wvl*z/(N*d1)
    x2,y2 = jnp.meshgrid(axis*d2,axis*d2)

    C=jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(jnp.einsum('...ij,ij->...ij',Uin,jnp.exp(1j*(k/(2*z))*(x1**2+y1**2))),axes=(-1, -2))),axes=(-1, -2))*(d1**2)

    Uout = jnp.einsum('...ij,ij->...ij',C,jnp.exp(1j*(k/(2*z))*(x2**2+y2**2))/(1j*wvl*z))

    return Uout

@jax.jit
def twoStepFresnel_jax(Uin, wvl, d1, d2, z):
    m = d2/d1
    Dz1 = z/(1-m+(m==1)*(m+1))
    Dz2= z-Dz1
    d1a = wvl*abs(Dz1)/(Uin.shape[-1]*d1)

    Uout1a = oneStepFresnel_jax(Uin,wvl,d1,Dz1)
    Uout = oneStepFresnel_jax(Uout1a,wvl,d1a,Dz2)

    return Uout

@jax.jit
def angularSpectrum_jax(Uin, wvl, d1, d2, z):
    N = Uin.shape[-1]  # Assume a square input field.
    k = 2 * jnp.pi / wvl  # Optical wavevector.
    
    (x1, y1) = jnp.meshgrid(d1 * (jnp.arange(N)-N//2),
                            d1 * (jnp.arange(N)-N//2))
    r1sq = (x1 ** 2 + y1 ** 2) + 1e-10
    
    # Spatial frequencies of the source plane.
    df1 = 1. / (N * d1)
    fX, fY = jnp.meshgrid(df1 * (jnp.arange(N)-N//2),
                          df1 * (jnp.arange(N)-N//2))
    fsq = fX ** 2 + fY ** 2
    # Scaling parameter.
    mag = d2 / d1
    
    # Observation-plane coordinates.
    x2, y2 = jnp.meshgrid(d2 * (jnp.arange(N)-N//2),
                          d2 * (jnp.arange(N)-N//2))
    r2sq = x2 ** 2 + y2 ** 2
    
    # Quadratic phase factors.
    Q1 = jnp.exp(-1j * k / 2. * (1 - mag) / z * r1sq)[None,...]
    Q2 = jnp.exp(1j * jnp.pi ** 2 * 2 * z / mag / k * fsq)[None,...]
    Q3 = jnp.exp(-1j * k / 2. * (mag - 1) / (mag * z) * r2sq)[None,...]
    
    outputComplexAmp = Q3 * jnp.fft.fftshift(
        jnp.fft.ifft2(
            jnp.fft.ifftshift(
                Q2 * (jnp.fft.fftshift(
                    jnp.fft.fft2(
                        jnp.fft.ifftshift(Q1 * Uin / mag, axes=(-1, -2)), axes=(-1, -2)
                    ), axes=(-1, -2)
                )), axes=(-1, -2)
            ), axes=(-1, -2)
        ), axes=(-1, -2)
    )
    
    return outputComplexAmp

@jax.jit
def lensAgainst_jax(Uin, wvl, d1, f):
    N = Uin.shape[-1]
    k = 2*jnp.pi/wvl
    fX = (jnp.arange(N)-N//2)/(N*d1)
    x2,y2 = jnp.meshgrid(wvl * f * fX, wvl * f * fX)
    C = jnp.fft.fftshift(
                    jnp.fft.fft2(
                        jnp.fft.ifftshift(Uin, axes=(-1, -2)), axes=(-1, -2)
                    ), axes=(-1, -2)
                )
    Uout = jnp.einsum('...ij,ij->...ij', C, jnp.exp( 1j*k/(2*f) * (x2**2 + y2**2) )/ (1j*wvl*f) )
    return Uout

@partial(jax.jit, static_argnames=['N', 'batch_size'])
def coarse_surface(prngkey,std, l0, del_len, N, batch_size=1):
    # std: standard deviation of the rough surface
    # del_len: spatial sampling interval
    # l0: correlation length
    # N: number of grid points
    tot_len = del_len * N   # Total length.
    del_freq,tot_freq = 1/tot_len,1/del_len # Grid frequency spacing and total frequency span.

    fx = (jnp.arange(N)-N//2)*del_freq
    (fx, fy) = jnp.meshgrid(fx, fx,indexing='xy')

    spectrum_ary = (std**2)*(l0**2/4*jnp.pi)*jnp.exp(-(jnp.pi**2)*(l0**2)*(fx**2+ fy**2))   # Power spectrum.
    spectrum_ary.at[N//2,N//2].set(0)
    random_ary = (jax.random.normal(prngkey,(2,batch_size,N,N))*jnp.asarray([1/jnp.sqrt(2),1j/jnp.sqrt(2)])[:,None,None,None]).sum(axis=0)
    prngkey = jax.random.split(prngkey,1)[0]

    freq_ary = 2*jnp.pi*tot_len*jnp.sqrt(spectrum_ary)[None,...] * random_ary

    surface_ary = (jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.fftshift(freq_ary,axes=(-1,-2)),axes=(-1,-2)),axes=(-1,-2)) * tot_freq**2).real
    return surface_ary.squeeze(),prngkey

def auto_random(func,seed=None, prng_key=None, return_key=False):
    if prng_key is None:
        if seed == None:
            seed = (time.time_ns() // 100)
        prngkey = jax.random.PRNGKey(seed)
    # The current random seed is determined by seed or by the current time.
    seed = (seed < 0) * (time.time_ns() // 100) + int(seed * (seed > 0))
    prngkey = jax.random.PRNGKey(seed)

    def inner_func(*args,**kwargs):
        nonlocal prngkey, return_key
        arg_ls = [prngkey]+list(args)
        # Inject the current PRNG key into the random function and update it after each call.
        phscrn, prngkey = func(*arg_ls,**kwargs)
        if return_key:
            return phscrn, prngkey
        else:
            return phscrn

    return inner_func

def gauss_beam(N, pixel_scale, waist):
    axis = jnp.linspace(-N / 2, N / 2, N) * pixel_scale
    EField = jnp.exp(-(axis[None, :] ** 2 + axis[:, None] ** 2) / waist ** 2)
    return EField
