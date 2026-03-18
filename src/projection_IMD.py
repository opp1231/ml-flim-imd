import numpy as np
import tifffile as tf
import os
import argparse
from pathlib import Path
from skimage.filters import gaussian, threshold_otsu
from skimage.exposure import match_histograms
import matplotlib as mpl
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from scipy import fftpack, ndimage
import re
import colorcet as cc
import pywt
import warnings

warnings.filterwarnings("ignore")

cmap_list = [temp.split(',')[0] for temp in [cc.get_aliases(name) for name in cc.all_original_names()]]

def parse_args():
    parser = argparse.ArgumentParser(description='Preidce Lifetimes with DLFLIM')
    parser.add_argument('--inpath', type=Path, help='path to the parent data folder')
    parser.add_argument('--savepath', type=Path, help='path to the save folder')
    parser.add_argument('--nz', nargs='?',default=1,type=int, help='number of z-slices in one volume')
    parser.add_argument('--nt', nargs='?',default=30,type=int, help='number of flim planes in one flim stack')
    parser.add_argument('--version', nargs='?',default=1,type=int, help='single or double decay')
    parser.add_argument('--min_tau', nargs='?',default=1.0,type=float, help='minimum tau')
    parser.add_argument('--max_tau', nargs='?',default=4.0,type=float, help='maximum tau')
    parser.add_argument('--gauss_sig', nargs='?',default=0.5,type=float, help='sigma for gaussian filtering intensity')
    parser.add_argument('--noise', nargs='?',default=10,type=int, help='noise level (see pystripe)')
    parser.add_argument('--sigmay', nargs='?',default=64,type=float, help='sigma_y (see pystripe)')
    parser.add_argument('--sigmax', nargs='?',default=128,type=float, help='sigma_x (see pystripe)')
    parser.add_argument('--start_t', nargs='?',default=1,type=int, help='starting time point for processing')
    parser.add_argument('--end_t', nargs='?',default=-1,type=int, help='ending time point for processing')
    parser.add_argument('--int_sat', nargs='?',default=100.0,type=float, help='intensity saturation as a percentage')
    parser.add_argument('--channel', nargs='?',default=0, type=int, choices=[0,1], help='channel to process')
    parser.add_argument('--cmap', nargs='?',default='rainbow4',type=str, help='colormap for imd')
    parser.add_argument('--project', '-p', default=False,action='store_true',dest='proj',help='run the sum projection')
    parser.add_argument('--destripe', '-ds', default=False,action='store_true',dest='destripe',help='run the destriping')
    parser.add_argument('--imd', '-im', default=False,action='store_true',dest='imd',help='run the intensity modulation')
    parser.add_argument('--total_norm', '-tn', default=False,action='store_true',dest='norm',
                        help='normalize intensity over the entire timecourse (false) or frame-by-frame (True, VERY slow)')
    args = parser.parse_args()
    return args

def input_flim_vol(file_path,nt,nz,sz_x=512,sz_y=512):
    A = np.fromfile(file_path,dtype='uint16',sep="")
    temp = np.zeros((nt,nz,sz_x,sz_y))
    for j in range(nz):
        for n in range(nt):
            temp[n,j,:,:] = np.reshape(A[(j*nt+n)*(sz_x*sz_y):(j*nt+n+1)*(sz_x*sz_y)],[sz_y,sz_x])
    
    return temp

def im2DLUT(Ac,Aint,cmap=None,cLim=None,intLim=None):
    
    if cmap is None:
        obj = 'magma'
        cmap = mpl.colormaps.get_cmap(obj)

    if cLim is None:
        cLim = np.array([np.amin(Ac), np.amax(Ac)])
        
    if intLim is None:
        intLim = np.percentile(Aint, [0.00,99.5])
        
    Nc = cmap.N
    divMap = rgb_to_hsv(np.array([cmap(i)[:-1] for i in range(Nc)]))
    divMap = np.reshape(divMap,[Nc,1,3])
    divMap = np.repeat(divMap,Nc,axis=1)

    obj = 'cet_CET_L1'
    intMap = mpl.colormaps.get_cmap(obj) 
    inT = np.mean(np.array([intMap(i)[:-1] for i in range(Nc)]),axis=1)
    divMap[...,-1] = np.repeat(inT[np.newaxis,:],Nc,axis=0)

    divMap = hsv_to_rgb(divMap)
    
    hueBins = np.linspace(cLim[0],cLim[-1],num=256)
    hueBins[0] = -np.inf
    hueBins[-1] = np.inf
    
    hue = np.digitize(Ac,hueBins)
    
    briBins = np.linspace(intLim[0],intLim[-1],num=256)
    briBins[0] = -np.inf
    briBins[-1] = np.inf
    
    bri = np.digitize(Aint,briBins)
    
    badVals = np.logical_or(np.isnan(hue),np.isnan(bri))
    hue[badVals] = 1
    bri[badVals] = 1

    LUTim = np.zeros_like(Ac)
    LUTim = np.repeat(LUTim[...,np.newaxis],3,axis=-1)
    LUTim[...,:] = divMap[np.newaxis,hue[...],bri[...],:]

    return divMap, LUTim

def wavedec(img, wavelet, level=None):
    """Decompose `img` using discrete (decimated) wavelet transform using `wavelet`

    Parameters
    ----------
    img : ndarray
        image to be decomposed into wavelet coefficients
    wavelet : str
        name of the mother wavelet
    level : int (optional)
        number of wavelet levels to use. Default is the maximum possible decimation

    Returns
    -------
    coeffs : list
        the approximation coefficients followed by detail coefficient tuple for each level

    """
    return pywt.wavedec2(img, wavelet, mode='symmetric', level=level, axes=(-2, -1))


def waverec(coeffs, wavelet):
    """Reconstruct an image using a multilevel 2D inverse discrete wavelet transform

    Parameters
    ----------
    coeffs : list
        the approximation coefficients followed by detail coefficient tuple for each level
    wavelet : str
        name of the mother wavelet

    Returns
    -------
    img : ndarray
        reconstructed image

    """
    return pywt.waverec2(coeffs, wavelet, mode='symmetric', axes=(-2, -1))

def fft(data, axis=-1, shift=True):
    """Computes the 1D Fast Fourier Transform of an input array

    Parameters
    ----------
    data : ndarray
        input array to transform
    axis : int (optional)
        axis to perform the 1D FFT over
    shift : bool
        indicator for centering the DC component

    Returns
    -------
    fdata : ndarray
        transformed data

    """
    fdata = fftpack.rfft(data, axis=axis)
    # fdata = fftpack.rfft(fdata, axis=0)
    if shift:
        fdata = fftpack.fftshift(fdata)
    return fdata


def ifft(fdata, axis=-1):
    # fdata = fftpack.irfft(fdata, axis=0)
    return fftpack.irfft(fdata, axis=axis)

def notch(n, sigma):
    """Generates a 1D gaussian notch filter `n` pixels long

    Parameters
    ----------
    n : int
        length of the gaussian notch filter
    sigma : float
        notch width

    Returns
    -------
    g : ndarray
        (n,) array containing the gaussian notch filter

    """
    if n <= 0:
        raise ValueError('n must be positive')
    else:
        n = int(n)
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    x = np.arange(n)
    g = 1 - np.exp(-x ** 2 / (2 * sigma ** 2))
    return g

def gaussian_filter(shape, sigma):
    """Create a gaussian notch filter

    Parameters
    ----------
    shape : tuple
        shape of the output filter
    sigma : float
        filter bandwidth

    Returns
    -------
    g : ndarray
        the impulse response of the gaussian notch filter

    """
    g = notch(n=shape[-1], sigma=sigma)
    g_mask = np.broadcast_to(g, shape).copy()
    return g_mask

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def foreground_fraction(img, center, crossover, smoothing):
    z = (img-center)/crossover
    f = sigmoid(z)
    return ndimage.gaussian_filter(f, sigma=smoothing)

def filter_subband(img, sigma, level, wavelet):
    img_log = np.log(1 + img)

    if level == 0:
        coeffs = wavedec(img_log, wavelet)
    else:
        coeffs = wavedec(img_log, wavelet, level)
    approx = coeffs[0]
    detail = coeffs[1:]

    width_frac = sigma / img.shape[0]
    coeffs_filt = [approx]
    for ch, cv, cd in detail:
        s = ch.shape[0] * width_frac
        fch = fft(ch, shift=False)
        g = gaussian_filter(shape=fch.shape, sigma=s)
        fch_filt = fch * g
        ch_filt = ifft(fch_filt)
        coeffs_filt.append((ch_filt, cv, cd))

    img_log_filtered = waverec(coeffs_filt, wavelet)
    return np.exp(img_log_filtered)-1

def filter_streaks(img, sigma, level=0, wavelet='db3', crossover=10, threshold=-1, flat=None, dark=0):
    """Filter horizontal streaks using wavelet-FFT filter

    Parameters
    ----------
    img : ndarray
        input image array to filter
    sigma : float or list
        filter bandwidth(s) in pixels (larger gives more filtering)
    level : int
        number of wavelet levels to use
    wavelet : str
        name of the mother wavelet
    crossover : float
        intensity range to switch between filtered background and unfiltered foreground
    threshold : float
        intensity value to separate background from foreground. Default is Otsu
    flat : ndarray
        reference image for illumination correction. Must be same shape as input images. Default is None
    dark : float
        Intensity to subtract from the images for dark offset. Default is 0.

    Returns
    -------
    fimg : ndarray
        filtered image

    """
    smoothing = 1

    if threshold == -1:
        try:
            threshold = threshold_otsu(img)
        except ValueError:
            threshold = 1

    img = np.array(img, dtype=np.float64)
    #
    # Need to pad image to multiple of 2
    #
    pady, padx = [_ % 2 for _ in img.shape]
    if pady == 1 or padx == 1:
        img = np.pad(img, ((0, pady), (0, padx)), mode="edge")

    # TODO: Clean up this logic with some dual-band CLI alternative
    sigma1 = sigma[0]  # foreground
    sigma2 = sigma[1]  # background
    if sigma1 > 0:
        if sigma2 > 0:
            if sigma1 == sigma2:  # Single band
                fimg = filter_subband(img, sigma1, level, wavelet)
            else:  # Dual-band
                background = np.clip(img, None, threshold)
                foreground = np.clip(img, threshold, None)
                background_filtered = filter_subband(background, sigma[1], level, wavelet)
                foreground_filtered = filter_subband(foreground, sigma[0], level, wavelet)
                # Smoothed homotopy
                f = foreground_fraction(img, threshold, crossover, smoothing=1)
                fimg = foreground_filtered * f + background_filtered * (1 - f)
        else:  # Foreground filter only
            foreground = np.clip(img, threshold, None)
            foreground_filtered = filter_subband(foreground, sigma[0], level, wavelet)
            # Smoothed homotopy
            f = foreground_fraction(img, threshold, crossover, smoothing=1)
            fimg = foreground_filtered * f + img * (1 - f)
    else:
        if sigma2 > 0:  # Background filter only
            background = np.clip(img, None, threshold)
            background_filtered = filter_subband(background, sigma[1], level, wavelet)
            # Smoothed homotopy
            f = foreground_fraction(img, threshold, crossover, smoothing=1)
            fimg = img * f + background_filtered * (1 - f)
        else:
            # sigma1 and sigma2 are both 0, so skip the destriping
            fimg = img

    if padx > 0:
        fimg = fimg[:, :-padx]
    if pady > 0:
        fimg = fimg[:-pady]
    return fimg

def sum_projection(ddir,time,nz=1,nt=30,noise_level=10,sig=0.5,sigmay=64,sigmax=128,destripe=True):

    data = input_flim_vol(os.path.join(ddir,time),nt=nt,nz=nz)

    if destripe:
        img_corr_py = np.zeros((data.shape[1],data.shape[2],data.shape[3]))
        # img_corr_py = vsnr2d(np.sum(data,axis=0), filters)
        for jj in range(data.shape[1]):
            img_corr_py[jj,:,:] = filter_streaks(np.sum(data[:,jj,:,:],axis=0), sigma=[sigmay,sigmax], level=noise_level, wavelet='db2')

    else:

        img_corr_py = np.sum(data,axis=0)

    if len(img_corr_py.shape) < 3:
        ret = gaussian(img_corr_py[np.newaxis,...],sigma=(0.0,sig,sig))
    
    else:

        ret = gaussian(img_corr_py,sigma=(0.0,sig,sig))

    return ret

def project_and_save(times,ddir,save_dir,nz=1,nt=30,nx=512,ny=512,sig=0.5,noise_level=10,sigmay=64,sigmax=128,destripe=True):

    data_clean = np.zeros((len(times),nz,nx,ny))

    for i, time in zip(range(len(times)),times):

        if len(times) > 1000:
            temp = i%100
        else:
            temp = i%10

        if not temp:
            t = re.split('_',re.split(r'TM',time)[-1])[0]
            print('Processing timepoint {}'.format(t))
        t = re.split('_',re.split(r'TM',time)[-1])[0]

        data_clean[i,...] = sum_projection(ddir,time,nz=nz,nt=nt,sig=sig,
                                            sigmay=sigmay,sigmax=sigmax,noise_level=noise_level,destripe=destripe)

        if i > 0:

            data_clean[i,...] = match_histograms(data_clean[i,...],data_clean[0,...],channel_axis=None)

    tf.imwrite(os.path.join(save_dir,'intensity_smooth.tif'),data_clean,metadata={'axes': 'TZYX'})

    return data_clean

if __name__ == '__main__':

    args = parse_args()

    obj = args.cmap

    cmap = mpl.colormaps.get_cmap('cet_' + obj)

    minima = args.min_tau
    maxima = args.max_tau
    int_sat = args.int_sat

    nt = args.nt
    nz = args.nz
    nx = 512
    ny = 512

    sigmay = args.sigmay
    sigmax = args.sigmax

    gauss_sig = args.gauss_sig

    noise_lvl = args.noise

    ddirs = [args.inpath]

    for ddir in ddirs:
        print('Processing files in directory {}'.format(ddir))
        # save_dir = os.path.join(ddir,'taus_cnn')
        save_dir = args.savepath

        if args.channel == 0:

            times = [f for f in os.listdir(ddir) if re.search(r'ANG000_CHN00',f)]
            

        elif args.channel == 1:

            times = [f for f in os.listdir(ddir) if re.search(r'ANG000_CHN01',f)]

        else:

            exit(f'error: channel must be 0 or 1')

        if len(times) == 1:
            
            times = times

        else:

            times = times[args.start_t:args.end_t]

        if args.proj:

            print('Creating sum projection of FLIM stacks...')

            data_clean = project_and_save(times=times,ddir=ddir,save_dir=save_dir,sig=gauss_sig,
                                            sigmay=sigmay,sigmax=sigmax,noise_level=noise_lvl,
                                            nz=nz,nt=nt,destripe=args.destripe)

        if args.imd:

            print('Creating intensity-modulated display of lifetime volumes...')

            if args.version == 1:

                tau_series = tf.imread(os.path.join(save_dir,'taus.tif'))


                if len(tau_series.shape) < 4 :

                    tau_series = tau_series[np.newaxis,...]

                else:

                    if tau_series.shape[0] > 1:

                        if args.end_t < 0:

                            if args.start_t > 1:

                                tau_series = tau_series[args.start_t:args.end_t,...]

                            elif args.end_t == -1:

                                tau_series = tau_series

                            else:

                                tau_series = tau_series[:args.end_t,...]
                                
                        else:
                            
                            if args.start_t > 1:

                                tau_series = tau_series[args.start_t:args.end_t,...]

                            elif args.end_t == -1:

                                tau_series = tau_series             
                            
                            else:

                                tau_series = tau_series[:args.end_t,...]

            elif args.version == 2:

                tau_series = tf.imread(os.path.join(save_dir,'taus_M.tif'))

                if len(tau_series.shape) < 4 :

                    tau_series = tau_series[np.newaxis,...]

                else:

                    if tau_series.shape[0] > 1:

                        if args.end_t < 0:

                            if args.start_t > 1:

                                tau_series = tau_series[args.start_t:args.end_t,...]

                            elif args.end_t == -1:

                                tau_series = tau_series  

                            else:

                                tau_series = tau_series[:args.end_t,...]

                        else:
                            
                            if args.start_t > 1:

                                tau_series = tau_series[args.start_t:args.end_t,...]

                            elif args.end_t == -1:

                                tau_series = tau_series  
                            
                            else:

                                tau_series = tau_series[:args.end_t,...]

            else:
                exit(f'error: version should be either 1 or 2')

            med_taus = gaussian(tau_series,sigma=(0.0,0.0,gauss_sig,gauss_sig))

            tau_imd_series = np.zeros((tau_series.shape[0],nz,nx,ny,3))

            if not args.proj:

                if os.path.isfile(os.path.join(save_dir,'intensity_smooth.tif')):

                    data_clean = tf.imread(os.path.join(save_dir,'intensity_smooth.tif'))

                    if data_clean.shape[0] != med_taus.shape[0]:
                        data_clean = data_clean[:tau_series.shape[0]+1,...]
                    # data_clean = data_clean[args.start_t:(args.start_t+tau_series.shape[0]+1),...]

                else:
                    print(f'Running sum projection anyways to facility IMD...')

                    data_clean = project_and_save(times=times,ddir=ddir,save_dir=save_dir,nz=nz,destripe=args.destripe)

            if not args.norm:

                divMap, color_norm_tau = im2DLUT(med_taus,data_clean,
                                                    cmap=cmap,
                                                    cLim=[args.min_tau,args.max_tau],intLim=[0.0,np.percentile(data_clean,args.int_sat)])

                tau_imd_series = color_norm_tau

                tf.imwrite(os.path.join(save_dir,'taus_imd_smooth.tif'),
                        tau_imd_series,metadata={'axes': 'TZYXC', 'mode': 'composite'},imagej=False)

            else:
                for j in range(tau_imd_series.shape[0]):

                    if len(range(tau_imd_series.shape[0])) > 1000:
                        temp = j % 100
                    else:
                        temp = j % 10

                    if not temp:
                        print('Processing timepoint {}'.format(j))

                    divMap, color_norm_tau = im2DLUT(med_taus[j,...],data_clean[j,...],
                                                        cmap=cmap,
                                                        cLim=[args.min_tau,args.max_tau],intLim=[0.0,np.percentile(data_clean,args.int_sat)])

                    tau_imd_series[j,...] = color_norm_tau

                tf.imwrite(os.path.join(save_dir,'taus_imd_smooth_perFrame.tif'),
                        tau_imd_series,metadata={'axes': 'TZYXC', 'mode': 'composite'},imagej=False)
