from PIL import Image
import skimage
from skimage import color, transform
import numpy as np
from scipy import fftpack
from io import BytesIO
from .transform_utils import randUnifC, randUnifI

#Color reduction
def color_reduction_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    scales = [np.random.randint(8, 201) for x in range(3)]
    multi_channel = np.random.choice(2) == 0
    params = [multi_channel] + [s / 200.0 for s in scales]
    if multi_channel:
        img = np.round(img * scales[0]) / scales[0]
    else:
        for i in range(3):
            img[:, :, i] = np.round(img[:, :, i] * scales[i]) / scales[i]
    if return_params:
        return img, params
    else:
        return img

def jpeg_compression_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    quality = np.random.randint(55,96)
    params = [quality / 100.0]
    pil_image = Image.fromarray((img * 255.0).astype(np.uint8) )
    f = BytesIO()
    pil_image.save(f, format='jpeg', quality=quality)
    jpeg_image = np.asarray(Image.open(f),).astype(np.float32) / 255.0
    img = jpeg_image
    if return_params:
        return img, params
    else:
        return img

def swirl_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    strength = (2.0 - 0.01) * np.random.random(1)[0] + 0.01
    c_x = np.random.randint(1, 257)
    c_y = np.random.randint(1, 257)
    radius = np.random.randint(10, 201)
    params = [strength / 2.0, c_x / 256.0, c_y / 256.0, radius / 200.0]
    img = transform.swirl(img, rotation=0,
                                    strength=strength, radius=radius, center=(c_x,
                                                                            c_y))
    if return_params:
        return img, params
    else:
        return img

def noise_injection_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    params = []
    options = ['gaussian', 'poisson', 'salt', 'pepper', 's&p', 'speckle']
    noise_type = np.random.choice(options, 1)[0]
    params.append(options.index(noise_type) / 6.0)
    per_channel = np.random.choice(2) == 0
    params.append(per_channel)
    if per_channel:
        for i in range(3):
            img[:, :, i] = skimage.util.random_noise(img[:, :, i], mode = noise_type )
    else:
        img = skimage.util.random_noise(img,mode = noise_type)
    if return_params:
        return img, params
    else:
        return img

def fft_perturbation_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    h, w, _ = img.shape
    point_factor = (1.02 - 0.98) * np.random.random((h, w)) + 0.98
    randomized_mask = [np.random.choice(2) == 0 for x in range(3)]
    keep_fraction = [(0.95 - 0.0) * np.random.random(1)[0] + 0.0 for x in range(3)]
    params = randomized_mask + keep_fraction
    for i in range(3):
        im_fft = fftpack.fft2(img[:, :, i])
        h, w = im_fft.shape
        if randomized_mask[i]:
            mask = np.ones(im_fft.shape[:2]) > 0
            im_fft[int(h * keep_fraction[i]):int(h * (1 - keep_fraction[i]))] = 0
            im_fft[:, int(w * keep_fraction[i]):int(w * (1 - keep_fraction[i]))] = 0
            mask = mask * (np.random.uniform(size=im_fft.shape[:2] ) >= keep_fraction[i])
            mask = ~mask
            im_fft = np.multiply(im_fft, mask)
        else:
            im_fft[int(h * keep_fraction[i]):int(h*(1-keep_fraction[i]))] = 0
            im_fft[:, int(w*keep_fraction[i]):int(w*(1-keep_fraction[i]))] = 0
        im_fft = np.multiply(im_fft, point_factor)
        im_new = fftpack.ifft2(im_fft).real
        im_new = np.clip(im_new, 0, 1)
        img[:, :, i] = im_new
    if return_params:
        return img, params
    else:
        return img

def zoom_in_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    h, w, _ = img.shape
    min_ = min(h, w) // 20
    max_ = min(h, w) // 4
    i_s = np.random.randint(min_, max_)
    i_e = np.random.randint(min_, max_)
    j_s = np.random.randint(min_, max_)
    j_e = np.random.randint(min_, max_)
    params = [i_s / max_, i_e / max_, j_s / max_, j_e / max_]
    i_e = h - i_e
    j_e = w - j_e
    # Crop the image...
    img = img[i_s:i_e, j_s:j_e, :]
    # ...now scale it back up
    img = skimage.transform.resize(img, (h, w, 3))
    if return_params:
        return img, params
    else:
        return img

def hsv_hub_perturbation_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    img = color.rgb2hsv(img)
    params = []
    # Hue
    img[:, :, 0] += randUnifC(-0.05, 0.05, params=params)
    # Saturation
    img[:, :, 1] += randUnifC(-0.25, 0.25, params=params)
    # Value
    img[:, :, 2] += randUnifC(-0.25, 0.25, params=params)
    img = np.clip(img, 0, 1.0)
    img = np.clip(color.hsv2rgb(img), 0, 1.0)
    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img   

def xyz_hub_perturbation_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    img = color.rgb2xyz(img)
    params = []

    # X
    img[:, :, 0] += randUnifC(-0.05, 0.05, params=params)
    # Y
    img[:, :, 1] += randUnifC(-0.05, 0.05, params=params)
    # Z
    img[:, :, 2] += randUnifC(-0.05, 0.05, params=params)
    img = np.clip(img, 0, 1.0)
    img = np.clip(color.xyz2rgb(img), 0, 1.0)
    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img
    
def lab_hub_perturbation_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    img = color.rgb2lab(img)
    params = []

    # L
    img[:, :, 0] += randUnifC(-5.0, 5.0, params=params)
    # a
    img[:, :, 1] += randUnifC(-2.0, 2.0, params=params)
    # b
    img[:, :, 2] += randUnifC(-2.0, 2.0, params=params)

    img[:, :, 0] = np.clip(img[:, :, 0], 0, 100.0)
    img[:, :, 1] = np.clip(img[:, :, 1], -128.0, 127.0)
    img[:, :, 2] = np.clip(img[:, :, 2], -128.0, 127.0)

    img = color.lab2rgb(img)
    img = np.clip(img, 0.0, 1.0)
    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img

def yuv_hub_perturbation_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    img = color.rgb2yuv(img)
    params = []
    # Y
    img[:, :, 0] += randUnifC(-0.05, 0.05, params=params)
    # U
    img[:, :, 1] += randUnifC(-0.02, 0.02, params=params)
    # V
    img[:, :, 2] += randUnifC(-0.02, 0.02, params=params)
    # U & V channels can have negative values; clip only Y
    img[:, :, 0] = np.clip(img[:, :, 0], 0, 1.0)
    img = color.yuv2rgb(img)
    img = np.clip(img, 0.0, 1.0)
    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img

def dist_equal_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    nbins = np.random.randint(40, 257)
    params = [nbins / 256.0]
    for i in range(3):
        img[:, :, i] = skimage.exposure.equalize_hist(img[:, :, i], nbins = nbins)
    img = np.clip(img, 0.0, 1.0)
    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img

def rescale_intensity_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    per_channel = np.random.choice(2) == 0
    params = [per_channel]
    low_precentile = [randUnifC(0.01, 0.04, params=params) for x in range(3)]
    hi_precentile = [randUnifC(0.96, 0.99, params=params)for x in range(3)]
    if per_channel:
        for i in range(3):
            p2, p98 = np.percentile(img[:, :, i],
                                    (low_precentile[i] * 100,
                                        hi_precentile[i] * 100))
            img[:, :, i] =skimage.exposure.rescale_intensity(img[:, :, i], in_range=(p2, p98))
    else:
        p2, p98 = np.percentile(img, (low_precentile[0] * 100, hi_precentile[0] * 100))
        img = skimage.exposure.rescale_intensity(img, in_range = (p2, p98) )
    img = np.clip(img, 0.0, 1.0)
    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img
    
def grayscale_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    ratios = np.random.rand(3)
    ratios /= ratios.sum()
    params = [x for x in ratios]
    img_g = img[:, :, 0] * ratios[0] + img[:, :, 1] * ratios[1] + img[:, :, 2] * ratios[2]
    for i in range(3):
        img[:, :, i] = img_g
    img = np.clip(img, 0.0, 1.0)
    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img
    
def color_grayscale_mix_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    ratios = np.random.rand(3)
    ratios /= ratios.sum()
    prop_ratios = np.random.rand(3)
    params = [x for x in ratios] + [x for x in prop_ratios]
    img_g = img[:, :, 0] * ratios[0] + img[:, :, 1] * ratios[1]+ img[:, :, 2] * ratios[2]
    for i in range(3):
        p = max(prop_ratios[i], 0.2)
    img[:, :, i] = img[:, :, i] * p + img_g * (1.0 - p)
    img = np.clip(img, 0.0, 1.0)
    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img

def random_grayscale_channels_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    params = []
    channels = [0, 1, 2]
    remove_channel = np.random.choice(3)
    channels.remove(remove_channel)
    params.append(remove_channel)
    ratios = np.random.rand(2)
    ratios /= ratios.sum()
    params.append(ratios[0])
    img_g = img[:, : ,channels[0]] * ratios[0] + img[:, :, channels[1]] * ratios[1]
    for i in channels:
        img[:, :, i] = img_g
    img = np.clip(img, 0.0, 1.0)
    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img

def random_grayscale_channel_color_mix_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    params = []
    channels = [0, 1, 2]
    to_alter = np.random.choice(3)
    channels.remove(to_alter)
    params.append(to_alter)
    ratios = np.random.rand(2)
    ratios /= ratios.sum()
    params.append(ratios[0])
    img_g = img[:, :, channels[0]] * ratios[0] + img[:, :, channels[1]] * ratios[1]
    # Lets mix it back in with the original channel
    p = (0.9 - 0.1) * np.random.random(1)[0] + 0.1
    params.append(p)
    img[:, :, to_alter] = img_g * p + img[:, :, to_alter] *(1.0 - p)
    img = np.clip(img, 0.0, 1.0)
    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img
    
def gaussian_filter_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    params = []
    if randUnifC(0, 1) > 0.5:
        sigma = [randUnifC(0.1, 3, params)] * 3
    else:
        sigma = [randUnifC(0.1, 3, params), randUnifC(0.1, 3, params), randUnifC(0.1, 3, params)]
    img[:, :, 0] = skimage.filters.gaussian(img[:, :, 0],sigma = sigma[0])
    img[:, :, 1] = skimage.filters.gaussian(img[:, :, 1], sigma = sigma[1])
    img[:, :, 2] = skimage.filters.gaussian(img[:, :, 2],sigma = sigma[2])
    img = np.clip(img, 0.0, 1.0) 
    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img
    
def median_blur_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    params = []
    if randUnifC(0, 1, params) > 0.5:
        radius = [randUnifI(2, 5, params)] * 3
    else:
        radius = [randUnifI(2, 5, params), randUnifI(2, 5, params), randUnifI(2, 5, params)]
    img = skimage.util.img_as_ubyte(img)
    
    for i in range(3):
        mask = skimage.morphology.disk(radius[i])
        img[:, :, i] = skimage.filters.rank.median(img[:, :, i], mask)
    img = img.astype(np.float32) / 255.0
    img = np.clip(img, 0.0, 1.0)
    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img

def mean_blur_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    params = []
    if randUnifC(0, 1, params) > 0.5:
        radius = [randUnifI(2, 5, params)] * 3
    else:
        radius = [randUnifI(2, 5, params), randUnifI(2, 5, params), randUnifI(2, 5, params)]
    img = skimage.util.img_as_ubyte(img)

    for i in range(3):
        mask = skimage.morphology.disk(radius[i])
        img[:, :, i] = skimage.filters.rank.mean(img[:, :, i], mask)
    img = img.astype(np.float32) / 255.0
    img = np.clip(img, 0.0, 1.0)

    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img

def mean_bilateral_blur_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    params = []
    radius = []
    ss = []
    for i in range(3):
        radius.append(randUnifI(2, 20, params=params))
        ss.append(randUnifI(5, 20, params=params))
        ss.append(randUnifI(5, 20, params=params))
    img = skimage.util.img_as_ubyte(img)

    for i in range(3):
        mask = skimage.morphology.disk(radius[i])
        img[:, :, i] = skimage.filters.rank.mean_bilateral(img[:, :, i], mask, s0 = ss[i], s1 = ss[3 + i])
    img = img.astype(np.float32) / 255.0
    img = np.clip(img, 0.0, 1.0)
    
    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img

def tv_chambolle_denoise_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    params = []
    weight = (0.25 - 0.05) * np.random.random(1)[0] + 0.05
    params.append(weight)
    img = skimage.restoration.denoise_tv_chambolle(img, weight = weight)
    img = np.clip(img, 0.0, 1.0)

    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img
    
def cycle_spin_denoise_transform(img, return_params=True):
    """
    Args:
    -img: numpy array of shape (H,W,3), range [0,1]
    """
    wavelets = ['db1','db2', 'haar', 'sym9']
    convert2ycbcr = np.random.choice(2) == 0
    wavelet = np.random.choice(wavelets)
    mode_ = np.random.choice(["soft", "hard"])
    denoise_kwargs = dict(convert2ycbcr=convert2ycbcr, wavelet=wavelet, mode=mode_, channel_axis=-1)
    max_shifts = np.random.choice([0, 1])
    params = [convert2ycbcr, wavelets.index(wavelet) /
                float(len(wavelets)), max_shifts / 5.0,
                (mode_ == "soft")]
    img = skimage.restoration.cycle_spin(img,
                                            func=skimage.restoration.denoise_wavelet,
                                            max_shifts=max_shifts, func_kw=denoise_kwargs,
                                            num_workers=1,
                                            channel_axis=-1)
    img = np.clip(img, 0.0, 1.0)

    assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3 and img.max() <= 1
    if return_params:
        return img, params
    else:
        return img

transform_config = {
    'color_reduction_transform': color_reduction_transform,
    'jpeg_compression_transform': jpeg_compression_transform,
    'swirl_transform': swirl_transform,
    'noise_injection_transform': noise_injection_transform,
    'jpeg_compression_transform': jpeg_compression_transform,
    'swirl_transform': swirl_transform,
    'noise_injection_transform': noise_injection_transform,
    'fft_perturbation_transform': fft_perturbation_transform,
    'zoom_in_transform': zoom_in_transform,
    'hsv_hub_perturbation_transform': hsv_hub_perturbation_transform,
    'xyz_hub_perturbation_transform': xyz_hub_perturbation_transform,
    'lab_hub_perturbation_transform': lab_hub_perturbation_transform,
    'yuv_hub_perturbation_transform': yuv_hub_perturbation_transform,
    'dist_equal_transform': dist_equal_transform,
    'rescale_intensity_transform': rescale_intensity_transform,
    'grayscale_transform': grayscale_transform,
    'color_grayscale_mix_transform': color_grayscale_mix_transform,
    'random_grayscale_channels_transform': random_grayscale_channels_transform,
    'random_grayscale_channel_color_mix_transform': random_grayscale_channel_color_mix_transform,
    'gaussian_filter_transform': gaussian_filter_transform,
    'median_blur_transform': median_blur_transform,
    'mean_blur_transform': mean_blur_transform,
    'mean_bilateral_blur_transform': mean_bilateral_blur_transform,
    'tv_chambolle_denoise_transform': tv_chambolle_denoise_transform,
    'cycle_spin_denoise_transform': cycle_spin_denoise_transform
    }
