import torch
import numpy as np
from PIL import Image


def randUnifC(low, high, params=None):
    p = np.random.uniform()
    if params is not None:
        params.append(p)
    return (high-low)*p + low

def randUnifI(low, high, params=None):
    p = np.random.uniform()
    if params is not None:
        params.append(p)
    return round((high-low)*p + low)

def randLogUniform(low, high, base=np.exp(1)):
    div = np.log(base)
    return base**np.random.uniform(np.log(low)/div,np.log(high)/div)

def load_img_as_array(img_path):
    img = Image.open(img_path)
    img_arr = np.array(img)[..., :3].astype(np.uint8) / 255
    img_arr = np.clip(img_arr, 0, 1)
    return img_arr

def convert_tensor_to_array(img_tensor):
    if len(img_tensor.shape) == 4 and img_tensor.shape[0]==1:
        img_tensor = img_tensor.squeeze(0)
    if img_tensor.max() > 1:
        img_tensor = img_tensor / 255.0
    img_arr = img_tensor.detach().cpu().numpy()
    img_arr = np.transpose(img_arr, (1, 2, 0))
    img_arr = np.clip(img_arr, 0, 1)
    return img_arr

def convert_array_to_tensor(img_arr):
    if img_arr.shape[0] != 3:
        img_arr = np.transpose(img_arr, (2, 0, 1))
    if img_arr.max() > 1:
        img_arr = img_arr / 255.0
    img_tensor = torch.from_numpy(img_arr).float()
    img_tensor = torch.clamp(img_tensor, 0, 1)
    return img_tensor

def save_array_as_img(img_arr, save_path=None):
    if img_arr.dtype != np.uint8:
        img_arr = (255 * img_arr).astype(np.uint8)
    if img_arr.shape[2] != 3:
        img_arr = np.transpose(img_arr, (1, 2, 0))

    img = Image.fromarray(img_arr)
    img.save(save_path)

__all__ = ['load_img_as_array', 'convert_tensor_to_array', \
           'convert_array_to_tensor', 'save_array_as_img']