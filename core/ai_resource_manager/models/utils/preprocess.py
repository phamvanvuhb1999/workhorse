import math

import cv2
import numpy as np


def image_padding(im: np.ndarray, value: int = 0):
    h, w, c = im.shape
    im_pad = np.zeros((max(32, h), max(32, w), c), np.uint8) + value
    im_pad[:h, :w, :] = im
    return im_pad


def resize_image(data: dict, limit_side_len: int = 960):
    """
    data = {
    'image': np.ndarray((H,W,3), dtype=np.uint8),
    }
    """
    img = data["image"]
    src_h, src_w, _ = img.shape
    if sum([src_h, src_w]) < 64:
        img = image_padding(img)
    # Resize
    h, w, c = img.shape

    # limit the max side
    if max(h, w) > limit_side_len:
        if h > w:
            ratio = float(limit_side_len) / h
        else:
            ratio = float(limit_side_len) / w
    else:
        ratio = 1.0

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)

    if int(resize_w) <= 0 or int(resize_h) <= 0:
        return None, (None, None)
    img = cv2.resize(img, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    data["image"] = img
    data["shape"] = np.array([src_h, src_w, ratio_h, ratio_w])
    return data


def normalize_image(
    data: dict, scale: float = 1.0 / 255.0, mean: tuple = (0.485, 0.456, 0.406), std: tuple = (0.229, 0.224, 0.225),
):
    """
    data = {
    'image': np.ndarray((H,W,3), dtype=np.uint8),
    }
    """
    if scale is not None:
        scale = eval(scale) if isinstance(scale, str) else scale
    else:
        scale = scale
    scale = np.float32(scale)
    mean = np.array(mean).reshape((1, 1, 3)).astype("float32")
    std = np.array(std).reshape((1, 1, 3)).astype("float32")

    img = data["image"]
    data["image"] = (img.astype("float32") * scale - mean) / std
    return data


def cvt_hwc_to_chw(data: dict):
    """
    data = {
    'image': np.ndarray((H,W,3), dtype=np.uint8),
    }
    """
    img = data["image"]
    data["image"] = img.transpose((2, 0, 1))
    return data


def get_data_by_keys(data: dict, keep_keys: list | None = None):
    """
    data = {
    'image': np.ndarray((H,W,3), dtype=np.uint8),
    }
    """
    if keep_keys is None:
        keep_keys = ["image", "shape"]
    return [data[key] for key in keep_keys]


def resize_norm_img(img, max_wh_ratio, rec_image_shape=[3, 48, 320]):
    imgC, imgH, imgW = rec_image_shape
    assert imgC == img.shape[2]
    imgW = int((imgH * max_wh_ratio))
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype("float32")
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def resize_with_padding(image, rec_image_shape=None):
    if not rec_image_shape:
        rec_image_shape = [3, 48, 320]
    original_height, original_width, channels = image.shape
    target_aspect_ratio = rec_image_shape[1] / rec_image_shape[2]

    original_aspect_ratio = original_height / original_width

    if original_aspect_ratio > target_aspect_ratio:
        new_height = rec_image_shape[1]
        new_width = int(new_height / original_height * original_width)
    else:
        new_width = rec_image_shape[2]
        new_height = int(new_width / original_width * original_height)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA).astype("float32")

    top = (rec_image_shape[1] - new_height) // 2
    bottom = rec_image_shape[1] - new_height - top
    left = (rec_image_shape[2] - new_width) // 2
    right = rec_image_shape[2] - new_width - left

    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # # Convert the image to [3, 48, 320] (C, H, W format)
    return np.transpose(padded_image, (2, 0, 1)) / 255
