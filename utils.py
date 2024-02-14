import cv2
import numpy as np
from typing import Tuple

def resize_with_pad(image: np.array, 
                    new_shape: Tuple[int, int]) -> np.array:
    """
    Maintains aspect ratio and resizes with padding.

    Args:
        image (np.array): Image to be resized
        new_shape (Tuple[int, int]): Expected (width, height) of new image

    Returns:
        np.array: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return image

def preprocess(frame: np.ndarray) -> np.ndarray:
    """Preprocessing steps on skin detected mask

    Args:
        frame (np.ndarray): input image/frame
    
    Returns:
        np.array: Morphological closed image
    """
    # explain the cv2.morphologyEx function
    # This function is used to perform morphological operations on the input image. That is, it is used to remove noise, smooth the image, and separate the objects in the image by performing operations like dilation, erosion, opening, closing, etc.

    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    return frame