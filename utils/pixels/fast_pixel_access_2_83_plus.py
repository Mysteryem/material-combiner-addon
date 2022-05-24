import numpy as np
from .pixel_types import pixel_dtype


def get_pixels(image):
    pixels = image.pixels
    buffer = np.empty(len(pixels), dtype=pixel_dtype)
    # Buffer must be flat when reading
    pixels.foreach_get(buffer)
    return buffer


def set_pixels(img, buffer):
    # buffer must be C contiguous and flattened when writing
    img.pixels.foreach_set(buffer.ravel())
