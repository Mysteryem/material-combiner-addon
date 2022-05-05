import os
import numpy as np

import bpy

# Type has to match the internal type used by Blender otherwise the internal data is iterated in a for loop, casting
# every element to the mismatched type, which is much slower than getting the internal data directly when the tpye
# matches the internal type.
pixel_dtype = np.single


def get_image(tex):
    return tex.image if tex and hasattr(tex, 'image') and tex.image else None


def get_image_path(img):
    path = bpy.path.abspath(img.filepath) if img else ''
    return path if os.path.isfile(path) and not path.lower().endswith(('.spa', '.sph')) else ''


# TODO: We could do some optimisations in the case of generated images as we can get the generated_color and use that
#       instead of getting a pixel buffer
def is_single_colour_generated(img):
    return not img.is_dirty and img.generated_type == 'BLANK'


# FIXME: Wasn't this used somewhere? Where did it go?
def is_valid_image(img):
    if img.filepath and img.filepath.lower().endswith(('.spa', '.sph')):
        return False
    else:
        return True


linear_colorspaces = {'Linear', 'Non-color', 'Raw'}
supported_colorspaces = linear_colorspaces | {'sRGB'}


def buffer_convert_linear_to_srgb(buffer):
    # Alpha is always linear, so get a view of only RGB.
    rgb_only_view = buffer[:, :, :3]

    is_small = rgb_only_view < 0.0031308

    small_rgb = rgb_only_view[is_small]
    # This can probably be optimised
    rgb_only_view[is_small] = np.where(small_rgb < 0.0, 0, small_rgb * 12.92)

    is_not_small = np.invert(is_small, out=is_small)

    rgb_only_view[is_not_small] = 1.055 * (rgb_only_view[is_not_small] ** (1.0 / 2.4)) - 0.055


def buffer_convert_srgb_to_linear(buffer):
    # Alpha is always linear, so get a view of only RGB.
    rgb_only_view = buffer[:, :, :3]

    is_small = rgb_only_view < 0.04045

    small_rgb = rgb_only_view[is_small]
    # This can probably be optimised
    rgb_only_view[is_small] = np.where(small_rgb < 0.0, 0, small_rgb / 12.92)

    is_not_small = np.invert(is_small, out=is_small)

    rgb_only_view[is_not_small] = ((rgb_only_view[is_not_small] + 0.055) / 1.055) ** 2.4


def get_pixel_buffer(img, atlas_colorspace='sRGB'):
    width, height = img.size
    channels = img.channels
    buffer = np.empty(width * height * channels, dtype=np.single)
    # Buffer must be flat when reading
    img.pixels.foreach_get(buffer)
    # View the buffer in a shape that better represents the data
    buffer.shape = (width, height, channels)
    # Blender treats the bottom left as (0, 0), but we want the top left to be treated as (0, 0), so view the buffer
    # with flipped y-axis
    buffer = buffer[:, ::-1, :]

    # Pixels are always read raw, meaning that changing the colorspace of the image has no effect on the pixels,
    # but if we want to combine a linear image into an sRGB image such that the linear image appears the same when
    # viewed in sRGB, we need to convert it to sRGB to linear, so that when it's viewed in sRGB, our initial
    # conversion from sRGB to linear and the conversion from raw (linear) to sRGB when viewed cancel each other out.
    img_color_space = img.colorspace_settings.name
    if atlas_colorspace == 'sRGB':
        if img_color_space == 'sRGB':
            return buffer
        elif img_color_space in linear_colorspaces:
            # Need to convert from sRGB to linear
            buffer_convert_srgb_to_linear(buffer)
            return buffer
        else:
            raise TypeError("Unsupported image colorspace {} for {}. Must be in {}.".format(img_color_space, img, supported_colorspaces))
    elif atlas_colorspace in linear_colorspaces:
        if img_color_space in linear_colorspaces:
            return buffer
        elif img_color_space == 'sRGB':
            # Need to convert from Linear to sRGB
            buffer_convert_linear_to_srgb(buffer)
            return buffer
    else:
        raise TypeError("Unsupported atlas colorspace {}. Must be in {}".format(atlas_colorspace, supported_colorspaces))


# Copy the image, resize the copy and then get the pixel buffer, the copied image is then destroyed
# The alternative would be to resize the passed in image and then reload it afterwards, but if the passed in image was
# dirty, then those dirty changes would be lost.
def get_resized_pixel_buffer(img, size):
    # Copy the input image
    copy = img.copy()
    # Scale (resize) the copy
    copy.scale(size[0], size[1])
    # Get the pixel buffer for the scaled copy
    buffer = get_pixel_buffer(copy)
    # Delete the scaled copy
    bpy.data.images.remove(copy)


def buffer_to_image(buffer, *, name):
    image = bpy.data.images.new(name, buffer.shape[0], buffer.shape[1], alpha=buffer.shape[2] == 4)
    write_pixel_buffer(image, buffer)
    return image


def write_pixel_buffer(img, buffer):
    width, height = img.size
    image_shape = (width, height, img.channels)
    if buffer.shape == image_shape:
        # buffer must be flattened when writing
        img.pixels.foreach_set(buffer.ravel())
    else:
        raise RuntimeError("Buffer shape {} does not match image shape {}".format(buffer.shape, image_shape))


def new_pixel_buffer(size, color=(0.0, 0.0, 0.0, 1.0)):
    """Create a new blank pixel buffer.
    The number of channels is determined based on the fill color.
    Default fill color is black with alpha.
    Compared to how pixels are usually accessed in Blender where (0,0) is the bottom left pixel, pixel buffers have the
    y-axis flipped so that (0,0) is the top left of the image
    :return: a new pixel buffer ndarray
    """
    width, height = size
    # rgba
    channels = len(color)
    if channels > 4 or channels == 0:
        raise TypeError("A color can have between 1 and 4 (inclusive) components, but found {}".format(channels))
    buffer = np.full((width, height, channels), fill_value=color, dtype=np.single)
    # Blender treats the bottom left as (0, 0), but we want the top left to be treated as (0, 0), so view the buffer
    # with flipped y-axis
    return buffer[:, ::-1, :]


def pixel_buffer_paste(target_buffer, source_buffer_or_pixel, corner_or_box):
    if isinstance(source_buffer_or_pixel, np.ndarray):
        source_dimensions = len(source_buffer_or_pixel.shape)
        if source_dimensions == 3:
            # Source is a buffer representing a 2D image, with the 3rd axis being the pixel data
            source_is_pixel = False
        elif source_dimensions == 1 and target_buffer.shape[-1] >= source_dimensions:
            # Source is the data for a single pixel, to be pasted into all the pixels in the box region
            source_is_pixel = True
        else:
            raise TypeError("source buffer or pixel could not be parsed for pasting")
    elif isinstance(source_buffer_or_pixel, (tuple, list)) and target_buffer.shape[-1] >= len(source_buffer_or_pixel):
        # Source is the data for a single pixel, to be pasted into all the pixels in the box region
        source_is_pixel = True
    else:
        raise TypeError("source pixel could not be parsed for pasting")

    def fit_box(box):
        # Fit the box to the image. This could be changed to raise an Error if the box doesn't fit.
        # Remember that box corners are cartesian coordinates where (0,0) is the top left corner of the top left pixel
        # and (1,1) is the bottom right corner of the top left pixel
        left, upper, right, lower = box
        left = max(left, 0)
        upper = max(upper, 0)
        right = min(right, target_buffer.shape[0] + 1)
        lower = min(lower, target_buffer.shape[1] + 1)
        return left, upper, right, lower

    if source_is_pixel:
        # When the source is a single pixel color, there must be a box to paste to
        if len(corner_or_box) != 4:
            raise TypeError("When pasting a pixel color, a box region to paste to must be supplied, but got: {}".format(corner_or_box))
        left, upper, right, lower = fit_box(corner_or_box)
        # Fill the box with corners (left, upper) and (right, lower) with the pixel color, filling in as many components
        # of the pixels as in the source pixel.
        # Remember that these corners are cartesian coordinates with (0,0) as the top left corner of the image.
        # A box with corners (0,0) and (1,1) only contains the pixels between (0,0) inclusive and (1,1) exclusive
        num_source_channels = len(source_buffer_or_pixel)
        target_buffer[left:right, upper:lower, :num_source_channels] = source_buffer_or_pixel
    else:
        # Parse a corner into a box
        if len(corner_or_box) == 2:
            # Only the top left corner to place the source buffer has been set, we will figure out the bottom right
            # corner
            left, upper = corner_or_box
            right = left + source_buffer_or_pixel.shape[0]
            lower = upper + source_buffer_or_pixel.shape[1]
        elif len(corner_or_box) == 4:
            left, upper, right, lower = corner_or_box
        else:
            raise TypeError("corner or box must be either a 2-tuple or 4-tuple, but was: {}".format(corner_or_box))

        if target_buffer.shape[-1] >= source_buffer_or_pixel.shape[-1]:
            fit_left, fit_upper, fit_right, fit_lower = fit_box((left, upper, right, lower))
            # TODO: Remove debug print
            if fit_left != left or fit_upper != upper or fit_right != right or fit_lower != lower:
                print('DEBUG: Image to be pasted did not fit into target image, {} -> {}'.format((left, upper, right, lower), (fit_left, fit_upper, fit_right, fit_lower)))
            # If the pasted buffer can extend outside the source image, we need to figure out the area which fits within
            # the source image
            source_left = fit_left - left
            source_upper = fit_upper - upper
            source_right = source_buffer_or_pixel.shape[0] - right + fit_right
            source_lower = source_buffer_or_pixel.shape[1] - lower + fit_lower
            num_source_channels = source_buffer_or_pixel.shape[2]
            target_buffer[fit_left:fit_right, fit_upper:fit_lower, :num_source_channels] = source_buffer_or_pixel[source_left:source_right, source_upper:source_lower]
        else:
            raise TypeError("Pixels in source have more channels than pixels in target, they cannot be pasted")
