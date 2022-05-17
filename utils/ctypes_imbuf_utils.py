import bpy
if bpy.app.version < (2, 80):
    raise RuntimeError("ctypes imbuf utils was attempted to be loaded on a Blender version older than 2.80")
if bpy.app.version >= (2, 83):
    raise RuntimeError("ctypes imbuf utils was attempted to be loaded on Blender version 2.83 or newer")

import ctypes
import imbuf
import numpy as np
import tempfile
import os
from .ctypes_utils import PyVarObject

# Used by Blender versions >= (2, 80) but < (2, 83) to save atlases as images faster.
# Versions older than 2.80 do not have the imbuf api and versions 2.83 and newer have access to Image.pixels.foreach_set
# so don't need this.

# Declare class to mirror ImBuf up to the rect and rect_float fields
# source/blender/imbuf/IMB_imbuf_types.h
class C_ImBuf(ctypes.Structure):
    """
      struct ImBuf *next, *prev; /**< allow lists of ImBufs, for caches or flipbooks */

      /* dimensions */
      /** Width and Height of our image buffer.
       * Should be 'unsigned int' since most formats use this.
       * but this is problematic with texture math in imagetexture.c
       * avoid problems and use int. - campbell */
      int x, y;

      /** Active amount of bits/bitplanes */
      unsigned char planes;
      /** Number of channels in `rect_float` (0 = 4 channel default) */
      int channels;

      /* flags */
      /** Controls which components should exist. */
      int flags;
      /** what is malloced internal, and can be freed */
      int mall;

      /* pixels */

      /** Image pixel buffer (8bit representation):
       * - color space defaults to `sRGB`.
       * - alpha defaults to 'straight'.
       */
      unsigned int *rect;
      /** Image pixel buffer (float representation):
       * - color space defaults to 'linear' (`rec709`).
       * - alpha defaults to 'premul'.
       * \note May need gamma correction to `sRGB` when generating 8bit representations.
       * \note Formats that support higher more than 8 but channels load as floats.
       */
      float *rect_float;
      [...]
    """
    pass


C_ImBuf._fields_ = [
    ('next', ctypes.POINTER(C_ImBuf)),
    ('prev', ctypes.POINTER(C_ImBuf)),
    ('x', ctypes.c_int),
    ('y', ctypes.c_int),
    ('planes', ctypes.c_ubyte),  # unsigned char -> ubyte? I guess this used to be called 'depth'
    ('channels', ctypes.c_int),
    ('flags', ctypes.c_int),
    ('mall', ctypes.c_int),
    ('rect', ctypes.POINTER(ctypes.c_uint)),  # looks like you can set rect to NULL to use rect_float in most cases
    ('rect_float', ctypes.POINTER(ctypes.c_float)),
]

# flags info
"""
enum {
  IB_rect = 1 << 0,
  IB_test = 1 << 1,
  IB_zbuf = 1 << 3,
  IB_mem = 1 << 4,
  IB_rectfloat = 1 << 5,
  IB_zbuffloat = 1 << 6,
  IB_multilayer = 1 << 7,
  IB_metadata = 1 << 8,
  IB_animdeinterlace = 1 << 9,
  IB_tiles = 1 << 10,
  IB_tilecache = 1 << 11,
  /** indicates whether image on disk have premul alpha */
  IB_alphamode_premul = 1 << 12,
  /** if this flag is set, alpha mode would be guessed from file */
  IB_alphamode_detect = 1 << 13,
  /* alpha channel is unrelated to RGB and should not affect it */
  IB_alphamode_channel_packed = 1 << 14,
  /** ignore alpha on load and substitute it with 1.0f */
  IB_alphamode_ignore = 1 << 15,
  IB_thumbnail = 1 << 16,
  IB_multiview = 1 << 17,
};"""


# declared in source/blender/python/generic/imbuf_py_api.c
class Py_ImBuf(PyVarObject):
    """
    typedef struct Py_ImBuf {
      PyObject_VAR_HEAD
          /* can be NULL */
          ImBuf *ibuf;
    } Py_ImBuf;
    """
    _fields_ = [
        ('ibuf', ctypes.POINTER(C_ImBuf))  # ('ibuf', ctypes.c_void_p)
    ]


# Compare against the basicsize to check
assert imbuf.types.ImBuf.__basicsize__ == ctypes.sizeof(Py_ImBuf)


def get_c_imbuf(imbuf_obj):
    py_imbuf = Py_ImBuf.from_address(id(imbuf_obj))
    # TODO: NULL check
    # If using c_void_p
    # c_imbuf = C_ImBuf.from_address(py_imbuf.ibuf)
    c_imbuf = py_imbuf.ibuf.contents
    return c_imbuf


def imbuf_to_np(image_buffer):
    c_image_buffer = get_c_imbuf(image_buffer)
    if c_image_buffer.rect:
        # Cast to ubyte, each uint is split into 4 ubytes, one for each channel
        pointer = ctypes.cast(c_image_buffer.rect, ctypes.POINTER(ctypes.c_ubyte))
    elif c_image_buffer.rect_float:
        pointer = c_image_buffer.rect_float
    else:
        raise TypeError("No pixels to copy, both rect and rect_float are NULL")
    np_from_pointer = np.ctypeslib.as_array(pointer, shape=(c_image_buffer.y, c_image_buffer.x, c_image_buffer.channels,))
    return np_from_pointer.copy()


# Assumes png
# Assumes 8-bit colour depth
# TODO: Support 16-bit colour depth?
#  It may be too difficult to mirror the fields all the way down to 'foptions', maybe we could include a tiny 16-bit
#  float png image, and load that to get an ImBuf with all the correct settings set up for us. Note that 16-bit png must
#  use rect_float according to the png export source/blender/imbuf/intern/png.c. We may also need to be careful of
#  colorspace settings when using float
def numpy_pixels_to_file(pixel_buffer: np.ndarray, filepath):
    if not filepath.endswith('.png'):
        raise TypeError("Only png is currently supported")
    if pixel_buffer.dtype != np.ubyte:
        if pixel_buffer.dtype == np.single:
            # Convert to unsigned char (unsigned byte)
            # This is the same conversion Blender uses internally, as defined in unit_float_to_uchar_clamp in
            # math_base_inline.c, albeit vectorized for numpy
            condition_list = [pixel_buffer <= np.single(0.0), pixel_buffer > (np.single(1.0) - np.single(0.5) / np.single(255.0))]
            choice_list = [np.single(0.0), np.single(255.0)]
            default_value = np.single(255.0) * pixel_buffer + np.single(0.5)
            pixel_buffer_ubyte = np.select(condition_list, choice_list, default=default_value).astype(np.ubyte)
        else:
            raise TypeError("Invalid pixel_buffer dtype: {}".format(pixel_buffer.dtype))
    else:
        pixel_buffer_ubyte = pixel_buffer

    height, width, channels = pixel_buffer_ubyte.shape

    if channels != 4:
        if channels > 4:
            raise TypeError("Too many channels: {}. Max is 4 (RGBA).".format(channels))
        else:
            raise RuntimeError("Padding channels up to 4 is not implemented")

    # Create a minimal ImBuf
    # The default flags are only IB_rect, which seems to be all that is needed
    image_buffer = imbuf.new((1, 1))
    # Get the corresponding ctypes object
    c_image_buffer = get_c_imbuf(image_buffer)

    # Store the old fields that we're going to change
    old_channels = c_image_buffer.channels
    # Won't be NULL
    old_rect = c_image_buffer.rect.content
    old_planes = c_image_buffer.planes
    old_x = c_image_buffer.x
    old_y = c_image_buffer.y

    try:
        c_image_buffer.channels = channels
        c_image_buffer.x = width
        c_image_buffer.y = height

        # ImBuf (at least for PNG) packs 4 ubyte into a single int
        # Types such as EXR would have to use rect_float instead
        # Alternative: ctypes.cast(c_image_buffer.rect, ctypes.c_void_p).value = pixel_buffer_ubyte.ctypes.data
        c_image_buffer.rect.contents = ctypes.c_uint.from_address(pixel_buffer_ubyte.ctypes.data)

        # 8 bits per channel
        # PNG-8 (Grayscale), PNG-24 (RGB) or PNG-32 (RGBA)
        c_image_buffer.planes = 8 * channels

        imbuf.write(image_buffer, filepath)
    finally:
        # ALWAYS set the fields back to the old values to avoid memory leaks/corruption
        c_image_buffer.channels = old_channels
        c_image_buffer.rect.content = old_rect
        c_image_buffer.planes = old_planes
        c_image_buffer.x = old_x
        c_image_buffer.y = old_y


# File name to use for the temporary image used to load pixels into images in a more performant manner than is usually
# possible on Blender 2.80 to 2.82
temp_png_image_name = "MaterialCombinerTemporaryAtlasImage.png"


# Convert pixel_buffer to imbuf
# Save imbuf to file
# Set filepath_raw of image to saved file
# Reload image
# Note that mark_dirty=True is slower, but it is useful if you don't immediately save the image afterwards and want to
# show to users that the image has unsaved changes.
# When use_temporary_file=True, the pixels are saved to a temporary file and then the image loads the pixels from that
# temporary file without saving. The temporary file is then deleted.
def numpy_pixels_to_image(img, pixel_buffer, write_to_image_filepath=False, temp_file_mark_dirty=False):
    if not write_to_image_filepath:
        image_file_path = os.path.join(tempfile.gettempdir(), temp_png_image_name)
        old_filepath_raw = img.filepath_raw
    elif img.source == 'FILE':
        # filepath could be relative, but filepath_from_user() will get the full filepath
        image_file_path = img.filepath_from_user()
    else:
        raise TypeError("{} image source is set to '{}' and not 'FILE'".format(img, img.source))
    numpy_pixels_to_file(pixel_buffer, image_file_path)
    try:
        if img.source != 'FILE':
            # Setting source reloads the image, so we don't need to do it ourselves
            img.filepath_raw = image_file_path
            img.source = 'FILE'
        else:
            # Setting filepath reloads the image when the source is already set to 'FILE'
            img.filepath = image_file_path
        if not write_to_image_filepath:
            # Evaluating bool(img.pixels) is a requirement to force Blender to load the image pixels internally. Without
            # doing so, the pixels won't be loaded until needed, which will usually be after the temporary file has been
            # deleted, but at that point, Blender can't load the pixels anymore because the image has been deleted.
            # Evaluating len(img.pixels) or getting/setting an individual pixel value would also function to force Blender
            # to load the pixels.
            if img.pixels and temp_file_mark_dirty:
                # Setting any pixels will mark the image as dirty, prompting the user to save or discard changes when viewed
                # in the image/uv editor.
                img.pixels[0] = img.pixels[0]
            # The temporary file isn't needed any more, delete it from the file system
            os.remove(image_file_path)
    finally:
        # If something were to go wrong, and we're not writing to the image's filepath, we need to restore the filepath
        # if it's been changed
        if not write_to_image_filepath:
            img.filepath_raw = old_filepath_raw
