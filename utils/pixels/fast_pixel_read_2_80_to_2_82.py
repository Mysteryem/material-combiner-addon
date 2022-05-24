import numpy as np

import bgl

from .pixel_types import pixel_gltype, pixel_dtype


# see https://blender.stackexchange.com/a/230242 for details
def get_pixels_gl_shared_buffer(image):
    pixels = image.pixels
    # Load the image into OpenGL and use that to get the pixels in a more performant manner
    # As per the documentation, the colours will be read in scene linear color space and have premultiplied or
    # straight alpha matching the image alpha mode.
    # see https://blender.stackexchange.com/a/230242 for details
    # Open GL will cache the image if we've used it previously, this means that if we update the image in Blender
    # it won't have updated in Open GL unless we free it first. There isn't really a way to know if the image has
    # changed since it was last cached, so we'll free it
    if image.bindcode:
        # If the open gl bindcode is set, then it's already been cached, so free it from open gl first
        image.gl_free()
    if image.gl_load():
        print("Could not load {} into Open GL, resorting to a slower method of getting pixels".format(image))
        return np.fromiter(pixels, dtype=pixel_dtype)
    bgl.glActiveTexture(bgl.GL_TEXTURE0)
    bgl.glBindTexture(bgl.GL_TEXTURE_2D, image.bindcode)
    buffer = np.empty(len(pixels), dtype=pixel_dtype)
    gl_buffer = bgl.Buffer(pixel_gltype, buffer.shape, buffer)
    bgl.glGetTexImage(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGBA, pixel_gltype, gl_buffer)
    return buffer
