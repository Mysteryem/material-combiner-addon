import os

import bpy


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


def save_generated_image_to_file(image, filepath, file_format=None):
    # Note that setting image.filepath and/or image.source = 'FILE' can't be done or it will reset the image
    image.filepath_raw = filepath
    if file_format:
        image.file_format = file_format
    image.save()
