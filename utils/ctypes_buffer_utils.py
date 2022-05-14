import bgl
import numpy as np
import ctypes

# This module is only used for Blender 2.79 and older where there is no fast method to get all of an Image's pixels into
# a numpy array
# When importing this module, make sure to surround it in a try block and have some sort of backup plan if it fails
# TODO: Split the bgl.Buffer specific code into its own module and leave what remains as a base module for potentially
#  other ctypes excursions
# TODO: Setting Image.pixels is slow up until Blender 2.83.
#  Could we also use ctypes to access and set the underlying memory of Image.pixels? Look for BPy_PropertyArrayRNA and
#  see https://developer.blender.org/D7053 for the commit that added foreach_get/foreach_set.
#  If we can figure out that access, it would likely be faster for getting pixels on Blender 2.82 and older too.

# To help with understanding this code, see:
# https://mcla.ug/blog/cpython-hackage.html
# https://github.com/CGCookie/addon_common/blob/b280/ext/bgl_ext.py

# Start with standard classes to mirror the basic CPython types


# Declare class to mirror PyObject type
# https://docs.python.org/3/c-api/structures.html#c.PyObject
# https://github.com/python/cpython/blob/10e5c66789a06dc9015a24968e96e77a75725a7a/Include/object.h#L104
class PyObject(ctypes.Structure):
    """typedef struct _object {
        _PyObject_HEAD_EXTRA
        Py_ssize_t ob_refcnt;
        struct _typeobject *ob_type;
    } PyObject;"""
    pass


# PyObject's Fields must be set separately as it references its own type
PyObject._fields_ = [
    ('ob_refcnt', ctypes.c_ssize_t),
    ('ob_type', ctypes.POINTER(PyObject)),
]

# _PyObject_HEAD_EXTRA expands out two extra fields sometimes
"""
#ifdef Py_TRACE_REFS
/* Define pointers to support a doubly-linked list of all live heap objects. */
#define _PyObject_HEAD_EXTRA            \
    struct _object *_ob_next;           \
    struct _object *_ob_prev;

#define _PyObject_EXTRA_INIT 0, 0,

#else
#define _PyObject_HEAD_EXTRA
#define _PyObject_EXTRA_INIT
#endif
"""
# Compare against object basicsize to check
if object.__basicsize__ != ctypes.sizeof(PyObject):
    # Change the fields to include the extra fields
    PyObject._fields_ = [
        ('_ob_next', ctypes.POINTER(PyObject)),
        ('_ob_prev', ctypes.POINTER(PyObject)),
        ('ob_refcnt', ctypes.c_ssize_t),
        ('ob_type', ctypes.POINTER(PyObject)),
    ]

# Ensure the size matches
assert object.__basicsize__ == ctypes.sizeof(PyObject)


# Declare class to mirror PyVarObject type
# https://docs.python.org/3/c-api/structures.html#c.PyVarObject
# https://github.com/python/cpython/blob/10e5c66789a06dc9015a24968e96e77a75725a7a/Include/object.h#L113
class PyVarObject(PyObject):
    """typedef struct {
        PyObject ob_base;
        Py_ssize_t ob_size; /* Number of items in variable part */
    } PyVarObject;"""
    _fields_ = [
        ('ob_size', ctypes.c_ssize_t),
    ]


# Declare class to mirror bgl.Buffer's _Buffer type
# The fields for bgl.Buffer haven't been changed since its introduction, but if they are changed in the future, then
# this class would need to be modified to match. Fortunately, we only need to use this when in Blender 2.79 and earlier,
# so it's not going to change.
class BglBuffer(PyVarObject):
    """https://github.com/blender/blender/blob/master/source/blender/python/generic/bgl.h
    For the PyObject_VAR_HEAD macro, see https://docs.python.org/3/c-api/structures.html#c.PyObject_VAR_HEAD
    /**
     * Buffer Object
     *
     * For Python access to OpenGL functions requiring a pointer.
     */
    typedef struct _Buffer {
      PyObject_VAR_HEAD
      PyObject *parent;

      int type; /* GL_BYTE, GL_SHORT, GL_INT, GL_FLOAT */
      int ndimensions;
      int *dimensions;

      union {
        char *asbyte;
        short *asshort;
        int *asint;
        float *asfloat;
        double *asdouble;

        void *asvoid;
      } buf;
    } Buffer;"""
    _fields_ = [
        # TODO: What's the difference between ctypes.py_object and ctypes.POINTER(PyObject)?
        ('parent', ctypes.py_object),
        ('type', ctypes.c_int),
        ('ndimensions', ctypes.c_int),
        ('dimensions', ctypes.POINTER(ctypes.c_int)),
        ('buf', ctypes.c_void_p),
    ]


# Ensure the size matches what Python tells us
assert bgl.Buffer.__basicsize__ == ctypes.sizeof(BglBuffer)


def np_array_from_bgl_buffer(bgl_buffer):
    """Convert a bgl.Buffer into a numpy ndarray, copying the data from the buffer into the ndarray

    :return: a new numpy ndarray with the same type and data as the input bgl.Buffer"""
    # no_copy is currently disabled due to garbage collection issues
    #     When no_copy is True, the original data of the input bgl.Buffer will be used directly by the numpy array and
    #      the bgl.Buffer will be detached from its data, rendering it unusable. Do not keep references to or use the
    #       bgl.Buffer after being passed to this function.
    #       (currently disabled due to garbage collection issues)
    #
    #     When no_copy is False, the data will be copied before being returned in the numpy array, which is slower
    #      but allows the input bgl.Buffer to continue being used.
    c_buf = BglBuffer.from_address(id(bgl_buffer))
    # Get the type of the buffer to find the type of the pointer to the data
    bgl_type = c_buf.type
    # See https://www.khronos.org/opengl/wiki/OpenGL_Type
    if bgl_type == bgl.GL_BYTE:
        c_data_type = ctypes.c_byte
    elif bgl_type == bgl.GL_SHORT:
        c_data_type = ctypes.c_int16
    elif bgl_type == bgl.GL_INT:
        c_data_type = ctypes.c_int32
    elif bgl_type == bgl.GL_FLOAT:
        c_data_type = ctypes.c_float
    elif bgl_type == bgl.GL_DOUBLE:
        c_data_type = ctypes.c_double
    else:
        raise TypeError("Unrecognised bgl.Buffer type constant {}".format(bgl_type))

    # Get a pointer to the data, the 'buf' field
    data_pointer = ctypes.pointer(c_data_type.from_address(c_buf.buf))
    # Numpy will infer the correct type for the array from the pointer
    np_from_pointer = np.ctypeslib.as_array(data_pointer, shape=tuple(bgl_buffer.dimensions))

    # FIXME: Data isn't getting garbage collected as intended with no_copy=True
    #  Maybe we can point the BglBuffer's data somewhere else so that it and numpy aren't sharing any more? The
    #  BglBuffer should then free its data when garbage collected, the question is, will numpy free its data when
    #  garbage collected? Possibly see https://github.com/numpy/numpy/issues/6511, in which case, there's nothing that
    #  can be done, as Blender 2.79's numpy version is 1.10.1
    # The numpy array is now SHARING the same memory as the data of the bgl.Buffer
    # Either the numpy array needs to copy the data or the bgl.Buffer needs to be disconnected from the underlying
    # BglBuffer C type and the numpy array be connected to it instead.
    # if no_copy:
    #     # If we want to use the data directly in the numpy array, we need to prevent it from being freed when the
    #     # bgl.Buffer is garbage collected. To do this, we can connect the BglBuffer to the numpy array by setting the
    #     # BglBuffer's parent to the numpy array.
    #     # When doing this, we need to adjust the reference counts manually so that garbage collection still works
    #
    #     # Get the functions for increasing and decreasing reference counts
    #     py_func_decrease_ref = ctypes.pythonapi.Py_DecRef
    #     py_func_increase_ref = ctypes.pythonapi.Py_IncRef
    #
    #     # Set the arguments for the functions
    #     ref_function_args = [ctypes.py_object]
    #     py_func_decrease_ref.argtypes = ref_function_args
    #     py_func_increase_ref.argtypes = ref_function_args
    #     # Set the return types for the functions (ctypes defaults to int, but we want no return type)
    #     py_func_decrease_ref.restype = None
    #     py_func_increase_ref.restype = None
    #
    #     # Increase the reference count of the numpy array
    #     py_func_increase_ref(np_from_pointer)
    #     # Decrease the reference count of the BglBuffer's current parent
    #     py_func_decrease_ref(c_buf.parent)
    #     # Set the numpy array as the new parent of the BglBuffer
    #     c_buf.parent = np_from_pointer
    #     c_buf.parent = ctypes.py_object.from_address(id(np_from_pointer)))
    #     # c_buf.parent = ctypes.pointer(PyObject.from_address(id(np_from_pointer)))
    #
    #     # It's now safe to return the numpy array as is, though the input bgl.Buffer can no longer be used
    #     return np_from_pointer
    # else:

    # We can't use the shared memory directly, because the bgl.Buffer can be garbage collected, freeing up the area
    # of memory that the numpy array is now using.
    # While we're in this function, there is clearly still a reference to the bgl.Buffer so we can safely make
    # a copy of all the data and return that instead.
    return np_from_pointer.copy()
