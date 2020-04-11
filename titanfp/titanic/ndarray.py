"""N-dimensional arrays and utilities."""


from collections.abc import Iterable


class ShapeError(ValueError):
    """Invalid shape."""


def reshape(a, recshape=None):
    """Convert a nested, non-string iterable into a flat list and its shape.
    Raggedly shaped data will be processed recursively by calling recshape.
    If recshape is None, then ragged data wil raise ShapeError.
    To leave ragged data unshaped, pass recshape=unshape.
    """
    if isinstance(a, Iterable) and not isinstance(a, str):
        data = []
        shape = False
        ragged = False
        shapes = None
        for elt in a:
            subtensor, subshape = reshape(elt, recshape=recshape)
            data.append(subtensor)
            if shape is False:
                shape = subshape
            elif ragged:
                shapes.append(subshape)
            elif shape != subshape:
                if recshape is None:
                    raise ShapeError('array has ragged shape: expecting {}, got {}'.format(repr(shape), repr(subshape)))
                ragged = True
                shapes = [shape] * len(data)
        if ragged:
            return [recshape(subtensor, subshape) if subshape else subtensor
                    for subtensor, subshape in zip(data, shapes)], (len(data),)
        else:
            if shape:
                return [elt for subtensor in data for elt in subtensor], (len(data), *shape)
            else:
                return data, (len(data),)
    else:
        return a, ()

def unshape(data, shape):
    """Expand a flat list and its shape into a nested list.
    """
    a = data
    for dim in reversed(shape[1:]):
        a = [a[chunk * dim:(chunk + 1) * dim] for chunk in range(len(a) // dim)]
    return a


def locate(shape, pos):
    idx = 0
    scale = 1
    for dim, coord in zip(reversed(shape), reversed(pos)):
        idx += coord * scale
        scale *= dim
    return idx

def position(shape, idx):
    quot = idx
    pos = []
    for dim in reversed(shape):
        quot, rem = divmod(quot, dim)
        pos.append(rem)
    return tuple(reversed(pos))

def check_bounds(shape, pos):
    if len(shape) != len(pos):
        raise IndexError('{} has invalid shape, expecting {}'.format(repr(pos), repr(shape)))
    for dim, coord in zip(shape, pos):
        if coord < 0 or dim <= coord:
            raise IndexError('{} out of range for shape {}'.format(repr(pos), repr(shape)))

def shape_size(shape):
    scale = 1
    for dim in shape:
        scale *= dim
    return scale
