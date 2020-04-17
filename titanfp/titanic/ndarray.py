"""N-dimensional arrays and utilities."""


from collections.abc import Iterable, Sequence, MutableSequence


class ShapeError(ValueError):
    """Invalid shape."""


def reshape(a, recshape=None):
    """Convert a nested, non-string iterable into a flat generator and its shape.
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
            return (recshape(subtensor, subshape) if subshape else subtensor
                    for subtensor, subshape in zip(data, shapes)), (len(data),)
        else:
            if shape:
                return (elt for subtensor in data for elt in subtensor), (len(data), *shape)
            else:
                return data, (len(data),)
    else:
        return a, ()

def unshape(data, shape):
    """Expand a flat iterable and its shape into a nested tuple.
    """
    a = data
    for dim in reversed(shape[1:]):
        a = zip(*[iter(a)]*dim)
    return tuple(a)

def unshape_list(data, shape):
    """Expand a flat list and its shape into a nested list.
    """
    a = data
    for dim in reversed(shape[1:]):
        a = [a[chunk * dim:(chunk + 1) * dim] for chunk in range(len(a) // dim)]
    return a

def describe(data, shape=None, descr=repr, sep=', ', lparen='(', rparen=')'):
    """Convert a shaped or unshaped iterable into a string, using the provided printing method and separators.
    """
    a = data
    if shape and len(shape) > 1:
        dim = shape[-1]
        shape = shape[:-1]
        a = zip(*[iter(a)]*dim)

    if isinstance(a,  Iterable) and not isinstance(a, str):
        return lparen + sep.join(describe(elt, shape=shape, descr=descr,
                                          sep=sep, lparen=lparen, rparen=rparen) for elt in a) + rparen
    else:
        return descr(a)

def locate(shape, pos):
    """Given a shape and a position vector, return the index of that position in the flat array.
    """
    idx = 0
    scale = 1
    for dim, coord in zip(reversed(shape), reversed(pos)):
        idx += coord * scale
        scale *= dim
    return idx

def position(shape, idx):
    """Given a shape and a flat index, return the corresponding position vector.
    """
    quot = idx
    pos = []
    for dim in reversed(shape):
        quot, rem = divmod(quot, dim)
        pos.append(rem)
    return tuple(reversed(pos))

def check_bounds(shape, pos):
    """Given a shape, check if a position vector is in bounds for that shape.
    Raises IndexError if the position is out of bounds.
    """
    for dim, coord in zip(shape, pos):
        if coord < 0 or dim <= coord:
            raise IndexError('{} out of range for shape {}'.format(repr(pos), repr(shape)))

def strict_bounds(shape, pos, strict=False):
    """Given a shape, check if a position vector is in bounds for that shape.
    Raises IndexError if the position is out of bounds or has the wrong number of dimensions.
    """
    if strict and len(shape) != len(pos):
        raise IndexError('{} has invalid shape, expecting {}'.format(repr(pos), repr(shape)))
    for dim, coord in zip(shape, pos):
        if coord < 0 or dim <= coord:
            raise IndexError('{} out of range for shape {}'.format(repr(pos), repr(shape)))

def shape_size(shape):
    """Compute the size of a shape (the len of the backing flat array).
    """
    if shape:
        scale = 1
        for dim in shape:
            scale *= dim
        return scale
    else:
        return 0

def check_size(data, shape):
    """Given a shape and a flat sequence, check if the sequence has the expected length.
    Raises ShapeError if the length is wrong.
    """
    if shape:
        scale = 1
        for dim in shape:
            scale *= dim
    else:
        scale = 0

    if len(data) != scale:
        raise ShapeError('shape {} should have total size {}, got {}'.format(repr(shape), repr(scale), repr(len(data))))

def fuse_lookup_old(shape, lookup):
    """Given a shape and a lookup, fuse concrete accesses at the beginning of the lookup.
    """
    strides, size = calc_strides_only(shape)

    start = 0
    fused = 0
    for dim, stride, query in zip(shape, strides, lookup):
        if isinstance(query, int):
            if query < 0:
                query = dim - query
            if query < 0 or dim <= query:
                raise IndexError('lookup {} out of range for dimension {} of shape {}'.format(repr(query), repr(fused), repr(shape)))
            start += stride * query
            fused += 1
        else:
            break

    return shape[fused:], strides[fused:], start, lookup[fused:]
    # size??

def calc_offset_old(shape, lookup):
    """Given a shape and a lookup, calculate explicit offsets for all slices in the lookup.
    """
    new_shape = []
    strides = []
    start = 0
    offset = []

    fused = list(zip(shape, lookup))
    scale = 1
    for dim in reversed(shape[len(lookup):]):
        new_shape.append(dim)
        strides.append(scale)
        scale *= dim
    for dim, query in reversed(fused):
        if isinstance(query, int):
            start += scale * query
        elif isinstance(query, slice):
            q_start, q_stop, q_stride = query.indices(dim)
            extent = q_stop - q_start
            new_shape.append(max(0, extent // q_stride))
            strides.append(scale)
            offset.append((q_start, q_stride))
        else:
            raise TypeError('index must be int or slice, got {}'.format(repr(query)))
        scale *= dim

    return (
        tuple(reversed(new_shape)),
        tuple(reversed(strides)),
        start,
        tuple(reversed(offset)),
        tuple(lookup[len(shape):]),
    )

def calc_strides(shape):
    """Calculate stride values for a shape.
    Returns the computed strides, and the overall size of the shape.
    """
    if shape:
        scale = 1
        strides = []
        for dim in reversed(shape):
            strides.append(scale)
            scale *= dim
        return tuple(reversed(strides)), scale
    else:
        return (), 0

def calc_offset(shape, strides, lookup):
    """Given a shape with strides and a lookup, calculate a start offset and new strides.
    Returns the new shape, the start offset, the new strides, and the rest of the lookup.
    """
    new_shape = []
    new_strides = []
    start = 0
    fused = 0

    for dim, stride, query in zip(shape, strides, lookup):
        if isinstance(query, int):
            if query < 0:
                query = dim - query
            if query < 0 or dim <= query:
                raise IndexError('lookup {} out of range for dimension {} of shape {}'
                                 .format(repr(query), repr(fused), repr(shape)))
            start += stride * query
        elif isinstance(query, slice):
            q_start, q_stop, q_stride = query.indices(dim)
            extent = q_stop - q_start
            new_shape.append(max(0, extent // q_stride))
            new_strides.append(stride * q_stride)
            start += stride * q_start
        else:
            raise TypeError('index must be int or slice, got {}'.format(repr(query)))
        fused += 1

    return (
        (*new_shape, *shape[fused:]),
        start,
        (*new_strides, *strides[fused:]),
        tuple(lookup[fused:]),
    )


def check_offset(data, shape, start, strides):
    """Check if a shape with a start offset and given strides is in bounds for some backing list.
    Raises ShapeError if the shape does not fit within the data.
    """
    min_offset = 0
    max_offset = 0
    for dim, stride in zip(shape, strides):
        offset = max(0, dim-1) * stride
        if offset < 0:
            min_offset += offset
        else:
            max_offset += offset
    if start + min_offset < 0 or len(data) <= start + max_offset:
        raise ShapeError('shape {} with strides {} extends from {} to {}, out of bounds for data with length {}'
                         .format(repr(shape), repr(strides), repr(start + min_offset), repr(start + max_offset),
                                 repr(len(data))))



class NDSeq(Sequence):
    """N-dimensional immutable sequence."""

    # core properties

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape

    @property
    def start(self):
        return 0

    # cached properties

    @property
    def strides(self):
        if self._dirty_strides:
            self._fixup_strides_size()
        return self._strides

    @property
    def size(self):
        if self._dirty_size:
            self._fixup_strides_size()
        return self._size

    def _fixup_strides_size(self):
        self._strides, self._size = calc_strides(self._shape)
        self._dirty_strides = False
        self._dirty_size = False

    def __init__(self, data=None, shape=None, strict=False):
        self._dirty_strides = True
        self._dirty_size = True
        if data:
            if shape:
                if isinstance(data, NDSeq):
                    self._data = list(data.data)
                else:
                    self._data = list(data)
                self._shape = shape
                if self.size != len(self.data):
                    raise ShapeError('data for shape {} should have size {}, got {}'
                                     .format(repr(shape), repr(self.size), repr(len(self.data))))
            else: # not shape
                if isinstance(data, NDSeq):
                    self._data = list(data.data)
                    self._shape = data.shape
                else:
                    if strict:
                        recshape = None
                    else:
                        # note that this depends on the default value for strict being false
                        recshape = type(self)
                    data_gen, shape = reshape(data, recshape=recshape)
                    self._data = list(data_gen)
                    self._shape = shape
        else: # not data
            if shape:
                strides, size = calc_strides(shape)
                self._data = [None]*size
                self._shape = shape
            else:
                self._data = []
                self._shape = ()

    # From the point of view of the sequence interface,
    # an NDSeq behaves like a sequence of other NDSeqs,
    # which are implemented as views of the same data.

    # def __contains__(self, item):
    #     raise NotImplementedError()

    # # better to write general iter and getitem here and inherit
    # def __iter__(self):
    #     if len(self.shape) > 1:
    #         dim, *subshape = self.shape
    #         stride, *substrides = self.strides
    #         return (NDSeqView(self.data, subshape, start=i*stride, strides=substrides) for i in range(dim))
    #     else:
    #         return iter(self.data)

    def __iter__(self):
        if len(self.shape) > 1:
            dim, *subshape = self.shape
            stride, *substrides = self.strides
            return (NDSeqView(self.data, subshape, start=self.start + (i*stride), strides=substrides) for i in range(dim))
        elif len(self.shape) == 1:
            length, = self.shape
            step, = self.strides
            if self.start == 0 and step == 1:
                return iter(self.data)
            else:
                return (self.data[self.start + (i*step)] for i in range(length))
        else:
            return iter([])

    # def __reversed__(self):
    #     raise NotImplementedError()

    def __len__(self):
        if self.shape:
            return self.shape[0]
        else:
            return 0

    def __getitem__(self, key):
        if isinstance(key, int):
            if self.shape:
                dim, *shape = self.shape
                stride, *strides = self.strides
                if key < 0:
                    key = dim - key
                if key < 0 or dim <= key:
                    raise IndexError('lookup {} out of range for dimension {} of shape {}'
                                     .format(repr(key), repr(shape)))
                start = stride * key
                lookup = ()
            else:
                raise IndexError('too many indices, got ({},) for shape {}'
                                 .format(repr(key), repr(self.shape)))
        elif isinstance(key, slice):
            if self.shape:
                dim, *shape = self.shape
                stride, *strides = self.strides
                k_start, k_stop, k_stride = key.indices(dim)
                extent = k_stop - k_start
                shape = (max(0, extent // k_stride), *shape)
                strides = (stride * k_stride, *strides)
                start = stride * k_start
                lookup = ()
            else:
                raise IndexError('too many indices, got ({},) for shape {}'
                                 .format(repr(key), repr(self.shape)))
        else:
            shape, start, strides, lookup = calc_offset(self.shape, self.strides, key)

        if shape:
            if lookup:
                # needs to reify
                # also, can actually coalesce
                # alternatively, we could keep lookups around and have a lookup merge logic
                raise NotImplementedError()
            else:
                return NDSeqView(self.data, shape, start=self.start + start, strides=strides)
        else:
            if lookup:
                return self.data[self.start + start][lookup]
            else:
                return self.data[self.start + start]

    def index(self, x, start=None, stop=None):
        raise NotImplementedError()

    def count(self, x):
        raise NotImplementedError()

    def tolist(self):
        return unshape_list(self.data, self.shape)

    def tostring(self, descr=repr, sep=', ', lparen='(', rparen=')'):
        return describe(self.data, self.shape, descr=descr, sep=sep, lparen=lparen, rparen=rparen)


class NDSeqView(NDSeq):
    """An offset view of an n-dimensional sequence."""

    # core properties

    @property
    def start(self):
        """The starting point of the data from the view, due to dereferenced dimensions.
        """
        return self._start

    # cached properties

    @property
    def size(self):
        if self._dirty_size:
            if self._dirty_strides:
                self._fixup_strides_size()
            else:
                self._fixup_size()
        return self._size

    def _fixup_size(self):
        self._size = shape_size(self._shape)

    def __init__(self, data, shape, start=0, strides=None):
        self._data = data
        self._shape = shape
        self._start = start
        if strides:
            self._strides = strides
            self._dirty_strides = False
            self._dirty_size = True
        else:
            self._dirty_strides = True
            self._dirty_size = True

        # might want to remove to improve performance if the check isn't needed
        check_offset(self._data, self._shape, self._start, self._strides)

    def __iter__(self):
        shape = self.shape
        if len(shape) > 1:
            dim, *subshape = shape
            stride, *substrides = self.strides
            return (NDSeqView(self.data, subshape, start=self.start + (i*stride), strides=substrides) for i in range(dim))
        elif len(shape) == 1:
            start = self.start
            length, = shape
            step, = self.strides
            if start == 0 and step == 1:
                return iter(self.data)
            else:
                return (self.data[start + (i*step)] for i in range(length))
        else:
            return iter([])



class NDArray(MutableSequence, NDSeq):
    """N-dimensional array."""

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise TypeError('cannot delete items from NDArray')

    def __iadd__(self, other):
        raise NotImplementedError()

    def reverse(self):
        raise NotImplementedError()

    def append(self, x):
        raise NotImplementedError()

    def insert(self, i, x):
        raise NotImplementedError()

    def extend(self):
        raise NotImplementedError()

    def pop(self, i=None):
        raise TypeError('cannot pop items from NDArray')

    def remove(self, x):
        raise TypeError('cannot remove items from NDArray')

    def clear(self):
        raise TypeError('cannot clear NDArray')
