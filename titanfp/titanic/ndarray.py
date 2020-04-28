"""N-dimensional arrays and utilities."""


# NOTE TODO: see https://rszalski.github.io/magicmethods/

from collections.abc import Iterable, Sequence, MutableSequence
import itertools


class ShapeError(ValueError):
    """Invalid shape."""

class Shaped(object):
    """A multi-dimensional sequence, which records the size of each dimension."""

    @property
    def data(self):
        raise NotImplementedError()

    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def size(self):
        raise NotImplementedError()

    @property
    def strides(self):
        raise NotImplementedError()

class View(object):
    """A view of another data structure, which can reify()
    to reconstruct an explicit backing.
    """

    def reify(self):
        raise NotImplementedError()


def reshape(a, recshape=None):
    """Convert a nested, non-string iterable into a flat generator and its shape.
    Raggedly shaped data will be processed recursively by calling recshape.
    If recshape is None, then ragged data wil raise ShapeError.
    To leave ragged data unshaped, pass recshape=unshape.
    """
    if isinstance(a, Shaped) and not isinstance(a, View):
        return a.data, a.shape
    elif isinstance(a, Iterable) and not isinstance(a, str):
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
                shapes = [shape] * (len(data) - 1)
                shapes.append(subshape)
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

def unshape_tuple(data, shape):
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

def unshape_gen(data, shape):
    """Expand a flat list and its shape into a nested generator.
    """
    a = data
    for dim in reversed(shape[1:]):
        a = (a[chunk * dim:(chunk + 1) * dim] for chunk in range(len(a) // dim))
    return a

def describe(a, descr=repr, sep=', ', lparen='(', rparen=')'):
    """Convert a shaped or unshaped iterable into a one-line string,
    using the provided printing method and separators.
    """
    if isinstance(a,  Iterable) and not isinstance(a, str):
        return ''.join([
            lparen,
            sep.join(describe(elt, descr=descr, sep=sep, lparen=lparen, rparen=rparen) for elt in a),
            rparen,
        ])
    else:
        return descr(a)

def dimsep_array(depth, height, lparen, rparen):
    if height <= 1:
        return ', '
    elif height <=3 :
        indent = ' ' * (len(lparen) * (depth+1))
        return ',{}{}'.format('\n' * (height-1), indent)
    else:
        indent = ' ' * (len(lparen) * (depth+1))
        return ',\n\n{}## {:d} ##\n\n{}'.format(indent, depth, indent)

def describe_nd(a, descr=repr, dimsep=dimsep_array, lparen='(', rparen=')', depth=0):
    """Convert a shaped or unshaped iterable into a string and a count of dimensions,
    using the provided printing method and separators.
    dimsep is a function that computes the separator given a logical depth and height
    from the top and bottom of the data structure, and the parentheses.
    """
    if isinstance(a, Iterable) and not isinstance(a, str):
        rows, heights = zip(*(describe_nd(elt, descr=descr, dimsep=dimsep,
                                          lparen=lparen, rparen=rparen, depth=depth+1) for elt in a))
        height = max(heights) + 1
        sep = dimsep(depth, height, lparen, rparen)
        return ''.join([
            lparen,
            sep.join(rows),
            rparen,
        ]), height
    else:
        return descr(a), 0

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
    Returns the start offset, the new shape, the new strides, and the rest of the lookup.
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
            start += q_start * stride
        else:
            raise TypeError('index must be int or slice, got {}'.format(repr(query)))
        fused += 1

    return (
        start,
        (*new_shape, *shape[fused:]),
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



# TODO notes

# how many implementations of reshape?
# Special cases for shaped / view?

# comparison
# mutable sequences

# it's very likely that the current calc_offset logic is subtly wrong for negative accesses


# TODO in an ideal world with infinite time

# hashing
# size changing mutation
# tests

# Python package?
# lazy sequences (including lookup chains, and size changing operations)



class NDSeq(Sequence, Shaped):
    """N-dimensional immutable sequence."""

    # informally, accessing the data properties (except start)
    # will cause a view to reify(), so methods that have a more efficient implementation
    # that doesn't depend on reification should use the "raw" values _data etc.

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    @property
    def strides(self):
        return self._strides

    @property
    def start(self):
        return 0

    def __init__(self, data=None, shape=None, strict=False):
        if data:
            if shape:
                if isinstance(data, NDSeq):
                    self._data = list(data.data)
                else:
                    self._data = list(data)
                self._shape = shape
                self._strides, self._size = calc_strides(self._shape)
                if self._size != len(self._data):
                    raise ShapeError('data for shape {} should have size {}, got {}'
                                     .format(repr(shape), repr(self.size), repr(len(self.data))))
            else: # not shape
                if isinstance(data, NDSeq):
                    self._data = list(data.data)
                    self._shape = data.shape
                    self._strides, self._size = calc_strides(self._shape)
                else:
                    if strict:
                        recshape = None
                    else:
                        # note that this depends on the default value for strict being false
                        recshape = type(self)
                    data_gen, shape = reshape(data, recshape=recshape)
                    self._data = list(data_gen)
                    self._shape = shape
                    self._strides, self._size = calc_strides(self._shape)
        else: # not data
            if shape:
                self._strides, self._size = calc_strides(shape)
                self._data = [None]*self._size
                self._shape = shape
            else:
                raise ValueError('shape cannot be empty')

    def __repr__(self):
        data = self.data
        shape = self.shape
        return '{}({}, {})'.format(type(self).__name__, repr(data), repr(shape))

    def __str__(self):
        s, height = describe_nd(self, descr=str, lparen='(', rparen=')')
        return s

    def __eq__(self, other):
        if isinstance(other, Iterable) and not isinstance(other, str):
            sentinel = object()
            return all(a == b for a, b in itertools.zip_longest(self, other, fillvalue=sentinel))
        else:
            return False

    def __ne__(self, other):
        if isinstance(other, Iterable) and not isinstance(other, str):
            sentinel = object()
            return any(a != b for a, b in itertools.zip_longest(self, other, fillvalue=sentinel))
        else:
            return True

    def __lt__(self, other):
        if isinstance(other, Iterable) and not isinstance(other, str):
            sentinel = object()
            for a, b in itertools.zip_longest(self, other, fillvalue=sentinel):
                if a is sentinel:
                    return True
                elif b is sentinel:
                    return False
                elif a < b:
                    return True
                elif b < a:
                    return False
            return False
        else:
            raise TypeError("'<' not supported between instances of '{}' and '{}'"
                            .format(type(self).__name__, type(other).__name__))


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
        dim, *subshape = self._shape
        stride, *substrides = self._strides
        start = self.start
        if subshape:
            return (NDSeqView(self._data, subshape, start=start + (i*stride), strides=substrides) for i in range(dim))
        else:
            return (self._data[start + (i*stride)] for i in range(dim))

    # def __reversed__(self):
    #     raise NotImplementedError()

    def __len__(self):
        # We don't actually know the shape, since the elements might be well-shaped subsequences
        # that will reify into a new sequence with more dimensions;
        # however, whatever happens, the size of the first dimension will never change.
        return self._shape[0]

    def __getitem__(self, key):
        if isinstance(key, int):
            dim, *subshape = self._shape
            stride, *substrides = self._strides
            if key < 0:
                key = dim + key
            if 0 <= key < dim:
                offset = stride * key
                lookup = ()
            else:
                raise IndexError('lookup {} out of range for dimension {} of shape {}'
                                 .format(repr(key), repr(shape)))
        elif isinstance(key, slice):
            dim, *subshape = self._shape
            stride, *substrides = self._strides
            k_start, k_stop, k_stride = key.indices(dim)
            extent = k_stop - k_start
            subshape = (max(0, extent // k_stride), *subshape)
            substrides = (stride * k_stride, *substrides)
            offset = k_start * stride
            lookup = ()
        else:
            offset, subshape, substrides, lookup = calc_offset(self._shape, self._strides, key)

        substart = self.start + offset
        if subshape:
            subseq = NDSeqView(self._data, subshape, start=substart, strides=substrides)
            if lookup:
                # could we make this more efficient, by not actually constructing this backing array?
                subdata = [elt[lookup] for elt in subseq.data]
                # we construct a concrete backing list for a view of the lookups;
                # because it's still a view, it can still try to reify to condense its shape.
                return NDSeqView(subdata, subshape)
            else:
                return subseq
        else:
            if lookup:
                return self._data[substart][lookup]
            else:
                return self._data[substart]

    def index(self, x, start=None, stop=None):
        raise NotImplementedError()

    def count(self, x):
        raise NotImplementedError()

    def totuple(self):
        return unshape_tuple(self.data, self.shape)

    def tolist(self):
        return unshape_list(self.data, self.shape)

    def tostr(self, descr=repr, sep=', ', lparen='(', rparen=')'):
        return describe(self, descr=descr, sep=sep, lparen=lparen, rparen=rparen)


class NDSeqView(NDSeq, View):
    """An offset view of an n-dimensional sequence."""

    real_cls = NDSeq

    @property
    def data(self):
        self.reify()
        return self._data

    @property
    def shape(self):
        self.reify()
        return self._shape

    @property
    def size(self):
        self.reify()
        return self._size

    @property
    def strides(self):
        self.reify()
        return self._strides

    @property
    def start(self):
        return self._start

    def reify(self):
        cls = self.real_cls
        data_gen, shape = reshape(self, recshape=cls)
        self._data = list(data_gen)
        self._shape = shape
        self._strides, self._size = calc_strides(self._shape)
        del self._start
        self.__class__ = cls

    def __init__(self, data, shape, start=0, strides=None):
        self._data = data
        self._shape = shape
        self._start = start
        if strides:
            self._strides = strides
            self._size = shape_size(self._shape)
        else:
            self._strides, self._size = calc_strides(self._shape)

        # might want to remove to improve performance if the check isn't needed
        check_offset(self._data, self._shape, self._start, self._strides)


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
