"""N-dimensional arrays."""

import random

def locate(shape, pos):
    idx = 0
    scale = 1
    for i in range(-1, -len(shape)-1, -1):
        s = shape[i]
        x = pos[i]
        idx += x * scale
        scale *= s
    return idx

def position(shape, idx):
    pos = []
    for i in range(-1, -len(shape)-1, -1):
        idx, x = divmod(idx, shape[i])
        pos.append(x)
    return tuple(reversed(pos))

def check_bounds(shape, pos):
    if len(shape) != len(pos):
        raise ValueError('{} has invalid shape, expecting {}'.format(repr(pos), repr(shape)))
    for i in range(len(shape)):
        s = shape[i]
        x = pos[i]
        if x < 0 or s <= x:
            raise IndexError('{} out of range for shape {}'.format(repr(pos), repr(shape)))

def shape_size(shape):
    scale = 1
    for dim in shape:
        scale *= dim
    return scale

def test_shape(dims, dimsz):
    shape = tuple([random.randint(1, dimsz) for i in range(dims)])
    pos = tuple([random.randint(0, dim-1) for dim in shape])

    idx = locate(shape, pos)
    pos2 = position(shape, idx)

    if pos != pos2:
        return shape, pos, idx, pos2
    else:
        return None

def testn(dims, dimsz, count, step=0):
    print('testing {} random shapes with {} dimensions up to size {}'.format(count, dims, dimsz))
    failures = 0
    for i in range(count):
        result = test_shape(dims, dimsz)
        if result is not None:
            print(result)
            failures += 1
    print('failed {} tests'.format(failures))

class NDArray(object):

    def __init__(self, shape, data=None):
        self.shape = shape
        self.size = shape_size(shape)
        if data is None:
            self.data = [None]*self.size
        else:
            if len(data) != self.size:
                raise ValueError('wrong data size {} for shape {}, expecting {}'.format(len(data), self.shape, self.size))
            self.data = data

    def to_list(self):
        data = self.data
        for i in range(-1, -len(self.shape)-1, -1):
            dim = self.shape[i]
            data = [data[chunk * dim:(chunk + 1) * dim] for chunk in range(len(data) // dim)]
        return data

    def to_tuple(self):
        return tuple(list(self))

    def __str__(self):
        return str(self.to_list())

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.shape) + ', data=' + repr(self.data) + ')'

    def __len__(self):
        return self.size

    def __getitem__(self, k):
        check_bounds(self.shape, k)
        return self.data[locate(self.shape, k)]

    def __setitem__(self, k, v):
        check_bounds(self.shape, k)
        self.data[locate(self.shape, k)] = v
