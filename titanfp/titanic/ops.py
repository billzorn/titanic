"""Standard operation codes, for multiple numerical backends."""

from enum import IntEnum, unique

class RM(IntEnum):
    ROUND_NEAREST_EVEN = 0
    RNE = 0
    ROUND_NEAREST_AWAY = 1
    RNA = 1
    ROUND_UP = 2
    RTP = 2
    ROUND_DOWN = 3
    RTN = 3
    ROUND_TO_ZERO = 4
    RTZ = 4
    ROUND_AWAY_ZERO = 5
    RAZ = 5

class OF(IntEnum):
    INFINITY = 1
    INF = 1
    CLAMP = 2
    WRAP = 3

@unique
class OP(IntEnum):
    add = 0
    sub = 1
    mul = 2
    div = 3
    neg = 4
    sqrt = 5
    fma = 6
    copysign = 7
    fabs = 8
    fdim = 9
    fmax = 10
    fmin = 11
    fmod = 12
    remainder = 13
    ceil = 14
    floor = 15
    nearbyint = 16
    round = 17
    trunc = 18
    acos = 19
    acosh = 20
    asin = 21
    asinh = 22
    atan = 23
    atan2 = 24
    atanh = 25
    cos = 26
    cosh = 27
    sin = 28
    sinh = 29
    tan = 30
    tanh = 31
    exp = 32
    exp2 = 33
    expm1 = 34
    log = 35
    log10 = 36
    log1p = 37
    log2 = 38
    cbrt = 39
    hypot = 40
    pow = 41
    erf = 42
    erfc = 43
    lgamma = 44
    tgamma = 45
