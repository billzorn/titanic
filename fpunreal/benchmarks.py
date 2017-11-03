import z3

RTZ = z3.RoundTowardZero()
RNE = z3.RoundNearestTiesToEven()

# (FPCore (x)
#  :name "NMSE example 3.1"
#  :cite (hamming-1987)
#  :pre (>= x 0)
#  (- (sqrt (+ x 1)) (sqrt x)))
def fpc_hamming_3_1(x, rm = RNE):
    return z3.fpSub(rm, z3.fpSqrt(rm, z3.fpAdd(rm, x, z3.FPVal(1.0, x.sort()))),
                    z3.fpSqrt(rm, x))

# (FPCore (x)
#  :name "NMSE example 3.6"
#  :cite (hamming-1987)
#  :pre (>= x 0)
#  (- (/ 1 (sqrt x)) (/ 1 (sqrt (+ x 1)))))
def fpc_hamming_3_6(x, rm = RNE):
    return z3.fpSub(rm, z3.fpDiv(rm, z3.FPVal(1.0, x.sort()), z3.fpSqrt(rm, x)),
                    z3.fpDiv(rm, z3.FPVal(1.0, x.sort()), z3.fpSqrt(rm, z3.fpAdd(rm, x, z3.FPVal(1.0, x.sort())))))

# (FPCore (x)
#  :name "NMSE problem 3.3.1"
#  :cite (hamming-1987)
#  :pre (!= x 0)
#  (- (/ 1 (+ x 1)) (/ 1 x)))
def fpc_hamming_3_3_1(x, rm = RNE):
    return z3.fpSub(rm, z3.fpDiv(rm, z3.FPVal(1.0, x.sort()), z3.fpAdd(rm, x, z3.FPVal(1.0, x.sort()))),
                    z3.fpDiv(rm, z3.FPVal(1.0, x.sort()), x))

# (FPCore (x)
#  :name "NMSE problem 3.3.3"
#  :cite (hamming-1987)
#  :pre (!= x 0)
#  (+ (- (/ 1 (+ x 1)) (/ 2 x)) (/ 1 (- x 1))))
def fpc_hamming_3_3_3(x, rm = RNE):
    return z3.fpAdd(rm, z3.fpSub(rm, z3.fpDiv(rm, z3.FPVal(1.0, x.sort()),
                                              z3.fpAdd(rm, x,  z3.FPVal(1.0, x.sort()))),
                                 z3.fpDiv(rm, z3.FPVal(2.0, x.sort()), x)),
                    z3.fpDiv(rm, z3.FPVal(1.0, x.sort()), z3.fpSub(rm, x, z3.FPVal(1.0, x.sort()))))

# (FPCore (x)
#   :name "Rosa's Benchmark"
#   :cite (darulova-kuncak-2014)
#   (- (* 0.954929658551372 x) (* 0.12900613773279798 (* (* x x) x))))
def fpc_rosa(x, rm = RNE):
    return z3.fpSub(rm, z3.fpMul(rm, z3.FPVal(0.954929658551372, x.sort()), x),
                    z3.fpMul(rm, z3.FPVal(0.12900613773279798, x.sort()),
                             z3.fpMul(rm, z3.fpMul(rm, x, x), x)))
