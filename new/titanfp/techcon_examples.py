from .fpbench import fpcparser
from .arithmetic import ieee754, sinking

quadratic = fpcparser.compile1(
"""(FPCore (a b c)
 :name "NMSE p42, positive"
 :cite (hamming-1987 herbie-2015)
 :fpbench-domain textbook
 :pre (and (>= (* b b) (* 4 (* a c))) (!= a 0))
 (/ (+ (- b) (sqrt (- (* b b) (* 4 (* a c))))) (* 2 a)))
""")

quadratic_herbified = fpcparser.compile1(
"""(FPCore (a b c)
 :herbie-status success
 :herbie-time 118637.2451171875
 :herbie-bits-used 3392
 :herbie-error-input ((256 28.609971950677362) (8000 33.90307227594979))
 :herbie-error-output ((256 5.078369297841056) (8000 6.594164753178634))
 :name "NMSE p42, positive"
 :pre (and (>= (* b b) (* 4 (* a c))) (!= a 0))
 (if (<= b -3.2964251401560902e+93)
     (- (/ b a))
     (if (<= b -9.121837495335558e-234)
         (/ 1 (/ (* a 2) (- (sqrt (- (* b b) (* c (* a 4)))) b)))
         (if (<= b 4.358108025323294e+96)
             (/ 1 (* (+ (sqrt (- (* b b) (* c (* a 4)))) b) (/ 2 (/ (- 4) (/ 1 c)))))
             (/ (- c) b)))))
""")

quadratic_regime = fpcparser.compile1(
"""(FPCore (a b c)
 (/ 1 (* (+ (sqrt (- (* b b) (* c (* a 4)))) b) (/ 2 (/ (- 4) (/ 1 c))))))
""")

arguments = [
    ('0.1', '2', '3'),
    ('0.001', '2', '3'),
    ('1e-9', '2', '3'),
    ('1e-15', '2', '3'),
    ('1e-16', '2', '3'),
    ('1e-17', '2', '3'),
]

if __name__ == '__main__':
    for args in arguments:
        ieee754_answer = ieee754.Interpreter.interpret(quadratic, args, ctx=ieee754.ieee_ctx(w=11, p=53))
        sinking_answer = sinking.Interpreter.interpret(quadratic, args, ctx=ieee754.ieee_ctx(w=11, p=53))
        print(*args, str(ieee754_answer), str(sinking_answer))

    print()

    for args in arguments:
        ieee754_answer = ieee754.Interpreter.interpret(quadratic_regime, args, ctx=ieee754.ieee_ctx(w=11, p=53))
        sinking_answer = sinking.Interpreter.interpret(quadratic_regime, args, ctx=ieee754.ieee_ctx(w=11, p=53))
        print(*args, str(ieee754_answer), str(sinking_answer))
