(declare-const x32 Float32)
(declare-const x Real)

(declare-const r32 Float32)
(declare-const r Real)
(declare-const c32 Float32)

(define-const one32 Float32 ((_ to_fp 8 24) RTZ 1.0))

(define-fun op32 ((x Float32)) Float32
  (fp.sub RNE (fp.sqrt RNE (fp.add RNE x one32)) (fp.sqrt RNE x)))

(define-fun op ((x Real)) Real
  (- (^ (+ x 1.0) 0.5) (^ x 0.5)))

(assert (and
	 (= x (fp.to_real x32))
	 
	 (= r32 (op32 x32))
	 (= r (op x))
	 (= c32 ((_ to_fp 8 24) RTZ x))

	 (not (= r32 c32))
	 ))

(check-sat)
(get-model)
