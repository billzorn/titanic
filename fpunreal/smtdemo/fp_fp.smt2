(declare-const x32 Float32)
(declare-const x64 Float64)

(declare-const r32 Float32)
(declare-const r64 Float64)
(declare-const c32 Float32)

(define-const one32 Float32 ((_ to_fp 8 24) RTZ 1.0))
(define-const one64 Float64 ((_ to_fp 11 53) RTZ 1.0))

(define-fun op32 ((x Float32)) Float32
  (fp.sub RNE (fp.sqrt RNE (fp.add RNE x one32)) (fp.sqrt RNE x)))

(define-fun op64 ((x Float64)) Float64
  (fp.sub RNE (fp.sqrt RNE (fp.add RNE x one64)) (fp.sqrt RNE x)))

(assert (and
	 (= x64 ((_ to_fp 11 53) RTZ x32))
	 
	 (= r32 (op32 x32))
	 (= r64 (op64 x64))
	 (= c32 ((_ to_fp 8 24) RTZ r64))
	 
	 (not (= r32 c32))	 
	 ))

(check-sat)
(get-model)
