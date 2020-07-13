"""FPCore benchmark templates for QuantiFind prototype"""

from ..arithmetic import evalctx, ieee754, posit

bf16 = ieee754.ieee_ctx(8, 16)
posit16_1 = posit.posit_ctx(1, 16)

sqrt_newton_template = '''(FPCore sqrt_bfloat_limit (a residual_bound)
 {overall_prec}

 (while* (and (! :titanic-analysis skip (< steps (# 10))) (>= (fabs residual) residual_bound))
  ([x a (! {diff_prec} (- x
                          (! {scale_prec} (/ residual (* 2 x)))))]
   [residual (! {res_prec} (- (* x x) a))
             (! {res_prec} (- (* x x) a))]
   [steps (! :titanic-analysis skip (# 0))
          (! :titanic-analysis skip (# (+ 1 steps)))])
  (cast x))

)
'''

sqrt_babylonian_template = '''(FPCore bab_bfloat_limit (a residual_bound)
 {overall_prec}

 (while* (and (! :titanic-analysis skip (< steps (# 10))) (>= (fabs residual) residual_bound))
  ([x a (! {diff_prec} (* 1/2 (+ x
                                 (! {scale_prec} (/ a x)))))]
   [residual (! {res_prec} (- (* x x) a))
             (! {res_prec} (- (* x x) a))]
   [steps (! :titanic-analysis skip (# 0))
          (! :titanic-analysis skip (# (+ 1 steps)))])
  (cast x)
 )

)
'''

def mk_sqrt(expbits, res_bits, diff_bits, scale_bits,
            overall_ctx=None, posit=False, babylonian=False):
    """Square root with Newton's method or optimized Babylonian method."""

    if overall_ctx is None:
        if posit:
            overall_prec = posit16_1.propstr()
        else:
            overall_prec = bf16.propstr()
    else:
        overall_prec = overall_ctx.propstr()

    if posit:
        res_ctx = posit.posit_ctx(expbits, res_bits)
        diff_ctx = posit.posit_ctx(expbits, diff_bits)
        scale_ctx = posit.posit_ctx(expbits, scale_bits)
    else:
        res_ctx = ieee754.ieee_ctx(expbits, expbits + res_bits)
        diff_ctx = ieee754.ieee_ctx(expbits, expbits + diff_bits)
        scale_ctx = ieee754.ieee_ctx(expbits, expbits + scale_bits)

    if babylonian:
        template = sqrt_babylonian_template
    else:
        template = sqrt_newton_template

    return template.format(
        overall_prec = overall_prec,
        res_prec = res_ctx.propstr(),
        diff_prec = diff_ctx.propstr(),
        scale_prec = scale_ctx.propstr(),
    )


dotprod_naive_template = '''(FPCore dotprod ((A n) (B m))
 :pre (== n m)
 {overall_prec}
  (for ([i n])
    ([accum 0 (! {sum_prec} (+ accum
                (! {mul_prec} (* (ref A i) (ref B i)))))])
    (cast accum)))
'''

dotprod_fused_template = '''(FPCore dotprod ((A n) (B m))
 :pre (== n m)
 {overall_prec}
  (for ([i n])
    ([accum 0 (! {sum_prec} (fma (ref A i) (ref B i) accum))])
    (cast accum)))
'''

dotprod_fused_unrounded_template = '''(FPCore dotprod ((A n) (B m))
 :pre (== n m)
 {overall_prec}
  (for ([i n])
    ([accum 0 (! {sum_prec} (fma (ref A i) (ref B i) accum))])
    accum))
'''

binsum_template = '''(FPCore addpairs ((A n))
 :pre (> n 1)
  (tensor ([i (# (/ (+ n 1) 2))])
    (let* ([k1 (# (* i 2))]
           [k2 (# (+ k1 1))])
      (if (< k2 n)
          (! {sum_prec} (+ (ref A k1) (ref A k2)))
          (ref A k1)))
  ))

(FPCore binsum ((A n))
  (while (> (size B 0) 1)
    ([B A (addpairs B)])
    (if (== (size B 0) 0) 0 (ref B 0))))
'''

nksum_template = '''(FPCore nksum ((A n))
 :name "Neumaier's improved Kahan Summation algorithm"
 {sum_prec}
  (for* ([i n])
    ([elt 0 (ref A i)]
     [t 0 (+ accum elt)]
     [c 0 (if (>= (fabs accum) (fabs elt))
              (+ c (+ (- accum t) elt))
              (+ c (+ (- elt t) accum)))]
     [accum 0 t])
    (+ accum c)))
'''

vec_prod_template = '''(FPCore vec-prod ((A n) (B m))
 :pre (== n m)
  (tensor ([i n])
    (! {mul_prec} (* (ref A i) (ref B i)))))
'''

dotprod_bin_template = (
    binsum_template + '\n' +
    vec_prod_template + '\n' +
'''(FPCore dotprod ((A n) (B m))
 :pre (== n m)
  (let ([result (binsum (vec-prod A B))])
    (! {overall_prec} (cast result))))
''')

dotprod_neumaier_template = (
    nksum_template + '\n' +
    vec_prod_template + '\n' +
'''(FPCore dotprod ((A n) (B m))
 :pre (== n m)
  (let ([result (nksum (vec-prod A B))])
    (! {overall_prec} (cast result))))
''')

dotprod_templates = {
    'naive' : dotprod_naive_template,
    'fused' : dotprod_fused_template,
    'unrounded' : dotprod_fused_unrounded_template,
    'bin' : dotprod_bin_template,
    'neumaier' : dotprod_neumaier_template,
}

def mk_dotprod(tempname, overall_ctx, mul_ctx, sum_ctx):
    """Dot product with control of accumulator precision."""

    if tempname in dotprod_templates:
        template = dotprod_templates[tempname]
    else:
        raise ValueError(f'unknown dot product template {tempname!r}')

    return template.format(
        overall_prec = overall_ctx.propstr() if overall_ctx else '',
        mul_prec = mul_ctx.propstr() if mul_ctx else '',
        sum_prec = sum_ctx.propstr() if sum_ctx else '',
    )


vec_scale_template = '''(FPCore vec-scale ((A n) x)
 (tensor ([i (# n)])
  (* (ref A i) x)))
'''

vec_add_template = '''(FPCore vec-add ((A n) (B m))
 :pre (== n m)
 (tensor ([i (# n)])
  (+ (ref A i) (ref B i))))
'''

rk4_template = '''(FPCore rk4-3d ((xyz 3) h)
 {rk_prec}
 (let* ([k1 (! {k1_prec} (vec-scale ({target_fn} xyz) h))]
        [k2 (! {k2_prec} (vec-scale ({target_fn} (vec-add xyz (vec-scale k1 1/2))) h))]
        [k3 (! {k3_prec} (vec-scale ({target_fn} (vec-add xyz (vec-scale k2 1/2))) h))]
        [k4 (! {k4_prec} (vec-scale ({target_fn} (vec-add xyz k3)) h))])
  (tensor ([i (# 3)])
   (+ (ref xyz i)
      (* 1/6
         (+ (+ (+ (ref k1 i) (* (ref k2 i) 2))
                  (* (ref k3 i) 2))
            (ref k4 i)))))))
'''

lorenz_template = '''(FPCore lorenz-3d ((xyz 3))
 {fn_prec}
 (let ([sigma 10]
       [beta 8/3]
       [rho 28]
       [x (ref xyz 0)]
       [y (ref xyz 1)]
       [z (ref xyz 2)])
  (array
      (* sigma (- y x))
      (- (* x (- rho z)) y)
      (- (* x y) (* beta z))
  )))
'''

rossler_template = '''(FPCore rossler-3d ((xyz 3))
 {fn_prec}
 (let ([a 0.432]
       [b 2]
       [c 4]
       [x (ref xyz 0)]
       [y (ref xyz 1)]
       [z (ref xyz 2)])
  (array
      (- (- y) z)
      (+ x (* a y))
      (- (+ b (* z x)) (* z c))
  )))
'''

chua_template = '''(FPCore chua-g (v)
 (if (< v -1) (+ (* -0.1 v) 3.9)
     (if (< v 1) (* -4.0 v)
         (- (* -0.1 v) 3.9))))

(FPCore chua-3d ((xyz 3))
 {fn_prec}
 (let ([inv_C1 10]
       [inv_C2 0.5]
       [inv_L 7]
       [G 0.7]
       [vc1 (ref xyz 0)]
       [vc2 (ref xyz 1)]
       [il (ref xyz 2)])
  (array
      (* inv_C1 (- (- (* G vc2) (* G vc1)) (chua-g vc1)))
      (* inv_C2 (+ (- (* G vc1) (* G vc2)) il))
      (* inv_L (- vc2))
  )))
'''

rk_main_template = '''(FPCore main ((initial-conditions 3) h steps)
 (tensor* ([step steps])
  ([xyz initial-conditions ({step_fn} xyz h)])
  xyz))
'''

rk_methods = {
    'rk4' : ('rk4-3d', rk4_template),
}

rk_equations = {
    'lorenz' : ('lorenz-3d', lorenz_template),
    'rossler' : ('rossler-3d', rossler_template),
    'chua' : ('chua-3d', chua_template),
}

rk_data = {
    'lorenz' : (
        '(array -12 -17/2 35) 1/64 240',
        (16.157760096498592, 19.29168560322699, 34.45572835102259),
        (31.339255067284, -123.60179554721446, 219.82848556462386),
    ),
    'rossler' : (
        '(array -2 -4 1/3) 1/16 270',
        (2.657846664768899, -3.7744623073353787, 0.8443226374976325),
        (2.930139669837746, 1.0272789480000155, 0.8667895560714332),
    ),
    'chua' : (
        '(array 10.2 2.2 -21.2) 1/32 280',
        (-11.144614016405676, -3.2978348452700295, 18.095962885507102),
        (4.7828401815438415, 6.301608732856075, 23.084843916890208),
    ),
}

def mk_rk(fn_ctx, rk_ctx, k1_ctx, k2_ctx, k3_ctx, k4_ctx, method='rk4', eqn='lorenz'):
    """RK methods for various chaotic attractors."""

    if method in rk_methods:
        mname, mtemp = rk_methods[method]
    else:
        raise ValueError(f'unknown method {method!r}')

    if eqn in rk_equations:
        eqname, eqtemp = rk_equations[eqn]
    else:
        raise ValueError(f'unknown equation {eqn!r}')

    template = '\n'.join([
        vec_scale_template,
        vec_add_template,
        eqtemp,
        mtemp,
        rk_main_template,
    ])

    return template.format(
        target_fn = eqname,
        step_fn = mname,
        fn_prec = fn_ctx.propstr(),
        rk_prec = rk_ctx.propstr(),
        k1_prec = k1_ctx.propstr(),
        k2_prec = k2_ctx.propstr(),
        k3_prec = k3_ctx.propstr(),
        k4_prec = k4_ctx.propstr(),
    )


blur_template = '''(FPCore fastblur-mask-3x3 ((img rows cols channels) (mask 3 3))
 {overall_prec}
 (let ([ymax (! :titanic-analysis skip (# (- rows 1)))]
       [xmax (! :titanic-analysis skip (# (- cols 1)))])
  (tensor ([y rows]
           [x cols])
   (for* ([my 3]
          [mx 3])
    ([y* 0 (! :titanic-analysis skip (# (+ y (- my 1))))]
     [x* 0 (! :titanic-analysis skip (# (+ x (- mx 1))))]
     [in-bounds? FALSE (! :titanic-analysis skip (and (<= 0 y* ymax) (<= 0 x* xmax)))]
     [mw 0 (if in-bounds? (! {mask_prec} (+ mw (ref mask my mx))) mw)]
     [w (tensor ([c channels]) 0)
        (if in-bounds?
            (tensor ([c channels])
              (! {accum_prec} (+ (ref w c)
                (! {mul_prec} (* (ref mask my mx) (ref img y* x* c)))))
            )
            w)])
    (tensor ([c channels]) (/ (ref w c) mw))))
 ))
'''

def mk_blur(overall_ctx, mask_ctx, accum_ctx, mul_ctx):
    """3x3 mask blur."""

    return blur_template.format(
        overall_prec = overall_ctx.propstr(),
        mask_prec = mask_ctx.propstr(),
        accum_prec = accum_ctx.propstr(),
        mul_prec = mul_ctx.propstr(),
    )
