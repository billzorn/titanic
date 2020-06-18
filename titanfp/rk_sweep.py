import math
import operator

from .titanic import ndarray
from .fpbench import fpcparser
from .arithmetic import mpmf, ieee754, evalctx, analysis
from .arithmetic.mpmf import Interpreter

from .sweep import search



eqn_core = '''(FPCore lorenz-3d ((xyz 3))
 :precision {fn_prec}
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

rk_core = ('''(FPCore vec-scale ((A n) x)
 (tensor ([i (# n)])
  (* (ref A i) x)))

(FPCore vec-add ((A n) (B m))
 :pre (== n m)
 (tensor ([i (# n)])
  (+ (ref A i) (ref B i))))
'''
+ eqn_core +
'''(FPCore rk4-3d ((xyz 3) h)
 :precision {rk_prec}
 (let* ([k1 (! :precision {k1_prec} (vec-scale ({target_fn} xyz) h))]
        [k2 (! :precision {k2_prec} (vec-scale ({target_fn} (vec-add xyz (vec-scale k1 1/2))) h))]
        [k3 (! :precision {k3_prec} (vec-scale ({target_fn} (vec-add xyz (vec-scale k2 1/2))) h))]
        [k4 (! :precision {k4_prec} (vec-scale ({target_fn} (vec-add xyz k3)) h))])
  (tensor ([i (# 3)])
   (+ (ref xyz i)
      (* 1/6
         (+ (+ (+ (ref k1 i) (* (ref k2 i) 2))
                  (* (ref k3 i) 2))
            (ref k4 i)))))))

(FPCore main ((initial-conditions 3) h steps)
 (tensor* ([step steps])
  ([xyz initial-conditions ({step_fn} xyz h)])
  xyz))
''')

rk_args = '''(array 1 1 1)
1/16
189
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mkplot(data, name='fig.png', title='Some chaotic attractor'):
    fig = plt.figure(figsize=(12, 9), dpi=80)
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    ax.plot(data[0], data[1], data[2], color='blue', lw=1)
    plt.savefig(name)


# .0002, 68500:
# 0.6281892410761881 1.1362559216491108 11.931824372016203
# 0.0945572632908741 0.1616933534076016 -0.6410587355265847

# .02, 685:
# 0.7275905023319384 1.3081569466209049 12.058504949154397
# 0.10792263132676083 0.1859532584808219 -0.6437818982542893

# real_state = (0.6281892410761881, 1.1362559216491108, 11.931824372016203)
# real_derivative = (0.0945572632908741, 0.1616933534076016, -0.6410587355265847)
# alg_state = (0.7275905023319384, 1.3081569466209049, 12.058504949154397)
# alg_derivative = (0.10792263132676083, 0.1859532584808219, -0.6437818982542893)





# new:
# 1/16, 189:

ref_state = (1.0299874625744918, 2.1055180341334374, 18.17152158199435)
ref_dstate = (10.755305715589456, 8.017691512596356, -46.28873370793634)

def avg_abserr(a1, a2):
    count = 0
    err = 0
    for e1, e2 in zip(a1, a2):
        if math.isfinite(e1) and math.isfinite(e2):
            err += abs(float(e2) - float(e1))
            count += 1
        else:
            return math.inf
    return err / count

def run_rk(cores, args):
    evaltor = Interpreter()
    als = analysis.BitcostAnalysis()
    evaltor.analyses = [als]
    main = cores[-1]
    for core in cores:
        evaltor.register_function(core)
        if core.ident and core.ident.lower() == 'main':
            main = core

    result_array = evaltor.interpret(main, args)

    return evaltor, als, result_array

rk_ebits = 8

def eval_rk(als, result_array, fn_prec):
    last = [e for e in result_array[-1]]

    formatted = eqn_core.format(fn_prec=f'(float {rk_ebits} {fn_prec + rk_ebits!s})')
    eqn = fpcparser.compile1(formatted)

    evaltor = Interpreter()
    dlast = evaltor.interpret(eqn, [ndarray.NDArray(last)])

    return (
        als.bits_requested,
        avg_abserr([float(str(x)) for x in last], ref_state),
        avg_abserr([float(str(x)) for x in dlast], ref_dstate),
    )

def setup_rk(fn_prec, rk_prec, k1_prec, k2_prec, k3_prec, k4_prec):
    formatted = rk_core.format(
        target_fn = 'lorenz-3d',
        step_fn = 'rk4-3d',
        fn_prec = f'(float {rk_ebits} {fn_prec + rk_ebits!s})',
        rk_prec = f'(float {rk_ebits} {rk_prec + rk_ebits!s})',
        k1_prec = f'(float {rk_ebits} {k1_prec + rk_ebits!s})',
        k2_prec = f'(float {rk_ebits} {k2_prec + rk_ebits!s})',
        k3_prec = f'(float {rk_ebits} {k3_prec + rk_ebits!s})',
        k4_prec = f'(float {rk_ebits} {k4_prec + rk_ebits!s})',
    )

    cores = fpcparser.compile(formatted)
    args = fpcparser.read_exprs(rk_args)
    return cores, args

def rk_stage(fn_prec, rk_prec, k1_prec, k2_prec, k3_prec, k4_prec):
    cores, args = setup_rk(fn_prec, rk_prec, k1_prec, k2_prec, k3_prec, k4_prec)
    evaltor, als, result_array = run_rk(cores, args)
    return eval_rk(als, result_array, fn_prec)

def rk_plot(fn_prec, rk_prec, k1_prec, k2_prec, k3_prec, k4_prec, name=None):
    cores, args = setup_rk(fn_prec, rk_prec, k1_prec, k2_prec, k3_prec, k4_prec)
    evaltor, als, result_array = run_rk(cores, args)

    results = [[], [], []]
    for x,y,z in result_array:
        r_x, r_y, r_z = results
        r_x.append(float(str(x)))
        r_y.append(float(str(y)))
        r_z.append(float(str(z)))


    title = f'fn={fn_prec!s}, rk={rk_prec!s}, k1={k1_prec!s}, k2={k2_prec!s}, k3={k3_prec!s}, k4={k4_prec!s}'

    if name is None:
        mkplot(results, name=title, title=title)
    else:
        mkplot(results, name=name, title=title)




def init_prec():
    return 16
def neighbor_prec(x):
    nearby = 2
    for neighbor in range(x-nearby, x+nearby+1):
        if 1 <= neighbor <= 24 and neighbor != x:
            yield neighbor

rk_inits = (init_prec,) * 6
rk_neighbors = (neighbor_prec,) * 6
rk_metrics = (operator.lt,) * 3

def run_random():
    return search.sweep_random_init(rk_stage, rk_inits, rk_neighbors, rk_metrics)






lo_frontier = [
    ((7, 8, 7, 6, 8, 6), (572496, 0.14302797157119254, 0.828087840083158)),
    ((6, 6, 6, 9, 4, 5), (539461, 0.6963279715711925, 5.971421173416491)),
    ((6, 3, 3, 5, 4, 6), (496429, 7.987994638237859, 1.3539103120407165)),
    ((4, 2, 3, 1, 2, 2), (439938, 3.230990973765907, 2.161421173416491)),
    ((3, 2, 2, 1, 1, 2), (426708, 5.654661304904526, 17.494754506749825)),
    ((4, 2, 2, 1, 1, 2), (435024, 2.134669663188198, 7.828087840083158)),
    ((1, 2, 1, 1, 1, 2), (408942, 10.98799463823786, 12.16142117341649)),
    ((2, 3, 2, 1, 1, 2), (427255, 1.6786720284288075, 14.505245493250174)),
    ((1, 2, 1, 1, 1, 1), (406863, 10.134669663188198, 16.161421173416493)),
    ((3, 1, 1, 1, 1, 1), (414632, 5.769009026234094, 20.842115970309813)),
    ((2, 1, 1, 1, 1, 1), (406316, 6.365330336811802, 28.171912159916843)),
    ((3, 3, 3, 3, 1, 2), (444265, 1.4356756929007604, 6.342115970309813)),
    ((1, 1, 1, 3, 1, 1), (405560, 6.365330336811802, 31.171912159916843)),
    ((1, 1, 1, 1, 1, 1), (398000, 6.6786720284288075, 37.17191215991684)),
    ((1, 2, 1, 3, 1, 1), (414423, 10.987994638237879, 12.16142117341469)),
    ((3, 4, 3, 3, 1, 2), (453128, 0.8019946382378592, 2.161421173416491)),
    ((1, 3, 1, 3, 1, 1), (423286, 10.987994638237774, 12.161421173415816)),
    ((1, 3, 1, 4, 1, 1), (427066, 7.237994638237859, 15.324550696356853)),
]

hi_frontier = [
    ((16, 17, 15, 16, 17, 16), (828789, 0.0003949715711923929, 0.0018840296901861582)),
    ((15, 16, 14, 16, 13, 16), (795356, 0.001372028428806941, 0.007002636976479219)),
    ((14, 17, 13, 15, 16, 16), (802329, 0.00040361548223385074, 0.0026106963568537367)),
    ((14, 17, 13, 14, 16, 16), (798549, 0.0021740262340933145, 0.006157840083157377)),
    ((10, 13, 10, 11, 14, 14), (703373, 0.01164235956742754, 0.06061597030981206)),
    ((12, 13, 11, 10, 14, 14), (717359, 0.006742359567426452, 0.02444364537404997)),
    ((14, 13, 11, 10, 14, 14), (733991, 0.0011279715711927836, 0.011417363023520958)),
    ((8, 9, 8, 10, 12, 11), (631444, 0.02800536176214045, 0.07144930364314621)),
    ((8, 9, 8, 10, 12, 9), (627286, 0.2159423595674265, 0.7344211734164915)),
    ((6, 9, 6, 8, 11, 10), (599125, 0.6030053617621408, 2.8254493036431456)),
    ((8, 9, 6, 6, 10, 11), (606496, 0.18639463823785876, 1.1414211734164912)),
    ((6, 9, 6, 9, 10, 8), (594967, 0.6393243070992402, 1.9127563546259505)),
    ((4, 2, 3, 1, 2, 2), (439938, 3.230990973765907, 2.161421173416491)),
    ((3, 2, 2, 1, 1, 2), (426708, 5.654661304904526, 17.494754506749825)),
    ((4, 2, 2, 1, 1, 2), (435024, 2.134669663188198, 7.828087840083158)),
    ((1, 2, 1, 1, 1, 2), (408942, 10.98799463823786, 12.16142117341649)),
    ((2, 3, 2, 1, 1, 2), (427255, 1.6786720284288075, 14.505245493250174)),
    ((1, 2, 1, 1, 1, 1), (406863, 10.134669663188198, 16.161421173416493)),
    ((3, 1, 1, 1, 1, 1), (414632, 5.769009026234094, 20.842115970309813)),
    ((2, 1, 1, 1, 1, 1), (406316, 6.365330336811802, 28.171912159916843)),
    ((3, 3, 3, 3, 1, 2), (444265, 1.4356756929007604, 6.342115970309813)),
    ((1, 1, 1, 3, 1, 1), (405560, 6.365330336811802, 31.171912159916843)),
    ((1, 1, 1, 1, 1, 1), (398000, 6.6786720284288075, 37.17191215991684)),
    ((1, 2, 1, 3, 1, 1), (414423, 10.987994638237879, 12.16142117341469)),
    ((3, 4, 3, 3, 1, 2), (453128, 0.8019946382378592, 2.161421173416491)),
    ((1, 3, 1, 3, 1, 1), (423286, 10.987994638237774, 12.161421173415816)),
    ((1, 3, 1, 4, 1, 1), (427066, 7.237994638237859, 15.324550696356853)),
]


