from . import ex_sqrt, ex_dotprod, ex_rk, ex_img

def run_baselines():
    prefix = 'out/baseline'
    ex_sqrt.sqrt_baseline(prefix)
    # dot product doesn't use the same baseline (not tuning normal FP types)
    ex_rk.rk_baseline(prefix)
    ex_img.img_baseline(prefix)

def run_experiments():
    prefix = 'out/sweep'
    ex_sqrt.sqrt_experiment(prefix, (2,8,2), (1,32,3), (3,5,2,24), 20, 100)
    #ex_dotprod.dotprod_experiment(prefix, (1,1024,5), (1,32), 1000, 1000, 10, 20)
    ex_rk.rk_experiment(prefix, (2,8,2), (3,24,3), (0,2,1), 20, 50)
    ex_img.img_experiment(prefix, (1,8,2), (3,16,3), (0,2,1), 10, 20)

def test_run_experiments():
    prefix = 'out/test'
    ex_sqrt.sqrt_experiment(prefix, (3,5,2), (5,12,3), (3,5,7,10), 10, 20)
    #ex_dotprod.dotprod_experiment(prefix, (1,1024,5), (1,32), 5, 100, 5, 5)
    ex_rk.rk_experiment(prefix, (8,8,0), (12,15,2), (1,1,0), 5, 5)
    ex_img.img_experiment(prefix, (8,8,0), (5,8,2), (1,1,0), 5, 5)


def qf_arith():
    prefix = 'out/s11/sweep'
    ex_rk.rk_experiment(prefix, (2,8,1), (3,24,1), (0,2,1), 20, 50, eq_name='lorenz')
    ex_img.img_experiment(prefix, (1,8,1), (3,16,1), (0,2,1), 10, 20)
    prefix = 'out/s12/sweep'
    ex_rk.rk_experiment(prefix, (2,8,1), (3,24,2), (0,2,1), 20, 50, eq_name='lorenz')
    ex_img.img_experiment(prefix, (1,8,1), (3,16,2), (0,2,1), 10, 20)
    prefix = 'out/s22/sweep'
    ex_rk.rk_experiment(prefix, (2,8,2), (3,24,2), (0,2,1), 20, 50, eq_name='lorenz')
    ex_img.img_experiment(prefix, (1,8,2), (3,16,2), (0,2,1), 10, 20)
    prefix = 'out/s23/sweep'
    ex_rk.rk_experiment(prefix, (2,8,2), (3,24,3), (0,2,1), 20, 50, eq_name='lorenz')
    ex_img.img_experiment(prefix, (1,8,2), (3,16,3), (0,2,1), 10, 20)
    prefix = 'out/s24/sweep'
    ex_rk.rk_experiment(prefix, (2,8,2), (3,24,4), (0,2,1), 20, 50, eq_name='lorenz')
    ex_img.img_experiment(prefix, (1,8,2), (3,16,4), (0,2,1), 10, 20)

def qf_arith_random():
    prefix = 'out/random/sweep'
    ex_rk.rk_random(prefix, (2,8,2), (3,24,3), (0,2,1), 20000, eq_name='lorenz')
    ex_img.img_random(prefix, (1,8,2), (3,16,3), (0,2,1), 10000)
    
if __name__ == '__main__':
    qf_arith_random()
