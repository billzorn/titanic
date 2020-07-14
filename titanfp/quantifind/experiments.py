from . import ex_sqrt, ex_dotprod, ex_rk, ex_img

def run_experiments():
    prefix = 'sweep'
    ex_sqrt.sqrt_experiment(prefix, (2,8,2), (1,32,3), (3,5,2,24), 20, 100)
    ex_dotprod.dotprod_experiment(prefix, (1,1024,5), (1,32), 1000, 1000, 10, 20)
    ex_rk.rk_experiment(prefix, (2,8,2), (3,24,3), (0,2,1), 20, 50)
    ex_img.img_experiment(prefix, (1,8,2), (3,16,3), (0,2,1), 10, 20)

def test_run_experiments():
    prefix = 'test'
    ex_sqrt.sqrt_experiment(prefix, (3,5,2), (5,12,3), (3,5,7,10), 10, 20)
    ex_dotprod.dotprod_experiment(prefix, (1,1024,5), (1,32), 5, 100, 5, 5)
    ex_rk.rk_experiment(prefix, (8,8,0), (12,15,2), (1,1,0), 5, 5)
    ex_img.img_experiment(prefix, (8,8,0), (5,8,2), (1,1,0), 5, 5)
