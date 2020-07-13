from . import ex_sqrt, ex_dotprod, ex_rk, ex_img

import traceback


def go():
    try:
        ex_sqrt.sqrt_experiment()
    except Exception:
        traceback.print_exc()

    try:
        ex_dotprod.dotprod_experiment()
    except Exception:
        traceback.print_exc()

    try:
        ex_rk.rk_experiment()
    except Exception:
        traceback.print_exc()

    try:
        ex_img.img_experiment()
    except Exception:
        traceback.print_exc()

