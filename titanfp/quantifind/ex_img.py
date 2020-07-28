"""Simple image processing examples."""

import operator
import math
import traceback

from ..titanic import ndarray
from ..fpbench import fpcparser
from ..arithmetic import mpmf, ieee754, posit, analysis

from . import search
from .utils import *
from .benchmarks import mk_blur

from PIL import Image
import numpy as np
from skimage import metrics as skim


def pixel(x):
    return max(0, min(int(x), 255))

def clamp(tensor):
    return ndarray.NDArray(shape=tensor.shape, data=map(pixel, tensor.data))

def npify(tensor):
    return np.array(clamp(tensor), dtype=np.uint8)

def imgify(tensor):
    return Image.fromarray(npify(tensor))

def ssim(a, b):
    return skim.structural_similarity(a, b, multichannel=True)

class ImgSettings(object):
    def __init__(self):
        input_img = Image.open('/home/bill/Documents/img32.png')
        np_img = np.array(input_img)
        img_data, img_shape = ndarray.reshape(np_img)
        nd_img = ndarray.NDArray(data=map(int, img_data), shape=img_shape)


        self.mask = '''(array (array 1/3 1/2 1/3)
                         (array 1/2 3/2 1/2)
                         (array 1/3 1/2 1/3))
        '''

        self.img_args = [nd_img] + fpcparser.read_exprs(self.mask)

        ref_evaltor = mpmf.Interpreter()
        ref_main = load_cores(ref_evaltor, mk_blur(f64, f64, f64, f64))
        ref_output = ref_evaltor.interpret(ref_main, self.img_args)
        output_img = imgify(ref_output)
        output_np = np.array(output_img)

        self.ref = output_np

        self.use_posit = True

    def cfg(self, use_posit):
        self.use_posit = use_posit

settings = ImgSettings()


def img_stage(ebits, overall_prec, mask_prec, accum_prec, mul_prec):
    if settings.use_posit:
        mk_ctx = posit.posit_ctx
        extra_bits = 0
    else:
        mk_ctx = ieee754.ieee_ctx
        extra_bits = ebits

    overall_ctx = mk_ctx(ebits, overall_prec + extra_bits)
    mask_ctx = mk_ctx(ebits, mask_prec + extra_bits)
    accum_ctx = mk_ctx(ebits, accum_prec + extra_bits)
    mul_ctx = mk_ctx(ebits, mul_prec + extra_bits)

    prog = mk_blur(overall_ctx, mask_ctx, accum_ctx, mul_ctx)

    evaltor = mpmf.Interpreter()
    als = analysis.BitcostAnalysis()
    main = load_cores(evaltor, prog, [als])
    result = evaltor.interpret(main, settings.img_args)

    err = ssim(settings.ref, npify(result))
    cost = als.bits_requested

    return cost, err

def img_ref_stage(overall_ctx, mask_ctx, accum_ctx, mul_ctx):
    prog = mk_blur(overall_ctx, mask_ctx, accum_ctx, mul_ctx)

    evaltor = mpmf.Interpreter()
    als = analysis.BitcostAnalysis()
    main = load_cores(evaltor, prog, [als])
    result = evaltor.interpret(main, settings.img_args)

    err = ssim(settings.ref, npify(result))
    cost = als.bits_requested

    return cost, err

def img_fenceposts():
    points = [
        ((describe_ctx(ctx),), img_ref_stage(*((ctx,) * 4)))
        for ctx in float_basecase + posit_basecase
    ]

    return [0], points, points


def img_experiment(prefix, ebit_slice, pbit_slice, es_slice, inits, retries):
    img_metrics = (operator.lt, operator.gt)
    init_ebits, neighbor_ebits = integer_neighborhood(*ebit_slice)
    init_pbits, neighbor_pbits = integer_neighborhood(*pbit_slice)

    # for posits
    init_es, neighbor_es = integer_neighborhood(*es_slice)

    img_inits = (init_ebits,) + (init_pbits,) * 4
    img_neighbors = (neighbor_ebits,) + (neighbor_pbits,) * 4

    settings.cfg(False)
    try:
        sweep = search.sweep_multi(img_stage, img_inits, img_neighbors, img_metrics, inits, retries, force_exploration=True)
        jsonlog(prefix + '_blur.json', *sweep, settings='Blur with floats')
    except Exception:
        traceback.print_exc()

    img_inits = (init_es,) + (init_pbits,) * 4
    img_neighbors = (neighbor_es,) + (neighbor_pbits,) * 4

    settings.cfg(True)
    try:
        sweep = search.sweep_multi(img_stage, img_inits, img_neighbors, img_metrics, inits, retries, force_exploration=True)
        jsonlog(prefix + '_blur_p.json', *sweep, settings='Blur with posits')
    except Exception:
        traceback.print_exc()


def img_baseline(prefix):
    img_bc_float = (float_basecase,) * 4
    img_bc_posit = (posit_basecase,) * 4
    img_metrics = (operator.lt, operator.gt)

    settings.cfg(False)
    try:
        sweep = search.sweep_exhaustive(img_ref_stage, img_bc_float, img_metrics)
        jsonlog(prefix + '_blur.json', *sweep, settings='Blur with floats baseline')
    except Exception:
        traceback.print_exc()

    try:
        sweep = img_fenceposts()
        jsonlog(prefix + '_blur_fenceposts.json', *sweep, settings='Blur with floats fenceposts')
    except Exception:
        traceback.print_exc()

    settings.cfg(True)
    try:
        sweep = search.sweep_exhaustive(img_ref_stage, img_bc_posit, img_metrics)
        jsonlog(prefix + '_blur_p.json', *sweep, settings='Blur with posits baseline')
    except Exception:
        traceback.print_exc()

    try:
        sweep = img_fenceposts()
        jsonlog(prefix + '_blur_p_fenceposts.json', *sweep, settings='Blur with posits fenceposts')
    except Exception:
        traceback.print_exc()
