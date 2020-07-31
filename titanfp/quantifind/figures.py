"""Frontier and 3d plotter."""

import os
import json
import re
import math
import operator
import traceback

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from adjustText import adjust_text

from .utils import *
from . import search

here = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(here, 'results')


class ExperimentData(object):
    """Data loader for stored experiment json logs."""

    _clean_re = re.compile(r'\W|^(?=\d)')
    def _clean(self, s):
        return self._clean_re.sub('_', s)

    _data_suffix = '.json'

    def __init__(self):
        files = filter(lambda name: name.endswith(self._data_suffix), os.listdir(data_dir))
        for fname in files:
            result_name = self._clean(fname[:-len(self._data_suffix)])
            if result_name in self.__dict__:
                raise ValueError(f'inappropriate result file name {fname!r}\n'
                                 f'  cleaned name {result_name!r} is already bound')
            with open(os.path.join(data_dir, fname), 'rt') as f:
                result_dict = json.load(f)

                frontier_as_lists = result_dict['frontier']
                frontier_as_tuples = [(tuple(a), tuple(b)) for (a, b) in frontier_as_lists]
                result_dict['frontier'] = frontier_as_tuples

                self.__dict__[result_name] = result_dict

data = ExperimentData()


def plot_frontier(fname, sources, new_metrics, plot_settings = [],
                  ref_pts = [], ref_lines = [], axis_titles = [],
                  complete_frontier = True, draw_ghosts = True, flip_axes = False):
    fig = plt.figure(figsize=(12,9), dpi=80)
    ax = fig.gca()

    print('generating', fname)

    try:
        plot_count = 0
        for source, metric_group, plot_settings_group in zip(sources, new_metrics, plot_settings):
            plot_count += 1
            frontier = source['frontier']
            all_points = source['configs']

            print(end='  ')

            for metrics, opts in zip(metric_group, plot_settings_group):
                filtered_frontier = search.filter_frontier(frontier, metrics)
                print(len(filtered_frontier), end=', ')

                x, y = [], []
                for cfg, measures in sorted(filtered_frontier, key = lambda t : t[1][0]):
                    a, b = measures
                    if flip_axes:
                        y.append(a)
                        x.append(b)
                    else:
                        x.append(a)
                        y.append(b)

                if plot_count == 1:
                    zidx = 98
                elif plot_count == 2:
                    zidx = 99
                else:
                    zidx = 100 - plot_count

                ax.plot(x, y, opts, zorder=zidx)

                if complete_frontier:
                    filtered_points = search.filter_metrics(frontier, metrics)
                    print(len(filtered_points), end=', ')
                    alpha = 0.5
                    x, y = [], []
                    for cfg, measures in sorted(filtered_points, key = lambda t : t[1][0]):
                        a, b = measures
                        if flip_axes:
                            y.append(a)
                            x.append(b)
                        else:
                            x.append(a)
                            y.append(b)

                    if plot_count == 1:
                        zidx = 68
                    elif plot_count == 2:
                        zidx = 69
                    else:
                        zidx = 70 - plot_count

                    ghost_opts = opts.rstrip('--').rstrip('-.').rstrip('-').rstrip(':')
                    ax.plot(x, y, ghost_opts, alpha=alpha, zorder=zidx)

                if draw_ghosts:
                    filtered_points = search.filter_metrics(all_points, metrics)
                    print(len(filtered_points), end=', ')
                    ghost_count = len(filtered_points)
                    alpha = min(100, math.sqrt(ghost_count)) / ghost_count
                    x, y = [], []
                    for cfg, measures in sorted(filtered_points, key = lambda t : t[1][0]):
                        a, b = measures
                        if flip_axes:
                            y.append(a)
                            x.append(b)
                        else:
                            x.append(a)
                            y.append(b)

                    if plot_count == 1:
                        zidx = 38
                    elif plot_count == 2:
                        zidx = 39
                    else:
                        zidx = 40 - plot_count

                    ghost_opts = opts.rstrip('--').rstrip('-.').rstrip('-').rstrip(':')
                    ax.plot(x, y, ghost_opts, alpha=alpha, zorder=zidx)

                print()

        texts = []
        for pt, label in ref_pts:
            if flip_axes:
                py, px = pt
            else:
                px, py = pt
            ax.scatter([px], [py], marker='o', color='red', zorder=100)
            texts.append(plt.text(
                px, py, label, zorder=101,
                #bbox=dict(facecolor='white', edgecolor='none', pad=1.0)
            ))
            adjust_text(texts)

        for y_value in ref_lines:
            ax.axhline(y_value, color='grey', zorder=0)

        if axis_titles:
            title, xlabel, ylabel = axis_titles
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

    except Exception:
        traceback.print_exc()

    finally:
        if not fname.lower().endswith('.pdf'):
            fname += '.pdf'
        with PdfPages(fname) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def convert_total_abits_to_avg(source):
    frontier = source['frontier']
    new_frontier = [(a, (*b[:-1], b[-1] / 417.0)) for a, b in frontier]
    source['frontier'] = new_frontier

    all_points = source['configs']
    new_points = [(a, (*b[:-1], b[-1] / 417.0)) for a, b in all_points]
    source['configs'] = new_points


def label_fenceposts(sweep, new_metrics):
    points = sweep['frontier']
    filtered = search.filter_metrics(points, new_metrics)

    fenceposts = []
    for data, measures in filtered:
        (label,) = data
        fenceposts.append((measures, label))

    return fenceposts


# same for newton and babylonian
sqrt_worst_ceiling = 6.632267911932321
sqrt_total_celing = 3318.686299105405
sqrt_avg_ceiling = sqrt_total_celing / 417.0

lorenz_avg_ceiling = 9.146828043582603
lorenz_davg_ceiling = 6.514926037030233
rossler_avg_ceiling = 14.207585778814689
rossler_davg_ceiling = 13.641604831520093
chua_avg_ceiling = 8.87539382493857
chua_davg_ceiling = 8.330294319415229

def all_plots():
    plot_dir = os.path.join(here, 'paper/figs')

    # return timeouts, infs, worst_bitcost, total_bitcost, worst_ulps, total_ulps, worst_abits, total_abits
    new_sqrt_metrics_worst = (None, None, None, operator.lt, None, None, operator.gt, None)
    new_sqrt_metrics_avg = (None, None, None, operator.lt, None, None, None, operator.gt)
    new_sqrt_metrics_infs = (None, operator.lt, None, operator.lt, None, None, None, None)
    new_sqrt_metrics_timeouts = (operator.lt, None, None, operator.lt, None, None, None, None)


    convert_total_abits_to_avg(data.sweep_newton_full)
    convert_total_abits_to_avg(data.sweep_newton_random)
    convert_total_abits_to_avg(data.baseline_newton)
    convert_total_abits_to_avg(data.baseline_newton_fenceposts)

    plot_frontier(os.path.join(plot_dir, 'sqrt_newton_infs'),
                  [data.sweep_newton_full, data.sweep_newton_random, data.baseline_newton],
                  [[new_sqrt_metrics_infs],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts = label_fenceposts(data.baseline_newton_fenceposts, new_sqrt_metrics_infs),
                  flip_axes = True,
                  axis_titles = ["square root with Newton's method", "bitcost", "infinities (out of 417 test cases)"])

    plot_frontier(os.path.join(plot_dir, 'sqrt_newton_timeouts'),
                  [data.sweep_newton_full, data.sweep_newton_random, data.baseline_newton],
                  [[new_sqrt_metrics_timeouts],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts = label_fenceposts(data.baseline_newton_fenceposts, new_sqrt_metrics_timeouts),
                  flip_axes = True)

    plot_frontier(os.path.join(plot_dir, 'sqrt_newton_avg'),
                  [data.sweep_newton_full, data.sweep_newton_random, data.baseline_newton],
                  [[new_sqrt_metrics_avg],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts = label_fenceposts(data.baseline_newton_fenceposts, new_sqrt_metrics_avg),
                  ref_lines=[sqrt_avg_ceiling])

    plot_frontier(os.path.join(plot_dir, 'sqrt_newton_worst'),
                  [data.sweep_newton_full, data.sweep_newton_random, data.baseline_newton],
                  [[new_sqrt_metrics_worst],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts=label_fenceposts(data.baseline_newton_fenceposts, new_sqrt_metrics_worst),
                  ref_lines=[sqrt_worst_ceiling])

    convert_total_abits_to_avg(data.sweep_babylonian_full)
    convert_total_abits_to_avg(data.sweep_babylonian_random)
    convert_total_abits_to_avg(data.baseline_babylonian)
    convert_total_abits_to_avg(data.baseline_babylonian_fenceposts)

    plot_frontier(os.path.join(plot_dir, 'sqrt_babylonian_infs'),
                  [data.sweep_babylonian_full, data.sweep_babylonian_random, data.baseline_babylonian],
                  [[new_sqrt_metrics_infs],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts = label_fenceposts(data.baseline_babylonian_fenceposts, new_sqrt_metrics_infs),
                  flip_axes = True)

    plot_frontier(os.path.join(plot_dir, 'sqrt_babylonian_timeouts'),
                  [data.sweep_babylonian_full, data.sweep_babylonian_random, data.baseline_babylonian],
                  [[new_sqrt_metrics_timeouts],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts = label_fenceposts(data.baseline_babylonian_fenceposts, new_sqrt_metrics_timeouts),
                  flip_axes = True)

    plot_frontier(os.path.join(plot_dir, 'sqrt_babylonian_avg'),
                  [data.sweep_babylonian_full, data.sweep_babylonian_random, data.baseline_babylonian],
                  [[new_sqrt_metrics_avg],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts=label_fenceposts(data.baseline_babylonian_fenceposts, new_sqrt_metrics_avg),
                  ref_lines=[sqrt_avg_ceiling])

    plot_frontier(os.path.join(plot_dir, 'sqrt_babylonian_worst'),
                  [data.sweep_babylonian_full, data.sweep_babylonian_random, data.baseline_babylonian],
                  [[new_sqrt_metrics_worst],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts=label_fenceposts(data.baseline_babylonian_fenceposts, new_sqrt_metrics_worst),
                  ref_lines=[sqrt_worst_ceiling])

    # # return quire_lo + quire_hi, infs, worst_ulps, avg_ulps, worst_abits, avg_abits
    # dotprod_avg_metrics = (operator.lt, None, None, None, None, operator.gt)

    # plot_frontier(os.path.join(plot_dir, 'dotprod_fused'),
    #               [data.sweep_dotprod_fused, data.sweep_dotprod_fused_unsigned],
    #               [[dotprod_avg_metrics],] * 2,
    #               plot_settings = [['C0x--'], ['C1+:']],
    #               ref_pts=[])

    # plot_frontier(os.path.join(plot_dir, 'dotprod_bin'),
    #               [data.sweep_dotprod_bin, data.sweep_dotprod_bin_unsigned],
    #               [[dotprod_avg_metrics],] * 2,
    #               plot_settings = [['C0x--'], ['C1+:']],
    #               ref_pts=[])

    # return als.bits_requested, worst_abits_last, avg_abits_last, worst_abits_dlast, avg_abits_dlast
    rk_avg_metrics = (operator.lt, None, operator.gt, None, None)
    rk_davg_metrics = (operator.lt, None, None, None, operator.gt)

    plot_frontier(os.path.join(plot_dir, 'rk_lorenz'),
                  [data.sweep_rk_lorenz, data.sweep_rk_lorenz_p, data.baseline_rk_lorenz, data.baseline_rk_lorenz_p],
                  [[rk_avg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
                  ref_pts=label_fenceposts(data.baseline_rk_lorenz_fenceposts, rk_avg_metrics),
                  ref_lines=[lorenz_avg_ceiling])

    plot_frontier(os.path.join(plot_dir, 'rk_lorenz_d'),
                  [data.sweep_rk_lorenz, data.sweep_rk_lorenz_p, data.baseline_rk_lorenz, data.baseline_rk_lorenz_p],
                  [[rk_davg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
                  ref_pts=label_fenceposts(data.baseline_rk_lorenz_fenceposts, rk_davg_metrics),
                  ref_lines=[lorenz_davg_ceiling])

    plot_frontier(os.path.join(plot_dir, 'rk_rossler'),
                  [data.sweep_rk_rossler, data.sweep_rk_rossler_p, data.baseline_rk_rossler, data.baseline_rk_rossler_p],
                  [[rk_avg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
                  ref_pts=label_fenceposts(data.baseline_rk_rossler_fenceposts, rk_avg_metrics),
                  ref_lines=[rossler_avg_ceiling])

    plot_frontier(os.path.join(plot_dir, 'rk_rossler_d'),
                  [data.sweep_rk_rossler, data.sweep_rk_rossler_p, data.baseline_rk_rossler, data.baseline_rk_rossler_p],
                  [[rk_davg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
                  ref_pts=label_fenceposts(data.baseline_rk_rossler_fenceposts, rk_davg_metrics),
                  ref_lines=[rossler_davg_ceiling])

    plot_frontier(os.path.join(plot_dir, 'rk_chua'),
                  [data.sweep_rk_chua, data.sweep_rk_chua_p, data.baseline_rk_chua, data.baseline_rk_chua_p],
                  [[rk_avg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
                  ref_pts=label_fenceposts(data.baseline_rk_chua_fenceposts, rk_avg_metrics),
                  ref_lines=[chua_avg_ceiling])

    plot_frontier(os.path.join(plot_dir, 'rk_chua_d'),
                  [data.sweep_rk_chua, data.sweep_rk_chua_p, data.baseline_rk_chua, data.baseline_rk_chua_p],
                  [[rk_davg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
                  ref_pts=label_fenceposts(data.baseline_rk_chua_fenceposts, rk_davg_metrics),
                  ref_lines=[chua_davg_ceiling])


    blur_metrics = (operator.lt, operator.gt)

    plot_frontier(os.path.join(plot_dir, 'blur'),
                  [data.sweep_blur, data.sweep_blur_p, data.baseline_blur, data.baseline_blur_p],
                  [[blur_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
                  ref_pts=label_fenceposts(data.baseline_blur_fenceposts, blur_metrics),
                  ref_lines=[1])
