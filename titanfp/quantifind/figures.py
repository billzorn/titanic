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

def nd_getter(*idxs):
    def nd_get(thing):
        tmp = thing
        for i in idxs:
            tmp = tmp[i]
        return tmp
    return nd_get

# color / marker / line
fmt_re = re.compile(r'(C\d+|[bgrcmykx])?([.,ov^<>sp*hH+xDd|_])?(--|-.|-|:)?')
def split_fmt_string(s):
    m = fmt_re.match(s)
    if m:
        return m.groups()
    else:
        return None

def convert_total_abits_to_avg(source):
    frontier = source['frontier']
    new_frontier = [(a, (*b[:-1], b[-1] / 417.0)) for a, b in frontier]
    source['frontier'] = new_frontier

    all_points = source['configs']
    new_points = [(gen, a, (*b[:-1], b[-1] / 417.0)) for gen, a, b in all_points]
    source['configs'] = new_points


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
# yes this is horrible
convert_total_abits_to_avg(data.sweep_newton_full)
convert_total_abits_to_avg(data.sweep_newton_random)
convert_total_abits_to_avg(data.baseline_newton)
convert_total_abits_to_avg(data.baseline_newton_fenceposts)
# do it for Babylonian too
convert_total_abits_to_avg(data.sweep_babylonian_full)
convert_total_abits_to_avg(data.sweep_babylonian_random)
convert_total_abits_to_avg(data.baseline_babylonian)
convert_total_abits_to_avg(data.baseline_babylonian_fenceposts)


def plot_density(fname, sources, metrics, plot_settings = [],
                 axis_titles = []):
    fig = plt.figure(figsize=(12,6), dpi=80)
    ax = fig.gca()

    print('creating density plot', fname)

    try:
        plot_count = 0
        for source, opts in zip(sources, plot_settings):
            all_points = source['configs']
            final_frontier = source['frontier']
            frontier = []
            plot_points = []
            gen_bounds = []

            current_gen = 0
            for i, point in enumerate(all_points):
                if len(point) == 2:
                    data, measures = point
                if len(point) == 3:
                    gen, data, measures = point
                    if gen > current_gen:
                        gen_bounds.append(i)
                        current_gen = gen
                    elif gen < current_gen:
                        print('regressed a generation???')
                        print(i, point)

                changed, frontier = search.update_frontier(frontier, (tuple(data), tuple(measures)), metrics)
                if changed or i == len(all_points) - 1:
                    points_from_final = 0
                    for current_point in frontier:
                        if current_point in final_frontier:
                            points_from_final += 1

                    plot_points.append((i, len(frontier), points_from_final))

            if plot_count == 1:
                zidx = 98
            elif plot_count == 2:
                zidx = 99
            else:
                zidx = 100 - plot_count

            x, y_size, y_final = zip(*plot_points)
            ax.plot(x, y_size, opts, fillstyle='none', zorder=zidx)
            ax.plot(x, y_final, opts, zorder=zidx)

            if gen_bounds:
                lw = 0.25
                if len(gen_bounds) >= 25:
                    lw = 0.1
                if len(gen_bounds) >= 80:
                    lw = 0.05

                opts_color, opts_marker, opts_line = split_fmt_string(opts)
                for bound in gen_bounds:
                    ax.axvline(bound, color=opts_color, linestyle=opts_line, linewidth=lw, zorder=zidx-50)

            print(f'  {len(plot_points)!s} points, {len(gen_bounds)!s} generations')

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



def integrate_frontier(frontier, x_left=None, x_right=None, y_floor=None, y_ceil=None,
                       x_idx=0, y_idx=1):
    x_first = None
    x_prev = None
    y_prev = None

    total_area = 0.0
    for cfg, meas in sorted(frontier, key=nd_getter(1, x_idx)):
        x, y = float(meas[x_idx]), float(meas[y_idx])
        if math.isfinite(x) and math.isfinite(y):
            if x_prev is None:
                assert x_left is None or x_left <= x
                x_first = x
                x_prev = x
                if y_floor is None:
                    y_floor = y
                y_prev = y
            else:
                assert x_prev <= x and y_prev <= y
                if x_right is None:
                    x_dist = x - x_prev
                else:
                    x_dist = min(x, x_right) - x_prev
                x_prev = x
                if y_ceil is None:
                    y_dist = y_prev - y_floor
                else:
                    y_dist = min(y_ceil, y_prev) - y_floor
                y_prev = y
                if y_dist > 0 and x_dist > 0:
                    # discount area under the floor
                    # (this can happen when we first find a point,
                    #  which happens to be worse than the best configuration
                    #  in the final frontier with the lowest cost we ever see.)
                    # also, discount area to the right of x_right,
                    # which can happen when a very expensive point is first discovered,
                    # which is subsequently improved upon even by a cheaper alternative.
                    total_area += x_dist * y_dist

                # if we've exceeded the ceiling, we actually want to stop early
                if y_ceil is not None and y >= y_ceil:
                    break

    # extend over to the right
    if x_right is not None and x_prev is not None and x_prev < x_right:
        x_dist = x_right - x_prev
        if y_ceil is None:
            y_dist = y_prev - y_floor
        else:
            y_dist = min(y_ceil, y_prev) - y_floor
        if y_dist > 0:
            # we already checked x_dist when deciding if we should run
            # this fixup code
            total_area += x_dist * y_dist

    return x_first, x_prev, y_floor, total_area

def plot_progress(fname, sources, new_metrics, ceiling=None,
                  plot_settings=[], axis_titles=[]):

    update_metrics = [m for m in new_metrics if m is not None]

    fig = plt.figure(figsize=(12,6), dpi=80)
    ax = fig.gca()

    x_idx, y_idx = 0, 1

    print('creating progress plot', fname)

    try:
        plot_count = 0
        for source, opts in zip(sources, plot_settings):
            all_points = source['configs']
            final_frontier = search.filter_frontier(source['frontier'], new_metrics)
            frontier = []
            plot_points = []
            plot_points_capped = []

            x_left, x_right, y_floor, ref_area = integrate_frontier(
                final_frontier, y_ceil=None, x_idx=x_idx, y_idx=y_idx,
            )

            last_ratio = 0

            if ceiling is not None:
                x_left_capped, x_right_capped, y_floor_capped, ref_area_capped = integrate_frontier(
                    final_frontier, y_ceil=ceiling, x_idx=x_idx, y_idx=y_idx,
                )

                last_ratio_capped = 0

            for i, point in enumerate(all_points):
                if len(point) == 2:
                    data, measures = point
                if len(point) == 3:
                    gen, data, measures = point

                filtered_measures = tuple(meas for meas, m in zip(measures, new_metrics) if m is not None)

                changed, frontier = search.update_frontier(frontier, (tuple(data), filtered_measures), update_metrics)
                if changed or i == len(all_points) - 1:
                    x_left_current, x_right_current, y_floor_current, current_area = integrate_frontier(
                        frontier, x_left=x_left, x_right=x_right, y_floor=y_floor, y_ceil=None, x_idx=x_idx, y_idx=y_idx,
                    )
                    # print(f'coverage: {current_area!s} / {ref_area!s}')
                    # print(f'  from {x_left_current!s} - {x_right_current!s} of {x_left!s} - {x_right!s}')

                    ratio = current_area / ref_area

                    if ratio < last_ratio:
                        print(f'  RATIO DECREASED!!! {last_ratio!s} -> {ratio!s}')

                    last_ratio = ratio
                    plot_points.append((i, ratio))

                    if ceiling is not None:
                        x_left_current, x_right_current, y_floor_current, current_area = integrate_frontier(
                            frontier, x_left=x_left_capped, x_right=x_right_capped, y_floor=y_floor_capped, y_ceil=ceiling, x_idx=x_idx, y_idx=y_idx,
                        )
                        # print(f' capped coverage: {current_area!s} / {ref_area!s}')
                        # print(f'  from {x_left_current!s} - {x_right_current!s} of {x_left!s} - {x_right!s}')

                        ratio = current_area / ref_area_capped

                        if ratio < last_ratio_capped:
                            print(f'  RATIO DECREASED!!! {last_ratio!s} -> {ratio!s}')

                        last_ratio_capped = ratio
                        plot_points_capped.append((i, ratio))

            if plot_count == 1:
                zidx = 96
            elif plot_count == 2:
                zidx = 98
            else:
                zidx = 100 - (2 * plot_count)

            x, y = zip(*plot_points)
            ax.plot(x, y, opts, ds='steps-post', zorder=zidx)
            print(f'  {len(plot_points)!s} points')

            if ceiling is not None:
                x, y = zip(*plot_points_capped)
                ax.plot(x, y, opts, ds='steps-post', fillstyle='none', zorder=zidx + 1)
                print(f'  {len(plot_points_capped)!s} points')

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


def plot_frontier(fname, sources, new_metrics, plot_settings = [],
                  extra_pts = [], ref_pts = [], ref_lines = [], axis_titles = [],
                  complete_frontier = True, draw_ghosts = True, flip_axes = False,
                  xlim=None, ylim=None):
    fig = plt.figure(figsize=(12,8), dpi=80)
    ax = fig.gca()

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)


    print('generating', fname)

    try:
        plot_count = 0
        for source, metric_group, plot_settings_group in zip(sources, new_metrics, plot_settings):
            plot_count += 1
            frontier = source['frontier']
            all_points = source['configs']

            print(f'{source["settings"]}--f: {len(frontier)!s}, all: {len(all_points)!s}')

            # print(end='  ')
            linend = '\n'

            for metrics, opts in zip(metric_group, plot_settings_group):
                filtered_frontier = search.filter_frontier(frontier, metrics)
                print('filtered:', len(filtered_frontier), end=linend)

                x, y = [], []
                for cfg, measures in sorted(filtered_frontier, key = lambda t : t[1][0]):
                    print('   ', cfg, measures)

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

                ax.plot(x, y, opts, ds='steps-post', zorder=zidx)

                if complete_frontier:
                    filtered_points = search.filter_metrics(frontier, metrics)
                    print('complete:', len(filtered_points), end=linend)
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
                    print('ghosts:', len(filtered_points), end=linend)
                    ghost_count = len(filtered_points)
                    alpha = min(100, math.sqrt(ghost_count)) / ghost_count
                    x, y = [], []
                    for point in sorted(filtered_points, key = lambda t : t[1][0]):
                        if len(point) == 2:
                            data, measures = point
                        if len(point) == 3:
                            gen, data, measures = point
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

        for pt, label in extra_pts:
            px, py = pt
            texts.append(plt.text(
                px, py, label, zorder=101,
            ))

        if texts:
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
        if draw_ghosts:
            if not fname.lower().endswith('.png'):
                fname += '.png'
            fig.savefig(fname, bbox_inches='tight')
        else:

            if not fname.lower().endswith('.pdf'):
                fname += '.pdf'
            with PdfPages(fname) as pdf:
                pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def label_fenceposts(sweep, new_metrics):
    points = sweep['frontier']
    filtered = search.filter_metrics(points, new_metrics)

    fenceposts = []
    for data, measures in filtered:
        (label,) = data
        fenceposts.append((measures, label))

    return fenceposts


# experiment metrics

sqrt_metrics = (operator.lt,) * 6 + (operator.gt,) * 2
rk_metrics = (operator.lt,) + (operator.gt,) * 4
img_metrics = (operator.lt, operator.gt)

# return timeouts, infs, worst_bitcost, total_bitcost, worst_ulps, total_ulps, worst_abits, total_abits
new_sqrt_metrics_worst = (None, None, None, operator.lt, None, None, operator.gt, None)
new_sqrt_metrics_avg = (None, None, None, operator.lt, None, None, None, operator.gt)
new_sqrt_metrics_both = (None, None, None, operator.lt, None, None, operator.gt, operator.gt)
new_sqrt_metrics_infs = (None, operator.lt, None, operator.lt, None, None, None, None)
new_sqrt_metrics_timeouts = (operator.lt, None, None, operator.lt, None, None, None, None)

# return als.bits_requested, worst_abits_last, avg_abits_last, worst_abits_dlast, avg_abits_dlast
rk_avg_metrics = (operator.lt, None, operator.gt, None, None)
rk_davg_metrics = (operator.lt, None, None, None, operator.gt)
rk_both_metrics = (operator.lt, None, operator.gt, None, operator.gt)

# reference results

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

# output location

plot_dir = os.path.join(here, 'paper/figs')
table_dir = os.path.join(here, 'paper/tables')

# Table labels

sqrt_labels = ['Exp', 'res', 'diff', 'scale',
               'bitcost', '(worst)', 'avg. acc']

rk_labels = ['Exp', 'fn', 'rk', 'k1', 'k2', 'k3', 'k4',
             'bitcost', 'acc. pos', 'acc. slope']

rk_baseline_labels = ['fn', 'rk', 'k1', 'k2', 'k3', 'k4',
             'bitcost', 'acc. pos', 'acc. slope']

blur_labels = ['Exp', 'overall', 'mask', 'accum', 'mul',
               'bitcost', 'ssim']

# extra plot labels

lorenz_pts_extra = [
    ((494760, 1.35), 'A'),
    ((569160, 4.69), 'B'),
    ((660280, 8.32), 'C'),

    ((379800, 3.26), 'D'),
    ((464520, 3.27), 'E'),
    ((680860, 7.96), 'F'),
]

rossler_pts_extra = [
    ((441742, 1.56), 'A'),
    ((539946, 5.93), 'B'),
    ((765244, 11.44), 'C'),

    ((366450, 2.28), 'D'),
    ((526735, 6.51), 'E'),
    ((768830, 11.10), 'F'),
]

chua_pts_extra = [
    ((705644, 4.59), 'A'),
    ((761631, 6.79), 'B'),
    ((918359, 9.46), 'C'),

    ((650696, 3.68), 'D'),
    ((733793, 7.25), 'E'),
    ((934971, 12.04), 'F'),
]

blur_pts_extra = [
        ((994604, 0.06), 'A'),
        ((1038784, 0.87), 'B'),
        ((1076176, 0.95), 'C'),
        ((1168632, 0.98), 'D'),
        ((1324988, 1.00), 'E'),

        ((488048, 0.27), 'F'),
        ((757320, 0.73), 'G'),
        ((1113568, 0.95), 'H'),
        ((1194140, 0.98), 'I'),
        ((1356236, 1.00), 'J'),

    ]

def all_plots(ghosts=False):
    plot_frontier(os.path.join(plot_dir, 'sqrt_newton_infs'),
                  [data.sweep_newton_full, data.sweep_newton_random, data.baseline_newton],
                  [[new_sqrt_metrics_infs],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts = label_fenceposts(data.baseline_newton_fenceposts, new_sqrt_metrics_infs),
                  flip_axes = True, draw_ghosts = ghosts,
                  axis_titles = ["Square root with Newton's method", "bitcost", "infinities (out of 417 test cases)"])

    plot_frontier(os.path.join(plot_dir, 'sqrt_newton_timeouts'),
                  [data.sweep_newton_full, data.sweep_newton_random, data.baseline_newton],
                  [[new_sqrt_metrics_timeouts],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts = label_fenceposts(data.baseline_newton_fenceposts, new_sqrt_metrics_timeouts),
                  flip_axes = True, draw_ghosts = ghosts,
                  axis_titles = ["Square root with Newton's method", "bitcost", "timeouts (out of 417 test cases)"])

    plot_frontier(os.path.join(plot_dir, 'sqrt_newton_avg'),
                  [data.sweep_newton_full, data.sweep_newton_random, data.baseline_newton],
                  [[new_sqrt_metrics_avg],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts = label_fenceposts(data.baseline_newton_fenceposts, new_sqrt_metrics_avg),
                  ref_lines=[sqrt_avg_ceiling],
                  draw_ghosts = ghosts,
                  axis_titles = ["Square root with Newton's method", "bitcost", "average bits of accuracy (out of 417 test cases)"])

    plot_frontier(os.path.join(plot_dir, 'sqrt_newton_worst'),
                  [data.sweep_newton_full, data.sweep_newton_random, data.baseline_newton],
                  [[new_sqrt_metrics_worst],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts=label_fenceposts(data.baseline_newton_fenceposts, new_sqrt_metrics_worst),
                  ref_lines=[sqrt_worst_ceiling],
                  draw_ghosts = ghosts,
                  axis_titles = ["Square root with Newton's method", "bitcost", "worst bits of accuracy (out of 417 test cases)"])

    plot_frontier(os.path.join(plot_dir, 'sqrt_babylonian_infs'),
                  [data.sweep_babylonian_full, data.sweep_babylonian_random, data.baseline_babylonian],
                  [[new_sqrt_metrics_infs],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts = label_fenceposts(data.baseline_babylonian_fenceposts, new_sqrt_metrics_infs),
                  flip_axes = True, draw_ghosts = ghosts,
                  axis_titles = ["Square root with Babylonian method", "bitcost", "infinities (out of 417 test cases)"])

    plot_frontier(os.path.join(plot_dir, 'sqrt_babylonian_timeouts'),
                  [data.sweep_babylonian_full, data.sweep_babylonian_random, data.baseline_babylonian],
                  [[new_sqrt_metrics_timeouts],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts = label_fenceposts(data.baseline_babylonian_fenceposts, new_sqrt_metrics_timeouts),
                  flip_axes = True, draw_ghosts = ghosts,
                  axis_titles = ["Square root with Babylonian method", "bitcost", "timeouts (out of 417 test cases)"])

    plot_frontier(os.path.join(plot_dir, 'sqrt_babylonian_avg'),
                  [data.sweep_babylonian_full, data.sweep_babylonian_random, data.baseline_babylonian],
                  [[new_sqrt_metrics_avg],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts=label_fenceposts(data.baseline_babylonian_fenceposts, new_sqrt_metrics_avg),
                  ref_lines=[sqrt_avg_ceiling],
                  draw_ghosts = ghosts,
                  axis_titles = ["Square root with Babylonian method", "bitcost", "average bits of accuracy (out of 417 test cases)"])

    plot_frontier(os.path.join(plot_dir, 'sqrt_babylonian_worst'),
                  [data.sweep_babylonian_full, data.sweep_babylonian_random, data.baseline_babylonian],
                  [[new_sqrt_metrics_worst],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts=label_fenceposts(data.baseline_babylonian_fenceposts, new_sqrt_metrics_worst),
                  ref_lines=[sqrt_worst_ceiling],
                  draw_ghosts = ghosts,
                  axis_titles = ["Square root with Babylonian method", "bitcost", "worst bits of accuracy (out of 417 test cases)"])

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

    plot_frontier(os.path.join(plot_dir, 'rk_lorenz'),
                  [data.sweep_rk_lorenz, data.sweep_rk_lorenz_p, data.baseline_rk_lorenz, data.baseline_rk_lorenz_p],
                  [[rk_avg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
                  extra_pts=lorenz_pts_extra,
                  ref_pts=label_fenceposts(data.baseline_rk_lorenz_fenceposts, rk_avg_metrics),
                  ref_lines=[lorenz_avg_ceiling],
                  draw_ghosts = ghosts,
                  axis_titles = ["Lorenz attractor, RK4", "bitcost", "bits of accuracy, final position"])

    plot_frontier(os.path.join(plot_dir, 'rk_lorenz_d'),
                  [data.sweep_rk_lorenz, data.sweep_rk_lorenz_p, data.baseline_rk_lorenz, data.baseline_rk_lorenz_p],
                  [[rk_davg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
                  ref_pts=label_fenceposts(data.baseline_rk_lorenz_fenceposts, rk_davg_metrics),
                  ref_lines=[lorenz_davg_ceiling],
                  draw_ghosts = ghosts,
                  axis_titles = ["Lorenz attractor, RK4", "bitcost", "bits of accuracy, final slope"])

    plot_frontier(os.path.join(plot_dir, 'rk_rossler'),
                  [data.sweep_rk_rossler, data.sweep_rk_rossler_p, data.baseline_rk_rossler, data.baseline_rk_rossler_p],
                  [[rk_avg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
                  extra_pts=rossler_pts_extra,
                  ref_pts=label_fenceposts(data.baseline_rk_rossler_fenceposts, rk_avg_metrics),
                  ref_lines=[rossler_avg_ceiling],
                  draw_ghosts = ghosts,
                  axis_titles = ["Rossler attractor, RK4", "bitcost", "bits of accuracy, final position"])

    plot_frontier(os.path.join(plot_dir, 'rk_rossler_d'),
                  [data.sweep_rk_rossler, data.sweep_rk_rossler_p, data.baseline_rk_rossler, data.baseline_rk_rossler_p],
                  [[rk_davg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
                  ref_pts=label_fenceposts(data.baseline_rk_rossler_fenceposts, rk_davg_metrics),
                  ref_lines=[rossler_davg_ceiling],
                  draw_ghosts = ghosts,
                  axis_titles = ["Rossler attractor, RK4", "bitcost", "bits of accuracy, final slope"])

    plot_frontier(os.path.join(plot_dir, 'rk_chua'),
                  [data.sweep_rk_chua, data.sweep_rk_chua_p, data.baseline_rk_chua, data.baseline_rk_chua_p],
                  [[rk_avg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
                  extra_pts=chua_pts_extra,
                  ref_pts=label_fenceposts(data.baseline_rk_chua_fenceposts, rk_avg_metrics),
                  ref_lines=[chua_avg_ceiling],
                  draw_ghosts = ghosts,
                  axis_titles = ["Chua attractor, RK4", "bitcost", "bits of accuracy, final position"])

    plot_frontier(os.path.join(plot_dir, 'rk_chua_d'),
                  [data.sweep_rk_chua, data.sweep_rk_chua_p, data.baseline_rk_chua, data.baseline_rk_chua_p],
                  [[rk_davg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
                  ref_pts=label_fenceposts(data.baseline_rk_chua_fenceposts, rk_davg_metrics),
                  ref_lines=[chua_davg_ceiling],
                  draw_ghosts = ghosts,
                  axis_titles = ["Chua attractor, RK4", "bitcost", "bits of accuracy, final slope"])

    plot_frontier(os.path.join(plot_dir, 'blur'),
                  [data.sweep_blur, data.sweep_blur_p, data.baseline_blur, data.baseline_blur_p],
                  [[img_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
                  extra_pts=blur_pts_extra,
                  ref_pts=label_fenceposts(data.baseline_blur_fenceposts, img_metrics),
                  ref_lines=[1],
                  draw_ghosts = ghosts,
                  axis_titles = ["3x3 mask blur", "bitcost", "structural similarity"])

def density_plots():
    plot_density(os.path.join(plot_dir, 'sqrt_newton_density'),
                 [data.sweep_newton_full, data.sweep_newton_random],
                 sqrt_metrics,
                 ['C0o-', 'C1o-'],
                 axis_titles = ["Square root with Newton's method", "configurations searched", "frontier size"])

    plot_density(os.path.join(plot_dir, 'sqrt_babylonian_density'),
                 [data.sweep_babylonian_full, data.sweep_babylonian_random],
                 sqrt_metrics,
                 ['C0o-', 'C1o-'],
                 axis_titles = ["Square root with Babylonian method", "configurations searched", "frontier size"])

    plot_density(os.path.join(plot_dir, 'rk_lorenz_density'),
                 [data.sweep_rk_lorenz, data.sweep_rk_lorenz_p],
                 rk_metrics,
                 ['C0s--', 'C1^:'],
                 axis_titles = ["Lorenz attractor, RK4", "configurations searched", "frontier size"])

    plot_density(os.path.join(plot_dir, 'rk_rossler_density'),
                 [data.sweep_rk_rossler, data.sweep_rk_rossler_p],
                 rk_metrics,
                 ['C0s--', 'C1^:'],
                 axis_titles = ["Rossler attractor, RK4", "configurations searched", "frontier size"])

    plot_density(os.path.join(plot_dir, 'rk_chua_density'),
                 [data.sweep_rk_chua, data.sweep_rk_chua_p],
                 rk_metrics,
                 ['C0s--', 'C1^:'],
                 axis_titles = ["Chua attractor, RK4", "configurations searched", "frontier size"])

    plot_density(os.path.join(plot_dir, 'blur_density'),
                 [data.sweep_blur, data.sweep_blur_p],
                 img_metrics,
                 ['C0s--', 'C1^:'],
                 axis_titles = ["3x3 mask blur", "configurations searched", "frontier size"])

def progress_plots(use_ceiling=False):
    sqrt_metrics = (operator.lt,) * 6 + (operator.gt,) * 2
    rk_metrics = (operator.lt,) + (operator.gt,) * 4
    img_metrics = (operator.lt, operator.gt)

    if use_ceiling:
        suffix = '_progress'
    else:
        suffix = '_progress'

    plot_progress(os.path.join(plot_dir, 'sqrt_newton_avg' + suffix),
                  [data.sweep_newton_full, data.sweep_newton_random],
                  new_sqrt_metrics_avg,
                  ceiling = sqrt_avg_ceiling if use_ceiling else None,
                  plot_settings = ['C0o-', 'C1+:'],
                  axis_titles = ["Square root with Newton's method", "configurations searched", "frontier coverage (avg)"])

    plot_progress(os.path.join(plot_dir, 'sqrt_newton_worst' + suffix),
                  [data.sweep_newton_full, data.sweep_newton_random],
                  new_sqrt_metrics_worst,
                  ceiling = sqrt_worst_ceiling if use_ceiling else None,
                  plot_settings = ['C0o-', 'C1+:'],
                  axis_titles = ["Square root with Newton's method", "configurations searched", "frontier coverage (worst)"])

    plot_progress(os.path.join(plot_dir, 'sqrt_babylonian_avg' + suffix),
                  [data.sweep_babylonian_full, data.sweep_babylonian_random],
                  new_sqrt_metrics_avg,
                  ceiling = sqrt_avg_ceiling if use_ceiling else None,
                  plot_settings = ['C0o-', 'C1+:'],
                  axis_titles = ["Square root with Babylonian method", "configurations searched", "frontier coverage (avg)"])

    plot_progress(os.path.join(plot_dir, 'sqrt_babylonian_worst' + suffix),
                  [data.sweep_babylonian_full, data.sweep_babylonian_random],
                  new_sqrt_metrics_worst,
                  ceiling = sqrt_worst_ceiling if use_ceiling else None,
                  plot_settings = ['C0o-', 'C1+:'],
                  axis_titles = ["Square root with Babylonian method", "configurations searched", "frontier coverage (worst)"])

    plot_progress(os.path.join(plot_dir, 'rk_lorenz' + suffix),
                  [data.sweep_rk_lorenz, data.sweep_rk_lorenz_p],
                  rk_avg_metrics,
                  ceiling = lorenz_avg_ceiling if use_ceiling else None,
                  plot_settings = ['C0s--', 'C1^:'],
                  axis_titles = ["Lorenz attractor, RK4", "configurations searched", "frontier coverage (position)"])

    plot_progress(os.path.join(plot_dir, 'rk_lorenz_d' + suffix),
                  [data.sweep_rk_lorenz, data.sweep_rk_lorenz_p],
                  rk_davg_metrics,
                  ceiling = lorenz_davg_ceiling if use_ceiling else None,
                  plot_settings = ['C0s--', 'C1^:'],
                  axis_titles = ["Lorenz attractor, RK4", "configurations searched", "frontier coverage (slope)"])

    plot_progress(os.path.join(plot_dir, 'rk_rossler' + suffix),
                  [data.sweep_rk_rossler, data.sweep_rk_rossler_p],
                  rk_avg_metrics,
                  ceiling = rossler_avg_ceiling if use_ceiling else None,
                  plot_settings = ['C0s--', 'C1^:'],
                  axis_titles = ["Rossler attractor, RK4", "configurations searched", "frontier coverage (position)"])

    plot_progress(os.path.join(plot_dir, 'rk_rossler_d' + suffix),
                  [data.sweep_rk_rossler, data.sweep_rk_rossler_p],
                  rk_davg_metrics,
                  ceiling = rossler_davg_ceiling if use_ceiling else None,
                  plot_settings = ['C0s--', 'C1^:'],
                  axis_titles = ["Rossler attractor, RK4", "configurations searched", "frontier coverage (slope)"])

    plot_progress(os.path.join(plot_dir, 'rk_chua' + suffix),
                  [data.sweep_rk_chua, data.sweep_rk_chua_p],
                  rk_avg_metrics,
                  ceiling = chua_avg_ceiling if use_ceiling else None,
                  plot_settings = ['C0s--', 'C1^:'],
                  axis_titles = ["Chua attractor, RK4", "configurations searched", "frontier coverage (position)"])

    plot_progress(os.path.join(plot_dir, 'rk_chua_d' + suffix),
                  [data.sweep_rk_chua, data.sweep_rk_chua_p],
                  rk_davg_metrics,
                  ceiling = chua_davg_ceiling if use_ceiling else None,
                  plot_settings = ['C0s--', 'C1^:'],
                  axis_titles = ["Chua attractor, RK4", "configurations searched", "frontier coverage (slope)"])

    plot_progress(os.path.join(plot_dir, 'blur' + suffix),
                  [data.sweep_blur, data.sweep_blur_p],
                  img_metrics,
                  plot_settings = ['C0s--', 'C1^:'],
                  axis_titles = ["3x3 mask blur", "configurations searched", "frontier coverage"])


def format_table_value(v):
    try:
        v, ceil = v
    except Exception:
        v, ceil = v, None

    if isinstance(v, int):
        s = str(v)
    elif isinstance(v, float):
        s = f'{v:.2f}'
    elif isinstance(v, str):
        s = v
    else:
        s = repr(v)

    s = s.replace('_', r'\_')

    if ceil is not None and v >= ceil:
        s = '{\color{orange}' + s + '}'

    return s

def format_table_row(cfg, meas, ceils=None):
    row = ' & '.join(map(format_table_value, cfg))
    if ceils is None:
        row = ' & '.join([row, ' & '.join(map(format_table_value, meas))])
    else:
        row = ' & '.join([row, ' & '.join(map(format_table_value, zip(meas, ceils)))])
    return '  ' + row


def dump_tex_table(fname, source, labels=None, filter_metrics=None, display_metrics=None,
                   ceils=None, key=None, reverse=False):
    frontier = source['frontier']

    if filter_metrics is not None:
        frontier = search.filter_frontier(frontier, filter_metrics, allow_inf=False, reconstruct_metrics=True)

    if display_metrics is not None:
        frontier = search.filter_metrics(frontier, display_metrics, allow_inf=True)

    left_cols, right_cols = 0, 0
    rows = []
    for cfg, meas in sorted(frontier, key=key, reverse=reverse):
        left_cols, right_cols = len(cfg), len(meas)
        rows.append(format_table_row(cfg, meas, ceils))

    cols = '||'.join(['|'.join(['c'] * left_cols), '|'.join(['c'] * right_cols)])

    if labels:
        header = ' ' + ' & '.join(labels) + r' \\' + '\n'
        header += r' \hline' + '\n'
    else:
        header = ''

    table = (r'\begin{longtable}{' + cols + '}\n'
            + header
            + (r' \\' + '\n').join(rows) + '\n'
            + r'\end{longtable}')

    if not fname.lower().endswith('.tex'):
            fname += '.tex'

    with open(fname, 'wt') as f:
        print(table, file=f, flush=True)


def tables():
    dump_tex_table(os.path.join(table_dir, 'sqrt_newton_full'),
                   data.sweep_newton_full,
                   labels=sqrt_labels,
                   filter_metrics=new_sqrt_metrics_avg, display_metrics=new_sqrt_metrics_both,
                   ceils=[None, sqrt_worst_ceiling, sqrt_avg_ceiling],
                   key=nd_getter(1, 0))

    dump_tex_table(os.path.join(table_dir, 'sqrt_newton_random'),
                   data.sweep_newton_random,
                   labels=sqrt_labels,
                   filter_metrics=new_sqrt_metrics_avg, display_metrics=new_sqrt_metrics_both,
                   ceils=[None, sqrt_worst_ceiling, sqrt_avg_ceiling],
                   key=nd_getter(1, 0))

    dump_tex_table(os.path.join(table_dir, 'sqrt_babylonian_full'),
                   data.sweep_babylonian_full,
                   labels=sqrt_labels,
                   filter_metrics=new_sqrt_metrics_avg, display_metrics=new_sqrt_metrics_both,
                   ceils=[None, sqrt_worst_ceiling, sqrt_avg_ceiling],
                   key=nd_getter(1, 0))

    dump_tex_table(os.path.join(table_dir, 'sqrt_babylonian_random'),
                   data.sweep_babylonian_random,
                   labels=sqrt_labels,
                   filter_metrics=new_sqrt_metrics_avg, display_metrics=new_sqrt_metrics_both,
                   ceils=[None, sqrt_worst_ceiling, sqrt_avg_ceiling],
                   key=nd_getter(1, 0))


    dump_tex_table(os.path.join(table_dir, 'rk_lorenz'),
                   data.sweep_rk_lorenz,
                   labels=rk_labels,
                   filter_metrics=rk_avg_metrics, display_metrics=rk_both_metrics,
                   ceils=[None, lorenz_avg_ceiling, lorenz_davg_ceiling],
                   key=nd_getter(1, 0))

    dump_tex_table(os.path.join(table_dir, 'rk_rossler'),
                   data.sweep_rk_rossler,
                   labels=rk_labels,
                   filter_metrics=rk_avg_metrics, display_metrics=rk_both_metrics,
                   ceils=[None, rossler_avg_ceiling, rossler_davg_ceiling],
                   key=nd_getter(1, 0))

    dump_tex_table(os.path.join(table_dir, 'rk_chua'),
                   data.sweep_rk_chua,
                   labels=rk_labels,
                   filter_metrics=rk_avg_metrics, display_metrics=rk_both_metrics,
                   ceils=[None, chua_avg_ceiling, chua_davg_ceiling],
                   key=nd_getter(1, 0))

    dump_tex_table(os.path.join(table_dir, 'rk_lorenz_p'),
                   data.sweep_rk_lorenz_p,
                   labels=rk_labels,
                   filter_metrics=rk_avg_metrics, display_metrics=rk_both_metrics,
                   ceils=[None, lorenz_avg_ceiling, lorenz_davg_ceiling],
                   key=nd_getter(1, 0))

    dump_tex_table(os.path.join(table_dir, 'rk_rossler_p'),
                   data.sweep_rk_rossler_p,
                   labels=rk_labels,
                   filter_metrics=rk_avg_metrics, display_metrics=rk_both_metrics,
                   ceils=[None, rossler_avg_ceiling, rossler_davg_ceiling],
                   key=nd_getter(1, 0))

    dump_tex_table(os.path.join(table_dir, 'rk_chua_p'),
                   data.sweep_rk_chua_p,
                   labels=rk_labels,
                   filter_metrics=rk_avg_metrics, display_metrics=rk_both_metrics,
                   ceils=[None, chua_avg_ceiling, chua_davg_ceiling],
                   key=nd_getter(1, 0))


    dump_tex_table(os.path.join(table_dir, 'blur'),
                   data.sweep_blur,
                   labels=blur_labels,
                   key=nd_getter(1, 0))
    dump_tex_table(os.path.join(table_dir, 'blur_p'),
                   data.sweep_blur_p,
                   labels=blur_labels,
                   key=nd_getter(1, 0))


    dump_tex_table(os.path.join(table_dir, 'baseline_rk_lorenz'),
                   data.baseline_rk_lorenz,
                   labels=rk_baseline_labels,
                   filter_metrics=rk_avg_metrics, display_metrics=rk_both_metrics,
                   ceils=[None, lorenz_avg_ceiling, lorenz_davg_ceiling],
                   key=nd_getter(1, 0))

    dump_tex_table(os.path.join(table_dir, 'baseline_rk_lorenz_p'),
                   data.baseline_rk_lorenz_p,
                   labels=rk_baseline_labels,
                   filter_metrics=rk_avg_metrics, display_metrics=rk_both_metrics,
                   ceils=[None, lorenz_avg_ceiling, lorenz_davg_ceiling],
                   key=nd_getter(1, 0))


def all_figs():
    all_plots()
    all_plots(True)
    density_plots()
    progress_plots()
    progress_plots(True)

    tables()





def test():
    # ghosts = False

    # plot_frontier(os.path.join(plot_dir, 'rk_lorenz'),
    #           [data.sweep_rk_lorenz, data.sweep_rk_lorenz_p, data.baseline_rk_lorenz, data.baseline_rk_lorenz_p],
    #           [[rk_avg_metrics],] * 4,
    #           plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
    #           extra_pts=lorenz_pts_extra,
    #           ref_pts=label_fenceposts(data.baseline_rk_lorenz_fenceposts, rk_avg_metrics),
    #           ref_lines=[lorenz_avg_ceiling],
    #           draw_ghosts = ghosts,
    #           axis_titles = ["Lorenz attractor, RK4", "bitcost", "bits of accuracy, final position"])

    ghosts = False

    blur_pts_extra = [
        #((994604, 0.06), 'A'),
        ((1038784, 0.87), 'B'),
        ((1076176, 0.95), 'C'),
        ((1168632, 0.98), 'D'),
        ((1324988, 1.00), 'E'),

        #((488048, 0.27), 'F'),
        #((757320, 0.73), 'G'),
        ((1113568, 0.95), 'H'),
        ((1194140, 0.98), 'I'),
        ((1356236, 1.00), 'J'),

    ]

    plot_frontier(os.path.join(plot_dir, 'blur_extra'),
                  [data.sweep_blur, data.sweep_blur_p, data.baseline_blur, data.baseline_blur_p],
                  [[img_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C1^:'], ['ks--'], ['k^:']],
                  extra_pts=blur_pts_extra,
                  #ref_pts=label_fenceposts(data.baseline_blur_fenceposts, img_metrics),
                  ref_lines=[1],
                  draw_ghosts = ghosts,
                  axis_titles = ["3x3 mask blur", "bitcost", "structural similarity"],
                  xlim=(1000000,2000000),
                  ylim=(0.85,1.025))
