"""Frontier and 3d plotter."""

import os
import json
import re
import operator
import traceback

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .utils import *
from . import search
#from . import ex_sqrt, ex_dotprod, ex_rk, ex_img

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


def plot_frontier(fname, sources, new_metrics, plot_settings = [], ref_pts = []):
    fig = plt.figure(figsize=(12,9), dpi=80)
    ax = fig.gca()

    try:
        for source, metric_group, plot_settings_group in zip(sources, new_metrics, plot_settings):
            frontier = source['frontier']

            for metrics, opts in zip(metric_group, plot_settings_group):
                filtered_frontier = search.filter_frontier(frontier, metrics)
                print(len(filtered_frontier))

                x, y = [], []
                for cfg, metrics in sorted(filtered_frontier, key = lambda t : t[1][0]):
                    a, b = metrics
                    x.append(a)
                    y.append(b)

                ax.plot(x, y, opts)

        for pt, label in ref_pts:
            px, py = pt
            ax.scatter([px], [py], marker='o', color='red')
            ax.annotate('  ' + label, pt)

    except Exception:
        traceback.print_exc()

    finally:
        if not fname.lower().endswith('.pdf'):
            fname += '.pdf'
        with PdfPages(fname) as pdf:
            pdf.savefig(fig)
        plt.close(fig)


def convert_total_abits_to_avg(source):
    frontier = source['frontier']
    new_frontier = [(a, (*b[:-1], b[-1] / 417.0)) for a, b in frontier]
    source['frontier'] = new_frontier


def all_plots():
    plot_dir = os.path.join(here, 'paper/figs')

    # return timeouts, infs, worst_bitcost, total_bitcost, worst_ulps, total_ulps, worst_abits, total_abits
    new_sqrt_metrics_worst = (None, None, None, operator.lt, None, None, operator.gt, None)
    new_sqrt_metrics_avg = (None, None, None, operator.lt, None, None, None, operator.gt)

    convert_total_abits_to_avg(data.sweep_newton_full)
    convert_total_abits_to_avg(data.sweep_newton_random)
    convert_total_abits_to_avg(data.baseline_newton)

    plot_frontier(os.path.join(plot_dir, 'sqrt_newton_avg'),
                  [data.sweep_newton_full, data.sweep_newton_random, data.baseline_newton],
                  [[new_sqrt_metrics_avg],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts = [])

    plot_frontier(os.path.join(plot_dir, 'sqrt_newton_worst'),
                  [data.sweep_newton_full, data.sweep_newton_random, data.baseline_newton],
                  [[new_sqrt_metrics_worst],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts=[])

    convert_total_abits_to_avg(data.sweep_babylonian_full)
    convert_total_abits_to_avg(data.sweep_babylonian_random)
    convert_total_abits_to_avg(data.baseline_babylonian)

    plot_frontier(os.path.join(plot_dir, 'sqrt_babylonian_avg'),
                  [data.sweep_babylonian_full, data.sweep_babylonian_random, data.baseline_babylonian],
                  [[new_sqrt_metrics_avg],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts=[])

    plot_frontier(os.path.join(plot_dir, 'sqrt_babylonian_worst'),
                  [data.sweep_babylonian_full, data.sweep_babylonian_random, data.baseline_babylonian],
                  [[new_sqrt_metrics_worst],] * 3,
                  plot_settings = [['C0o-'], ['C1+:'], ['ks--']],
                  ref_pts=[])

    # return quire_lo + quire_hi, infs, worst_ulps, avg_ulps, worst_abits, avg_abits
    dotprod_avg_metrics = (operator.lt, None, None, None, None, operator.gt)

    plot_frontier(os.path.join(plot_dir, 'dotprod_fused'),
                  [data.sweep_dotprod_fused, data.sweep_dotprod_fused_unsigned],
                  [[dotprod_avg_metrics],] * 2,
                  plot_settings = [['C0x--'], ['C1+:']],
                  ref_pts=[])

    plot_frontier(os.path.join(plot_dir, 'dotprod_bin'),
                  [data.sweep_dotprod_bin, data.sweep_dotprod_bin_unsigned],
                  [[dotprod_avg_metrics],] * 2,
                  plot_settings = [['C0x--'], ['C1+:']],
                  ref_pts=[])

    # return als.bits_requested, worst_abits_last, avg_abits_last, worst_abits_dlast, avg_abits_dlast
    rk_avg_metrics = (operator.lt, None, operator.gt, None, None)
    rk_davg_metrics = (operator.lt, None, None, None, operator.gt)

    plot_frontier(os.path.join(plot_dir, 'rk_lorenz'),
                  [data.sweep_rk_lorenz, data.sweep_rk_lorenz_p, data.baseline_rk_lorenz, data.baseline_rk_lorenz_p],
                  [[rk_avg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C0^:'], ['ks--'], ['k^:']],
                  ref_pts=[])

    plot_frontier(os.path.join(plot_dir, 'rk_lorenz_d'),
                  [data.sweep_rk_lorenz, data.sweep_rk_lorenz_p, data.baseline_rk_lorenz, data.baseline_rk_lorenz_p],
                  [[rk_davg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C0^:'], ['ks--'], ['k^:']],
                  ref_pts=[])

    plot_frontier(os.path.join(plot_dir, 'rk_rossler'),
                  [data.sweep_rk_rossler, data.sweep_rk_rossler_p, data.baseline_rk_rossler, data.baseline_rk_rossler_p],
                  [[rk_avg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C0^:'], ['ks--'], ['k^:']],
                  ref_pts=[])

    plot_frontier(os.path.join(plot_dir, 'rk_rossler_d'),
                  [data.sweep_rk_rossler, data.sweep_rk_rossler_p, data.baseline_rk_rossler, data.baseline_rk_rossler_p],
                  [[rk_davg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C0^:'], ['ks--'], ['k^:']],
                  ref_pts=[])

    plot_frontier(os.path.join(plot_dir, 'rk_chua'),
                  [data.sweep_rk_chua, data.sweep_rk_chua_p, data.baseline_rk_chua, data.baseline_rk_chua_p],
                  [[rk_avg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C0^:'], ['ks--'], ['k^:']],
                  ref_pts=[])

    plot_frontier(os.path.join(plot_dir, 'rk_chua_d'),
                  [data.sweep_rk_chua, data.sweep_rk_chua_p, data.baseline_rk_chua, data.baseline_rk_chua_p],
                  [[rk_davg_metrics],] * 4,
                  plot_settings = [['C0s--'], ['C0^:'], ['ks--'], ['k^:']],
                  ref_pts=[])


    plot_frontier(os.path.join(plot_dir, 'blur'),
                  [data.sweep_blur, data.sweep_blur_p, data.baseline_blur, data.baseline_blur_p],
                  [[(operator.lt, operator.gt)],] * 4,
                  plot_settings = [['C0s--'], ['C0^:'], ['ks--'], ['k^:']],
                  ref_pts=[((1000000,0.3),'foobar')])
