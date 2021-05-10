"""Vizualization support for Pareto frontiers."""

import os

import matplotlib.pyplot as plt

from .utils import *
from . import search


class SearchData(object):
    """Data loader for stored search data."""

    checkpoint_name = 'latest.json'
    final_name = 'final.json'
    snapshot_name = 'frontier.json'

    def __init__(self, search_dir):
        self.search_dir = search_dir
        self.checkpoint_path = os.path.join(self.search_dir, self.checkpoint_name)
        self.final_path = os.path.join(self.search_dir, self.final_name)
        self.snapshot_path = os.path.join(self.search_dir, self.snapshot_name)

        self.settings = None
        self.state = None
        self.frontier = None
        self.is_final = False
        self._load_files()

    def _load_json(self, fpath):
        with open(fpath, 'rt') as f:
            data = json.load(f)
        return data

    def _load_files(self):
        data = {}
        if os.path.exists(self.final_path):
            self.is_final = True
            data.update(self._load_json(self.final_path))
        else:
            self.is_final = False
            if os.path.exists(self.checkpoint_path):
                data.update(self._load_json(self.checkpoint_path))
            if os.path.exists(self.snapshot_path):
                data.update(self._load_json(self.snapshot_path))

        if 'settings' in data:
            self.settings = search.SearchSettings.from_dict(data['settings'])
        if 'state' in data:
            self.state = search.SearchState.from_dict(data['state'])
        if 'frontier' in data:
            self.frontier = [(tuple(a), tuple(b)) for a, b in data['frontier']]
        elif self.state is not None:
            self.frontier = self.state.frontier


    def refresh(self, full=False):
        if full:
            self._load_files()
        else:
            # only reload the frontier
            if os.path.exists(self.snapshot_path):
                data = self._load_json(self.snapshot_path)
                if 'frontier' in data:
                    self.frontier = [(tuple(a), tuple(b)) for a, b in data['frontier']]

    def __repr__(self):
         if self.settings is not None:
             settings_str = 'settings'
         else:
             settings_str = ''
         if self.state is not None:
             state_str = 'state'
         else:
             state_str = ''
         if self.frontier is not None:
             frontier_str = 'frontier'
         else:
             frontier_str = ''
         strs = [s for s in (settings_str, state_str, frontier_str) if s != '']
         if len(strs) == 0:
             descr = 'nothing'
         else:
             descr = ', '.join(strs)
         return f'<{type(self).__name__} object at {hex(id(self))} with {descr}>'

    def __str__(self):
        lines = []
        lines.append(f'{type(self).__name__}')
        if self.settings is not None:
            lines.append(str(self.settings))
        if self.state is not None:
            lines.append(str(self.state))
        if self.frontier is not None:
            lines.append(f'{len(self.frontier)} points in the frontier.')
        return '\n'.join(lines)


def displaycfg(cfg, qos):
    cfg_str = repr(cfg)
    qos_strs = ['{:.2f}'.format(x) if isinstance(x, float) else str(x) for x in qos]
    qos_str = ', '.join(qos_strs)
    return f'{cfg_str}\n->  {qos_str}'

def plot_frontier(source, metric_fns, xidx, yidx,
                  prefilter=None, draw_ghosts=True, opts='C0s--', interactive=True):

    ghost_alpha = 0.4

    fig = plt.figure(figsize=(12,8), dpi=80)
    ax = fig.gca()

    frontier = source.frontier
    cfgs = {}
    for rec in frontier:
        cfg, qos = rec
        cfgs[cfg] = rec

    if prefilter is not None:
        frontier = filter_frontier(frontier, prefilter)

    reduced_frontier, ghosts = reconstruct_frontier(frontier, metric_fns)
    sortkey = lambda t: t[1][xidx]

    x, y, plot_cfgs = [], [], []
    for cfg, qos in sorted(reduced_frontier, key=sortkey):
        x.append(qos[xidx])
        y.append(qos[yidx])
        plot_cfgs.append(cfg)

    plot_line = ax.plot(x, y, opts, ds='steps-post')[0]

    if draw_ghosts:
        x, y, ghost_cfgs = [], [], []
        for cfg, qos in sorted(ghosts, key=sortkey):
            x.append(qos[xidx])
            y.append(qos[yidx])
            ghost_cfgs.append(cfg)

        ghost_opts = opts.rstrip('--').rstrip('-.').rstrip('-').rstrip(':')
        ghost_line = ax.plot(x, y, ghost_opts, alpha=ghost_alpha)[0]
    else:
        ghost_line = None

    if interactive:

        xytext = (-20,20)
        annot = ax.annotate('', xy=(0,0), xytext=xytext, textcoords='offset points',
                            bbox=dict(boxstyle='round', fc='w'),
                            arrowprops=dict(arrowstyle='->'))
        annot.set_visible(False)
        annot.set_zorder(1000)

        def update_annot(event):
            cont, dat = plot_line.contains(event)
            if cont:
                ind = dat['ind'][0]
                cfg, qos = cfgs[plot_cfgs[ind]]
            elif ghost_line is not None:
                cont, dat = ghost_line.contains(event)
                if cont:
                    ind = dat['ind'][0]
                    cfg, qos = cfgs[ghost_cfgs[ind]]
            else:
                cont = False

            if not cont:
                annot.set_visible(False)
                return

            # # weird magic to keep the annotation in the figure
            # w, h = fig.get_size_inches() * fig.dpi
            # ws = (event.x > w/2.0) * -1 + (event.x <= w/2.0)
            # hs = (event.y > h/2.0) * -1 + (event.y <= h/2.0)
            # annot.xytext = (xytext[0]*ws, xytext[1]*hs)
            # # apparently this doesn't work the same way for xytext?
            # # it seems to work for xybox

            # update the annotation
            annot.xy = (qos[xidx], qos[yidx])
            annot.set_text(displaycfg(cfg, qos))
            annot.get_bbox_patch().set_alpha(0.9)
            annot.set_visible(True)

        def hover(event):
            if event.inaxes == ax:
                update_annot(event)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', hover)

    plt.show()
