"""Vizualization support for Pareto frontiers."""

import os
import math
import re

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .utils import *
from . import search


class SearchData(object):
    """Data loader for stored search data."""

    checkpoint_name = 'latest.json'
    final_name = 'final.json'
    snapshot_name = 'frontier.json'
    checkpoint_dirname = 'checkpoints'
    checkpoint_re = re.compile('gen([0-9]+)' + re.escape('.json'))
    checkpoint_key = lambda s: int(checkpoint_re.fullmatch(s).group(1))

    def __init__(self, search_dir):
        self.search_dir = search_dir
        self.checkpoint_dir = os.path.join(self.search_dir, self.checkpoint_dirname)
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

    def latest_checkpoint(self):
        """get the generation number of the most recently saved full checkpoint"""
        final = os.path.exists(self.final_path)
        if not os.path.exists(self.checkpoint_dir):
            return None, final

        latest = None
        for name in os.listdir(self.checkpoint_dir):
            m = self.checkpoint_re.fullmatch(name)
            if m:
                cp_num = m.group(1)
                if latest is None or cp_num > latest:
                    latest = cp_num
        return latest, final

    def refresh(self, full=False):
        """Reread the checkpoint files from disk.
        If full is False, only read the frontier, not the other checkpoint data.
        """
        if full:
            self._load_files()
        else:
            # only reload the frontier
            if os.path.exists(self.snapshot_path):
                data = self._load_json(self.snapshot_path)
                if 'frontier' in data:
                    print('refreshed frontier')
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

class LivePlot(object):
    """Persistent interactive plot."""

    def __init__(self, source, metric_fns, xidx, yidx,
                 prefilter = None,
                 draw_ghosts = True,
                 opts = 'C0s--',
                 refresh = None):

        # settings
        self.source = source
        self.metric_fns = metric_fns
        self.xidx = xidx
        self.yidx = yidx
        self.prefilter = prefilter
        self.draw_ghosts = draw_ghosts
        self.opts = opts
        if refresh is True:
            self.refresh = 1000
        else:
            self.refresh = refresh
        # hard-coded
        self.new_opts = 'go'
        self.old_opts = 'ko'
        self.old_alpha = 0.1
        self.ghost_opts = opts.rstrip('--').rstrip('-.').rstrip('-').rstrip(':')
        self.ghost_alpha = 0.4

        # plot stuff
        self.fig = plt.figure(figsize=(12,8), dpi=80)
        self.ax = self.fig.gca()
        self.anim = None

        # record keeping for interactive display
        self.known_cfgs = {}
        self.new_line_record = None
        self.frontier_line_record = None
        self.ghost_line_record = None
        self.old_line_record = None

        latest_gen, final = self.source.latest_checkpoint()
        self.latest_gen = latest_gen

        # launch interaction
        if self.source.frontier:
            self.redraw(self.source.frontier)
            self.setup_annotation()

            if self.refresh:
                self.setup_refresh(self.refresh)


    def active_lines(self):
        if self.new_line_record is not None:
            yield self.new_line_record
        if self.frontier_line_record is not None:
            yield self.frontier_line_record
        if self.ghost_line_record is not None:
            yield self.ghost_line_record
        if self.old_line_record is not None:
            yield self.old_line_record

    def redraw(self, frontier):
        """Clear the plot and draw a new frontier"""

        if self.prefilter is not None:
            frontier = filter_frontier(frontier, prefilter)

        known_cfgs = {}
        for rec in frontier:
            cfg, qos = rec
            known_cfgs[cfg] = rec

        current_frontier, ghosts = reconstruct_frontier(frontier, self.metric_fns, check=False)
        frontier_line_data = self.align(current_frontier)
        if self.draw_ghosts:
            ghost_line_data = self.align(ghosts)

        # clear old info
        self.ax.clear()
        self.new_line_record = None
        self.frontier_line_record = None
        self.ghost_line_record = None
        self.old_line_record = None

        # set new info
        self.known_cfgs = known_cfgs

        line, = self.ax.plot([], [], self.new_opts)
        self.new_line_record = ([], [], [], line)

        line, = self.ax.plot([], [], self.old_opts, alpha=self.old_alpha)
        self.old_line_record = ([], [], [], line)

        xs, ys, cfgs = frontier_line_data
        line, = self.ax.plot(xs, ys, self.opts, ds='steps-post')
        self.frontier_line_record = (xs, ys, cfgs, line)

        if self.draw_ghosts:
            xs, ys, cfgs = ghost_line_data
            line, = self.ax.plot(xs, ys, self.ghost_opts, alpha=self.ghost_alpha)
            self.ghost_line_record = (xs, ys, cfgs, line)
        else:
            line, = self.ax.plot([], [], self.ghost_opts, alpha=self.ghost_alpha)
            self.ghost_line_record = ([], [], [], line)

    def align(self, frontier):
        """Sort a frontier into a list of x and y coordinates, and a list of cfgs"""
        xidx, yidx = self.xidx, self.yidx
        sortkey = lambda t: t[1][xidx]

        xs, ys, cfgs = [], [], []
        for cfg, qos in sorted(frontier, key=sortkey):
            xs.append(qos[xidx])
            ys.append(qos[yidx])
            cfgs.append(cfg)
        return xs, ys, cfgs

    def setup_annotation(self):
        """Start the interactive label on hover."""

        annot = self.ax.annotate('', xy=(0,0), xytext=(-20, 20), textcoords='offset points',
                                 bbox=dict(boxstyle='round', fc='w'),
                                 arrowprops=dict(arrowstyle='->'))
        annot.set_visible(False)
        annot.get_bbox_patch().set_alpha(0.9)
        annot.set_zorder(1000)

        def update_annot(event):
            contained = False
            for xs, ys, cfgs, line in self.active_lines():
                contained, details = line.contains(event)
                if contained:
                    event_pt = event.x, event.y
                    nearby_idxs = details['ind']
                    idx = nearby_idxs[0]
                    pt = xs[idx], ys[idx]
                    nearest = dist(event_pt, pt)
                    for i in nearby_idxs[1:]:
                        pt = xs[i], ys[i]
                        distance = dist(event_pt, pt)
                        if distance < nearest:
                            nearest = distance
                            idx = i
                    cfg = cfgs[idx]
                    break

            if not contained:
                annot.set_visible(False)
                return

            cfg, qos = self.known_cfgs[cfg]
            annot.xy = (qos[self.xidx], qos[self.yidx])
            annot.set_text(displaycfg(cfg, qos))
            annot.set_visible(True)

        def hover(event):
            if event.inaxes == self.ax:
                update_annot(event)
                self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect('motion_notify_event', hover)

    def setup_refresh(self, interval_ms):
        """Start the periodic refresh."""

        def update(i):
            full_update = False
            latest_gen, final = self.source.latest_checkpoint()
            if ((latest_gen is not None and self.latest_gen is None)
                or (latest_gen is not None and self.latest_gen is not None and latest_gen > self.latest_gen)):
                self.latest_gen = latest_gen
                full_update = True

                # new gen; update everything
                self.source.refresh(full=True)
                cp_frontier = self.source.state.frontier

                new_cfgs = {}
                for rec in cp_frontier:
                    cfg, qos = rec
                    if cfg not in self.known_cfgs:
                        new_cfgs[cfg] = rec

                current_frontier, ghosts = reconstruct_frontier(cp_frontier, self.metric_fns, check=False)
                frontier_line_data = self.align(current_frontier)
                if self.draw_ghosts:
                    ghost_line_data = self.align(ghosts)

                # different from the state frontier, might have been partial updates
                frontier = self.source.frontier

            else:
                # continuing previous gen in progress
                self.source.refresh(full=False)
                frontier = self.source.frontier
                new_cfgs = {}

            # if frontier is None:
            #     return []

            new_frontier = []
            for rec in frontier:
                cfg, qos = rec
                if cfg not in self.known_cfgs:
                    new_cfgs[cfg] = rec
                    new_frontier.append(rec)

            new_line_data = self.align(new_frontier)

            # update
            updated_artists = []
            self.known_cfgs.update(new_cfgs)

            xs, ys, cfgs, line = self.new_line_record
            added_xs, added_ys, added_cfgs = new_line_data
            xs += added_xs
            ys += added_ys
            cfgs += added_cfgs
            self.new_line_record, _ = (xs, ys, cfgs, line), line.set_data(xs, ys)
            updated_artists.append(line)

            if full_update:
                xs, ys, cfgs, line = self.frontier_line_record
                xs, ys, cfgs = frontier_line_data
                self.frontier_line_record, _ = (xs, ys, cfgs, line), line.set_data(xs, ys)
                updated_artists.append(line)

                if self.draw_ghosts:
                     xs, ys, cfgs, line = self.ghost_line_record
                     xs, ys, cfgs = ghost_line_data
                     self.ghost_line_record, _ = (xs, ys, cfgs, line), line.set_data(xs, ys)
                     updated_artists.append(line)

                # copy new line to old line, and clear new line
                new_xs, new_ys, new_cfgs, new_line = self.new_line_record
                old_xs, old_ys, old_cfgs, old_line = self.old_line_record
                old_xs += new_xs
                old_ys += new_ys
                old_cfgs += new_cfgs
                self.old_line_record, _ = (old_xs, old_ys, old_cfgs, old_line), old_line.set_data(old_xs, old_ys)
                self.new_line_record, _ = ([], [], [], new_line), new_line.set_data([], [])
                updated_artists.append(old_line)


            self.ax.relim()
            self.ax.autoscale_view()
            return updated_artists

        self.anim = animation.FuncAnimation(self.fig, update, interval=interval_ms)

        
class ComparePlot(object):
    """Compare frontiers across multiple data sources with same metric fns."""

    def __init__(self, sources, metric_fns, xidx, yidx,
                 prefilter = None,
                 draw_ghosts = True,
                 opts = 's--',
                 refresh = None):

        # settings
        self.sources = sources
        self.metric_fns = metric_fns
        self.xidx = xidx
        self.yidx = yidx
        self.prefilter = prefilter
        self.draw_ghosts = draw_ghosts
        self.opts = opts
        # hardcoded
        self.ghost_opts = opts.rstrip('--').rstrip('-.').rstrip('-').rstrip(':')
        self.ghost_alpha = 0.1

        # plot stuff
        self.fig = plt.figure(figsize=(12,8), dpi=80)
        self.ax = self.fig.gca()

        # go
        for i, source in enumerate(self.sources):
            self.draw(source.frontier, i)

    def align(self, frontier):
        """Sort a frontier into a list of x and y coordinates, and a list of cfgs"""
        xidx, yidx = self.xidx, self.yidx
        sortkey = lambda t: t[1][xidx]

        xs, ys, cfgs = [], [], []
        for cfg, qos in sorted(frontier, key=sortkey):
            xs.append(qos[xidx])
            ys.append(qos[yidx])
            cfgs.append(cfg)
        return xs, ys, cfgs
        
    def draw(self, frontier, color_idx):
        """Add a frontier to the plot"""

        opts = 'C' + str(color_idx) + self.opts
        ghost_opts = 'C' + str(color_idx) + self.ghost_opts
        
        if self.prefilter is not None:
            frontier = filter_frontier(frontier, prefilter)

        current_frontier, ghosts = reconstruct_frontier(frontier, self.metric_fns, check=False)
        
        xs, ys, cfgs = self.align(current_frontier)
        self.ax.plot(xs, ys, opts, ds='steps-post')
        
        if self.draw_ghosts:
            xs, ys, cfgs = self.align(ghosts)
            self.ax.plot(xs, ys, ghost_opts, alpha=self.ghost_alpha)

            
class XferPlot(object):
    """Compare metrics on a single frontier."""

    def __init__(self, source, metric_fns, xidx, yidx, xidx2,
                 prefilter = None,
                 draw_ghosts = True,
                 opts = 's--',
                 refresh = None):

        # settings
        self.source = source
        #self.metric_fns = metric_fns
        self.xidx = xidx
        self.yidx = yidx
        self.xidx2 = xidx2
        self.metric_fns = [f if i == xidx or i == yidx else None for i, f in enumerate(metric_fns)]
        self.metric_fns2 = [f if i == xidx2 or i == yidx else None for i, f in enumerate(metric_fns)]
        self.prefilter = prefilter
        self.draw_ghosts = draw_ghosts
        self.opts = opts
        # hardcoded
        self.ghost_opts = opts.rstrip('--').rstrip('-.').rstrip('-').rstrip(':')
        self.ghost_alpha = 0.1

        # plot stuff
        self.fig = plt.figure(figsize=(12,8), dpi=80)
        self.ax = self.fig.gca()
        self.ax2 = self.ax.twiny()

        # go
        self.draw(source.frontier)
        
    def align(self, frontier, xidx):
        """Sort a frontier into a list of x and y coordinates, and a list of cfgs"""
        yidx =  self.yidx
        sortkey = lambda t: t[1][xidx]

        xs, ys, cfgs = [], [], []
        for cfg, qos in sorted(frontier, key=sortkey):
            xs.append(qos[xidx])
            ys.append(qos[yidx])
            cfgs.append(cfg)
        return xs, ys, cfgs
        
    def draw(self, frontier):
        """Add a frontier to the plot"""

        color_idx = 0
        opts = 'C' + str(color_idx) + self.opts
        ghost_opts = 'C' + str(color_idx) + self.ghost_opts
        
        if self.prefilter is not None:
            frontier = filter_frontier(frontier, prefilter)

        current_frontier, ghosts = reconstruct_frontier(frontier, self.metric_fns, check=False)
        
        xs1, ys1, cfgs1 = self.align(current_frontier, self.xidx)
        self.ax.plot(xs1, ys1, opts, ds='steps-post')
        
        if self.draw_ghosts:
            xs, ys, cfgs = self.align(ghosts, self.xidx)
            self.ax.plot(xs, ys, ghost_opts, alpha=self.ghost_alpha)

        # the twin
        color_idx = 1
        opts = 'C' + str(color_idx) + self.opts
        ghost_opts = 'C' + str(color_idx) + self.ghost_opts
        
        # if self.prefilter is not None:
        #     frontier = filter_frontier(frontier, prefilter)

        current_frontier, ghosts = reconstruct_frontier(frontier, self.metric_fns2, check=False)
        
        xs2, ys2, cfgs2 = self.align(current_frontier, self.xidx2)
        self.ax2.plot(xs2, ys2, opts, ds='steps-post')
        
        if self.draw_ghosts:
            xs, ys, cfgs = self.align(ghosts, self.xidx2)
            self.ax2.plot(xs, ys, ghost_opts, alpha=self.ghost_alpha)

        # draw arrows
        x1_range = self.ax.get_xlim()
        x2_range = self.ax2.get_xlim()

        self.ax.set_xlim((0, x1_range[1]))
        self.ax2.set_xlim((0, x2_range[1]))
        
        # x1_range = self.ax.get_xlim()
        # x2_range = self.ax2.get_xlim()
        
        # ax1_size = x1_range[1] - x1_range[0]
        # ax2_size = x2_range[1] - x2_range[0]
        # scale = ax1_size / ax2_size
        
        # for cfg, qos in frontier:
        #     if cfg in cfgs1 or cfg in cfgs2:
        #         x1 = cfg[self.xidx]
        #         x2 = cfg[self.xidx2]
        #         y = cfg[self.yidx]

        #         x2_scaled = x2 * scale

        #         self.ax.arrow(x1, y, x2_scaled - x1, 0)
