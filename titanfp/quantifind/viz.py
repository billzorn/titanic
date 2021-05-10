"""Vizualization support for Pareto frontiers."""

import os

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
