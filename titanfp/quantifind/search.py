"""Common code for parameter search."""

import os
import time
import itertools
import collections
import operator
import multiprocessing
import threading
import random
import math
import re

from .utils import *


# general utilities

# this bfs cartesian product is adapted from here:
# https://stackoverflow.com/questions/42288203/generate-itertools-product-in-different-order
# but the partitioning scheme is rewritten to be at least ~50x faster
def _partition_rec(n, max_position, total, idx, prefix):
    """Recursive helper for the breadth first partitioning scheme.
    Takes n and a total weight for the rest of the max_position list,
    as well as the current idx to work on,
    and the prefix of the configuration assembled so far.

    Yields all continuations from that prefix;
    the yield itself is in tail position.

    Assumes the rest of the list has at least two positions to work on.
    """
    this_pos = max_position[idx]
    total_rest = total - this_pos
    if idx == len(max_position) - 2:
        # base case, unrolled once
        for i in range(max(n - total_rest, 0), min(n, this_pos) + 1):
            yield prefix + (i, n-i)
    else:
        # recursive case
        for i in range(max(n - total_rest, 0), min(n, this_pos) + 1):
            yield from _partition_rec(n-i, max_position, total_rest, idx+1, prefix+(i,))

def breadth_first_partitions(n, max_position):
    """Split n total weight (will be used to compute index values)
    over k different buckets, where k is the len(max_position),
    such that no bucket i has weight > max_position[i].

    Yield every possible partitioning.
    """

    # to explain the idiom:
    #   range(max(n - total_rest, 0), min(n, this_pos) + 1)
    #
    # there are only so many ways to take things.
    #
    # say we will take i things for a bucket with size this_pos.
    # there will be total-this_pos things to take in the rest of the array (total_rest)
    # and we will need to take n - i of them (n_rest)
    # this is only possible if 0 <= n_rest <= total_rest
    #
    # for that to hold, we have 0 <= n - i <= total - this_pos
    # -->  -n <= -i <= total - this_pos - n
    # -->  n >= i >= n + this_pos - total
    #
    # so i is at least (n - total_rest), and at most n
    # at the low end, we can't take negative things, so it is bounded by 0,
    # and at the high end, it is also limited by our bound this_pos on the current value
    #
    # that wasn't so hard now was it

    if len(max_position) == 1:
        # true base case
        if 0 <= n <= max_position[0]:
            yield (n,)
    elif len(max_position) == 2:
        # base case, unrolled once
        this_pos, total_rest = max_position
        for i in range(max(n - total_rest, 0), min(n, this_pos) + 1):
            yield (i, n-i)
    elif len(max_position) >= 3:
        # recursive case, unrolled once
        total = sum(max_position)
        this_pos = max_position[0]
        total_rest = total - this_pos
        for i in range(max(n - total_rest, 0), min(n, this_pos) + 1):
            yield from _partition_rec(n-i, max_position, total_rest, 1, (i,))

def breadth_first_product(*sequences):
    """Breadth First Search Cartesian Product"""
    sequences = [list(seq) for seq in sequences]
    max_position = [len(i)-1 for i in sequences]
    for i in range(sum(max_position) + 1):
        for positions in breadth_first_partitions(i, max_position):
            yield tuple(map(operator.getitem, sequences, positions))

def reorder_for_bfs(seq, elt):
    """Reorder a list to contain the closest things to some starting point first.

    The starting point must be an element of the list.
    If it is the first item, return the list as-is.
    If it is the last item, reverse the list.
    Otherwise, return the representative first, then the item after it,
    then the one before it, then the next one after it, and so on.

    Returns an iterator for the list (or reversed list, or the interleaved reordering)
    """
    seq = list(seq)
    idx = seq.index(elt)
    if idx == 0:
        yield from iter(seq)
    elif idx == len(seq) - 1:
        yield from reversed(seq)
    else:
        yield seq[idx]
        for i in range(1, len(seq)):
            upper = idx + i
            if upper < len(seq):
                yield seq[upper]
            lower = idx - i
            if lower >= 0:
                yield seq[lower]

def center_ranges(input_ranges):
    """Given a list of sequences, create a list of new sequences
    which are all the same length, by repeating the element
    on the ends of any sequences shorter than the max length.
    If any of the sequences is empty, they must all be empty
    or a ValueError is raised (since there is nothing to pad with).
    """
    ranges = [tuple(r) for r in input_ranges]
    maxlen = 0
    empty = False
    for rng in ranges:
        if len(rng) > maxlen:
            maxlen = len(rng)
        elif len(rng) == 0:
            empty = True
    if empty and maxlen > 0:
        raise ValueError(f'cannot center empty range: {repr(ranges)}')
    for i in range(len(ranges)):
        rng = ranges[i]
        if len(rng) < maxlen:
            pad_count = maxlen - len(rng)
            left_pad = right_pad = pad_count // 2
            if left_pad + right_pad < pad_count:
                right_pad += 1
            ranges[i] = ((rng[0],) * left_pad) + rng + ((rng[-1],) * right_pad)
    return ranges

def nearby_points(cfg, neighbor_fns,
                  bfs_neighbors=False, combine=False, product=False, randomize=False):
    """Generator function to yield "nearby" configurations to cfg.
    Each neighbor generator in neighbor_fns will be called individually,
    and a new configuration yielded that replaces that element of cfg
    with each possible neighbor.

    If bfs_neighbors is True, then do this in BFS order
    (first change the first parameter by one step, then the next one by one step, etc.)

    If combine is True, also yield "combined" neighbor configurations
    that call all of the neighbor generators at once.

    If product is True, instead yield the full cartesian product
    of every possible neighbor point.
    Neighbors are iterated in believable way,
    so that consuming part of the generator makes sense
    in order to reach for more nearby points to explore.
    """
    if product:
        # pick an order to iterate
        if randomize:
            all_neighbors = [list(f(x)) for x, f in zip(cfg, neighbor_fns)]
            for neighbors in all_neighbors:
                random.shuffle(neighbors)
        else:
            all_neighbors = [reorder_for_bfs(f(x), x) for x, f in zip(cfg, neighbor_fns)]

        # and go
        yield from breadth_first_product(*all_neighbors)
    else:
        if bfs_neighbors:
            # this horrible code visits the "axial hypercube" neighbors in bfs order
            stopiter_sentinel = object()
            gens = [(i, reorder_for_bfs(f(x), x))
                    for i, (x, f) in enumerate(zip(cfg, neighbor_fns))]
            new_gens = []
            if randomize:
                while gens:
                    new_gens.clear()
                    random.shuffle(gens)
                    for i, gen in gens:
                        elt = next(gen, stopiter_sentinel)
                        if elt is not stopiter_sentinel:
                            new_gens.append((i, gen))
                            if elt != cfg[i]:
                                nearby_cfg = list(cfg)
                                nearby_cfg[i] = elt
                                yield tuple(nearby_cfg)
                    gens, new_gens = new_gens, gens
            else:
                while gens:
                    new_gens.clear()
                    for i, gen in gens:
                        elt = next(gen, stopiter_sentinel)
                        if elt is not stopiter_sentinel:
                            new_gens.append((i, gen))
                            if elt != cfg[i]:
                                nearby_cfg = list(cfg)
                                nearby_cfg[i] = elt
                                yield tuple(nearby_cfg)
                    gens, new_gens = new_gens, gens
        else:
            # equivalent for the exhaustive case, but not in bfs order
            for i, (x, f) in enumerate(zip(cfg, neighbor_fns)):
                for nearby_x in f(x):
                    if nearby_x != x:
                        nearby_cfg = list(cfg)
                        nearby_cfg[i] = nearby_x
                        yield tuple(nearby_cfg)
        if combine:
            all_neighbors = [f(x) for x, f in zip(cfg, neighbor_fns)]
            # to explain the opaque zip(*) logic below:
            # We start with a list of "nearby" parameters, for each variable in the cfg:
            # [[1,2,3], [7], [5,6]]
            # center_ranges pads this out so each list is the same length:
            # [[1,2,3], [7,7,7], [5,6,6]]
            # and then the zip(*) idiom re-slices this list of possible parameters into a list of configurations:
            # [[1,7,5], [2,7,6], [3,7,6]]
            # The zip generator yields new tuples, so we don't need to do anything else to package its outputs.
            yield from zip(*center_ranges(all_neighbors))


class SearchSettings(object):
    """Settings container for QuantiFind search."""

    initial_gen_size = 1

    restart_size_target = 0
    restart_gen_target = 0

    pop_random_weight = 0
    pop_mutant_weight = 0
    pop_crossed_weight = 0
    pop_local_weight = 1

    pop_random_target = None
    pop_mutant_target = None
    pop_crossed_target = None
    pop_local_target = None

    pop_weight_scale = 0

    mutation_probability = 0.5
    crossover_probability = 0.5

    def __init__(self, profile=None,
                 initial_gen_size=None,
                 restart_size_target = None,
                 restart_gen_target = None,
                 pop_weights = None,
                 pop_targets = None,
                 pop_weight_scale = None,
                 mutation_probability = None,
                 crossover_probability = None):
        # set defaults based on profile
        if profile is None or profile == 'local': # default
            self.initial_gen_size = 1
            self.restart_size_target = 0
            self.restart_gen_target = 0
            self.pop_random_weight = 0
            self.pop_mutant_weight = 0
            self.pop_crossed_weight = 0
            self.pop_local_weight = 1
            self.pop_random_target = None
            self.pop_mutant_target = None
            self.pop_crossed_target = None
            self.pop_local_target = None
            self.pop_weight_scale = 0
            self.mutation_probability = 0.5
            self.crossover_probability = 0.5
        elif profile == 'balanced':
            self.initial_gen_size = 1
            self.restart_size_target = 0
            self.restart_gen_target = 0
            self.pop_random_weight = 1
            self.pop_mutant_weight = 1
            self.pop_crossed_weight = 1
            self.pop_local_weight = 3
            self.pop_random_target = None
            self.pop_mutant_target = None
            self.pop_crossed_target = None
            self.pop_local_target = None
            self.pop_weight_scale = 0
            self.mutation_probability = 0.5
            self.crossover_probability = 0.5
        else:
            raise ValueError(f'unknown search profile {repr(profile)}')

        if initial_gen_size is not None:
            self.initial_gen_size = initial_gen_size
        if restart_size_target is not None:
            self.restart_size_target = restart_size_target
        if restart_gen_target is not None:
            self.restart_gen_target = restart_gen_target
        if pop_weights is not None:
            self.pop_random_weight, self.pop_mutant_weight, self.pop_crossed_weight, self.pop_local_weight = pop_weights
        if pop_targets is not None:
            self.pop_random_target, self.pop_mutant_target, self.pop_crossed_target, self.pop_local_target = pop_targets
        if mutation_probability is not None:
            self.pop_weight_scale = pop_weight_scale
        if mutation_probability is not None:
            self.mutation_probability = mutation_probability
        if crossover_probability is not None:
            self.crossover_probability = crossover_probability

    def __repr__(self):
        cls = type(self)
        fields = []
        if self.initial_gen_size != cls.initial_gen_size:
            fields.append(f'initial_gen_size={repr(self.initial_gen_size)}')
        if self.restart_size_target != cls.restart_size_target:
            fields.append(f'restart_size_target={repr(self.restart_size_target)}')
        if self.restart_gen_target != cls.restart_gen_target:
            fields.append(f'restart_gen_target={repr(self.restart_gen_target)}')
        if (self.pop_random_weight != cls.pop_random_weight or
            self.pop_mutant_weight != cls.pop_mutant_weight or
            self.pop_crossed_weight != cls.pop_crossed_weight or
            self.pop_local_weight != cls.pop_local_weight):
            fields.append(f'pop_weights=({repr(self.pop_random_weight)},'
                          f'{repr(self.pop_mutant_weight)},'
                          f'{repr(self.pop_crossed_weight)},'
                          f'{repr(self.pop_local_weight)})')
        if (self.pop_random_target != cls.pop_random_target or
            self.pop_mutant_target != cls.pop_mutant_target or
            self.pop_crossed_target != cls.pop_crossed_target or
            self.pop_local_target != cls.pop_local_target):
            fields.append(f'pop_targets=({repr(self.pop_random_target)},'
                          f'{repr(self.pop_mutant_target)},'
                          f'{repr(self.pop_crossed_target)},'
                          f'{repr(self.pop_local_target)})')
        if self.pop_weight_scale != cls.pop_weight_scale:
            fields.append(f'pop_weight_scale={repr(self.pop_weight_scale)}')
        if self.mutation_probability != cls.mutation_probability:
            fields.append(f'mutation_probability={repr(self.mutation_probability)}')
        if self.crossover_probability != cls.crossover_probability:
            fields.append(f'crossover_probability={repr(self.crossover_probability)}')
        sep = ', '
        return f'{cls.__name__}({sep.join(fields)})'

    def __str__(self):
        cls = type(self)
        fields = []
        if self.initial_gen_size != cls.initial_gen_size:
            fields.append(f'  initial_gen_size: {str(self.initial_gen_size)}')
        if self.restart_size_target != cls.restart_size_target:
            fields.append(f'  restart_size_target: {str(self.restart_size_target)}')
        if self.restart_gen_target != cls.restart_gen_target:
            fields.append(f'  restart_gen_target: {str(self.restart_gen_target)}')
        if (self.pop_random_weight != cls.pop_random_weight or
            self.pop_mutant_weight != cls.pop_mutant_weight or
            self.pop_crossed_weight != cls.pop_crossed_weight or
            self.pop_local_weight != cls.pop_local_weight):
            fields.append(f'  pop_weights:\n'
                          f'    random:  {str(self.pop_random_weight)}\n'
                          f'    mutant:  {str(self.pop_mutant_weight)}\n'
                          f'    crossed: {str(self.pop_crossed_weight)}\n'
                          f'    local:   {str(self.pop_local_weight)}')
        if (self.pop_random_target != cls.pop_random_target or
            self.pop_mutant_target != cls.pop_mutant_target or
            self.pop_crossed_target != cls.pop_crossed_target or
            self.pop_local_target != cls.pop_local_target):
            fields.append(f'  pop_targets:\n'
                          f'    random:  {str(self.pop_random_target)}\n'
                          f'    mutant:  {str(self.pop_mutant_target)}\n'
                          f'    crossed: {str(self.pop_crossed_target)}\n'
                          f'    local:   {str(self.pop_local_target)}')
        if self.pop_weight_scale != cls.pop_weight_scale:
            fields.append(f'  pop_weight_scale: {str(self.pop_weight_scale)}')
        if self.mutation_probability != cls.mutation_probability:
            fields.append(f'  mutation_probability: {str(self.mutation_probability)}')
        if self.crossover_probability != cls.crossover_probability:
            fields.append(f'  crossover_probability: {str(self.crossover_probability)}')
        sep = '\n'
        if len(fields) > 0:
            return f'{cls.__name__}:\n{sep.join(fields)}'
        else:
            return f'{cls.__name__}:'

    def to_dict(self):
        d = {}
        d.update(self.__dict__)
        return d

    @classmethod
    def from_dict(cls, d):
        new_settings = cls.__new__(cls)
        new_settings.__dict__.update(d)
        return new_settings


class SearchState(object):
    """State container for QuantiFind search."""

    def __init__(self):
        # The key search state breaks down into two main pools of configurations:
        #   the "frontier" is the pareto frontier of configurations we have explored so far,
        self.frontier = []
        #   and the "horizon" is the set of configurations we have decided to explore next.
        self.horizon = collections.deque()

        # We also keep around a list of every configuration we have ever run, in order,
        self.history = []
        # and an index to look them up by configuration parameters (partly for caching reasons):
        self.cache = {}

        # Finally, we can track the history of the frontier:
        # each time we add a point, we track it, as well as the set of points it replaced.
        self.frontier_log = []

        # Informal type information and invariants:
        #
        # self.frontier contains pairs of tuples (config_parameters, metric_values)
        # as does self.history.
        #
        # self.frontier_log is a list of tuples:
        #   [(config_parameters, metric_values), replaced, replaced_by]
        # where replaced is a list of configurations this one replaced in the frontier
        # (stored by log index in the cache),
        # and replaced_by is None if this point is still on the live frontier,
        # or the index of the point that replaced it if it isn't.
        #
        # self.horizon is a deque of configurations (just config_parameters) to explore next.
        # self.cache is a dict that maps each config_parameters to its current position:
        #   [history_idx, frontier_log_idx, hits, source_id]
        # if history_idx is None, then this configuration is still on the horizon (being run).
        # if frontier_log_idx is None, then this configuration never made it into the frontier.
        # hits is the number of times we found this configuration in local/random search,
        # and tried to add it to the horizon.
        # source_id is an integer indicating how we first came to add this configuration
        # to the horizon:
        #   0 - random
        #   1 - mutant
        #   2 - crossover
        #   3 - local search
        #   4 - targeted exhaustive search
        #
        # A single configuration (i.e. config_parameters) can be found:
        #   in self.cache once, if we've ever thought of running it
        #   in self.horizon, if we've planned to run it but haven't tried putting it into the frontier yet
        #   in self.history exactly once, if we have run it and tried putting it into the frontier
        #   in self.frontier_log up to once, if it made it into the frontier
        #
        # self.frontier is essentially a cache of the current frontier,
        # and is entirely reproducibly from self.frontier_log.

        # some other misc record keepting:

        # count of total points explored, and new frontier points found, for each generation
        self.generations = []

        # for the stopping criteria: the count of "initial" configs run and "initial" generations
        #   A generation is initial if it was created while the search was exhausted;
        #   i.e. the previous generation failed to add anything to the Pareto frontier.
        #   A configuration is initial if it was part of an initial generation.
        self.initial_cfgs = 0
        self.initial_gens = 0

        # additional data specific to this search, e.g. serializable test inputs
        self.additional_data = {}

    def __repr__(self):
        return f'<{type(self).__name__} object at {hex(id(self))} with {len(self.cache)} configurations>'

    def __str__(self):
        return (
            f'{type(self).__name__}:\n'
            f'  total cfgs:   {len(self.cache)}\n'
            f'  horizon:      {len(self.horizon)}\n'
            f'  history:      {len(self.history)}\n'
            f'  frontier log: {len(self.frontier_log)}\n'
            f'  frontier:     {len(self.frontier)}\n'
            f'  running for {len(self.generations)} generations'
        ) + (f'\n  {len(self.additional_data)} additional data records' if len(self.additional_data) > 0 else '')

    def to_dict(self):
        d = {
            'frontier': list(self.frontier),
            'horizon': list(self.horizon),
            'history': list(self.history),
            'cache': list(self.cache.items()),
            'frontier_log': list(self.frontier_log),
            'generations': list(self.generations),
            'initial_cfgs': self.initial_cfgs,
            'initial_gens': self.initial_gens,
            'additional_data': self.additional_data,
        }
        return d

    @classmethod
    def from_dict(cls, d):
        new_state = cls.__new__(cls)
        new_state.__dict__['frontier'] = [(tuple(a), tuple(b)) for a, b in d['frontier']]
        new_state.__dict__['horizon'] = [(tuple(a), tuple(b)) for a, b in d['horizon']]
        new_state.__dict__['history'] = [(tuple(a), tuple(b)) for a, b in d['history']]
        new_state.__dict__['cache'] = dict((tuple(k), list(v)) for k, v in d['cache'])
        new_state.__dict__['frontier_log'] = [[(tuple(a), tuple(b)), v1, v2] for (a, b), v1, v2 in d['frontier_log']]
        new_state.__dict__['generations'] = [tuple(a) for a in d['generations']]
        new_state.__dict__['initial_cfgs'] = d['initial_cfgs']
        new_state.__dict__['initial_gens'] = d['initial_gens']
        new_state.__dict__['additional_data'] = d['additional_data']
        return new_state

    def __getitem__(self, i):
        # try to look something up in the history
        if isinstance(i, int):
            # if it's an integer, assume they mean a history index
            return self.history[i]
        else:
            # otherwise, assume it's a tuple of config_parameters, and look up in the cache
            hidx, fidx, hits, reason = self.cache[i]
            if hidx is None:
                # it hasn't been run yet, so put in None for metric_values
                return (i, None)
            else:
                return self.history[hidx]

    def check(self, metric_fns=None, verbose=True):
        """Check the search state for consistency."""
        consistent = True

        if verbose:
            print('Checking horizon, history, and frontier log... ')

        deduped_horizon = {}
        for cfg in self.horizon:
            if cfg in deduped_horizon:
                if verbose:
                    print(f'-- CHECK SEARCHSTATE: duplicated {repr(cfg)} in horizon --')
                deduped_horizon[cfg][0] += 1
            else:
                deduped_horizon[cfg] = [1, False]

        good_history = [False] * len(self.history)
        deduped_history = {}
        for i, (cfg, qos) in enumerate(self.history):
            if verbose and cfg in deduped_history:
                print(f'-- CHECK SEARCHSTATE: duplicated {repr(cfg)} in history --')
            deduped_history[cfg] = i

        good_frontier_log = [False] * len(self.frontier_log)
        deduped_frontier_log = {}
        for i, ((cfg, qos), replaced, replaced_by) in enumerate(self.frontier_log):
            if verbose and cfg in deduped_frontier_log:
                print(f'-- CHECK SEARCHSTATE: duplicated {repr(cfg)} in frontier log --')
            deduped_frontier_log[cfg] = i

        if verbose:
            print(f'  ({len(deduped_horizon)} {len(deduped_history)} {len(deduped_frontier_log)})')
            print('Checking the cache... ')

        for cfg, record in self.cache.items():
            hidx, fidx, hits, source_id = record
            actual_hidx = deduped_history.get(cfg, None)
            actual_fidx = deduped_frontier_log.get(cfg, None)

            # it's either on the horizon somewhere
            if cfg in deduped_horizon:
                # hopefully hidx and fidx are None
                if verbose and (hidx is not None or cfg in deduped_history):
                    print(f'-- CHECK SEARCHSTATE: configuration {repr(cfg)} on the horizon '
                          f'reports hidx={hidx}, last seen at {actual_hidx} --')
                if verbose and (fidx is not None or cfg in deduped_frontier_log):
                    print(f'-- CHECK SEARCHSTATE: configuration {repr(cfg)} on the horizon '
                          f'reports fidx={fidx}, last seen at {actual_fidx} --')
            elif hidx is None:
                # or, if it isn't on the horizon, and it has no hidx, something is wrong
                consistent = False
                if verbose:
                    print(f'-- CHECK SEARCHSTATE: configuration {repr(cfg)} is not recorded in history or on the horizon --')
                    if cfg in deduped_history:
                        print(f'--    why was it seen at {actual_hidx} ? --')
                    if fidx is not None or cfg in deduped_frontier_log:
                        print(f'--    why does it report fidx={fidx}, last seen at {actual_fidx} ? --')

            # now check the consistency of the hidx and fidx, regardless of the horizon
            if hidx != actual_hidx:
                consistent = False
                if verbose:
                    print(f'-- CHECK SEARCHSTATE: configuration {repr(cfg)} '
                          f'reports hidx={hidx}, BUT was last seen at {actual_hidx} --')
            elif hidx is not None:
                if good_history[hidx] and verbose:
                    print(f'-- CHECK SEARCHSTATE: configuration {repr(cfg)} '
                          f'at hidx={hidx} has already been recorded --')
                good_history[hidx] = True

            if fidx != actual_fidx:
                consistent = False
                if verbose:
                    print(f'-- CHECK SEARCHSTATE: configuration {repr(cfg)} '
                          f'reports fidx={fidx}, BUT was last seen at {actual_fidx} --')
            elif fidx is not None:
                if good_frontier_log[fidx] and verbose:
                    print(f'-- CHECK SEARCHSTATE: configuration {repr(cfg)} '
                          f'at fidx={fidx} has already been recorded --')
                good_frontier_log[fidx] = True

        for i, good in enumerate(good_history):
            if not good:
                cfg, qos = self.history[i]
                if cfg not in self.cache:
                    consistent = False
                    if verbose:
                        print(f'-- CHECK SEARCHSTATE: uncached configuration {repr(cfg)} at hidx {i} --')
                elif verbose:
                    print(f'-- CHECK SEARCHSTATE: configuration {repr(cfg)} at hidx {i} is not linked from cache --')

        for i, good in enumerate(good_frontier_log):
            if not good:
                (cfg, qos), replaced, replaced_by = self.frontier_log[i]
                if cfg not in self.cache:
                    consistent = False
                    if verbose:
                        print(f'-- CHECK SEARCHSTATE: uncached configuration {repr(cfg)} at fidx {i} --')
                elif verbose:
                    print(f'-- CHECK SEARCHSTATE: configuration {repr(cfg)} at fidx {i} is not linked from cache --')

        if verbose:
            print(f'  ({len(self.cache)})')

        # check the frontier
        if metric_fns is not None:
            if verbose:
                print('Checking the frontier... ')

            if self.frontier:
                frontier, removed = reconstruct_frontier(self.frontier, metric_fns, check=True, verbose=verbose)
                if removed:
                    consistent = False
                    if verbose:
                        print('-- CHECK SEARCHSTATE: some configurations were removed while reconstructing the frontier --')
                        for thing in removed:
                            print('--> ', repr(thing))
            else:
                frontier, removed = reconstruct_frontier(self.history, metric_fns, check=False, verbose=verbose)

            frontier_cfgs = set(map(operator.itemgetter(0), frontier))

            # note that this isn't a true consistency check between the frontier, the history, and the frontier log...
            # we only check that the frontier agrees with the log,
            # and only rebuild from history if there is not provided frontier at all.

            for i, record in enumerate(self.frontier_log):
                (cfg, qos), replaced, replaced_by = record
                for replaced_idx in replaced:
                    _, _, replaced_by_me = self.frontier_log[replaced_idx]
                    if replaced_by_me != i:
                        if verbose:
                            print(f'-- CHECK SEARCHSTATE: fidx {i} replaced {replaced_idx}, which claims to have been replaced by {replaced_by_me} --')
                # only checking replaced -> replaced by, not the other way
                if replaced_by is None:
                    if cfg not in frontier_cfgs:
                        consistent = False
                        if verbose:
                            print(f'-- CHECK SEARCHSTATE: configuration {repr(cfg)} at fidx {i} is missing from the frontier --')

            if verbose:
                print(f'  ({len(frontier)})')

        if verbose:
            print('Checking generations...')

        total_cfgs = 0
        frontier_cfgs = 0
        initial_gens = 0
        initial_cfgs = 0

        is_initial_gen = True
        for horizon_size, new_frontier_points in self.generations:
            total_cfgs += horizon_size
            frontier_cfgs += new_frontier_points
            if is_initial_gen:
                initial_gens += 1
                initial_cfgs += horizon_size
            is_initial_gen = (new_frontier_points == 0)

        if total_cfgs != len(self.history):
            consistent = False
            if verbose:
                print(f'-- CHECK SEARCHSTATE: generation log reports {total_cfgs} configurations were run, '
                      f'but history only has {len(self.history)} --')
        # less important checks don't have to trip consistent
        if frontier_cfgs != len(self.frontier_log):
            if verbose:
                print(f'-- CHECK SEARCHSTATE: generation log reports {frontier_cfgs} configurations were added to the frontier, '
                      f'but frontier log only has {len(self.frontier_log)} --')
        if initial_gens != self.initial_gens:
            if verbose:
                print(f'-- CHECK SEARCHSTATE: generation log reports {initial_gens} initial generations, state has {self.initial_gens} --')
        if initial_cfgs != self.initial_cfgs:
            if verbose:
                print(f'-- CHECK SEARCHSTATE: generation log reports {initial_cfgs} total initial configurations, state has {self.initial_cfgs} --')

        if verbose:
            print(f'  ({len(self.generations)})')

        if verbose:
            if consistent:
                print('Done. State is consistent.')
            else:
                print('WARNING! State is inconsistent!')
        return consistent

    def poke_cache(self, cfg):
        """Poke the cache to see if this configuration has been seen.
        If cfg has been seen before, increment the hit count and return False.
        Else if cfg is new, return True.
        """
        if cfg in self.cache:
            self.cache[cfg][2] += 1
            return False
        else:
            return True

    def add_to_horizon(self, batch, reason, check=True, verbose=True):
        """Add batch configurations to the horizon, also recording them in the cache for reason.
        If check is True, first check if the configurations are already in the cache and skip them;
        Otherwise replace any existing configurations with new cache records.
        """
        if check:
            repeat_cfgs = 0
            for cfg in batch:
                if cfg in self.cache:
                    if verbose:
                        print(f'-- configuration {repr(cfg)} is already in the cache --')
                    self.cache[cfg][2] += 1
                    repeat_cfgs += 1
                else:
                    self.cache[cfg] = [None, None, 1, reason]
                    self.horizon.append(cfg)
            if repeat_cfgs > 0:
                print(f'WARNING: skipped {repeat_cfgs} configurations that were already in the cache')
            return repeat_cfgs
        else:
            repeat_cfgs = 0
            for cfg in batch:
                if cfg in self.cache:
                    hits = self.cache[cfg][2] + 1
                    repeat_cfgs += 1
                else:
                    hits = 1
                self.cache[cfg] = [None, None, hits, reason]
                self.horizon.append(cfg)
            return repeat_cfgs

    def get_from_horizon(self, n=None):
        """Get n configurations from the horizon to work on next;
        presumably this is a batch to process,
        and after retrieving them they will be committed.
        """
        if n is None:
            return list(self.horizon)
        else:
            # turn the islice generator into a list immediately,
            # so that we can't break it by mutating the deque
            return list(itertools.islice(self.horizon, n))

    def commit_to_history(self, result, metric_fns, verbose=True):
        """Commit an evaluated configuration to the history,
        updating the Pareto frontier in the process.
        Does not update the state's generation info.
        """
        cfg, qos = result
        if len(self.horizon) > 0:
            horizon_cfg = self.horizon.popleft()
        else:
            horizon_cfg = None
        if verbose and cfg != horizon_cfg:
            print(f'-- configuration {repr(cfg)} is not equal to {repr(horizon_cfg)} from the horizon --')

        if cfg in self.cache:
            record = self.cache[cfg]
        else:
            record = [None, None, 0, -1]
            self.cache[cfg] = record
            if verbose:
                print(f'-- configuration {repr(cfg)} was never seen in the cache; adding --')

        hidx = len(self.history)
        self.history.append(result)
        record[0] = hidx

        keep, new_frontier, removed = update_frontier(self.frontier, result, metric_fns)
        if keep:
            fidx = len(self.frontier_log)
            ridxs = []
            record[1] = fidx

            for removed_result in removed:
                removed_cfg, removed_qos = removed_result
                removed_record = self.cache.get(removed_cfg, None)
                if removed_record is not None:
                    ridx = removed_record[1]
                    ridxs.append(ridx)
                    self.frontier_log[ridx][2] = fidx
                elif verbose:
                    print(f'-- cache is missing {repr(removed_result)} which was removed from the frontier --')

            self.frontier_log.append([result, ridxs, None])
        self.frontier = new_frontier
        return keep


# new search algo, inspired by "GOFAI" and specifically genetic algorithms
# this general idea was brought to me by Max Willsey

# we have roughly 3 ways to generate configurations to explore:
# 1. completely at random
# 2. by "mutating" existing points:
#   a. randomly (taking some subset of the values and changing them to random values)
#   b. in a local way "exploring the neighborhood"
#      as an aside, we believe that this local, exhaustive search is good from previous experiments
# 3. by "crossover", where we trade values between existing configurations

# some strategies to use these generators:

# "exhaustive local search with restarts"
# 1. when no progress is made (or to start) run a controlled size generation of purely random configurations
# 2. while progress is being made, explore locally
# 3. repeat for a quantity of random restarts
# this is what we were doing before, but a little smarter in retrospect

# "balanced mixed search"
# for each generation, try to ensure an equal mix of the following kinds of points:
# 1. random
# 2. crossed
# 3. points on the frontier
# each should be forced up to some minimum, probably the number of threads for efficient parallelism,
# possibly capped

# local exploration could be substituted for 3 to limit it, or allowed to run without a limit
# 3 isn't really controllable, but we can increase the minimum sizes to match it if the frontier grows

# obviously we start with an initial gen of the minimum size of random points,
# and the frontier develops from there.

# halt the search when some number of generations (which at the end will be just random, and possibly crossed)
# fail to make any improvements.


class Sweep(object):
    """QuantiFind search driver object."""

    def __init__(self, eval_fn, init_fns, neighbor_fns, metric_fns,
                 settings=None, state=None, cores=None, batch=None, retry_attempts=1,
                 verbosity=3):
        self.eval_fn = eval_fn
        self.init_fns = init_fns
        self.neighbor_fns = neighbor_fns
        self.metric_fns = metric_fns
        if settings is None:
            self.settings = SearchSettings()
        else:
            self.settings = settings
        if state is None:
            self.state = SearchState()
        else:
            self.state = state
        self.cores = cores
        self.batch = batch
        self.retry_attempts = retry_attempts
        self.verbosity = verbosity

        # handle this with a context manager
        self.pool = None

        # logging setup
        self.keep_checkpoints = 2
        self.checkpoint_suffix = '.json'
        self.checkpoint_fmt = 'gen{:d}' + self.checkpoint_suffix
        self.checkpoint_re = re.compile('gen([0-9]+)' + re.escape(self.checkpoint_suffix))
        self.checkpoint_key = lambda s: self.checkpoint_re.fullmatch(s).group(1)
        self.checkpoint_tmpdir = '.tmp'
        self.checkpoint_outdir = 'checkpoints'
        self.snapshot_name = 'frontier' + self.checkpoint_suffix
        # currently set in the run method
        self.logdir = None

    def __enter__(self):
        self.pool = multiprocessing.Pool(self.cores)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def __repr__(self):
        return f'<{type(self).__name__} object at {hex(id(self))} with {len(self.state.cache)} configurations>'

    def __str__(self):
        lines = []
        lines.append(f'{type(self).__name__}:')
        lines.append(f'  evaluation function: {repr(self.eval_fn)}')
        for line in str(self.settings).split('\n'):
            lines.append('  ' + line)
        for line in str(self.state).split('\n'):
            lines.append('  ' + line)
        if self.cores is not None:
            lines.append(f'  cores:      {self.cores}')
        if self.batch is not None:
            lines.append(f'  batch size: {self.batch}')
        if self.retry_attempts > 0:
            lines.append(f'  retries:    {self.retry_attempts}')
        if self.verbosity >= 0:
            lines.append(f'  verbosity:  {self.verbosity}')
        if self.pool is not None:
            lines.append(f'  with worker pool {repr(self.pool)}')
        return '\n'.join(lines)

    def checkpoint(self, logdir, name=None, overwrite=True):
        """Save a checkpoint of the current settings and search state somewhere under the specified directory."""
        fname = self.checkpoint_fmt.format(len(self.state.generations))
        work_dir = os.path.join(logdir, self.checkpoint_tmpdir)
        target_dir = os.path.join(logdir, self.checkpoint_outdir)

        if name is None:
            link_path = None
        else:
            linkname = name + self.checkpoint_suffix
            link_path = os.path.join(logdir, linkname)

        if self.verbosity >= 0:
            print(f'Saving checkpoint for gen {len(self.state.generations)} to {target_dir}...')

        data = {
            'settings': self.settings.to_dict(),
            'state': self.state.to_dict(),
            'frontier': list(self.state.frontier),
        }
        log_and_copy(data, fname, work_dir=work_dir, target_dir=target_dir, link=link_path,
                     cleanup_re=self.checkpoint_re, keep_files=self.keep_checkpoints, key=self.checkpoint_key)

        if self.verbosity >= 0:
            print('Checkpoint saved, done.')

    def snapshot_frontier(self, logdir):
        """Save a snapshot of the current frontier."""
        fname = self.snapshot_name
        work_dir = os.path.join(logdir, self.checkpoint_tmpdir)
        target_dir = logdir

        # if self.verbosity >= 0:
        #     print(f'Saving snapshot at gen {len(self.state.generations)} to {target_dir}...')

        data = {
            'frontier': list(self.state.frontier),
        }
        log_and_copy(data, fname, work_dir=work_dir, target_dir=target_dir)

        # if self.verbosity >= 0:
        #     print('Snapshot saved, done.')

    # batch generation methods will "poke" the current cache state,
    # but do not create any new entries or add things to the horizon.

    def random_batch(self, size):
        """Return a new batch of completely random configurations to explore."""
        if self.verbosity >= 2:
            print(f'    generating a new batch of {size} random configurations...')

        batch = set()
        hits = 0
        for _ in range(self.retry_attempts + 1):
            for _ in range(size):
                cfg = tuple(f() for f in self.init_fns)
                if self.state.poke_cache(cfg) and cfg not in batch:
                    batch.add(cfg)
                    if len(batch) >= size:
                        break
                else:
                    hits += 1
            if len(batch) >= size:
                break

        if self.verbosity >= 2:
            print(f'    generated {len(batch)} random configurations ({hits} hit cache).')
        return batch

    def mutant_batch(self, size):
        """Return a new batch of mutated configurations, based on the current frontier."""
        if len(self.state.frontier) < 1:
            if self.verbosity >= 2:
                print(f'    unable to generate mutated configurations; no configurations in frontier')
            return set()

        if self.verbosity >= 2:
            print(f'    generating a new batch of {size} mutated configurations...')

        batch = set()
        hits = 0
        for _ in range(self.retry_attempts + 1):
            for _ in range(size):
                res1, = random.sample(self.state.frontier, 1)
                cfg1, qos1 = res1
                # mutate
                p = self.settings.mutation_probability
                cfg = tuple(f() if random.random() < p else x
                            for x, f in zip(cfg1, self.init_fns))
                if self.state.poke_cache(cfg) and cfg not in batch:
                    batch.add(cfg)
                    if len(batch) >= size:
                        break
                else:
                    hits += 1
            if len(batch) >= size:
                break

        if self.verbosity >= 2:
            print(f'    generated {len(batch)} mutated configurations ({hits} hit cache).')
        return batch

    def cross_batch(self, size):
        """Return a new batch of configurations using crossover, based on the current frontier."""
        if len(self.state.frontier) < 2:
            if self.verbosity >= 2:
                print(f'    unable to generate configurations with crossover; must have at least two configurations in frontier')
            return set()

        if self.verbosity >= 2:
            print(f'    generating a new batch of {size} crossed configurations...')

        batch = set()
        hits = 0
        for _ in range(self.retry_attempts + 1):
            for _ in range(size):
                res1, res2 = random.sample(self.state.frontier, 2)
                cfg1, qos1 = res1
                cfg2, qos2 = res2
                # crossover
                p = self.settings.crossover_probability
                cfg = tuple(y if random.random() < p else x
                            for x, y in zip(cfg1, cfg2))
                if self.state.poke_cache(cfg) and cfg not in batch:
                    batch.add(cfg)
                    if len(batch) >= size:
                        break
                else:
                    hits += 1
            if len(batch) >= size:
                break

        if self.verbosity >= 2:
            print(f'    generated {len(batch)} crossed configurations ({hits} hit cache).')
        return batch

    def neighborhood(self, axes_first=True):
        """Generator for the full space of configurations "nearby" to the Pareto frontier.
        May produce the same configuration multiple times, but not too many times.
        If axes_first is False, then skip the pre-pass that explores hypercubes
        defined by extending along axes.
        """
        stopiter_sentinel = object()
        if axes_first:
            gens = [nearby_points(cfg, self.neighbor_fns, bfs_neighbors=True, combine=True, product=False)
                    for cfg, _ in self.state.frontier]
            new_gens = []
            while gens:
                random.shuffle(gens)
                new_gens.clear()
                for gen in gens:
                    elt = next(gen, stopiter_sentinel)
                    if elt is not stopiter_sentinel:
                        new_gens.append(gen)
                        yield elt
                # swap lists in place
                gens, new_gens = new_gens, gens
        # now go through the full (breadth-first) cartesian product
        # in the same way
        gens = [nearby_points(cfg, self.neighbor_fns, product=True)
                for cfg, _ in self.state.frontier]
        new_gens = []
        while gens:
            random.shuffle(gens)
            new_gens.clear()
            for gen in gens:
                elt = next(gen, stopiter_sentinel)
                if elt is not stopiter_sentinel:
                    new_gens.append(gen)
                    yield elt
            # swap lists in place
            gens, new_gens = new_gens, gens

    def local_neighborhood(self, min_size=None, max_size=None):
        """Explore the local neighborhood of the current pareto frontier.
        First return points by extending each axis, then all axes together.
        Stop early if max_size is hit.
        If min_size is still not hit, look for additional points
        in the full cartesian products of neighbors of the frontier.
        """
        if len(self.state.frontier) < 1:
            if self.verbosity >= 2:
                print(f'    unable to generate neighboring configurations; no configurations in frontier')
            return set()

        if self.verbosity >= 2:
            if min_size is not None or max_size is not None:
                print(f'    generating local neighborhood ({min_size} - {max_size})...')
            else:
                print(f'    generating local neighborhood...')

        gens = [nearby_points(cfg, self.neighbor_fns, bfs_neighbors=(max_size is not None), combine=True, product=False)
                for cfg, _ in self.state.frontier]

        batch = set()
        hits = 0
        if max_size is None:
            for cfg in itertools.chain(*gens):
                if self.state.poke_cache(cfg) and cfg not in batch:
                    batch.add(cfg)
                else:
                    hits += 1
        else: # max_size is not None
             for cfg in itertools.chain(*gens):
                if self.state.poke_cache(cfg) and cfg not in batch:
                    batch.add(cfg)
                    if len(batch) >= max_size:
                        break
                else:
                    hits += 1

        if min_size is not None and len(batch) < min_size:
            if self.verbosity >= 2:
                print(f'    local search only found {len(batch)} points, exploring full cartesian product...')

            for cfg in self.neighborhood(axes_first=False):
                if self.state.poke_cache(cfg) and cfg not in batch:
                    batch.add(cfg)
                    if len(batch) >= min_size:
                        break
                else:
                    hits += 1

        if self.verbosity >= 2:
            print(f'    generated local neighborhood of {len(batch)} configurations ({hits} hit cache).')
        return batch

    def local_batch(self, size, axes_first=True):
        """Return a new batch of nearby configurations, based on the current frontier.
        If axes_first is True, first explore along the parameter axes.
        """
        if len(self.state.frontier) < 1:
            if self.verbosity >= 2:
                print(f'    unable to generate neighboring configurations; no configurations in frontier')
            return set()

        if self.verbosity >= 2:
            print(f'    generating a new batch of {size} neighboring configurations...')

        batch = set()
        hits = 0
        for cfg in self.neighborhood(axes_first=axes_first):
            if self.state.poke_cache(cfg) and cfg not in batch:
                batch.add(cfg)
                if len(batch) >= size:
                    break
            else:
                hits += 1

        if self.verbosity >= 2:
            print(f'    generated {len(batch)} nieghboring configurations ({hits} hit cache).')
        return batch

    def exhaustive_batch(self, searchspace, center_cfg=None, max_size=None):
        """Return a new batch of configurations, exhaustively exploring searchspace.
        If center_cfg is provided, explore manhattan spheres around it in order;
        if max_size is provided, then stop after that many points.
        """
        if self.verbosity >= 2:
            if max_size is None:
                print(f'    exhaustively enumerating configurations...')
            else:
                print(f'    exhaustively enumerating up to {max_size} configurations...')

        if center_cfg is None:
            space = [list(parameter_axis) for parameter_axis in searchspace]
            gen = itertools.product(*space)
        else:
            space = [reorder_for_bfs(parameter_axis, x) for parameter_axis, x in zip(searchspace, center_cfg)]
            gen = breadth_first_product(*space)

        batch = set()
        hits = 0
        if max_size is None:
            for cfg in gen:
                if self.state.poke_cache(cfg) and cfg not in batch:
                    batch.add(cfg)
                else:
                    hits += 1
        else:
            for cfg in gen:
                if self.state.poke_cache(cfg) and cfg not in batch:
                    batch.add(cfg)
                    if len(batch) >= max_size:
                        break
                else:
                    hits += 1

        if self.verbosity >= 2:
            print(f'    exhaustively generated {len(batch)} configurations ({hits} hit cache).')
        return batch

    def relative_pop(self, this_weight, ref_weight, ref_size, target_bounds):
        """Scale ref_size as a fraction this_weight of ref_weight.
        If target_bounds are provided as a tuple (min_bound, max_bound),
        then return a number in that range.
        Never return less than 1, unless ref_size or this_weight is 0.
        """
        unbounded = ref_size * this_weight
        if unbounded != 0:
            unbounded = (unbounded // ref_weight) + 1
        if target_bounds is not None:
            min_bound, max_bound = target_bounds
            if min_bound is not None:
                unbounded = max(min_bound, unbounded)
            if max_bound is not None:
                unbounded = min(unbounded, max_bound)
        return unbounded

    def expand_horizon(self):
        """Expand the horizon by adding new configurations.

        This is the only place that self.state.horizon is modified,
        besides the exhaustive and random search methods;
        batches are generated sequentially for each population
        based on the population constraints
        and individually added to the horizon (and the cache) in order:
          - local
          - crossed
          - mutant
          - purely random

        In addition to modifying the state, returns the number of configurations
        added to the horizon;
        if this is zero (i.e. we couldn't expand the horizon)
        the search should probably end, as it's hard to see how it will make progress.
        """
        settings = self.settings
        pop_random_weight = settings.pop_random_weight
        pop_mutant_weight = settings.pop_mutant_weight
        pop_crossed_weight = settings.pop_crossed_weight
        pop_local_weight = settings.pop_local_weight
        pop_weight = pop_random_weight + pop_mutant_weight + pop_crossed_weight + pop_local_weight
        pop_weight_scale = settings.pop_weight_scale
        if pop_weight <= 0:
            if self.verbosity >= 1:
                print(f'  Unable to expand horizon; zero population weight')
            return 0

        if self.verbosity >= 1:
            print(f'  Expanding the horizon...')

        # decide on a population scheme
        if pop_weight_scale <= 0:
            # weight relatively so that pop_local_weight ~= size of the local neighborhood
            if pop_local_weight <= 0:
                if self.verbosity >= 1:
                    print(f'  Unable to expand horizon; no local population to weight against')
                return 0

            # first we need the local neighborhood
            if settings.pop_local_target:
                min_size, max_size = settings.pop_local_target
            else:
                min_size, max_size = None, None
            neighbors = self.local_neighborhood(min_size, max_size)
            new_cfg_count = len(neighbors)
            self.state.add_to_horizon(neighbors, 3, verbose=self.verbosity>=3)

            if pop_random_weight + pop_mutant_weight + pop_crossed_weight == 0:
                if self.verbosity >= 1:
                    print(f'  Added {new_cfg_count} new configurations from the local neighborhood.')
                return new_cfg_count
            else:
                pop_ref_size = len(neighbors)
                pop_ref_weight = pop_local_weight
        else:
            # weight relatively so that pop_weight / pop_weight_scale ~= size of the frontier
            pop_ref_size = len(self.state.frontier) * pop_weight_scale
            pop_ref_weight = pop_weight
            # placeholder - the code will handle this later
            neighbors = None
            new_cfg_count = 0

        if neighbors is not None:
            local_size = len(neighbors)
        else:
            local_size = self.relative_pop(pop_local_weight, pop_ref_weight, pop_ref_size, settings.pop_local_target)
        crossed_size = self.relative_pop(pop_crossed_weight, pop_ref_weight, pop_ref_size, settings.pop_crossed_target)
        mutant_size = self.relative_pop(pop_mutant_weight, pop_ref_weight, pop_ref_size, settings.pop_mutant_target)
        random_size = self.relative_pop(pop_random_weight, pop_ref_weight, pop_ref_size, settings.pop_random_target)

        if self.verbosity >= 1:
            print(f'  Looking for about {pop_ref_size} new configurations...')
            if self.verbosity >= 2:
                if local_size > 0:
                    if neighbors is not None:
                        print(f'  - Local:   {local_size} (already added)')
                    else:
                        print(f'  - Local:   {local_size}')
                if crossed_size > 0:
                    print(f'  - Crossed: {crossed_size}')
                if mutant_size > 0:
                    print(f'  - Mutant:  {mutant_size}')
                if random_size > 0:
                    print(f'  - Random:  {random_size}')

        if neighbors is None and local_size > 0:
            batch = self.local_batch(local_size)
            new_cfg_count += len(batch)
            self.state.add_to_horizon(batch, 3, verbose=self.verbosity>=3)
        if crossed_size > 0:
            batch = self.cross_batch(crossed_size)
            new_cfg_count += len(batch)
            self.state.add_to_horizon(batch, 2, verbose=self.verbosity>=3)
        if mutant_size > 0:
            batch = self.mutant_batch(mutant_size)
            new_cfg_count += len(batch)
            self.state.add_to_horizon(batch, 1, verbose=self.verbosity>=3)
        if random_size > 0:
            batch = self.random_batch(random_size)
            new_cfg_count += len(batch)
            self.state.add_to_horizon(batch, 0, verbose=self.verbosity>=3)

        if self.verbosity >= 1:
            print(f'  Added {new_cfg_count} new configurations to the horizon.')
        return new_cfg_count

    def explore_randomly(self, n):
        """Try to add up to n random configurations to the frontier.
        This is a good fallback if we have no current frontier to guide the search,
        or the entire local area has been exhausted.
        """
        if self.verbosity >= 1:
            print(f'  Looking for {n} random configurations to add to the horizon...')

        batch = self.random_batch(n)
        self.state.add_to_horizon(batch, 0, verbose=self.verbosity>=3)

        if self.verbosity >= 1:
            print(f'  Added {len(batch)} new random configurations to the horizon.')
        return len(batch)

    def explore_exhaustively(self, searchspace, center_cfg=None, max_size=None):
        """Explore a space of parameters exhaustively.
        Searchspace should be a list of generators, one for each parameter,
        giving every possible value in order.
        If center_cfg is provided, then order the search outward (in terms of manhattan spheres)
        from that point.
        """
        if self.verbosity >= 1:
            print(f'  Exploring exhaustively...')

        batch = self.exhaustive_batch(searchspace, center_cfg=center_cfg, max_size=max_size)
        self.state.add_to_horizon(batch, 4, verbose=self.verbosity>=3)

        if self.verbosity >= 1:
            print(f'  Added {len(batch)} exhaustive configurations to the horizon.')
        return len(batch)

    def process_batch(self, pool):
        """Run a batch of configurations from the horizon,
        and commit the results to the state.
        This should probably? be reimplemented to take advantage of the appropriate
        Pool.map or Pool.imap / async functionality. Or not."""
        if self.verbosity >= 2:
            if self.batch is not None:
                print(f'    processing a batch of {self.batch} configurations...')
            else:
                print(f'    processing the entire horizon...')

        cfgs = self.state.get_from_horizon(self.batch)
        async_results = []
        for cfg in cfgs:
            async_results.append(pool.apply_async(self.eval_fn, cfg))

        if self.verbosity >= 2:
            print(f'    dispatched {len(async_results)} evaluations...')

        new_frontier_points = 0
        pending_point = False
        last_snapshot = time.time()
        for cfg, ares in zip(cfgs, async_results):
            qos = ares.get()
            result = cfg, qos
            if self.state.commit_to_history(result, self.metric_fns, verbose=self.verbosity>=3):
                new_frontier_points += 1
                pending_point = True
                if self.verbosity >= 3:
                    print('!', end='', flush=True)
            else:
                if self.verbosity >= 3:
                    print('.', end='', flush=True)

            if self.logdir is not None:
                now = time.time()
                if pending_point and now > last_snapshot + 2.0:
                    pending_point = False
                    last_snapshot = now
                    self.snapshot_frontier(self.logdir)
                    print('X', end='', flush=True)

        if self.logdir is not None:
            if pending_point:
                self.snapshot_frontier(self.logdir)
                print('X', end='', flush=True)

        if self.verbosity >= 3:
            print(flush=True)

        if self.verbosity >= 2:
            print(f'    processed {len(async_results)} configurations, added {new_frontier_points} to the frontier.')
        return new_frontier_points

    def run_generation(self, pool=None):
        """Run the next generation of configurations currently on the horizon.
        Handles the generation records in the state.
        """
        gen_idx = len(self.state.generations)
        horizon_size = len(self.state.horizon)
        new_frontier_points = 0
        self.state.generations.append((horizon_size, new_frontier_points))

        if self.verbosity >= 1:
            print(f'  Evaluating the horizon for generation {gen_idx}...')

        if pool is None:
            pool = self.pool

        if pool is None:
            with multiprocessing.Pool(self.cores) as pool:
                while len(self.state.horizon) > 0:
                    new_frontier_points += self.process_batch(pool)
                    self.state.generations[gen_idx] = (horizon_size, new_frontier_points)
        else:
            while len(self.state.horizon) > 0:
                new_frontier_points += self.process_batch(pool)
                self.state.generations[gen_idx] = (horizon_size, new_frontier_points)

        if self.verbosity >= 1:
            print(f'  Evaluated {horizon_size} configurations for generation {gen_idx}, adding {self.state.generations[gen_idx]} to the frontier.')

        return new_frontier_points

    def cleanup_horizon(self, pool=None):
        """Empty the horizon, in case only part of a generation was evaluated.
        This is almost the same as running a generation,
        but it doesn't append a new generation record.
        """
        horizon_remaining = len(self.state.horizon)
        if horizon_remaining == 0:
            return 0

        gen_idx = len(self.state.generations) - 1
        if gen_idx < 0:
            gen_idx = 0
            self.state.generations.append((horizon_remaining, 0))

        horizon_size, old_frontier_points = self.state.generations[gen_idx]
        new_frontier_points = 0

        if self.verbosity >= 1:
            print(f'  Cleaning up {horizon_remaining} configurations left on the horizon at generation {gen_idx}...')

        if pool is None:
            pool = self.pool

        if pool is None:
            with multiprocessing.Pool(self.cores) as pool:
                while len(self.state.horizon) > 0:
                    new_frontier_points += self.process_batch(pool)
                    self.state.generations[gen_idx] = (horizon_size, old_frontier_points + new_frontier_points)
        else:
            while len(self.state.horizon) > 0:
                new_frontier_points += self.process_batch(pool)
                self.state.generations[gen_idx] = (horizon_size, old_frontier_points + new_frontier_points)

        if self.verbosity >= 1:
            print(f'  Cleaned up {horizon_remaining} configurations for generation {gen_idx}, adding {new_frontier_points} to the frontier.')
        return total_new_points

    def _do_checkpoint(self):
        """used in run_search"""
        if self.verbosity >= 0:
            print(flush=True)
        self.checkpoint(self.logdir, name='latest')
        if self.verbosity >= 0:
            print(flush=True)
    def _final_checkpoint(self):
        """used in run_search"""
        if self.logdir is not None:
            self._do_checkpoint()
            out_dir = os.path.join(self.logdir, self.checkpoint_outdir)
            outname = self.checkpoint_fmt.format(len(self.state.generations))
            out_path = os.path.join(out_dir, outname)
            if os.path.exists(out_path):
                # do the symlink thing
                tmp_dir = os.path.join(self.logdir, self.checkpoint_tmpdir)
                linkname = 'final' + self.checkpoint_suffix
                tmp_path = os.path.join(tmp_dir, linkname)
                link_target = os.path.relpath(out_path, self.logdir)
                link_path = os.path.join(self.logdir, linkname)
                os.symlink(link_target, tmp_path)
                os.replace(tmp_path, link_path)
    def run_search(self, checkpoint_dir=None, check=False):
        """Run the Pareto frontier exploration sweep!"""
        if self.verbosity >= 0:
            print('Running QuantiFind sweep...')
            if checkpoint_dir is not None:
                print(f'Saving checkpoints to {checkpoint_dir}.')
            if self.verbosity >= 2:
                print(self)

        self.logdir = checkpoint_dir

        self.cleanup_horizon()
        is_initial_gen = (len(self.state.history) == 0)

        while True:
            if self.verbosity >= 1:
                print(flush=True)

            # try to expand the horizon
            new_cfgs = self.expand_horizon()

            # no normal way to expand locally; try randomly as a backup
            if new_cfgs <= 0:
                new_cfgs = self.explore_randomly(self.settings.initial_gen_size)

            if new_cfgs <= 0:
                if self.verbosity >= 0:
                    print('Exhausted the search space. Done.')
                if self.verbosity >= 2:
                    print('final ', end='')
                    print(self.state)
                    print(flush=True)
                self._final_checkpoint()
                return self.state.frontier

            if is_initial_gen:
                self.state.initial_gens += 1
                self.state.initial_cfgs += new_cfgs

            if self.verbosity >= 1:
                print(flush=True)

            new_frontier_points = self.run_generation()
            is_initial_gen = (new_frontier_points == 0)

            # stopping criteria
            if is_initial_gen:
                if (self.state.initial_gens >= self.settings.restart_gen_target and
                    self.state.initial_cfgs >= self.settings.restart_size_target):
                    if self.verbosity >= 0:
                        print(f'Searched for {self.state.initial_gens} initial generations '
                              f'with {self.state.initial_cfgs} total initial configurations. Done.')
                    if self.verbosity >= 2:
                        print('final ', end='')
                        print(self.state)
                        print(flush=True)
                    self._final_checkpoint()
                    return self.state.frontier
            else:
                if self.verbosity >= 3:
                    print('The frontier changed.')
                    print_frontier(self.state.frontier)

            if self.verbosity >= 2:
                print(flush=True)
                print('current ', end='')
                print(self.state)

                print(flush=True)
                print('progress:')
                print(f'  initial generations:    {self.state.initial_gens} / {self.settings.restart_gen_target}')
                print(f'  initial configurations: {self.state.initial_cfgs} / {self.settings.restart_size_target}')

            if check:
                if self.verbosity >= 0:
                    print(flush=True)
                consistent = self.state.check(metric_fns=self.metric_fns, verbose=self.verbosity>=0)
                if self.verbosity >= 0:
                    print(flush=True)

            if self.logdir is not None:
                self._do_checkpoint()
