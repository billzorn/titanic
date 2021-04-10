"""Common code for parameter search."""


# Goal: find a distribution of "good" configurations according to a set of metrics

# Kinds of metrics:
# - quality
# - cost
# Ways to search:
# - quality must be with in bound, find best cost (the most common way)
# - cost must be wtihin bound, do as well as you can

# This means each metric either has a bound or doesn't
# When evaluating a configuration, there are two modes:
# - bound metric is satisfied
#   - improve "cost" metrics
# - bound metric not satisfied
#   - improve bound metric

# Search procedure
#
# Shotgun: pick a random configuration, based on a set of annotation sites and format hints
# Evaluate: check the configuration's evaluation
# Hillclimb: Look at nearby configurations (move each annotation [up to]? k steps)
#   Among those configurations, look for the "most passing" configuration
#   - if there is a passing configuration, use the cheapest
#   - otherwise, use the least bad
#   probably move each point at least k1 steps, and at most k2 steps, in each direction
#   - start by moving k1
#   - then, if there isn't a passing configuration, keep going up to k2?
#   - no, just search fixed k in each direction, it's simpler
#   Now update our evaluated starting config and keep climbing

# Keeping a set of "good" configurations
#
# We have a set of metrics, some of which have hard bounds
# Most of them we can also order
# Our goal is to only consider options that satisfy all the bounds; of those we want the best
# Obviously, only keep configurations that satisfy the bounds
# Of those that do, look at all other cost metrics we care about
# keep a "frontier set" that are each better at some subset of metrics than any other member of frontier
# search until no point is near a point that can be improved, and after a certain number of random inits


# functions we need:
# init - randomly pick a starting point, based on a precision hint
# next/prev on types - get higher and lower precision, in a sequence
# compare metric - given two values, which is better
# bound metric - given a value, is it ok

# there's an optimization frontier - things might split

# Hillclimbing with random restarts

# cost metric isn't about the chip - we have proxies
# spin it as an advantage!


import itertools
import multiprocessing
import random
import math

from .utils import describe_ctx


def center_ranges(input_ranges):
    maxlen = 0
    output_ranges = []

    # turn the generator into a list, so we can iterate it multiple times
    input_ranges = [tuple(r) for r in input_ranges]
    for rng in input_ranges:
        new_range = list(rng)
        if len(new_range) == 0:
            raise ValueError('cannot center an empty range')
            # return input_ranges
        if len(new_range) > maxlen:
            maxlen = len(new_range)
        output_ranges.append(new_range)

    for i in range(len(output_ranges)):
        rng = output_ranges[i]
        if len(rng) < maxlen:
            pad_count = maxlen - len(rng)
            left_pad = right_pad = pad_count // 2
            if left_pad + right_pad < pad_count:
                right_pad += 1
            output_ranges[i] = ([rng[0]] * left_pad) + rng + ([rng[-1]] * right_pad)

    return output_ranges


def compare_results(m1, m2, metrics):
    lt = False
    gt = False
    for a1, a2, cmp_lt in zip(m1, m2, metrics):
        if cmp_lt(a1, a2):
            lt = True
        elif cmp_lt(a2, a1):
            gt = True

    if lt:
        if gt:
            return None
        else:
            return -1
    else:
        if gt:
            return 1
        else:
            return 0


def update_frontier(frontier, result, metrics):

    keep = True
    new_frontier = []
    frontier_cfgs = set()

    result_data, result_m = result

    for frontier_data, frontier_m in frontier:
        comparison = compare_results(result_m, frontier_m, metrics)

        if comparison is None or comparison == 0:
            # the points are incomparable; keep both
            if frontier_data not in frontier_cfgs:
                new_frontier.append((frontier_data, frontier_m))
                frontier_cfgs.add(frontier_data)


        elif comparison > 0:
            # some existing result is better than the new one;
            # we don't need the new one
            if frontier_data not in frontier_cfgs:
                new_frontier.append((frontier_data, frontier_m))
                frontier_cfgs.add(frontier_data)
            keep = False

        # else: # comparison < 0
        #     pass
        #     # new result is less, which is strictly better;
        #     # keep it and throw out this result

    if keep and result_data not in frontier_cfgs:
        new_frontier.append(result)
        frontier_cfgs.add(result_data)

    #check_frontier(new_frontier, metrics)

    return keep, new_frontier


def check_frontier(frontier, metrics, verbosity=3):

    broken = False
    new_frontier = []
    frontier_cfgs = set()

    for i1 in range(len(frontier)):
        res1_data, res1_m = frontier[i1]
        keep = True

        for i2 in range(len(frontier)):
            if i1 != i2:

                res2_data, res2_m = frontier[i2]

                comparison = compare_results(res1_m, res2_m, metrics)

                if not (comparison is None or comparison == 0):
                    broken = True

                    if comparison > 0:
                        if verbosity >= 2:
                            print(f'Discarding point {i1!s} {frontier[i1]!r} from frontier')
                            print(f'  point {i2!s} {frontier[i2]!r} is strictly better')
                        keep = False
                        break

        if keep and res1_data not in frontier_cfgs:
            new_frontier.append((res1_data, res1_m))
            frontier_cfgs.add(res1_data)

    if broken and verbosity >= 1:
        print(f'Discarded {len(frontier) - len(new_frontier)} points in total')

    return broken, new_frontier


def print_frontier(frontier):
    print('[')
    for frontier_data, frontier_m in frontier:
        print(f'    ({frontier_data!r}, {frontier_m!r}),')
    print(']')


def sweep_random_init(stage_fn, inits, neighbors, metrics,
                      previous_sweep=None, force_exploration=False, verbosity=3):
    initial_cfg = tuple(f() for f in inits)
    initial_result = stage_fn(*initial_cfg)
    visited_points = [(initial_cfg, initial_result)]

    if verbosity >= 1:
        print(f'Random init: the initial point is {initial_cfg!r} : {initial_result!r}')

    if previous_sweep is None:
        all_cfgs = {initial_cfg}
        frontier = [(initial_cfg, initial_result)]
        gen_log = [0]
        improved_frontier = True
    else:
        explore_area = False
        gen_log, all_cfgs, frontier = previous_sweep
        if initial_cfg in all_cfgs:
            if verbosity >= 1:
                print('  this point has already been explored, ', end='')
            explore_area = True
        else:
            all_cfgs.add(initial_cfg)
            updated, frontier = update_frontier(frontier, (initial_cfg, initial_result), metrics)
            if updated:
                gen_log.append(0)
                improved_frontier = True
            else:
                if verbosity >= 1:
                    print('  this point is not interesting, ', end='')
                explore_area = True

        if explore_area:
            if force_exploration:
                if verbosity >= 1:
                    print('exploring anyway')
                gen_log.append(-1)
                frontier.append((initial_cfg, initial_result))
                improved_frontier = True
            else:
                if verbosity >= 1:
                    print('aborting')
                return (False, visited_points, *previous_sweep)

    with multiprocessing.Pool() as pool:
        while improved_frontier:
            improved_frontier = False

            if verbosity >= 1:
                print(f'generation {gen_log[-1]!s}: ({len(all_cfgs)!s} cfgs total) ')
                print_frontier(frontier)
                print(flush=True)

            gen_log[-1] += 1

            async_results = []
            skipped = 0
            for cfg, result in frontier:
                # work on individual points
                for i in range(len(cfg)):
                    x = cfg[i]
                    f = neighbors[i]
                    for new_x in f(x):
                        new_cfg = list(cfg)
                        new_cfg[i] = new_x
                        new_cfg = tuple(new_cfg)
                        if new_cfg not in all_cfgs:
                            async_results.append((new_cfg, pool.apply_async(stage_fn, new_cfg)))
                            all_cfgs.add(new_cfg)
                        else:
                            skipped += 1

                # work on all points together
                for combined_cfg in zip(*center_ranges(f(x) for f, x in zip(neighbors, cfg))):
                    if combined_cfg not in all_cfgs:
                        async_results.append((combined_cfg, pool.apply_async(stage_fn, combined_cfg)))
                        all_cfgs.add(combined_cfg)
                    else:
                        skipped += 1

            if verbosity >= 2:
                print(f'dispatched {len(async_results)!s} evaluations for generation {gen_log[-1]!s}, {skipped!s} skipped')

            for i, (new_cfg, ares) in enumerate(async_results):
                new_res = ares.get()
                visited_points.append((new_cfg, new_res))
                updated, frontier = update_frontier(frontier, (new_cfg, new_res), metrics)
                improved_frontier |= updated

                if verbosity >= 3:
                    print(f' -- {i+1!s} -- ran {new_cfg!r}, got {new_res!r}')
                    if updated:
                        print('The frontier changed:')
                        print_frontier(frontier)

            broken, frontier = check_frontier(frontier, metrics, verbosity=verbosity)
            if broken and explore_area and gen_log[-1] == 0 and verbosity >= 1:
                print('  this is expected due to forced exploration')

    if verbosity >= 1:
        print(f'improvement stopped at generation {gen_log[-1]!s}: ')
        print_frontier(frontier)
        print(flush=True)

    return (gen_log[-1] > 0), visited_points, gen_log, all_cfgs, frontier

def sweep_multi(stage_fn, inits, neighbors, metrics, max_inits, max_retries,
                force_exploration=False, verbosity=3):
    if verbosity >= 1:
        print(f'Multi-sweep: sweeping over {max_inits!s} random initializations, and up to {max_retries!s} ignored points')

    improved, visited_points, gens, cfgs, frontier = sweep_random_init(stage_fn, inits, neighbors, metrics, verbosity=verbosity)

    visited_points = [(0, a, b) for a, b in visited_points]

    attempts = 0
    successes = 0
    failures = 0
    while successes < max_inits and failures < max_retries:
        if verbosity >= 2:
            print(f'\n == attempt {attempts+1!s}, {len(cfgs)!s} cfgs, {len(frontier)!s} elements in frontier ==\n')

        improved, new_visited_points, gens, cfgs, frontier = sweep_random_init(stage_fn, inits, neighbors, metrics, verbosity=verbosity,
                                                                               previous_sweep=(gens, cfgs, frontier), force_exploration=force_exploration)

        visited_points.extend((attempts, a, b) for a, b in new_visited_points)

        if verbosity >= 2:
            print(f'\n == finished attempt {attempts+1!s}, improved? {improved!s} ==\n')

        attempts += 1
        if improved:
            successes += 1
        else:
            failures += 1

    return gens, visited_points, frontier

def sweep_exhaustive(stage_fn, cfgs, metrics, verbosity=3):
    if verbosity >= 1:
        print(f'Exhaustive sweep for stage {stage_fn!r}')

    all_cfgs = set()
    visited_points = []
    frontier = []
    with multiprocessing.Pool() as pool:
        async_results = []
        for cfg in itertools.product(*cfgs):
            str_cfg = tuple(describe_ctx(ctx) for ctx in cfg)
            if str_cfg not in all_cfgs:
                all_cfgs.add(str_cfg)
                async_results.append((str_cfg, pool.apply_async(stage_fn, cfg)))

        if verbosity >= 2:
            print(f'dispatched {len(async_results)!s} evaluations')

        for i, (cfg, ares) in enumerate(async_results):
            res = ares.get()
            visited_points.append((0, cfg, res))
            updated, frontier = update_frontier(frontier, (cfg, res), metrics)

            if verbosity >= 3:
                print(f' -- {i+1!s} -- ran {cfg!r}, got {res!r}')
                if updated:
                    print('The frontier changed:')
                    print_frontier(frontier)

    if verbosity >= 1:
        print('Done. final frontier:')
        print_frontier(frontier)

    return [1], visited_points, frontier

def sweep_random(stage_fn, inits, metrics, points, batch_size=1000, verbosity=3):
    if verbosity >= 1:
        print(f'Random sweep over {points!s} points')

    explored_points = set()
    visited_points = []
    frontier = []

    with multiprocessing.Pool() as pool:
        batchnum = 1
        while len(explored_points) < points:
            if verbosity >= 1:
                print(f'starting batch {batchnum!s}, explored {len(explored_points)!s} so far')
                print_frontier(frontier)
                print(flush=True)

            async_results = []
            skipped = 0

            for i in range(batch_size):
                cfg = tuple(f() for f in inits)
                if cfg in explored_points:
                    skipped += 1
                else:
                    async_results.append((cfg, pool.apply_async(stage_fn, cfg)))
                    explored_points.add(cfg)
                    if len(explored_points) >= points:
                        break

            if async_results:
                if verbosity >= 2:
                    print(f'dispatched {len(async_results)!s} evaluations for batch {batchnum!s}, {skipped!s} skipped')

                for cfg, ares in async_results:
                    res = ares.get()
                    visited_points.append((cfg, res))
                    updated, frontier = update_frontier(frontier, (cfg, res), metrics)

                    if verbosity >= 3:
                        print(f' -- {len(visited_points)!s} -- ran {cfg!r}, got {res!r}')
                        if updated:
                            print('The frontier changed:')
                            print_frontier(frontier)
            else:
                if verbosity >= 1:
                    print('could not find any new configurations, aborting')
                    break

    if verbosity >= 1:
        print('Done. final frontier:')
        print_frontier(frontier)

    return [1], visited_points, frontier

def filter_metrics(points, metrics, allow_inf=False):
    new_points = []

    for point in points:
        if len(point) == 2:
            data, measures = point
            filtered_measures = tuple(meas for meas, m in zip(measures, metrics) if m is not None)
            if allow_inf or all(map(math.isfinite, filtered_measures)):
                new_points.append((data, filtered_measures))
        if len(point) == 3:
            gen, data, measures = point
            filtered_measures = tuple(meas for meas, m in zip(measures, metrics) if m is not None)
            if allow_inf or all(map(math.isfinite, filtered_measures)):
                new_points.append((gen, data, filtered_measures))
    return new_points


def filter_frontier(frontier, metrics, allow_inf=False, reconstruct_metrics=False):
    new_metrics = [m for m in metrics if m is not None]

    new_frontier = []
    for i, (data, measures) in enumerate(frontier):
        filtered_measures = tuple(meas for meas, m in zip(measures, metrics) if m is not None)

        if reconstruct_metrics:
            filtered_data = i
        else:
            filtered_data = data

        if allow_inf or all(map(math.isfinite, filtered_measures)):
            _, new_frontier = update_frontier(new_frontier, (filtered_data, filtered_measures), new_metrics)

    if reconstruct_metrics:
        reconstructed_frontier = [frontier[i] for i, measures in new_frontier]
        new_frontier = reconstructed_frontier

    return new_frontier













# new general utilities

# this bfs cartesian product is adapted from here:
# https://stackoverflow.com/questions/42288203/generate-itertools-product-in-different-order
#
# filtering out of bounds points in the partitioning scheme,
# instead of using a try/catch, improves performance by ~2x for "believable" cases...
def breadth_first_partitions(n, k, max_position):
    for c in itertools.combinations(range(n+k-1), k-1):
        positions = []
        safe = True
        for a, b, limit in zip((-1,)+c, c+(n+k-1,), max_position):
            position = b-a-1
            if position > limit:
                safe = False
                break
            else:
                positions.append(position)
        if safe:
            yield positions

def breadth_first_product(*sequences):
    """Breadth First Search Cartesian Product"""
    sequences = [list(seq) for seq in sequences]

    max_position = [len(i)-1 for i in sequences]
    for i in range(sum(max_position)):
        for positions in breadth_first_partitions(i, len(sequences), max_position):
            yield tuple(map(lambda seq, pos: seq[pos], sequences, positions))
    yield tuple(map(lambda seq, pos: seq[pos], sequences, max_position))

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

def nearby_points(cfg, neighbor_fns, combine=False, product=False, randomize=False):
    """Generator function to yield "nearby" configurations to cfg.
    Each neighbor generator in neighbor_fns will be called individually,
    and a new configuration yielded that replaces that element of cfg
    with each possible neighbor.

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
        for i, (x, f) in enumerate(zip(cfg, neighbor_fns)):
            for nearby_x in f(x):
                if nearby_x != x:
                    nearby_cfg = list(cfg)
                    nearby_cfg[i] = nearby_x
                    yield tuple(nearby_cfg)
        if combine:
            all_neighbors = [f(x) for x, f in zip(cfg, neighbor_fns)]
            # to explain the opaque yield from call below:
            # We start with a list of "nearby" parameters, for each variable in the cfg:
            # [[1,2,3], [7], [5,6]]
            # center_ranges pads this out so each list is the same length:
            # [[1,2,3], [7,7,7], [5,6,6]]
            # and then the zip(*) idiom re-slices this list of possible parameters into a list of configurations:
            # [[1,7,5], [2,7,6], [3,7,6]]
            # The zip generator yields new tuples, so we don't need to do anything else to package its outputs.
            yield from zip(*center_ranges(all_neighbors))


# new genetic algorithm stuff
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


class SearchSettings(object):
    """Settings container for QuantiFind search."""

    initial_gen_size = 1

    restart_size_target = 0
    restart_gen_target = 0

    pop_random_weight = 0
    pop_mutant_weight = 0
    pop_local_weight = 1
    pop_crossed_weight = 0

    pop_random_target = None
    pop_mutant_target = None
    pop_local_target = None
    pop_crossed_target = None

    mutation_probability = 0.5
    crossover_probability = 0.5

    def __init__(self, profile=None,
                 initial_gen_size=None,
                 restart_size_target = None,
                 restart_gen_target = None,
                 pop_weights = None,
                 pop_targets = None,
                 mutation_probability = None,
                 crossover_probability = None):
        # set defaults based on profile
        if profile is None or profile == 'local': # default
            self.initial_gen_size = 1
            self.restart_size_target = 0
            self.restart_gen_target = 0
            self.pop_random_weight = 0
            self.pop_mutant_weight = 0
            self.pop_local_weight = 1
            self.pop_crossed_weight = 0
            self.pop_random_target = None
            self.pop_mutant_target = None
            self.pop_local_target = None
            self.pop_crossed_target = None
            self.mutation_probability = 0.5
            self.crossover_probability = 0.5
        elif profile == 'balanced':
            self.initial_gen_size = 1
            self.restart_size_target = 0
            self.restart_gen_target = 0
            self.pop_random_weight = 1
            self.pop_mutant_weight = 1
            self.pop_local_weight = 3
            self.pop_crossed_weight = 1
            self.pop_random_target = None
            self.pop_mutant_target = None
            self.pop_local_target = None
            self.pop_crossed_target = None
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
            self.pop_random_weight, self.pop_mutant_weight, self.pop_local_weight, self.pop_crossed_weight = pop_weights
        if pop_targets is not None:
            self.pop_random_target, self.pop_mutant_target, self.pop_local_target, self.pop_crossed_target = pop_targets
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
            self.pop_local_weight != cls.pop_local_weight or
            self.pop_crossed_weight != cls.pop_crossed_weight):
            fields.append(f'pop_weights=({repr(self.pop_random_weight)},'
                          f'{repr(self.pop_random_weight)},'
                          f'{repr(self.pop_local_weight)},'
                          f'{repr(self.pop_crossed_weight)})')
        if (self.pop_random_target != cls.pop_random_target or
            self.pop_mutant_target != cls.pop_mutant_target or
            self.pop_local_target != cls.pop_local_target or
            self.pop_crossed_target != cls.pop_crossed_target):
            fields.append(f'pop_targets=({repr(self.pop_random_target)},'
                          f'{repr(self.pop_random_target)},'
                          f'{repr(self.pop_local_target)},'
                          f'{repr(self.pop_crossed_target)})')            
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
            self.pop_local_weight != cls.pop_local_weight or
            self.pop_crossed_weight != cls.pop_crossed_weight):
            fields.append(f'  pop_weights:\n'
                          f'    random:  {str(self.pop_random_weight)}\n'
                          f'    mutant:  {str(self.pop_random_weight)}\n'
                          f'    local:   {str(self.pop_local_weight)}\n'
                          f'    crossed: {str(self.pop_crossed_weight)}')
        if (self.pop_random_target != cls.pop_random_target or
            self.pop_mutant_target != cls.pop_mutant_target or
            self.pop_local_target != cls.pop_local_target or
            self.pop_crossed_target != cls.pop_crossed_target):
            fields.append(f'  pop_targets:\n'
                          f'    random:  {str(self.pop_random_target)}\n'
                          f'    mutant:  {str(self.pop_random_target)}\n'
                          f'    local:   {str(self.pop_local_target)}\n'
                          f'    crossed: {str(self.pop_crossed_target)}')
        if self.mutation_probability != cls.mutation_probability:
            fields.append(f'  mutation_probability: {str(self.mutation_probability)}')
        if self.crossover_probability != cls.crossover_probability:
            fields.append(f'  crossover_probability: {str(self.crossover_probability)}')
        sep = '\n'
        return f'{cls.__name__}\n{sep.join(fields)}'
            
    @classmethod
    def from_dict(cls, d):
        new_settings = cls.__new__(cls)
        new_settings.__dict__.update(d)
        return new_settings
        


def sweep_genetic(stage_fn, inits, neighbors, metrics, verbosity=3):
    if verbosity >= 1:
        print(f'Multi-sweep: sweeping over {max_inits!s} random initializations, and up to {max_retries!s} ignored points')

    improved, visited_points, gens, cfgs, frontier = sweep_random_init(stage_fn, inits, neighbors, metrics, verbosity=verbosity)

    visited_points = [(0, a, b) for a, b in visited_points]

    attempts = 0
    successes = 0
    failures = 0
    while successes < max_inits and failures < max_retries:
        if verbosity >= 2:
            print(f'\n == attempt {attempts+1!s}, {len(cfgs)!s} cfgs, {len(frontier)!s} elements in frontier ==\n')

        improved, new_visited_points, gens, cfgs, frontier = sweep_random_init(stage_fn, inits, neighbors, metrics, verbosity=verbosity,
                                                                               previous_sweep=(gens, cfgs, frontier), force_exploration=force_exploration)

        visited_points.extend((attempts, a, b) for a, b in new_visited_points)

        if verbosity >= 2:
            print(f'\n == finished attempt {attempts+1!s}, improved? {improved!s} ==\n')

        attempts += 1
        if improved:
            successes += 1
        else:
            failures += 1

    return gens, visited_points, frontier
