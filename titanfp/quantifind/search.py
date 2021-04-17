"""Common code for parameter search."""

DONT_HAMMER_LEVIATHAN=32

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
import collections
import operator
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
    pop_local_weight = 1
    pop_crossed_weight = 0

    pop_random_target = None
    pop_mutant_target = None
    pop_local_target = None
    pop_crossed_target = None

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
            self.pop_local_weight = 1
            self.pop_crossed_weight = 0
            self.pop_random_target = None
            self.pop_mutant_target = None
            self.pop_local_target = None
            self.pop_crossed_target = None
            self.pop_weight_scale = 0
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
            self.pop_random_weight, self.pop_mutant_weight, self.pop_local_weight, self.pop_crossed_weight = pop_weights
        if pop_targets is not None:
            self.pop_random_target, self.pop_mutant_target, self.pop_local_target, self.pop_crossed_target = pop_targets
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
        # (stored by index in the cache),
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

        # count of new frontier points found for each generation
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
            f'  total cfgs: {len(self.cache)}\n'
            f'  horizon:    {len(self.horizon)}\n'
            f'  history:    {len(self.history)}\n'
            f'  frontier:   {len(self.frontier)}\n'
            f'  running for {len(self.generations)} generations'
        ) + (f'\n  {len(self.additional_data)} additional data records' if len(self.additional_data) > 0 else '')

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
        self.history.append(cfg)
        record[0] = hidx

        keep, new_frontier = update_frontier(self.frontier, result, metric_fns)
        if keep:
            fidx = len(self.frontier_log)
            # TODO: we need update_frontier to track what got removed
            self.frontier_log.append([result, [], None])
            record[1] = fidx

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
                 settings=None, state=None, cores=DONT_HAMMER_LEVIATHAN, batch=None, retry_attempts=1,
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
                old_cfg, _ = random.sample(self.state.frontier, 1)
                # mutate
                p = self.settings.mutation_probability
                cfg = tuple(f() if random.random() < p else x
                            for x, f in zip(old_cfg, self.init_fns))
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
                    if len(batch) >= max_size:
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
                cfg1, cfg2 = map(operator.itemgetter(0), random.sample(self.state.frontier, 2))
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
            return min(max(min_bound, unbounded), max_bound)
        else:
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
        pop_local_weight = settings.pop_local_weight
        pop_crossed_weight = settings.pop_crossed_weight
        pop_weight = pop_random_weight + pop_mutant_weight +  pop_local_weight + pop_crossed_weight
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
        for cfg, ares in zip(cfgs, async_results):
            qos = ares.get()
            result = cfg, qos
            if self.state.commit_to_history(result, self.metric_fns, verbose=self.verbosity>=3):
                new_frontier_points += 1
                if self.verbosity >= 3:
                    print('!', end='', flush=True)
            else:
                if self.verbosity >= 3:
                    print('.', end='', flush=True)
        if self.verbosity >= 3:
            print(flush=True)

        if self.verbosity >= 2:
            print(f'    processed {len(async_results)} configurations, added {new_frontier_points} to the frontier.')
        return new_frontier_points

    def run_generation(self, pool=None):
        """Run the next generation of configurations currently on the horizon.
        Handles the generation records in the state.
        """
        horizon_size = len(self.state.horizon)
        gen_idx = len(self.state.generations)
        self.state.generations.append(0)

        if self.verbosity >= 1:
            print(f'  Evaluating the horizon for generation {len(self.state.generations)}...')

        if pool is None:
            pool = self.pool
        if pool is None:
            with multiprocessing.Pool(self.cores) as pool:
                while len(self.state.horizon) > 0:
                        new_frontier_points = self.process_batch(pool)
                        self.state.generations[gen_idx] += new_frontier_points
        else:
            while len(self.state.horizon) > 0:
                new_frontier_points = self.process_batch(pool)
                self.state.generations[gen_idx] += new_frontier_points

        if self.verbosity >= 1:
            print(f'  Evaluated {horizon_size} configurations for generation {gen_idx}, adding {self.state.generations[gen_idx]} to the frontier.')
        return self.state.generations[gen_idx]

    def cleanup_horizon(self, pool=None):
        """Empty the horizon, in case only part of a generation was evaluated.
        This is almost the same as running a generation,
        but it doesn't append a new generation record.
        """
        horizon_size = len(self.state.horizon)
        if horizon_size == 0:
            return 0

        gen_idx = len(self.state.generations) - 1
        if gen_idx < 0:
            gen_idx = 0
            self.state.generations.append(0)

        elif self.verbosity >= 1:
            print(f'  Cleaning up {horizon_size} configurations left on the horizon at generation {gen_idx}...')

        total_new_points = 0
        if pool is None:
            pool = self.pool
        if pool is None:
            with multiprocessing.Pool(self.cores) as pool:
                while len(self.state.horizon) > 0:
                    new_frontier_points = self.process_batch(pool)
                    total_new_points += new_frontier_points
                    self.state.generations[gen_idx] += new_frontier_points
        else:
            while len(self.state.horizon) > 0:
                new_frontier_points = self.process_batch(pool)
                total_new_points += new_frontier_points
                self.state.generations[gen_idx] += new_frontier_points

        if self.verbosity >= 1:
            print(f'  Cleaned up {horizon_size} configurations for generation {gen_idx}, adding {total_new_points} to the frontier.')
        return total_new_points

    def run_search(self):
        """Run the Pareto frontier exploration sweep!"""
        if self.verbosity >= 0:
            print('Running QuantiFind sweep...')
            if self.verbosity >= 2:
                print(self)

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
                    return self.state.frontier
            else:
                if self.verbosity >= 3:
                    print('The frontier changed.')
                    print_frontier(self.state.frontier)
