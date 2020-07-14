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
    fromtier_cfgs = set()

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
                return (False, *previous_sweep)

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

            if verbosity >= 2:
                print(f'dispatched {len(async_results)!s} evaluations for generation {gen_log[-1]!s}, {skipped!s} skipped')

            for i, (new_cfg, ares) in enumerate(async_results):
                new_res = ares.get()
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

    return (gen_log[-1] > 0), gen_log, all_cfgs, frontier


def sweep_multi(stage_fn, inits, neighbors, metrics, max_inits, max_retries,
                force_exploration=False, verbosity=3):
    if verbosity >= 1:
        print(f'Multi-sweep: sweeping over {max_inits!s} random initializations, and up to {max_retries!s} ignored points')

    improved, gens, cfgs, frontier = sweep_random_init(stage_fn, inits, neighbors, metrics, verbosity=verbosity)

    attempts = 0
    successes = 0
    failures = 0
    while successes < max_inits and failures < max_retries:
        if verbosity >= 2:
            print(f'\n == attempt {attempts+1!s}, {len(cfgs)!s} cfgs, {len(frontier)!s} elements in frontier ==\n')

        improved, gens, cfgs, frontier = sweep_random_init(stage_fn, inits, neighbors, metrics, verbosity=verbosity,
                                                           previous_sweep=(gens, cfgs, frontier), force_exploration=force_exploration)

        if verbosity >= 2:
            print(f'\n == finished attempt {attempts+1!s}, improved? {improved!s} ==\n')

        attempts += 1
        if improved:
            successes += 1
        else:
            failures += 1

    return gens, cfgs, frontier


def filter_frontier(frontier, metrics):
    new_metrics = [m for m in metrics if m is not None]

    new_frontier = []
    for data, measures in frontier:
        filtered_measures = tuple(meas for meas, m in zip(measures, metrics) if m is not None)
        _, new_frontier = update_frontier(new_frontier, (data, filtered_measures), new_metrics)

    return new_frontier
