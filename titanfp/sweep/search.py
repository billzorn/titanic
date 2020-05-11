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

    # probably don't need this?
    # s = range(len(metrics))
    # powerset = itertools.chain.from_iterable(
    #     itertools.combinations(s, r) for r in range(len(s)+1)
    # )

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
    for frontier_data, frontier_m in frontier:
        result_data, result_m = result
        comparison = compare_results(result_m, frontier_m, metrics)

        if comparison is None:
            # the points are incomparable; keep both
            new_frontier.append((frontier_data, frontier_m))


        elif comparison >= 0:
            # some existing result is at least as good as the new one;
            # we don't need the new one
            new_frontier.append((frontier_data, frontier_m))
            keep = False

        # else: # comparison < 0
        #     pass
        #     # new result is less, which is strictly better;
        #     # keep it and throw out this result

    if keep:
        new_frontier.append(result)

    #check_frontier(new_frontier, metrics)

    return keep, new_frontier


def check_frontier(frontier, metrics):

    broken = False
    new_frontier = []

    for i1 in range(len(frontier)):
        keep = True

        for i2 in range(i1+1, len(frontier)):

            res1_data, res1_m = frontier[i1]
            res2_data, res2_m = frontier[i2]

            comparison = compare_results(res1_m, res2_m, metrics)

            if comparison != None:
                broken = True
                print('how did this get here?')
                print(i1, repr(frontier[i1]))
                print(i2, repr(frontier[i2]))
                print(flush=True)

                if comparison >= 0:
                    keep = False
                    break

        if keep:
            new_frontier.append(frontier[i1])

    return broken, new_frontier


def print_frontier(frontier):
    print('[')
    for frontier_data, frontier_m in frontier:
        print(f'    ({frontier_data!r}, {frontier_m!r}),')
    print(']')


def sweep_random_init(stage_fn, inits, neighbors, metrics, verbose=3):
    initial_cfg = [f() for f in inits]
    initial_result = stage_fn(*initial_cfg)

    frontier = [(initial_cfg, initial_result)]
    improved_frontier = True
    gen_number = 0

    with multiprocessing.Pool() as pool:
        while improved_frontier:
            improved_frontier = False

            if verbose:
                print(f'generation {gen_number!s}: ')
                print_frontier(frontier)
                print(flush=True)

            gen_number += 1

            async_results = []
            for cfg, result in frontier:
                for i in range(len(cfg)):
                    x = cfg[i]
                    f = neighbors[i]
                    for new_x in f(x):
                        new_cfg = list(cfg)
                        new_cfg[i] = new_x
                        async_results.append((new_cfg, pool.apply_async(stage_fn, new_cfg)))

            if verbose >= 2:
                print(f'dispatched {len(async_results)!s} evaluations for generation {gen_number!s}')

            for i, (new_cfg, ares) in enumerate(async_results):
                new_res = ares.get()
                updated, frontier = update_frontier(frontier, (new_cfg, new_res), metrics)
                improved_frontier |= updated

                if verbose >= 3:
                    print(f' -- {i+1!s} -- ran {new_cfg!r}, got {new_res!r}')
                    if updated:
                        print('The frontier changed:')
                        print_frontier(frontier)

    if verbose:
        print(f'improvement stopped at generation {gen_number!s}: ')
        print_frontier(frontier)
        print(flush=True)

    return frontier
