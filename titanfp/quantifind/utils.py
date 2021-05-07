"""Sweep helpers."""

import os
import sys
import random
import json

from ..fpbench import fpcparser
from ..arithmetic import ieee754, posit, evalctx


# convenient rounding contexts

f8 = ieee754.ieee_ctx(3, 8)
f16 = ieee754.ieee_ctx(5, 16)
bf16 = ieee754.ieee_ctx(8, 16)
f32 = ieee754.ieee_ctx(8, 32)
f64 = ieee754.ieee_ctx(11, 64)

f4k = ieee754.ieee_ctx(20, 4096)

posit8_0 = posit.posit_ctx(0, 8)
posit16_1 = posit.posit_ctx(1, 16)
posit16_2 = posit.posit_ctx(2, 16)
posit32_2 = posit.posit_ctx(2, 32)
posit64_3 = posit.posit_ctx(3, 64)

float_basecase = (f8, f16, bf16, f32)
posit_basecase = (posit8_0, posit16_1, posit16_2, posit32_2)

def describe_ctx(ctx):
    if isinstance(ctx, evalctx.IEEECtx):
        if ctx.es == 3 and ctx.nbits == 8:
            return 'float8'
        elif ctx.es == 5 and ctx.nbits == 16:
            return 'float16'
        elif ctx.es == 8 and ctx.nbits == 16:
            return 'bfloat16'
        elif ctx.es == 8 and ctx.nbits == 32:
            return 'float32'
        elif ctx.es == 11 and ctx.nbits == 32:
            return 'float64'
        else:
            return f'(float {ctx.es!s} {ctx.nbits!s})'

    elif isinstance(ctx, evalctx.PositCtx):
        if ctx.es == 0 and ctx.nbits == 8:
            return 'posit8_0'
        elif ctx.es == 1 and ctx.nbits == 16:
            return 'posit16_1'
        elif ctx.es == 2 and ctx.nbits == 16:
            return 'posit16_2'
        elif ctx.es == 2 and ctx.nbits == 32:
            return 'posit32_2'
        elif ctx.es == 3 and ctx.nbits == 64:
            return 'posit64_3'
        else:
            return f'(posit {ctx.es!s} {ctx.nbits!s})'

    else:
        return ctx.propstr()


# import Pareto frontier manipulation logic

def compare_with_metrics(x1, x2, metric_fns, distinguish_incomparable=False):
    """Compare two points x1 and x2 by calling an element of metric_fns
    for each coordinate.
    If a metric_fn is None, then ignore the coordinate.

    By default, return -1 if x1 is strictly less, 1 if x1 is strictly greater,
    and 0 if the points are equal or unordered.

    If distinguish_incomparable is True,
    only return 0 for points that are equal;
    for unordered but not equal ("incomparable") points return None instead.
    """
    lt = False
    gt = False
    for a1, a2, cmp_lt in zip(x1, x2, metric_fns):
        if cmp_lt is not None:
            if cmp_lt(a1, a2):
                lt = True
                if gt:
                    break
            elif cmp_lt(a2, a1):
                gt = True
                if lt:
                    break

    if lt:
        if gt:
            if distinguish_incomparable:
                return None
            else:
                return 0
        else: # not gt
            return -1

    else: # not lt
        if gt:
            return 1
        else: # not gt
            return 0

def update_frontier(frontier, result, metric_fns, check=False):
    """Given a Pareto frontier and a new point,
    try to add that point to the frontier.

    Return a triple of:
      whether or not the new point changed the frontier,
      the new frontier,
      and a list of all the points that were removed from the frontier in the process.

    Assuming a well-formed frontier, it should be impossible to remove points
    if the new point is not on the new frontier;
    passing check=True avoids exiting early based on this assumption.
    """
    cfg, qos = result

    keep = True
    new_frontier = []
    removed_points = []

    for i, frontier_result in enumerate(frontier):
        frontier_cfg, frontier_qos = frontier_result
        comparison = compare_with_metrics(qos, frontier_qos, metric_fns, distinguish_incomparable=True)

        if comparison:
            # the comparison is 1 or -1; these points are ordered

            if comparison > 0:
                # existing point is better; we don't need the new one
                new_frontier.append(frontier_result)
                keep = False
            else: # comparison < 0
                # existing point is strictly worse; get rid of it
                removed_points.append(frontier_result)

        else:
            # the comparison is 0 or None; the points are unordered but may be equal

            # we always keep the existing point in this case
            new_frontier.append(frontier_result)

            if comparison == 0 and cfg == frontier_cfg:
                # if we have seen exactly this same point, avoid duplicating it
                keep = False

        if not (check or keep):
            # we can return early, assuming this was indeed a Pareto frontier to begin with;
            # keep the rest of the points
            new_frontier.extend(frontier[i+1:])
            return keep, new_frontier, removed_points

    if keep:
        new_frontier.append(result)
    return keep, new_frontier, removed_points

def reconstruct_frontier(frontier, metric_fns, check=False, verbose=True):
    """Reconstruct the given frontier, possibly using a different set of metrics.
    Pass the check option along to the underlying update_frontier function.
    """
    new_frontier = []
    all_removed = []
    for result in frontier:
        changed, next_frontier, removed_points = update_frontier(new_frontier, result, metric_fns, check=check)
        if check and verbose and not changed:
            print(f'-- point {repr(result)} is not on the frontier --')
        new_frontier = next_frontier
        all_removed.extend(removed_points)
    return new_frontier, all_removed

def merge_frontiers(frontier1, frontier2, metric_fns, check=False):
    """Combine two frontiers into a single new frontier.
    Equivalent to reconstructing the two frontiers appended together,
    but more efficient.
    """
    if len(frontier1) >= len(frontier2):
        larger = frontier1
        smaller = frontier2
    else:
        larger = frontier2
        smaller = frontier1

    if check:
        frontier, all_removed = reconstruct_frontier(larger, metric_fns, check=check)
    else:
        frontier, all_removed = larger, []

    for result in smaller:
        changed, frontier, removed = update_frontier(frontier, result, metric_fns, check=check)
        all_removed.extend(removed)

    return frontier, all_removed

def filter_frontier(frontier, pred_fns, cfg_pred_fns=None):
    """Filter out points from the frontier that don't pass all the pred_fns.
    If a pred_fn is None, ignore that metric.
    """
    new_frontier = []
    for result in frontier:
        cfg, qos = result
        keep = True
        if not all(f(x) for x, f in zip(qos, pred_fns) if f is not None):
            keep = False
        if cfg_pred_fns is not None:
            if not all(f(x) for x, f in zip(cfg, cfg_pred_fns) if f is not None):
                keep = False
        if keep:
            new_frontier.append(result)
    return new_frontier

def check_frontier(frontier, metric_fns, verbose=True):
    """Quadratically check the frontier for well-formedness,
    explaining any discrepancies and returning a count;
    this should only be used for debugging.
    """
    badness = 0
    for i1 in range(len(frontier)):
        for i2 in range(i1+1, len(frontier)):
            res1, res2 = frontier[i1], frontier[i2]
            cfg1, qos1 = res1
            cfg2, qos2 = res2
            comparison = compare_with_metrics(qos1, qos2, metric_fns, distinguish_incomparable=True)
            if comparison:
                badness += 1
                if verbose:
                    if comparison < 0:
                        print(f'-- Not a Pareto frontier! point {repr(res1)} < {repr(res2)} --')
                    else: # comparison > 0
                        print(f'-- Not a Pareto frontier! point {repr(res2)} < {repr(res1)} --')
            else:
                if comparison == 0 and cfg1 == cfg2:
                    badness += 1
                    if verbose:
                        print(f'-- Duplicate point {repr(res1)} at index {i1} and {i2} --')
    return badness

def print_frontier(frontier):
    print('[')
    for frontier_data, frontier_m in frontier:
        print(f'    ({frontier_data!r}, {frontier_m!r}),')
    print(']')


# safe json logging (suitable for calling in a thread)

def log_and_copy(data, fname, work_dir='.tmp', target_dir='.', link=None,
                 cleanup_re=None, keep_files=0, key=None):
    """Write data as json to fname.
    Do this in a way that is safe for async readers,
    i.e. by writing a temporary file first (in work_dir)
    and then renaming it to that actual fname in the target dir.

    If link is not None, also create a symlink to the final file.
    The path for the link is independent of work_dir and target_dir.

    If cleanup_re is specified, then remove all but keep_files
    with names matching the re from the target dir,
    keeping the most recent files if key is None,
    or otherwise the "greatest" files sorting by that key.
    """

    # write in temporary directory
    os.makedirs(work_dir, exist_ok=True)
    work_path = os.path.join(work_dir, fname)

    with open(work_path, 'wt') as f:
        json.dump(data, f, indent=None, separators=(',', ':'))
        print(file=f, flush=True)

    # check for files to cleanup, then rename and optionally link
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, fname)
    if cleanup_re is not None:
        old_files = os.listdir(target_dir)
    os.replace(work_path, target_path)

    if link is not None:
        link_dir = os.path.dirname(link)
        link_target = os.path.relpath(target_path, link_dir)
        # borrow work_path again for a temporary link
        os.symlink(link_target, work_path)
        os.replace(work_path, link)

    # cleanup
    if cleanup_re is not None:
        to_clean = [name for name in old_files if cleanup_re.fullmatch(name)]
        to_clean.sort(key=key)
        for name in to_clean[:len(to_clean)-keep_files]:
            clean_path = os.path.join(target_dir, name)
            os.remove(clean_path)

    # this may or may not actually be a good idea...
    if sys.platform.startswith('linux'):
        os.sync()


# other random stuff

def linear_ulps(x, y):
    smaller_n = min(x.n, y.n)
    x_offset = x.n - smaller_n
    y_offset = y.n - smaller_n

    x_m = x.m << x_offset
    y_m = y.m << y_offset

    return x_m - y_m

def load_cores(interpreter, cores, analyses=None):
    if isinstance(cores, str):
        cores = fpcparser.compile(cores)

    main = cores[-1]
    for core in cores:
        interpreter.register_function(core)
        if core.ident and core.ident.lower() == 'main':
            main = core

    if analyses:
        interpreter.analyses = analyses

    return main


def neighborhood(lo, hi, near):
    def neighbors(x):
        for n in range(x-near, x+near+1):
            if lo <= n <= hi:
                yield n
    return neighbors

def static_neighborhood(v):
    def init_static():
        return v
    def neighbor_static(x):
        yield v
    return init_static, neighbor_static

def integer_neighborhood(lo, hi, near):
    def init_random():
        return random.randint(lo, hi)
    return init_random, neighborhood(lo, hi, near)


def jsonlog(fname, gens, cfgs, frontier, settings=None):
    data = {'generations' : gens,
            'configs' : list(cfgs),
            'frontier' : frontier}
    if settings:
        data['settings'] = str(settings)

    with open(fname, 'wt') as f:
        json.dump(data, f)
        print(file=f, flush=True)
