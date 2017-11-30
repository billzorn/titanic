import os
import time
import multiprocessing

import z3
# z3.set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)
# print('using z3 {}'.format(z3.get_version_string()))

import fpcast as ast

default_lo_sort = 16
default_hi_sort = 32
default_rm = z3.RoundNearestTiesToEven()

# To allow for more understandable debug messages
class SolverID(object):
    def __init__(self):
        self.i = 0
        self.lock = multiprocessing.Lock()
    def __call__(self):
        with self.lock:
            self.i += 1
            return self.i
# Use this instance
sid = SolverID()

# Standardized variable names inside solver formulas. Also used to extract
# values from the models.
arglo_str = '_arglo_'
arghi_str = '_arghi_'
argsep = '_'

def arglo(i, name):
    return '{}{:d}{}{}'.format(arglo_str, i, argsep, name)
def arghi(i, name):
    return '{}{:d}{}{}'.format(arghi_str, i, argsep, name)

def get_arglo(s):
    if s.startswith(arglo_str):
        i_name = s[len(arglo_str):].split(argsep)
        return int(i_name[0]), argsep.join(i_name[1:])
    else:
        return None, None
def get_arghi(s):
    if s.startswith(arghi_str):
        i_name = s[len(arghi_str):].split(argsep)
        return int(i_name[0]), argsep.join(i_name[1:])
    else:
        return None, None

reslo = 'lo_result'
reshi = 'hi_result'
resexp = 'expected_result'
resulps = 'ulps'

# Serialize all information to dicts of strings so that results can be passed
# around easily in a multiprocess environment.
def serialize_model(m):
    return {str(k) : str(m[k]) for k in m}

def serialize_stats(stats):
    return {str(k) : str(v) for k, v in stats}

def describe_results(results):
    if results:
        descr, status, elapsed, model, stats = results
        header = 'Report for {}: ({:.2f}s)'.format(descr, elapsed)
        if status == 'sat':
            body = '  arguments:'
            nargs = 0
            arg_data = {}
            for k in model:
                i, argname = get_arglo(k)
                if i is not None:
                    arg_data[i] = (argname, model[k])
                    nargs += 1
            for i in range(nargs):
                argname, argval = arg_data[i]
                body += '\n    {} = {}'.format(argname, get_z3fp(argval))
            body += '\n  low precision: {}'.format(get_z3fp(model[reslo]))
            body += '\n  exp precision: {}'.format(get_z3fp(model[resexp]))
            body += '\n  full precision: {}'.format(get_z3fp(model[reshi]))
            if resulps in model:
                body += '\n    {} ulps'.format(model[resulps])
        else:
            body = '  {}'.format(status)
        footer = '  statistics:'
        for k in ['conflicts', 'decisions', 'max memory', 'restarts']:
            if k in stats:
                footer += '\n    {}: {}'.format(k, stats[k])
        return '\n'.join((header, body, footer))

# symbolic units in the last place difference

def z3fp_to_ordinal(x, sort, rm = default_rm):
    z3sort = ast.z3_sort(sort)
    x_prime = z3.fpToFP(rm, x, z3sort) if x.sort() != z3sort else x
    return z3.If(x_prime < z3.FPVal(0.0, x_prime.sort()),
                 -z3.fpToIEEEBV(-x_prime),
                 z3.fpToIEEEBV(z3.fpAbs(x_prime)))

def z3ulps(x, y, sort, rm = default_rm):
    xz = z3fp_to_ordinal(x, sort, rm=rm)
    yz = z3fp_to_ordinal(y, sort, rm=rm)
    return z3.If(xz < yz, yz - xz, xz - yz)

# temporary! - restricts prescision to at most float64
z3fp_constants = {
    '+oo' : float('+inf'),
    '-oo' : float('-inf'),
    'NaN' : float('nan'),
}
def get_z3fp(v):
    if v in z3fp_constants:
        return z3fp_constants[v]
    else:
        return float(eval(v))

class FPSolver(object):
    def __init__(self, core, lo_sort = default_lo_sort, hi_sort = default_hi_sort,
                 rm = default_rm, timeout_ms = 3600 * 1000, verbosity = 1):

        # debug identification
        self.sid = sid()

        self.core = core
        self.lo = ast.z3_sort(lo_sort)
        self.hi = ast.z3_sort(hi_sort)
        self.rm = rm
        # can be improved
        self.default_maxulps = (2 ** (self.lo.ebits() + self.lo.sbits())) - 1

        # these can be changed
        self.timeout_ms = timeout_ms
        self.verbosity = verbosity

        # store result of last query
        self.last_results = None

        # arguments and results
        self.lo_args = {}
        self.hi_args = {}
        for i, arg in enumerate(self.core.args):
            self.lo_args[arg] = z3.FP(arglo(i, arg), self.lo)
            self.hi_args[arg] = z3.FP(arghi(i, arg), self.hi)
        self.lo_result = z3.FP(reslo, self.lo)
        self.hi_result = z3.FP(reshi, self.hi)
        self.expected_result = z3.FP(resexp, self.lo)

        # ulps
        self.ulps_result = z3.BitVec(resulps, self.lo.ebits() + self.lo.sbits())

        self.solver = z3.Solver()
        self._init_solver()

    def _init_solver(self):
        # preconditions on domain
        if self.core.pre:
            self.solver.add(self.core.pre.apply_z3(self.lo_args, self.lo, rm=self.rm))

        # argument equivalence between sorts; i think all comparisons need to be ==, not fpEQ, to avoid -0.0 == 0.0 etc.
        for arg in self.core.args:
            self.solver.add(self.hi_args[arg] == z3.fpToFP(self.rm, self.lo_args[arg], self.hi))

        # expression evaluation
        self.solver.add(self.lo_result == self.core.e.apply_z3(self.lo_args, self.lo, rm=self.rm))
        self.solver.add(self.hi_result == self.core.e.apply_z3(self.hi_args, self.hi, rm=self.rm))

        # result downcast
        self.solver.add(self.expected_result == z3.fpToFP(self.rm, self.hi_result, self.lo))

    def _run_solver(self, *assumptions):
        self.solver.set('timeout', self.timeout_ms)
        start = time.time()
        status = self.solver.check(*assumptions)
        if status == z3.unknown:
            status = '{}: {}'.format(str(status), self.solver.reason_unknown())
        end = time.time()
        elapsed = end - start

        if status == z3.sat:
            return str(status), elapsed, serialize_model(self.solver.model()), serialize_stats(self.solver.statistics())
        else:
            return str(status), elapsed, {}, serialize_stats(self.solver.statistics())

    # it seems that these ways of incrementalizing make some queries impossible to solve...

    def query_pred(self, descr, e):
        if self.verbosity >= 1:
            print('Checking {} (solver on {:d}) ...'.format(descr, os.getpid()))

        b = z3.Bool(descr)
        self.solver.add(z3.Implies(b, e))

        results = self._run_solver(b)
        self.last_results = (descr, *results,)
        return self.last_results

    def query_push(self, descr, e):
        if self.verbosity >= 1:
            print('Checking {} (solver on {:d}) ...'.format(descr, os.getpid()))

        self.solver.push()
        self.solver.add(e)

        results = self._run_solver()

        self.solver.pop()

        self.last_results = (descr, *results,)
        return self.last_results

    # this isn't actually incremental, but it returns answers sometimes
    def query_reset(self, descr, e):
        if self.verbosity >= 1:
            print('Checking {} (solver on {:d}) ...'.format(descr, os.getpid()))

        self.solver.add(e)

        results = self._run_solver()

        self.solver.reset()
        self._init_solver()

        self.last_results = (descr, *results,)
        return self.last_results

    def query(self, descr, e):
        return self.query_reset(descr, e)

    def describe_last(self):
        if self.last_results:
            res = describe_results(self.last_results)
        else:
            res = 'No queries run.'
        if self.verbosity >= 1:
            print(res)
        return res

    def get_model(self):
        if self.last_results:
            descr, status, elapsed, model, stats = self.last_results
            if status == 'sat':
                return model
        return None

    # all queries are set up separately, so the solver can be reused

    def check_NaN(self, lo = None):
        if lo is True:
            descr = 'r=NaN'
            e = z3.And(z3.fpIsNaN(self.lo_result), z3.Not(z3.fpIsNaN(self.expected_result)))
        elif lo is False:
            descr = 'c=NaN'
            e = z3.And(z3.Not(z3.fpIsNaN(self.lo_result)), z3.fpIsNaN(self.expected_result))
        else:
            descr = 'NaN'
            e = z3.Or(z3.And(z3.fpIsNaN(self.lo_result), z3.Not(z3.fpIsNaN(self.expected_result))),
                      z3.And(z3.Not(z3.fpIsNaN(self.lo_result)), z3.fpIsNaN(self.expected_result)))
        return self.query(descr, e)

    def check_Inf(self, lo = None, neg = None):
        if lo is True:
            if neg is True:
                descr = 'r=-Inf'
                e = z3.And(z3.fpIsInf(self.lo_result), z3.fpIsNegative(self.lo_result), z3.Not(self.expected_result == self.lo_result))
            elif neg is False:
                descr = 'r=+Inf'
                e = z3.And(z3.fpIsInf(self.lo_result), z3.fpIsPositive(self.lo_result), z3.Not(self.expected_result == self.lo_result))
            else:
                descr = 'r=Inf'
                e = z3.And(z3.fpIsInf(self.lo_result), z3.Not(self.expected_result == self.lo_result))
        elif lo is False:
            if neg is True:
                descr = 'c=-Inf'
                e = z3.And(z3.fpIsInf(self.expected_result), z3.fpIsNegative(self.expected_result), z3.Not(self.expected_result == self.lo_result))
            elif neg is False:
                descr = 'c=+Inf'
                e = z3.And(z3.fpIsInf(self.expected_result), z3.fpIsPositive(self.expected_result), z3.Not(self.expected_result == self.lo_result))
            else:
                descr = 'c=Inf'
                e = z3.And(z3.fpIsInf(self.expected_result), z3.Not(self.expected_result == self.lo_result))
        else:
            if neg is True:
                descr = '-Inf'
                e = z3.And(z3.Or(z3.And(z3.fpIsInf(self.lo_result), z3.fpIsNegative(self.lo_result)),
                                 z3.And(z3.fpIsInf(self.expected_result), z3.fpIsNegative(self.expected_result))),
                           z3.Not(self.expected_result == self.lo_result))
            elif neg is False:
                descr = '+Inf'
                e = z3.And(z3.Or(z3.And(z3.fpIsInf(self.lo_result), z3.fpIsPositive(self.lo_result)),
                                 z3.And(z3.fpIsInf(self.expected_result), z3.fpIsPositive(self.expected_result))),
                           z3.Not(self.expected_result == self.lo_result))
            else:
                descr = 'Inf'
                e = z3.And(z3.Or(z3.fpIsInf(self.lo_result), z3.fpIsInf(self.expected_result)),
                           z3.Not(self.expected_result == self.lo_result))
        return self.query(descr, e)

    def check_zero(self, neg = None):
        if neg is True:
            descr = 'r=-0.0, c=+0.0'
            e = z3.And(self.lo_result == z3.FPVal('-0.0', self.lo), self.expected_result == z3.FPVal('+0.0', self.lo))
        elif neg is False:
            descr = 'r=+0.0, c=-0.0'
            e = z3.And(self.lo_result == z3.FPVal('+0.0', self.lo), self.expected_result == z3.FPVal('-0.0', self.lo))
        else:
            descr = 'r=-0.0, c=+0.0 OR r=+0.0, c=-0.0'
            e = z3.Or(z3.And(self.lo_result == z3.FPVal('+0.0', self.lo), self.expected_result == z3.FPVal('-0.0', self.lo)),
                      z3.And(self.lo_result == z3.FPVal('-0.0', self.lo), self.expected_result == z3.FPVal('+0.0', self.lo)))
        return self.query(descr, e)

    def _init_ulps(self):
        # we have independent checks for all bad behavior involving NaN, Inf, and mismatched signs at zero.
        # ulps inherently treat +-zero the same, so we only need explicit conditions about NaN and Inf
        self.solver.add(z3.Not(z3.fpIsNaN(self.lo_result)),
                        z3.Not(z3.fpIsNaN(self.expected_result)),
                        z3.Not(z3.fpIsInf(self.lo_result)),
                        z3.Not(z3.fpIsInf(self.expected_result)),
                        self.ulps_result == z3ulps(self.lo_result, self.expected_result, self.lo, rm=self.rm))

    def check_ulps(self, ulps):
        descr = '>= {:d} ulps'.format(ulps)
        e = z3.UGE(self.ulps_result, z3.BitVecVal(ulps, self.ulps_result.size()))
        self._init_ulps()
        return self.query_reset(descr, e)

    def ulps_incremental_begin(self):
        self._init_ulps()

    def ulps_incremental(self, ulps):
        descr = 'incremental >= {:d} ulps'.format(ulps)
        e = z3.UGE(self.ulps_result, z3.BitVecVal(ulps, self.ulps_result.size()))
        return self.query_push(descr, e)

    def ulps_incremental_end(self):
        self.solver.reset()
        self._init_solver()

    def binsearch_ulps(self, ulps_requested = None, incremental = False):
        ulps_target = 1
        ulps_lo = 1
        ulps_hi = ulps_requested
        ulps_scale = 2

        stored_results = [None, None]
        total_time = [0]

        def get_ulps(ulps):
            if incremental:
                results = self.ulps_incremental(ulps)
            else:
                results = self.check_ulps(ulps)
            descr, status, elapsed, model, stats = results
            total_time[0] += elapsed
            if status == 'sat':
                stored_results[0] = results
                return int(model[resulps])
            elif status == 'unsat':
                stored_results[1] = results
                return None
            else:
                self.describe_last()
                raise ValueError('Binsearch: unknown query')

        if self.verbosity >= 1:
            print('Binsearch ulps up to {}, incremental={} ...'.format(ulps_hi, incremental))

        if incremental:
            self.ulps_incremental_begin()

        if ulps_requested is None:
            # determine ulps hi with logsearch
            while ulps_hi is None:
                ulps = get_ulps(ulps_target)
                if ulps is None:
                    ulps_hi = ulps_target
                else:
                    ulps_lo = ulps
                    ulps_target = min(ulps_lo * ulps_scale, self.default_maxulps)
        else:
            # check requested number of ulps first
            ulps_target = ulps_requested
            ulps = get_ulps(ulps_target)
            if ulps is None:
                ulps_hi = ulps_target
            else:
                ulps_lo = ulps

        while ulps_hi > ulps_lo + 1:
            ulps_target = ((ulps_hi - ulps_lo) // 2) + ulps_lo
            ulps = get_ulps(ulps_target)
            if ulps is None:
                ulps_hi = ulps_target
            else:
                ulps_lo = ulps

        if incremental:
            self.ulps_incremental_end()

        if self.verbosity >= 1:
            print('Binsearch finished in {}s.'.format(total_time[0]))
            print(describe_results(stored_results[0]))
            print(describe_results(stored_results[1]))

        if stored_results[0]:
            self.last_results = stored_results[0]
            return int(stored_results[0][3][resulps])
        else:
            return None

    def ransearch_point(self, ulps, point):
        pass
        # needs ordinals


if __name__ == '__main__':
    import fpcparser

    simple_core_str = '''(FPCore (x)
 :name "NMSE example 3.1"
 :cite (hamming-1987)
 :pre (>= x 0)
 (- (sqrt (+ x 1)) (sqrt x)))
'''
    core_str = '''(FPCore (a b c)
 :name "NMSE p42, positive"
 :cite (hamming-1987)
 :pre (and (>= (* b b) (* 4 (* a c))) (!= a 0))
 (/ (+ (- b) (sqrt (- (* b b) (* 4 (* a c))))) (* 2 a)))
'''
    core = fpcparser.compile(simple_core_str)[0]

    s = FPSolver(core, timeout_ms = 10000)

    s.check_NaN()
    s.describe_last()

    s2 = FPSolver(fpcparser.compile(simple_core_str)[0], timeout_ms = 20000)

    s2.check_NaN(lo = True)
    s2.describe_last()
    s2.timeout_ms = 50
    s2.check_NaN(lo = False)
    s2.describe_last()
    s2.timeout_ms = 10000
    s2.check_NaN(lo = False)
    s2.describe_last()

    s.check_Inf(lo = True, neg = True)
    s.describe_last()
    s.check_Inf(lo = True, neg = False)
    s.describe_last()
    s.check_Inf(lo = True, neg = None)
    s.describe_last()
    s.check_Inf(lo = False, neg = True)
    s.describe_last()
    s.check_Inf(lo = False, neg = False)
    s.describe_last()
    s.check_Inf(lo = False, neg = None)
    s.describe_last()
    s.check_Inf(lo = None, neg = True)
    s.describe_last()
    s.check_Inf(lo = None, neg = False)
    s.describe_last()
    s.check_Inf(lo = None, neg = None)
    s.describe_last()

    s.check_zero(neg = True)
    s.describe_last()
    s.check_zero(neg = False)
    s.describe_last()
    s.check_zero(neg = None)
    s.describe_last()

    s.check_ulps(10)
    s.describe_last()
    s.check_ulps(1000)
    s.describe_last()
    s.check_ulps(33000)
    s.describe_last()
    s.check_ulps(65535)
    s.describe_last()

    s.timeout_ms = 5000
    s.ulps_incremental_begin()
    s.ulps_incremental(1)
    s.describe_last()

    s.timeout_ms = 40000
    s.ulps_incremental(1)
    s.describe_last()
    s.ulps_incremental(10)
    s.describe_last()
    s.ulps_incremental(1000)
    s.describe_last()
    s.ulps_incremental(2000)
    s.describe_last()
    s.ulps_incremental(10000)
    s.describe_last()
    s.ulps_incremental(33000)
    s.describe_last()
    s.ulps_incremental_end()

    s.check_Inf()
    s.describe_last()

    s.timeout_ms = 3600 * 1000
    ur = s.binsearch_ulps()
    ui = s.binsearch_ulps(incremental=True)

    print(ur, ui, ur == ui)
