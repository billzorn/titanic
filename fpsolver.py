import os
import time

import z3
# z3.set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)
# print('using z3 {}'.format(z3.get_version_string()))

import mathlib
from fpcore import FPCore

default_lo_sort = 16
default_hi_sort = 32

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
                i, argname = mathlib.get_arglo(k)
                if i is not None:
                    arg_data[i] = (argname, model[k])
                    nargs += 1
            for i in range(nargs):
                argname, argval = arg_data[i]
                body += '\n    {} = {}'.format(argname, mathlib.get_z3fp(argval))
            body += '\n  low precision: {}'.format(mathlib.get_z3fp(model[mathlib.reslo]))
            body += '\n  exp precision: {}'.format(mathlib.get_z3fp(model[mathlib.resexp]))
            body += '\n  full precision: {}'.format(mathlib.get_z3fp(model[mathlib.reshi]))
            if mathlib.resulps in model:
                body += '\n    {} ulps'.format(model[mathlib.resulps])
        else:
            body = '  {}'.format(status)
        footer = '  statistics:'
        for k in ['conflicts', 'decisions', 'max memory', 'restarts']:
            if k in stats:
                footer += '\n    {}: {}'.format(k, stats[k])
        return '\n'.join((header, body, footer))

class FPSolver(object):
    def __init__(self, core, lo_sort = default_lo_sort, hi_sort = default_hi_sort,
                 rm = mathlib.default_rm, timeout_ms = 3600 * 1000, verbosity = 1):

        # don't change these after initialization
        if not isinstance(core, FPCore):
            self.core = FPCore(core)
        else:
            self.core = core
        self.lo = mathlib.z3_sort(lo_sort)
        self.hi = mathlib.z3_sort(hi_sort)
        self.rm = rm

        # these can be changed
        self.timeout_ms = timeout_ms
        self.verbosity = verbosity

        # store result of last query
        self.last_results = None

        # arguments and results
        self.lo_args = {}
        self.hi_args = {}
        for i, arg in enumerate(self.core.args):
            self.lo_args[arg] = z3.FP(mathlib.arglo(i, arg), self.lo)
            self.hi_args[arg] = z3.FP(mathlib.arghi(i, arg), self.hi)
        self.lo_result = z3.FP(mathlib.reslo, self.lo)
        self.hi_result = z3.FP(mathlib.reshi, self.hi)
        self.expected_result = z3.FP(mathlib.resexp, self.lo)

        # ulps
        self.ulps_result = z3.BitVec(mathlib.resulps, self.lo.ebits() + self.lo.sbits())

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
        self.solver.add(self.lo_result == self.core.expr.apply_z3(self.lo_args, self.lo, rm=self.rm))
        self.solver.add(self.hi_result == self.core.expr.apply_z3(self.hi_args, self.hi, rm=self.rm))

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
            print(describe_results(self.last_results))
        else:
            print('No queries run.')

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
        self.query(descr, e)

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
        self.query(descr, e)

    def _init_ulps(self):
        # we have independent checks for all bad behavior involving NaN, Inf, and mismatched signs at zero.
        # ulps inherently treat +-zero the same, so we only need explicit conditions about NaN and Inf
        self.solver.add(z3.Not(z3.fpIsNaN(self.lo_result)),
                        z3.Not(z3.fpIsNaN(self.expected_result)),
                        z3.Not(z3.fpIsInf(self.lo_result)),
                        z3.Not(z3.fpIsInf(self.expected_result)),
                        self.ulps_result == mathlib.z3ulps(self.lo_result, self.expected_result, self.lo, rm=self.rm))

    def check_ulps(self, ulps):
        descr = '>= {:d} ulps'.format(ulps)
        e = z3.UGE(self.ulps_result, z3.BitVecVal(ulps, self.ulps_result.size()))
        self._init_ulps()
        self.query_reset(descr, e)

    def ulps_incremental_begin(self):
        self._init_ulps()

    def ulps_incremental(self, ulps):
        descr = 'incremental >= {:d} ulps'.format(ulps)
        e = z3.UGE(self.ulps_result, z3.BitVecVal(ulps, self.ulps_result.size()))
        self.query_push(descr, e)

    def ulps_incremental_end(self):
        self.solver.reset()
        self._init_solver()


if __name__ == '__main__':
    from fpcore import FPCore

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
    core = FPCore(core_str)

    s = FPSolver(core, timeout_ms = 10000)

    s.check_NaN()
    s.describe_last()

    s2 = FPSolver(core_str, timeout_ms = 20000)

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
