"""Dynamic analysis for fpcore interpreters."""


from ..titanic import utils
from ..fpbench import fpcast as ast
from ..fpbench import fpcparser


class AnalysisError(utils.TitanicError):
    """Titanic dynamic analysis error."""


def bitcost(x):
    try:
        return x.ctx.nbits
    except AttributeError:
        return 0


class DefaultAnalysis(object):
    """FPCore default run-time analysis.
    Tracks execution counts and contexts per node of the AST.
    Nodes are tracked by id(), as value nodes support equality
    on their values and may not be otherwise distinguishable.
    """

    def __init__(self):
        self.node_map = {}

    def track(self, e, ctx, inputs, result):
        if not (isinstance(e, ast.ValueExpr) or isinstance(e, ast.Ctx)):
            eid = id(e)
            if eid not in self.node_map:
                self.node_map[eid] = DefaultRecord(e)
            self.node_map[eid].record(ctx)

    def report(self):
        s = 'Default titanic analysis:\n'
        lines = []
        for k, record in self.node_map.items():
            line = '  ' + str(record.e.depth_limit(3)) + ' : ' + str(record.evals)
            lines.append(line)
        return s + '\n'.join(lines)

class DefaultRecord(object):
    """Analysis record for a single node in the ast."""

    def __init__(self, e):
        self.e = e
        self.ctxs = {}
        self.evals = 0

    def record(self, ctx):
        s = ctx.propstr()
        if s in self.ctxs:
            self.ctxs[s] += 1
        else:
            self.ctxs[s] = 1
        self.evals += 1

    def to_props(self):
        ctx_counts = ['(({:s}) {:d})'.format(propstr, count) for propstr, count in self.ctxs.items()]
        s = ':titanic-eval-count {:d} :titanic-eval-ctxs ({:s})'.format(self.evals, ' '.join(ctx_counts))
        return fpcparser.read_props(s)


class BitcostAnalysis(object):
    """FPCore bitcost analysis.
    Tracks bits consumed and produced by operations.
    """

    def __init__(self):
        self.bits_constant = 0
        self.bits_variable = 0
        self.bits_requested = 0
        self.bits_computed = 0
        self.bits_referenced = 0
        self.bits_quantized = 0

    def track(self, e, ctx, inputs, result):
        if isinstance(e, ast.NaryExpr):
            if isinstance(e, ast.Dim) or isinstance(e, ast.Size):
                pass
            elif isinstance(e, ast.UnknownOperator):
                pass
            elif isinstance(e, ast.Ref):
                self.bits_referenced += bitcost(result)
            elif isinstance(e, ast.Cast):
                inbits = bitcost(inputs[0])
                outbits = bitcost(result)
                if inbits != 0 and outbits != 0:
                    self.bits_quantized += (outbits - inbits)
            else:
                if inputs:
                    for arg in inputs:
                        self.bits_requested += bitcost(arg)
                self.bits_computed += bitcost(result)
        elif isinstance(e, ast.Val):
            self.bits_constant += bitcost(result)
        elif isinstance(e, ast.Var):
            self.bits_variable += bitcost(result)
        elif isinstance(e, ast.ControlExpr):
            pass
        else:
            pass

    def report(self):
        s = 'Titanic bitcost analysis:\n'
        s += '  {:d} bits requested as inputs by operations\n'.format(self.bits_requested)
        s += '  {:d} bits computed as outputs by operations\n'.format(self.bits_computed)
        s += '  {:d} net bits quantized by format conversion\n'.format(self.bits_quantized)
        s += '  {:d} bits read from constants\n'.format(self.bits_constant)
        s += '  {:d} bits read from bound variables\n'.format(self.bits_variable)
        s += '  {:d} bits read from arrays\n'.format(self.bits_referenced)
        return s


class RangeAnalysis(object):
    """FPCore analysis for dynamic range of values that come out of operations
    """

    def __init__(self):
        self.node_map = {}
        self.report_from = set()

    def track(self, e, ctx, inputs, result):
        eid = id(e)
        if eid not in self.node_map:
            self.node_map[eid] = RangeRecord(e)
        self.node_map[eid].record(result)
        if isinstance(e, ast.Ctx):
            if 'report' in e.props and e.props['report'] == 'here':
                self.report_from.add(eid)

    def report(self):
        return [(str(self.node_map[eid].e.depth_limit(3)), self.node_map[eid].exponents) for eid in self.report_from]

class RangeRecord(object):
    """Analysis record for a single node in the ast."""

    def __init__(self, e):
        self.e = e
        self.exponents = []

    def record(self, result):
        self.exponents.append(result.e)
