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
        if not isinstance(e, ast.Ctx):
            eid = id(e)
            if eid not in self.node_map:
                self.node_map[eid] = DefaultRecord(e)
            self.node_map[eid].record(ctx)

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
        for propstr, count in self.ctxs.items():
            props = fpcparser.read_props(propstr)
            print(propstr, count, props)


class BitcostAnalysis(object):
    """FPCore bitcost analysis.
    Tracks bits consumed and produced by operations.
    """

    def __init__(self):
        self.bits_constant = 0
        self.bits_requested = 0
        self.bits_computed = 0
        self.bits_referenced = 0
        self.bits_quantized = 0

    def track(self, e, ctx, inputs, result):
        if isinstance(e, ast.NaryExpr):
            if isinstance(e, ast.Dim) or isinstance(e, ast.Size):
                pass
            elif isinstance(e, ast.Ref):
                self.bits_referenced += bitcost(result)
            elif isinstance(e, ast.Cast):
                inbits = bitcost(inputs[0])
                outbits = bitcost(result)
                if inbits != 0 and outbits != 0:
                    quantized += (outbits - inbits)
            else:
                if inputs:
                    for arg in inputs:
                        self.bits_requested += bitcost(arg)
                self.bits_computed += bitcost(result)
        elif isinstance(e, ast.ValueExpr):
            self.bits_constant += bitcost(result)
        elif isinstance(e, ast.ControlExpr):
            pass
        else:
            pass
