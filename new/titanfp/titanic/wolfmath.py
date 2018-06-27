"""Interface with Wolfram Mathematica."""

import re

import pexpect
from pexpect.replwrap import REPLWrapper

from .integral import bitmask
from .sinking import Sink
from . import ops


def digital_to_math(x):
    if x.isnan:
        raise ValueError('math does not have NaN')
    elif x.isinf:
        raise ValueError('math does not have inf')
    elif x.is_zero():
        return '0'
    elif x.is_integer() and x.exp < 0:
        return str(x.m >> -x.exp)
    elif x.is_integer() and x.exp < 32:
        return str(x.m << x.exp)
    else:
        return '(' + str(x.m) + '*2^' + str(x.exp) + ')'


def math_to_digital(digits):
    m = _mathdigits.match(digits)
    if m is None:
        raise ValueError('unable to get digits, math returned:\n{}'.format(digits))

    if len(m.group(1)) == 0:
        c = 0
    elif not m.group(1).startswith('1'):
        if '1' in m.group(1):
            raise ValueError('bad digits? got:\n{}'.format(digits))
        else:
            c = 0
    else:
        c = int(_digitsep.sub('', m.group(1)), base=2)

    exp = int(m.group(2)) - c.bit_length()
    negative = m.group(3) == 'True'
    inexact = m.group(4) == 'True'
    if inexact:
        # the result is always truncated toward zero
        rc = 1
    else:
        rc = 0

    return Sink(
        negative=negative,
        c=c,
        exp=exp,
        inexact=inexact,
        rc=rc,
    )


def _remainder(x1, x2):
    raise ValueError('remainder: unimplemented')

def _nearbyint(x):
    raise ValueError('nearbyint: unimplemented')

def _round(x):
    raise ValueError('round: unimplemented')

math_ops = [
    lambda x1, x2: '(' + x1 + ' + ' + x2 + ')',
    lambda x1, x2: '(' + x1 + ' - ' + x2 + ')',
    lambda x1, x2: '(' + x1 + ' * ' + x2 + ')',
    lambda x1, x2: '(' + x1 + ' / ' + x2 + ')',
    lambda x: '(-' + x + ')',
    lambda x: 'Sqrt[' + x + ']',
    lambda x1, x2, x3: '(' + x1 + ' * ' + x2 + ' + ' + x3 + ')',
    lambda x1, x2: '(Abs[' + x1 + '] * Sign[' + x2 + '])',
    lambda x: 'Abs[' + x + ']',
    lambda x1, x2: 'Max[0, (' + x1 + ' - ' + x2 + ')]',
    lambda x1, x2: 'Max[' + x1 + ', ' + x2 + ']',
    lambda x1, x2: 'Min[' + x1 + ', ' + x2 + ']',
    lambda x1, x2: '(Mod[Abs[' + x1 + '], Abs[' + x2 + ']] * Sign[' + x1 + '])',
    _remainder,
    lambda x: 'Ceiling[' + x + ']',
    lambda x: 'Floor[' + x + ']',
    _nearbyint,
    _round,
    lambda x: 'IntegerPart[' + x + ']',
    lambda x: 'ArcCos[' + x + ']',
    lambda x: 'ArcCosh[' + x + ']',
    lambda x: 'ArcSin[' + x + ']',
    lambda x: 'ArcSinh[' + x + ']',
    lambda x: 'ArcTan[' + x + ']',
    lambda x1, x2: 'ArcTan[' + x1 + ', ' + x2 + ']',
    lambda x: 'ArcTanh[' + x + ']',
    lambda x: 'Cos[' + x + ']',
    lambda x: 'Cosh[' + x + ']',
    lambda x: 'Sin[' + x + ']',
    lambda x: 'Sinh[' + x + ']',
    lambda x: 'Tan[' + x + ']',
    lambda x: 'Tanh[' + x + ']',
    lambda x: 'Exp[' + x + ']',
    lambda x: '(2 ^ ' + x + ')',
    lambda x: '(Exp[' + x + '] - 1)',
    lambda x: 'Log[' + x + ']',
    lambda x: 'Log10[' + x + ']',
    lambda x: 'Log[(1 + ' + x + ')]',
    lambda x: 'Log2[' + x + ']',
    lambda x: 'CubeRoot[' + x + ']',
    lambda x1, x2: 'Sqrt[' + x1 + '^2 + ' + x2 + '^2]',
    lambda x1, x2: '(' + x1 + ' ^ ' + x2 + ')',
    lambda x: 'Erf[' + x + ']',
    lambda x: 'Erfc[' + x + ']',
    lambda x: 'LogGamma[' + x + ']',
    lambda x: 'Gamma[' + x + ']',
]


def _mathprompt(i):
    return 'In[{:d}]:= '.format(i)

_outprompt = re.compile(r'Out\[[0-9]+\]= ')
_mathdigits = re.compile(r'\{\{\{([01, ]*)\}, ([+-]?[0-9]+)\}, (True|False), (True|False)\}')
_digitsep = re.compile(r'[, ]')


class MathRepl(object):
    def __init__(self, max_extra_precision = 100000):
        self.i = 1
        self.repl = REPLWrapper('math -rawterm', _mathprompt(self.i), None)
        self.run('SetOptions["stdout", PageWidth -> Infinity]')
        self.run('$MaxExtraPrecision = {}'.format(max_extra_precision))

    def __enter__(self):
        return self

    def __exit__(self):
        self.exit_repl()

    def run(self, cmd):
        self.i += 1
        self.repl.prompt = _mathprompt(self.i)
        # print(cmd)
        # print('--')
        output = self.repl.run_command(cmd)
        # print(output)
        # print('==\n')
        prefmatch = _outprompt.search(output)
        if prefmatch is None:
            raise ValueError('missing output prompt, got:\n{}'.format(output))
        return output[prefmatch.end():].strip()

    def exit_repl(self):
        try:
            output = self.run('Exit')
        except pexpect.EOF:
            pass
        else:
            print('failed to exit math, got:{}'.format(output))

    def evaluate_to_digits(self, expr, prec=54):
        if len(expr) <= 0:
            raise ValueError('cannot evaluate empty expression')

        cmd = 'With[{{EXACT = {:s}}}, With[{{DIGITS = RealDigits[EXACT, 2, {:d}]}}, With[{{ROUNDED = FromDigits[DIGITS, 2], NEGATIVE = EXACT < 0}}, {{DIGITS, NEGATIVE, If[NEGATIVE, -ROUNDED != EXACT, ROUNDED != EXACT]}}]]]'.format(expr, prec)
        output = self.run(cmd)

        return output

        # m = _mathdigits.match(output)
        # if m is None:
        #     raise ValueError('unable to parse output, got:\n{}'.format(output))

        # if len(m.group(1)) == 0:
        #     c = 0
        # elif not m.group(1).startswith('1'):
        #     if '1' in m.group(1):
        #         raise ValueError('bad digits? got:\n{}'.format(output))
        #     else:
        #         c = 0
        # else:
        #     c = int(_digitsep.sub('', m.group(1)), base=2)

        # exp = int(m.group(2)) - c.bit_length()
        # negative = m.group(3) == 'True'
        # inexact = m.group(4) == 'True'

        # cbits = c.bit_length()
        # n = exp - 1
        # e = n + cbits
        # target_n = max(e - max_p, min_n)
        # xbits = target_n - n

        # # copy pasta from gmpmath rounding code

        # if c > 0:
        #     sig = c >> xbits
        #     half_x = c & bitmask(xbits)
        #     half = half_x >> (xbits - 1)
        #     x = half_x & bitmask(xbits - 1)
        # else:
        #     sig = 0
        #     half_x = 0
        #     half = 0
        #     x = 0

        # # Now we need to decide how to round. The value we have in sig was rounded toward zero, so we
        # # look at the half bit and the inexactness of the operation to decide if we should round away.

        # if half > 0:
        #     # greater than halfway away implied by inexactness, or demonstrated by nonzero xbits
        #     if x > 0 or inexact:
        #         # if we have no precision, round away by increasing n
        #         if max_p == 0:
        #             target_n += 1
        #         else:
        #             sig += 1
        #     # TODO: hardcoded RNE
        #     elif sig & 1 > 0:
        #         sig += 1

        # # fixup extra precision from a carry out
        # # TODO in theory this could all be abstracted away with .away()
        # if sig.bit_length() > max_p:
        #     # TODO assert
        #     if sig & 1 > 0:
        #         raise AssertionError('cannot fixup extra precision: {} {}'.format(repr(op), repr(args))
        #                              + '  {}, m={}, exp={}, decoded as {} {} {} {}'.format(
        #                                  repr(candidate), repr(m), repr(exp), negative, sig, half, x))
        #     sig >>= 1
        #     target_n += 1

        # result_inexact = half > 0 or x > 0 or inexact
        # result_sided = True

        # return Sink(c=sig,
        #             exp=target_n + 1,
        #             negative=negative,
        #             inexact=result_inexact,
        #             sided=result_sided,
        #             full=False)


_repls = []
def compute(opcode, *args, prec=54, repl=None):
    op = math_ops[opcode]
    inputs = [digital_to_math(arg) for arg in args]
    formula = op(*inputs)

    if repl is None:
        if len(_repls) <= 0:
            _repls.append(MathRepl())
        repl = _repls[0]

    digits = repl.evaluate_to_digits(formula, prec=prec)

    return math_to_digital(digits)
