"""Interface with Wolfram Mathematica."""

import re

import pexpect
from pexpect.replwrap import REPLWrapper

from .integral import bitmask
from .digital import Digital
from . import ops

p4nb = """setpositenv[{n_Integer /; n >= 2, e_Integer /; e >= 0}] := 
  ({nbits, es} = {n, e}; npat = 2^nbits; useed = 2^2^es; 
   {minpos, maxpos} = {useed^(-nbits + 2), useed^(nbits - 2)}; 
   qsize = 2^Ceiling[Log[2, (nbits - 2)*2^(es + 2) + 5]]; 
   qextra = qsize - (nbits - 2)*2^(es + 2); );
positQ[p_Integer] := Inequality[0, LessEqual, p, Less, npat];
twoscomp[sign_, p_] := Mod[If[sign > 0, p, npat - p], npat];
signbit[p_ /; positQ[p]] := IntegerDigits[p, 2, nbits][[1]];
regimebits[p_ /; positQ[p]] := Module[{q = twoscomp[1 - signbit[p], p], bits, 
    bit2, npower, tempbits}, bits = IntegerDigits[q, 2, nbits]; bit2 = bits[[2]]; 
    tempbits = Join[Drop[bits, 1], {1 - bit2}]; 
    npower = Position[tempbits, 1 - bit2, 1, 1][[1]] - 1; 
    Take[bits, {2, Min[npower + 1, nbits]}]];
regimevalue[bits_] := If[bits[[1]] == 1, Length[bits] - 1, -Length[bits]];
exponentbits[p_ /; positQ[p]] := Module[{q = twoscomp[1 - signbit[p], p], bits, 
    startbit}, startbit = Length[regimebits[q]] + 3; 
    bits = IntegerDigits[q, 2, nbits]; If[startbit > nbits, {}, 
     Take[bits, {startbit, Min[startbit + es - 1, nbits]}]]];
fractionbits[p_ /; positQ[p]] := Module[{q = twoscomp[1 - signbit[p], p], bits, 
    startbit}, startbit = Length[regimebits[q]] + 3 + es; 
    bits = IntegerDigits[q, 2, nbits]; If[startbit > nbits, {}, 
     Take[bits, {startbit, nbits}]]];
p2x[p_ /; positQ[p]] := Module[{s = (-1)^signbit[p], 
    k = regimevalue[regimebits[p]], e = exponentbits[p], f = fractionbits[p]}, 
   e = Join[e, Table[0, es - Length[e]]]; e = FromDigits[e, 2]; 
    If[f == {}, f = 1, f = 1 + FromDigits[f, 2]/2^Length[f]]; 
    Which[p == 0, 0, p == npat/2, ComplexInfinity, True, s*useed^k*2^e*f]];
positableQ[x_] := Abs[x] == Infinity || Element[x, Reals];
x2p[x_ /; positableQ[x]] := Module[{i, p, e = 2^(es - 1), y = Abs[x]}, 
   Which[y == 0, 0, y == Infinity, BitShiftLeft[1, nbits - 1], True, 
    If[y >= 1, p = 1; i = 2; While[y >= useed && i < nbits, 
        {p, y, i} = {2*p + 1, y/useed, i + 1}]; p = 2*p; i++, 
      p = 0; i = 1; While[y < 1 && i <= nbits, {y, i} = {y*useed, i + 1}]; 
       If[i >= nbits, p = 2; i = nbits + 1, p = 1; i++]]; 
     While[e > 1/2 && i <= nbits, p = 2*p; If[y >= 2^e, y /= 2^e; p++]; e /= 2; 
       i++]; y--; While[y > 0 && i <= nbits, y = 2*y; p = 2*p + Floor[y]; 
       y -= Floor[y]; i++]; p *= 2^(nbits + 1 - i); i++; i = BitAnd[p, 1]; 
     p = Floor[p/2]; p = Which[i == 0, p, y == 1 || y == 0, p + BitAnd[p, 1], 
       True, p + 1]; Mod[If[x < 0, npat - p, p], npat]]];"""

bdig = """bdigits[x_, n_] := With[{EXACT = x, SIMPLIFIED = Simplify[x]}, 
   Which[SIMPLIFIED === Indeterminate, Indeterminate, 
    Abs[SIMPLIFIED] == Infinity, Infinity, True, 
    With[{DIGITS = RealDigits[EXACT, 2, n]}, 
     With[{ROUNDED = FromDigits[DIGITS, 2], NEGATIVE = EXACT < 0}, 
      {DIGITS, NEGATIVE, If[NEGATIVE, -ROUNDED != EXACT, ROUNDED != EXACT]}]]]];"""

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
    if digits == 'Infinity':
        return Digital(inf=True)
    elif digits == 'Indeterminate':
        return Digital(nan=True)
    
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

    return Digital(
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
        self.log_f = open('mathlog.txt', 'wt')
        P = pexpect.spawn('math -rawterm', encoding='ascii', logfile=self.log_f)
        self.repl = REPLWrapper(P, _mathprompt(self.i), None)
        self.run('SetOptions["stdout", PageWidth -> Infinity]')
        self.run('$MaxExtraPrecision = {}'.format(max_extra_precision))
        self.run(bdig)

    def __enter__(self):
        return self

    def __exit__(self):
        self.exit_repl()

    def run(self, cmd):
        self.i += 1
        self.repl.prompt = _mathprompt(self.i)
        #print(cmd)
        #print('--')
        output = self.repl.run_command(cmd.replace('\n', ' ').replace('\r', ' '))
        #print(output)
        #print('==\n')
        prefmatch = _outprompt.search(output)
        if prefmatch is None:
            return None
            #raise ValueError('missing output prompt, got:\n{}'.format(output))
        return output[prefmatch.end():].strip()

    def exit_repl(self):
        try:
            output = self.run('Exit')
        except pexpect.EOF:
            pass
        else:
            print('failed to exit math, got:{}'.format(output))
        self.log_f.close()

    def evaluate_to_digits(self, expr, prec=54):
        if len(expr) <= 0:
            raise ValueError('cannot evaluate empty expression')

        cmd = 'bdigits[{:s}, {:d}]'.format(expr, prec)
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

        # return Digital(c=sig,
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

def runmath(args_formula, *args, prec=53):
    argnames, formstr = args_formula
    if len(argnames) != len(args):
        raise ValueError('argument number mismatch: got {}, {}'.format(repr(argnames), repr(args)))

    withstr = ', '.join(name + ' = ' + digital_to_math(arg) for name, arg in zip(argnames, args))
    
    formula = 'With[{{{:s}}}, {:s}]'.format(withstr, formstr)

    print(formula)
    
    if len(_repls) <= 0:
        _repls.append(MathRepl())
    repl = _repls[0]

    digits = repl.evaluate_to_digits(formula, prec=prec+1)
    return math_to_digital(digits)
