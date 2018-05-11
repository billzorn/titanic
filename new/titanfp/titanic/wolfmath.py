"""Interface with Wolfram Mathematica."""

import re

import pexpect
from pexpect.replwrap import REPLWrapper

from .integral import bitmask
from .sinking import Sink

def _mathprompt(i):
    return 'In[{:d}]:= '.format(i)

_outprompt = re.compile(r'Out\[[0-9]+\]= ')
_sinkdigits = re.compile(r'\{\{([01, ]*)\}, ([+-]?[0-9]+), \{(True|False), (True|False)\}\}')
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
        output = self.repl.run_command(cmd)
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

    def evaluate_to_sink(self, expr, min_n = -1075, max_p = 53):
        cmd = 'With[{{EXACT = {:s}}}, With[{{DIGITS = RealDigits[EXACT, 2, {:d}]}}, With[{{ROUNDED = FromDigits[DIGITS, 2]}}, Append[DIGITS, {{EXACT < 0, ROUNDED != EXACT}}]]]]'.format(expr, max_p + 1)
        output = self.run(cmd)

        m = _sinkdigits.match(output)
        if m is None:
            raise ValueError('unable to parse output, got:\n{}'.format(output))

        if len(m.group(1)) == 0:
            c = 0
        elif not m.group(1).startswith('1'):
            if '1' in m.group(1):
                raise ValueError('bad digits? got:\n{}'.format(output))
            else:
                c = 0
        else:
            c = int(_digitsep.sub('', m.group(1)), base=2)

        exp = int(m.group(2)) - c.bit_length()
        negative = m.group(3) == 'True'
        inexact = m.group(4) == 'True'

        cbits = c.bit_length()
        n = exp - 1
        e = n + cbits
        target_n = max(e - max_p, min_n)
        xbits = target_n - n

        # copy pasta from gmpmath rounding code

        if c > 0:
            sig = c >> xbits
            half_x = c & bitmask(xbits)
            half = half_x >> (xbits - 1)
            x = half_x & bitmask(xbits - 1)
        else:
            sig = 0
            half_x = 0
            half = 0
            x = 0
        
        # Now we need to decide how to round. The value we have in sig was rounded toward zero, so we
        # look at the half bit and the inexactness of the operation to decide if we should round away.

        if half > 0:
            # greater than halfway away implied by inexactness, or demonstrated by nonzero xbits
            if x > 0 or inexact:
                # if we have no precision, round away by increasing n
                if max_p == 0:
                    target_n += 1
                else:
                    sig += 1
            # TODO: hardcoded RNE
            elif sig & 1 > 0:
                sig += 1

        # fixup extra precision from a carry out
        # TODO in theory this could all be abstracted away with .away()
        if sig.bit_length() > max_p:
            # TODO assert
            if sig & 1 > 0:
                raise AssertionError('cannot fixup extra precision: {} {}'.format(repr(op), repr(args))
                                     + '  {}, m={}, exp={}, decoded as {} {} {} {}'.format(
                                         repr(candidate), repr(m), repr(exp), negative, sig, half, x))
            sig >>= 1
            target_n += 1

        result_inexact = half > 0 or x > 0 or inexact
        result_sided = True

        return Sink(c=sig,
                    exp=target_n + 1,
                    negative=negative,
                    inexact=result_inexact,
                    sided=result_sided,
                    full=False)
