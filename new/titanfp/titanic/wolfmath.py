"""Interface with Wolfram Mathematica."""

import re

import pexpect
from pexpect.replwrap import REPLWrapper

from .sinking import Sink

def _mathprompt(i):
    return 'In[{:d}]:= '.format(i)

_outprompt = re.compile(r'Out\[[0-9]+\]= ')
_sinkdigits = re.compile(r'\{\{([01, ]*)\}, ([+-]?[0-9+]), \{(True|False), (True|False)\}\}')
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
            raise ValueError('bad digits? got:\n{}'.format(output))
        else:
            c = int(_digitsep.sub('', m.group(1)), base=2)

        exp = int(m.group(2)) - c.bit_length()
        negative = m.group(3) == 'True'
        inexact = m.group(4) == 'True'

        # print(cmd)
        # print(output)
        # print(m)

        return Sink(c=c, exp=exp, negative=negative, inexact=inexact)
