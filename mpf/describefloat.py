#!/usr/bin/env python

from bv import BV
from real import Real
import core
import conv

def describe_float(x, w, p, rm):
    assert isinstance(x, str) or isinstance(x, Real) or isinstance(x, BV)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2
    assert rm == core.RTN or rm == core.RTP or rm == core.RTZ or rm == core.RNE or rm == core.RNA

    # format constants
    umax = ((2 ** w) - 1) * (2 ** (p - 1))
    emax = (2 ** (w - 1)) - 1
    emin = 1 - emax
    fmax = (Real(2) ** emax) * (Real(2) - ((Real(2) ** (1 - p)) / Real(2)))
    prec = max(28, 2 ** w, p * 2)

    # implicit, packed, and real
    if isinstance(x, str):
        # raw bitvector input, hacky
        s = x.strip().lower()
        if s.startswith('0x') or s.startswith('0b'):
            if s.startswith('0x'):
                b = int(s, 16)
                n = (len(s) - 2) * 4
            else: # s.startswith('0b')
                b = int(s, 2)
                n = len(s) - 2
            assert n == w + p
            B = BV(b, n)
            S, E, T = core.packed_to_implicit(B, w, p)
            R = core.implicit_to_real(S, E, T)
            r = R
        else:
            r = Real(x)
            S, E, T = conv.str_to_implicit(x, w, p, rm)
            B = core.implicit_to_packed(S, E, T)
            R = core.implicit_to_real(S, E, T)
    elif isinstance(x, BV):
        assert B.n == w + p
        B = x
        S, E, T = core.packed_to_implicit(B, w, p)
        R = core.implicit_to_real(S, E, T)
        r = R
    else: # isinstance(x, Real)
        r = x
        S, E, T = core.real_to_implicit(x, w, p, rm)
        B = core.implicit_to_packed(S, E, T)
        R = core.implicit_to_real(S, E, T)

    # implicit bit
    Se, Ee, C = core.implicit_to_explicit(S, E, T)

    # integral values
    s = core.uint(S)
    e = core.uint(E) - emax
    c = core.uint(C)

    # ordinals and rounding info
    if R.isnan:
        i = None
        below = None
        above = None
    else:
        i = core.implicit_to_ordinal(S, E, T)
        below, above = core.binsearch_nearest_ordinals(r, w, p)

        if below == above:
            exact = True
            below = max(i - 1, -umax)
            above = min(i + 1, umax)
        else:
            exact = False

        Sa, Ea, Ta = core.ordinal_to_implicit(above, w, p)
        Sb, Eb, Tb = core.ordinal_to_implicit(below, w, p)
        guess_above = core.implicit_to_real(Sa, Ea, Ta)
        guess_below = core.implicit_to_real(Sb, Eb, Tb)
        difference_above = guess_above - r
        difference_below = r - guess_below

    # print a bunch of stuff, may want to make this separate
    if w == 5 and p == 11:
        format_name = ' (binary16)'
    elif w == 8 and p == 24:
        format_name = ' (binary32)'
    elif w == 11 and p == 53:
        format_name = ' (binary64)'
    else:
        format_name = ''

    D = conv.implicit_to_dec(S, E, T)

    print('  input   : {}'.format(repr(x)))
    print('  format  : w={}, p={}, emax={}, emin={}, umax={}, prec={}, rm={}{}'
          .format(w, p, emax, emin, umax, prec, rm, format_name))
    print('            the largest representable magnitude is about ~{:1.4e}'
          .format(conv.real_to_dec(fmax, prec)))
    print('  binary  : {} {} ({}) {}'
          .format(str(S)[2:], str(E)[2:], str(C[p-1]), str(T)[2:]))
    print('  approx  : ~{:1.8e}'.format(D))
    print('  decimal : {}'.format(D))
    print('  rational: {} /'.format(R.numerator))
    print('            {}'.format(R.denominator))
    print('  ordinal : {}'.format(i))

    if exact:
        print('  this representation is exact.')
        print('  next    : ~{:1.16e}, ~{:1.16e} away'
              .format(conv.real_to_dec(guess_above, prec), conv.real_to_dec(difference_above, prec)))
        print('  prev    : ~{:1.16e}, ~{:1.16e} away'
              .format(conv.real_to_dec(guess_below, prec), conv.real_to_dec(difference_below, prec)))
    else:
        print('  this representation is not exact.')
        print('  above   : ~{:1.16e}, ~{:1.16e} away'
              .format(conv.real_to_dec(guess_above, prec), conv.real_to_dec(difference_above, prec)))
        print('  below   : ~{:1.16e}, ~{:1.16e} away'
              .format(conv.real_to_dec(guess_below, prec), conv.real_to_dec(difference_below, prec)))
        if i == above:
            print('  we rounded up.')
        else:
            print('  we rounded down.')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', type=int, default=11,
                        help='exponent bits')
    parser.add_argument('-p', type=int, default=53,
                        help='significand bits')
    parser.add_argument('-rm', choices={core.RTN, core.RTP, core.RTZ, core.RNE, core.RNA}, default=core.RNE,
                        help='IEEE 754 rounding mode')
    parser.add_argument('x',
                        help='string to describe')
    args = parser.parse_args()

    describe_float(args.x, args.w, args.p, args.rm)
    exit(0)
