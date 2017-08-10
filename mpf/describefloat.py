#!/usr/bin/env python

from bv import BV
from real import FReal
import core
import conv

def describe_format(w, p):
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2

    umax = ((2 ** w) - 1) * (2 ** (p - 1))
    emax = (2 ** (w - 1)) - 1
    emin = 1 - emax
    fmax_scale = FReal(2) - ((FReal(2) ** (1 - p)) / FReal(2))
    fmax = (FReal(2) ** emax) * fmax_scale
    prec = conv.bdb_round_trip_prec(p)
    full_prec = conv.dec_full_prec(w, p)

    return {
        'w'          : w,
        'p'          : p,
        'umax'       : umax,
        'emax'       : emax,
        'emin'       : emin,
        'fmax_scale' : fmax_scale,
        'fmax'       : fmax,
        'prec'       : prec,
        'full_prec'  : full_prec,
    }

def describe_float(S, E, T):
    assert isinstance(S, BV)
    assert S.n == 1
    assert isinstance(E, BV)
    assert E.n >= 2
    assert isinstance(T, BV)

    w = E.n
    p = T.n + 1
    _, _, C = core.implicit_to_explicit(S, E, T)
    B = core.implicit_to_packed(S, E, T)

    s = S.uint
    emax = (2 ** (w - 1)) - 1
    e = E.uint - emax
    c = C.uint
    implicit_bit = C[p-1]
    c_prime = T.uint

    R = core.implicit_to_real(S, E, T)

    # ordinals
    if R.isnan:
        ieee_class = 'nan'
        i = None
        i_prev = None
        R_prev = None
        i_next = None
        R_next = None
    else:
        if R.isinf:
            ieee_class = 'inf'
        elif R.iszero:
            ieee_class = 'zero'
        elif E.uint == 0:
            ieee_class = 'subnormal'
        else:
            ieee_class = 'normal'

        umax = ((2 ** w) - 1) * (2 ** (p - 1))
        i = core.implicit_to_ordinal(S, E, T)

        i_prev = max(i-1, -umax)
        R_prev = core.implicit_to_real(*core.ordinal_to_implicit(i_prev, w, p))

        i_next = min(i+1, umax)
        R_next = core.implicit_to_real(*core.ordinal_to_implicit(i_next, w, p))

        # -0 compliant nextafter behavior
        if R_next.iszero:
            R_next = -R_next

    # rounding
    rounding_info = {}
    if not R.isnan:
        for rm in core.RNE, core.RNA, core.RTZ, core.RTP, core.RTN:
            lower, lower_inclusive, upper, upper_inclusive = conv.implicit_to_rounding_envelope(S, E, T, rm)
            # slow, duplicates the work of building the envelope, but meh
            prec, lowest, midlo, midhi, highest, e = conv.shortest_dec(R, S, E, T, rm, round_correctly=False)
            rounding_info[str(rm)] = {
                'lower'           : lower,
                'lower_inclusive' : lower_inclusive,
                'upper'           : upper,
                'upper_inclusive' : upper_inclusive,
                'prec'    : prec,
                'lowest'  : lowest,
                'midlo'   : midlo,
                'midhi'   : midhi,
                'highest' : highest,
                'e'       : e,
            }

    return {
        'w' : w,
        'p' : p,
        'S' : S,
        'E' : E,
        'T' : T,
        'C' : C,
        'B' : B,
        's' : s,
        'e' : e,
        'c' : c,
        'implicit_bit' : implicit_bit,
        'c_prime'      : c_prime,
        'R'            : R,
        'ieee_class'   : ieee_class,
        'i'      : i,
        'i_prev' : i_prev,
        'R_prev' : R_prev,
        'i_next' : i_next,
        'R_next' : R_next,
        'rounding_info' : rounding_info,
    }

def describe_real(x, w, p):
    assert isinstance(x, int) or isinstance(x, BV) or isinstance(x, FReal) or isinstance(x, str)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2

    # After the parsing logic, we have the following variables:
    # input_repr : repr(what was passed to x)
    #          x : the input x, parsed into an int, BV, or FReal
    #          R : an FReal with the value of the raw input
    #      i_below : the ordinal below R
    #      i_above : the ordinal above R

    # Note that the sign of 0 is missing when only considering i_above/i_below,
    # but it can be recovered from R.

    # Note also that all of this information can all be determined
    # without considering a particular rounding mode.

    # Finally, i_below and i_above will be None if the input is NaN.

    # Invoke some custom parsing logic on strings to figure out what they are.
    # First, generate input_repr since we might reassign here to x.
    input_repr = repr(x)
    if isinstance(x, str):
        x = conv.str_to_ord_bv_real(x)

    # integers are interpreted as ordinals (and never produce -0)
    if isinstance(x, int):
        R, i_below, i_above = conv.ordinal_to_bounded_real(x, w, p)

    # bitvectors are interpreted as packed representations
    elif isinstance(x, BV):
        R, i_below, i_above = conv.bv_to_bounded_real(x, w, p)

    # reals are just themselves
    elif isinstance(x, FReal):
        R, i_below, i_above = conv.real_to_bounded_real(x, w, p)

    # what is this?
    else:
        raise ValueError('expected an int, BV, or FReal; given {}; parsed to {}'
                         .format(input_repr, repr(x)))

    if i_below is None or i_below == i_above:
        # what do we do for NaN?
        exact = True
        if i_below is None:
            S, E, T = core.real_to_implicit(R, w, p, core.RNE)
            i = None
        else:
            S, E, T = core.ordinal_to_implicit(i_below, w, p)
            # negative zero
            if R.iszero and R.sign == -1:
                S = BV(1, 1)
            i = i_below
    else:
        exact = False
        Sb, Eb, Tb = core.ordinal_to_implicit(i_below, w, p)
        R_below = core.implicit_to_real(Sb, Eb, Tb)
        Sa, Ea, Ta = core.ordinal_to_implicit(i_above, w, p)
        R_above = core.implicit_to_real(Sa, Ea, Ta)

        # -0 compliant nextafter behavior
        if R_above.iszero:
            R_above = -R_above

        difference_below = R - R_below
        difference_above = R_above - R

        # rounding
        rounding_info = {}
        for rm in core.RNE, core.RNA, core.RTZ, core.RTP, core.RTN:
            S, E, T = core.ieee_round_to_implicit(R, i_below, i_above, w, p, rm)
            i = core.implicit_to_ordinal(S, E, T)
            lower, lower_inclusive, upper, upper_inclusive = conv.implicit_to_rounding_envelope(S, E, T, rm)
            # slow, duplicates the work of building the envelope, but meh
            prec, lowest, midlo, midhi, highest, e = conv.shortest_dec(R, S, E, T, rm, round_correctly=True)
            rounding_info[str(rm)] = {
                'lower'           : lower,
                'lower_inclusive' : lower_inclusive,
                'upper'           : upper,
                'upper_inclusive' : upper_inclusive,
                'prec'    : prec,
                'lowest'  : lowest,
                'midlo'   : midlo,
                'midhi'   : midhi,
                'highest' : highest,
                'e'       : e,
                'S' : S,
                'E' : E,
                'T' : T,
                'i' : i,
            }

    if exact:
        return {
            'w'          : w,
            'p'          : p,
            'input_repr' : input_repr,
            'input'      : x,
            'R'          : R,
            'i_below'    : i_below,
            'i_above'    : i_above,
            'exact'      : exact,
            'S' : S,
            'E' : E,
            'T' : T,
            'i' : i,
        }
    else:
        return {
            'w'          : w,
            'p'          : p,
            'input_repr' : input_repr,
            'input'      : x,
            'R'          : R,
            'i_below'    : i_below,
            'i_above'    : i_above,
            'exact'      : exact,
            'R_below' : R_below,
            'R_above' : R_above,
            'difference_below' : difference_below,
            'difference_above' : difference_above,
            'rounding_info' : rounding_info,
        }

    # # print a bunch of stuff, may want to make this separate
    # if w == 5 and p == 11:
    #     format_name = ' (binary16)'
    # elif w == 8 and p == 24:
    #     format_name = ' (binary32)'
    # elif w == 11 and p == 53:
    #     format_name = ' (binary64)'
    # else:
    #     format_name = ''

    # D = conv.implicit_to_dec(S, E, T)

    # print('  input   : {}'.format(repr(x)))
    # print('  format  : w={}, p={}, emax={}, emin={}, umax={}, prec={}, rm={}{}'
    #       .format(w, p, emax, emin, umax, prec, rm, format_name))
    # print('            the largest representable magnitude is about ~{:1.4e}'
    #       .format(conv.real_to_dec(fmax, prec)))
    # print('  binary  : {} {} ({}) {}'
    #       .format(str(S)[2:], str(E)[2:], str(C[p-1]), str(T)[2:]))
    # print('  approx  : ~{:1.8e}'.format(D))
    # print('  decimal : {}'.format(D))
    # print('  rational: {} /'.format(R.rational_numerator))
    # print('            {}'.format(R.rational_denominator))
    # print('  ordinal : {}'.format(i))

    # if exact:
    #     print('  this representation is exact.')
    #     print('  next    : ~{:1.16e}, ~{:1.16e} away'
    #           .format(conv.real_to_dec(guess_above, prec), conv.real_to_dec(difference_above, prec)))
    #     print('  prev    : ~{:1.16e}, ~{:1.16e} away'
    #           .format(conv.real_to_dec(guess_below, prec), conv.real_to_dec(difference_below, prec)))
    # else:
    #     print('  this representation is not exact.')
    #     print('  above   : ~{:1.16e}, ~{:1.16e} away'
    #           .format(conv.real_to_dec(guess_above, prec), conv.real_to_dec(difference_above, prec)))
    #     print('  below   : ~{:1.16e}, ~{:1.16e} away'
    #           .format(conv.real_to_dec(guess_below, prec), conv.real_to_dec(difference_below, prec)))
    #     if i == above:
    #         print('  we rounded up.')
    #     else:
    #         print('  we rounded down.')


def explain_format(d):
    w = d['w']
    p = d['p']
    k = w + p
    if conv.ieee_split_w_p(k) == (w, p):
        format_name = 'binary{:d}'.format(k)
    else:
        format_name = 'custom'

    s = '"{}": w={:d}, p={:d}, emax={:d}, emin={:d}, umax={}\n'.format(
        format_name, w, p, d['emax'], d['emin'], d['umax'])

    s += '  largest representable: {} = (2**{:d})*({})\n'.format(
        conv.real_to_string(d['fmax'], prec=8, exact=False), d['emax'], str(d['fmax_scale']))

    s += '  decimal precision {:d} (round trip), {:d} (exact)'.format(
        d['prec'], d['full_prec'])

    return s

def explain_dict(d, ident=0):
    s = ''
    for k in d:
        x = d[k]
        if isinstance(x, dict):
            s += ' '*ident + repr(k) + ' :\n'
            s += explain_dict(x, ident=ident+2)
        else:
            s += ' '*ident + repr(k) + ' : ' + repr(x) + '\n'
    return s


def explain_float(d):
    S_str = str(d['S']).lower().replace('0b','')
    E_str = str(d['E']).lower().replace('0b','')
    T_str = str(d['T']).lower().replace('0b','')
    ibit_str  = str(d['implicit_bit'])

    s = '   ' + ' '*len(E_str) + 'implicit bit\n'
    s += 'S E' + ' '*len(E_str) + u'\u2193 T\n'
    s += '{} {} {} {}\n'.format(S_str, E_str, ibit_str, T_str)
    s += '\n'

    s += '  (-1)**{:d} * 2**({:d}) * ({:d} * 2**({:d}))\n'.format(d['s'], d['e'], d['c'], 1-d['p'])
    approx12_str = conv.real_to_string(d['R'], prec=12, exact=False)
    if approx12_str.startswith(u'\u2248'):
        s += '  {} {}\n'.format(approx12_str[0], approx12_str[1:])
        s += '  = {}\n'.format(conv.real_to_string(d['R'], exact=True))
    else:
        s += '  = {}\n'.format(approx12_str)

    return s

def explain_real(d):
    return str(d['R'])


# anecdotally, for (kinda) acceptable performance we need to limit ourselves to:
# w <= 20
# p <= 1024
# 1024 characters of input
# scientific notation exponent <= 200000

def explain_all(x, w, p):

    fmt_descr = describe_format(w, p)
    r_descr = describe_real(x, w, p)
    if r_descr.get('exact', False):
        S, E, T = r_descr['S'], r_descr['E'], r_descr['T']
        f_descr = describe_float(S, E, T)
    else:
        f_descr = None

    s = explain_format(fmt_descr) + '\n\n' + explain_real(r_descr)

    if f_descr is not None:
        s += '\n\n' + explain_float(f_descr)

    # temporary, dump dict contents
    s += '\n\n\n\n'

    s += 'fmt_descr:\n'
    s += explain_dict(fmt_descr, 2)
    s += '\n'

    s += 'r_descr:\n'
    s += explain_dict(r_descr, 2)
    s += '\n'

    if f_descr is None:
        s += 'f_descr:\n  None\n'
    else:
        s += 'f_descr:\n'
        s += explain_dict(f_descr, 2)
        s += '\n'

    return s

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', type=int, default=11,
                        help='exponent bits')
    parser.add_argument('-p', type=int, default=53,
                        help='significand bits')
    # parser.add_argument('-rm', choices={core.RTN, core.RTP, core.RTZ, core.RNE, core.RNA}, default=core.RNE,
    #                     help='IEEE 754 rounding mode')
    parser.add_argument('x',
                        help='string to describe')
    args = parser.parse_args()

    print(explain_all(args.x, args.w, args.p))
