#!/usr/bin/env python

from bv import BV
from real import FReal
import core
import conv

import operator

def weblink(s, S, E, T):
    w = E.n
    p = T.n + 1
    Bs = str(core.implicit_to_packed(S, E, T))
    href = '"/demo?s={}&w={:d}&p={:d}"'.format(Bs, w, p)
    return '<a href={}>{}</a>'.format(href, s)

ieee_rm_names = {
    core.RTN : 'roundTowardNegative',
    core.RTP : 'roundTowardPositive',
    core.RTZ : 'roundTowardZero',
    core.RNE : 'roundTiesToEven',
    core.RNA : 'roundTiesToAway',
}

def approx_or_exact(R, prec = conv.default_prec, spacer = '', exact_str = '='):
    R_approx = conv.real_to_string(R, prec=prec, exact=False)
    isapprox = R_approx.startswith(conv.approx_str)
    if not isapprox:
        return isapprox, exact_str + spacer + R_approx
    else:
        return isapprox, conv.approx_str + spacer + R_approx[1:]

def summarize_with(R, prec = conv.default_prec, spacer1 = ' ', spacer2 = ' ', exact_str = '='):
    is_approx, R_approx = approx_or_exact(R, prec=prec, spacer=spacer1, exact_str=exact_str)
    if is_approx:
        return R_approx + spacer2 + '=' + spacer2 + conv.real_to_string(R, prec=prec, exact=True)
    else:
        return R_approx

darrow = u'\u2193'

def unicode_fbits(S, E, T, ibit):
    S_str = str(S).lower().replace('0b','')
    E_str = str(E).lower().replace('0b','')
    T_str = str(T).lower().replace('0b','')
    ibit_str  = str(ibit)

    s = '   ' + ' '*len(E_str) + 'implicit bit\n'
    s += 'S E' + ' '*len(E_str) + darrow + ' T\n'
    s += '{} {} {} {}'.format(S_str, E_str, ibit_str, T_str)
    return s


topc = u'\u252c'
botc = u'\u2534'
topo = u'\u2564'
boto = u'\u2567'
vert = u'\u2502'
tic = u'\u253c'
bullet = u'\u2022'

def vertical_nl_line(R, tag, name, s, pri):
    if len(tag) == 0:
        tag = ' '
    else:
        tag = tag[0]
    return tag + name + s

def vertical_nl_combine(line1, line2):
    R1, tag1, name1, s1, pri1 = line1
    R2, tag2, name2, s2, pri2 = line2

    if len(name1) > 0:
        if len(name2) > 0:
            name = name1 + ' = ' + name2
        else:
            name = name1
    else:
        name = name2

    if pri1 > pri2:
        if len(tag1) > 0:
            tag = tag1
        else:
            tag = tag2
        R = R1
        s = s1
        pri = pri1
    else:
        if len(tag2) > 0:
            tag = tag2
        else:
            tag = tag1
        R = R2
        s = s2
        pri = pri2

    return R, tag, name, s, pri

def unicode_double_vertical_nl(lines, start_idx, end_idx, lmid_idx = None,
                               ltop = topc, rtop = topc, lbot = botc, rbot = botc):
    lmax = len(lines) - 1
    s = ''
    for idx, line in enumerate(lines):
        if idx == 0:
            ll = ltop
        elif idx == lmax:
            ll = lbot
        elif lmid_idx is not None and idx == lmid_idx:
            ll = tic
        else:
            ll = vert

        if idx == start_idx:
            rl = rtop
        elif start_idx < idx and idx < end_idx:
            rl = vert
        elif idx == end_idx:
            rl = rbot
        else:
            rl = ' '

        s += ' {} {}{}'.format(ll, rl, line)
        if idx < lmax:
            s += '\n'

    return s

leftc = u'\u251c'
rightc = u'\u2524'
hori = u'\u2500'
uarrow = u'\u2191'
def unicode_horizontal_nl(left, R, right, width,
                          note = '', prec = 12, mirror = False):
    assert isinstance(left, FReal) or isinstance(left, tuple)
    assert isinstance(right, FReal) or isinstance(right, tuple)

    if isinstance(left, tuple):
        left, Sl, El, Tl = left
        link_left = True
    else:
        link_left = False

    if isinstance(right, tuple):
        right, Sr, Er, Tr = right
        link_right = True
    else:
        link_right = False

    assert isinstance(R, FReal)
    assert left <= R and R <= right
    assert isinstance(width, int)
    assert width >= 3

    # lol
    if left == right:
        return real_to_string(R, exact=True) + '\n' + vert

    span = right - left
    left_offset = (R - left) / span
    # no floor, should be fine for positive
    effw = width-2
    int_offset = int((left_offset * effw).numeric_value(conv.ndig(effw) + 10))
    int_offset = min(int_offset, width-3)

    # more lol
    if mirror:
        left, right = right, left
        int_offset = effw - 1 - int_offset

    mid_at_left = None

    if left == R:
        left_label = note + conv.real_to_string(R, exact=True)
        len_left_label = len(left_label)
        mid_at_left = True
    else:
        left_label = conv.real_to_string(left, prec=prec, exact=False)
        len_left_label = len(left_label)
        if link_left:
            left_label = weblink(left_label, Sl, El, Tl)
        mid_label = note + conv.real_to_string(R, exact=True)

    if right == R:
        right_label = note + conv.real_to_string(R, exact=True)
        len_right_label = len(right_label)
        mid_at_left = False
    else:
        right_label = conv.real_to_string(right, prec=prec, exact=False)
        len_right_label = len(right_label)
        if link_right:
            right_label = weblink(right_label, Sr, Er, Tr)
        mid_label = note + conv.real_to_string(R, exact=True)

    if mid_at_left is not None:
        right_label_offset = max(0, width - len_right_label)
        nl = leftc + hori*(width-2) + rightc
        if mid_at_left:
            return left_label + '\n' + darrow + '\n' + nl + '\n' + ' '*right_label_offset + right_label
        else:
            return left_label + '\n' + nl + '\n' + ' '*(width-1) + uarrow + '\n' + ' '*right_label_offset + right_label
    else:
        nl = leftc + hori*int_offset + tic + hori*(width-3-int_offset) + rightc
        mid_label_offset = int_offset + 1
        if len_left_label + 2 < mid_label_offset:
            right_label_offset = max(0, width - len_right_label)
            return (' '*mid_label_offset + mid_label + '\n' +
                    left_label + ' '*(mid_label_offset-len_left_label) + darrow + '\n'
                    + nl + '\n'
                    + ' '*right_label_offset + right_label)
        else:
            right_label_offset = max(1, width - mid_label_offset - 1 - len_right_label)
            return (' '*mid_label_offset + mid_label + '\n' +
                    ' '*mid_label_offset + darrow + ' '*right_label_offset + right_label + '\n'
                    + nl + '\n'
                    + left_label)

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
            prec, lowest, midlo, midhi, highest, r_e = conv.shortest_dec(R, S, E, T, rm, round_correctly=False)
            rounding_info[str(rm)] = {
                'rm' : rm,
                'lower'           : lower,
                'lower_inclusive' : lower_inclusive,
                'upper'           : upper,
                'upper_inclusive' : upper_inclusive,
                'prec'    : prec,
                'lowest'  : lowest,
                'midlo'   : midlo,
                'midhi'   : midhi,
                'highest' : highest,
                'e'       : r_e,
                'S' : S,
                'E' : E,
                'T' : T,
                'i'        : i,
                'R_center' : R,
                'i_prev'   : i_prev,
                'R_prev'   : R_prev,
                'i_next'   : i_next,
                'R_next'   : R_next,
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
        umax = ((2 ** w) - 1) * (2 ** (p - 1))
        rounding_info = {}
        for rm in core.RNE, core.RNA, core.RTZ, core.RTP, core.RTN:
            S, E, T = core.ieee_round_to_implicit(R, i_below, i_above, w, p, rm)
            # we need this info to fully describe the envelope
            i = core.implicit_to_ordinal(S, E, T)
            R_center = core.implicit_to_real(*core.ordinal_to_implicit(i, w, p))
            i_prev = max(i-1, -umax)
            R_prev = core.implicit_to_real(*core.ordinal_to_implicit(i_prev, w, p))
            i_next = min(i+1, umax)
            R_next = core.implicit_to_real(*core.ordinal_to_implicit(i_next, w, p))
            # -0 compliant nextafter behavior
            if R_next.iszero:
                R_next = -R_next
            lower, lower_inclusive, upper, upper_inclusive = conv.implicit_to_rounding_envelope(S, E, T, rm)
            # slow, duplicates the work of building the envelope, but meh
            prec, lowest, midlo, midhi, highest, e = conv.shortest_dec(R, S, E, T, rm, round_correctly=True)
            rounding_info[str(rm)] = {
                'rm' : rm,
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
                'i'        : i,
                'R_center' : R_center,
                'i_prev'   : i_prev,
                'R_prev'   : R_prev,
                'i_next'   : i_next,
                'R_next'   : R_next,
                'R' : R,
                'i_below' : i_below,
                'i_above' : i_above,
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

def explain_dict(d, indent=0):
    s = ''
    for k in d:
        x = d[k]
        if isinstance(x, dict):
            s += ' '*indent + repr(k) + ' :\n'
            s += explain_dict(x, indent=indent+2)
        else:
            s += ' '*indent + repr(k) + ' : ' + repr(x) + '\n'
    return s

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

def explain_rm(d):
    prec = d['prec']
    if 'R' in d:
        R = d['R']
        R_label = 'R'
        R_spacer = ' '
        R_exact = '='
    else:
        R = d['R_center']
        R_label = ''
        R_spacer = ''
        R_exact = ''
    _, R_approx = approx_or_exact(R, prec+1)

    s = 'rounding envelope for {} around R{}:\n'.format(ieee_rm_names[d['rm']], R_approx)
    s += '{:d} digit(s) of decimal precision required to round-trip\n'.format(prec)

    # well this is fun
    lower = d['lower']
    lower_inclusive = d['lower_inclusive']
    upper = d['upper']
    upper_inclusive = d['upper_inclusive']
    lowest = d['lowest']
    midlo = d['midlo']
    midhi = d['midhi']
    highest = d['highest']
    e = d['e']
    S = d['S']
    E = d['E']
    T = d['T']
    i = d['i']
    R_center = d['R_center']
    i_prev = d['i_prev']
    R_prev = d['R_prev']
    i_next = d['i_next']
    R_next = d['R_next']
    i_below = d.get('i_below', None)
    i_above = d.get('i_above', None)

    # set up some arguments to pass to the numberline generator
    if upper_inclusive:
        envtop = topc
    else:
        envtop = topo
    if lower_inclusive:
        envbot = botc
    else:
        envbot = boto

    w = E.n
    p = T.n + 1

    Sp, Ep, Tp = core.ordinal_to_implicit(i_prev, w, p)
    if R_prev.iszero and R_prev.sign == -1:
        Sp = BV(1, 1)
    Sn, En, Tn = core.ordinal_to_implicit(i_next, w, p)
    if R_next.iszero and R_next.sign == -1:
        Sn = BV(1, 1)

    # left number line:
    # - next
    # - upper (could be =next =center)
    # - center (could be =upper or =lower)
    # - lower (could be =center or =prev)
    # - prev
    lnl = [
        (R_next,   '', '', weblink(summarize_with(R_next, prec+1, spacer1='', exact_str=''), Sn, En, Tn), 1,),
        (upper,    '', '', summarize_with(upper, prec+1, spacer1='', exact_str=''),                       0,),
        (R_center, '', '', weblink(summarize_with(R_center, prec+1, spacer1='', exact_str=''), S, E, T),  1,),
        (lower,    '', '', summarize_with(lower, prec+1, spacer1='', exact_str=''),                       0,),
        (R_prev,   '', '', weblink(summarize_with(R_prev, prec+1, spacer1='', exact_str=''), Sp, Ep, Tp), 1,),
    ]

    # right number line: (could all be equal, R could be anywhere in this ordering)
    # - highest
    # - midhi
    # - midlo
    # - lowest
    #   ?? R
    rnl = [(R, '', R_label + R_spacer, summarize_with(R, prec+1, spacer1=R_spacer, exact_str=R_exact), 2,),]
    # trim identical decimal tags using (hopefully) fast integer compare
    old_c = None
    for c in (midhi, midlo, lowest,):
        if old_c is None or c != old_c:
            old_c = c
            R_c = FReal(c) * (FReal(10)**e)
            if c == midhi or c == midlo:
                tag = bullet
            else:
                tag = ''
            rnl.append((R_c, tag, '', conv.pow10_to_str(c, e), 0,))

    # special cases for top and bottom
    if R_next > upper:
        start_idx = 2
        lines = [vertical_nl_line(*lnl[0]), '',]
        lnl[:] = lnl[1:]
    else:
        start_idx = 0
        lines = []

    if R_prev < lower:
        post_lines = ['', vertical_nl_line(*lnl[-1]),]
        lnl[:] = lnl[:-1]
    else:
        post_lines = []

    # sort
    protos = sorted(lnl + rnl, key=operator.itemgetter(0), reverse=True)
    # dedup and append
    lmid_idx = None
    old_proto = None
    for proto in protos:
        if old_proto is None:
            old_proto = proto
        else:
            Ro, _, _, _, _ = old_proto
            R, _, _, _, _ = proto
            if R == Ro:
                old_proto = vertical_nl_combine(old_proto, proto)
            else:
                if Ro == R_center:
                    lmid_idx = len(lines)
                lines.append(vertical_nl_line(*old_proto))
                old_proto = proto
    if old_proto is not None:
        Ro, _, _, _, _ = old_proto
        if Ro == R_center:
            lmid_idx = len(lines)
        lines.append(vertical_nl_line(*old_proto))

    end_idx = len(lines) - 1
    lines.extend(post_lines)

    s += unicode_double_vertical_nl(lines, start_idx, end_idx, lmid_idx=lmid_idx,
                                    rtop=envtop, rbot=envbot)
    return s

def explain_nl(R, S, E, w, p, fwidth=100, ewidth=80, enote = ''):
    if R.iszero:
        R_left = core.implicit_to_real(BV(1,1), E, BV(-1,p-1))
        R_right = core.implicit_to_real(BV(0,1), E, BV(-1,p-1))
        mirror = False
        mirror_str = ''
    elif R > 0:
        R_left = core.implicit_to_real(S, E, BV(0,p-1))
        R_right = core.implicit_to_real(S, E, BV(-1,p-1))
        mirror = False
        mirror_str = ''
    else: # R < 0
        R_left = core.implicit_to_real(S, E, BV(-1,p-1))
        R_right = core.implicit_to_real(S, E, BV(0,p-1))
        mirror = True
        mirror_str = ', mirrored'

    emax = (2 ** (w - 1)) - 1
    emin = 1 - emax
    e = max(E.uint - emax, emin)

    s = 'fractional position (linear scale)\n'
    s += unicode_horizontal_nl(R_left, R, R_right, fwidth, note='R=')
    s += '\n\nexponential position (log scale' + mirror_str + ')' + enote + '\n'
    s += unicode_horizontal_nl(FReal(emin), FReal(e), FReal(emax), ewidth, note='e=', mirror=mirror)
    return s

def explain_float(d, summary_length = 12):

    w = d['w']
    p = d['p']
    R = d['R']

    s = 'floating point representation:\n'
    s += unicode_fbits(d['S'], d['E'], d['T'], d['implicit_bit']) + '\n\n'

    s += 'IEEE 754 binary representation:\n'
    B = d['B']
    s += '  ' + str(B) + '\n'
    k = w + p
    if conv.ieee_split_w_p(k) == (w, p):
        s += ('  {:#0' + str((k//4)+2) + 'x}\n').format(B.uint)
    s += '\n'

    if not R.isnan:
        s += 'ordinal (ulps away from zero):\n'
        s += '  ' + str(d['i']) + '\n\n'

    s += 'real value:\n'
    if R.isnan:
        s += '  {} (with payload: {})\n\n'.format(conv.real_to_pretty_string(R),
                                                    conv.real_to_string(R, show_payload=True))
    elif R.isinf:
        s += '  {} ( {} )\n\n'.format(conv.real_to_pretty_string(R),
                                      conv.real_to_string(R))
    elif R.iszero:
        s += '  ' + conv.real_to_string(R) + '\n\n'
    else:
        s += '  (-1)**{:d} * 2**({:d}) * ({:d} * 2**({:d}))\n'.format(d['s'], d['e'], d['c'], 1-d['p'])
        summary_is_approx, R_summary = approx_or_exact(R, prec=summary_length, spacer=' ')
        if summary_is_approx:
            s += '  ' + R_summary + ' = ' + conv.real_to_string(R, prec=summary_length, exact=True) + '\n'
            s += '  = ' + str(R)
        else:
            s += '  ' + R_summary

    # number lines
    if not R.isnan or R.isinf:
        s += '\n\n' + explain_nl(R, d['S'], d['E'], d['w'], d['p'])

    # rounding envelopes
    if 'rounding_info' in d:
        for k, r_info in d['rounding_info'].items():
            s += '\n\n' + explain_rm(r_info)

    return s

def explain_input(d):
    s = 'received input (w={:d}, p={:d}):\n'.format(d['w'], d['p'])
    s += '  ' + d['input_repr'] + '\n\n'

    pretty_repr = conv.real_to_pretty_string(d['R'])
    pretty_repr = pretty_repr.replace('\n', '\n  ')
    s += '  ' + pretty_repr

    return s

def explain_real(d):
    R = d['R']
    if R.isnan or d['exact']:
        s = 'this real value has an exact floating point representation'
    else:
        w = d['w']
        p = d['p']
        i_below = d['i_below']
        i_above = d['i_above']
        R_below = d['R_below']
        R_above = d['R_above']
        diff_below = d['difference_below']
        diff_above = d['difference_above']

        s = 'nearby floating point values:\n'
        s += '  ordinal ' + str(i_above) + '\n'
        s += '       above ' + summarize_with(R_above, 8) + '\n'
        s += '  difference ' + summarize_with(diff_above, 8) + '\n'
        s += '           R ' + summarize_with(R, 8) + '\n'
        s += '  difference ' + summarize_with(diff_below, 8) + '\n'
        s += '       below ' + summarize_with(R_below, 8) + '\n'
        s += '  ordinal ' + str(i_below) + '\n\n'

        if diff_above < diff_below:
            s += 'relative position: (linear scale, above is closer)\n'
        elif diff_above == diff_below:
            s += 'relative position: (linear scale, exactly between)\n'
        else:
            s += 'relative position: (linear scale, below is closer)\n'

        Sb, Eb, Tb = core.ordinal_to_implicit(i_below, w, p)
        if R_below.iszero and R_below.sign == -1:
            Sb = BV(1, 1)
        Sa, Ea, Ta = core.ordinal_to_implicit(i_above, w, p)
        if R_above.iszero and R_above.sign == -1:
            Sa = BV(1, 1)

        RSET_below = (R_below, Sb, Eb, Tb,)
        RSET_above = (R_above, Sa, Ea, Ta,)

        s += unicode_horizontal_nl(RSET_below, R, RSET_above, 100, note='R=')

        # rounding envelopes
        if 'rounding_info' in d:
            for k, r_info in d['rounding_info'].items():
                s += '\n\n' + explain_rm(r_info)

    return s

# anecdotally, for (kinda) acceptable performance we need to limit ourselves to:
# w <= 20
# p <= 1024
# 1024 characters of input
# scientific notation exponent <= 200000

def explain_all(x, w, p, show_format=False):

    if show_format:
        fmt_descr = describe_format(w, p)
        return explain_format(fmt_descr)

    else:
        r_descr = describe_real(x, w, p)
        if r_descr.get('exact', False):
            S, E, T = r_descr['S'], r_descr['E'], r_descr['T']
            f_descr = describe_float(S, E, T)
        else:
            f_descr = None

        s = explain_input(r_descr)

        if f_descr is None:
            s += '\n\n' + explain_real(r_descr)
        else:
            s += '\n\n' + explain_float(f_descr)

        return s

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', type=int, default=11,
                        help='exponent bits')
    parser.add_argument('-p', type=int, default=53,
                        help='significand bits')
    parser.add_argument('x', nargs='?', default=None,
                        help='string to describe')
    parser.add_argument('-f', action='store_true',
                        help='show format information')
    args = parser.parse_args()

    if not args.f and args.x is None:
        print('no input string; nothing to do')
    else:
        print(explain_all(args.x, args.w, args.p, args.f))
