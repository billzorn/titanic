#!/usr/bin/env python

import operator

from bv import BV
from real import FReal
import core
import conv
import webcontent

# old
def strlink(s, s_link, w, p):
    href = '"/demo?s={}&w={:d}&p={:d}"'.format(s_link, w, p)
    return '<a href={}>{}</a>'.format(href, s)

def weblink(s, S, E, T):
    w = E.n
    p = T.n + 1
    Bs = str(core.implicit_to_packed(S, E, T))
    return strlink(s, Bs, w, p)

# new
def stup(S, E, T):
    return '({}, {}, {})'.format(str(S), str(E), str(T))

def permalink(s_link, w, p, text='link'):
    return webcontent.link(text, s_link, w, p)

ieee_rm_names = {
    core.RTN : 'roundTowardNegative',
    core.RTP : 'roundTowardPositive',
    core.RTZ : 'roundTowardZero',
    core.RNE : 'roundTiesToEven',
    core.RNA : 'roundTiesToAway',
}
ieee_rm_longest = max(len(name) for name in ieee_rm_names.values())

def approx_or_exact(R, prec = conv.default_prec, spacer = '', exact_str = '='):
    R_approx = conv.real_to_string(R, prec=prec, exact=False)
    isapprox = R_approx.startswith(conv.approx_str)
    if isapprox:
        return isapprox, conv.approx_str + spacer + R_approx[1:]
    else:
        return isapprox, exact_str + spacer + R_approx

def summarize_with(R, prec = conv.default_prec, spacer1 = ' ', spacer2 = ' ', exact_str = '=', linkdata = None):
    is_approx, R_approx = approx_or_exact(R, prec=prec, spacer=spacer1, exact_str=exact_str)
    if linkdata is not None:
        s_link, w, p = linkdata
        R_approx = permalink(s_link, w, p, text=R_approx)
    if is_approx:
        return R_approx + spacer2 + '=' + spacer2 + conv.real_to_string(R, prec=prec, exact=True)
    else:
        return R_approx

darrow = u'\u2193'

def unicode_fbits(R, S, E, T, C, ibit, prec=12):
    S_str = str(S).lower().replace('0b','')
    E_str = str(E).lower().replace('0b','')
    T_str = str(T).lower().replace('0b','')
    ibit_str  = str(ibit)

    w = E.n
    p = C.n
    emax = (2 ** (w - 1)) - 1
    e = E.uint - emax
    c_prime = T.uint
    c = C.uint

    e_str = 'e={:d}'.format(e)
    c_prime_str = "c'={:d}".format(c_prime)

    s = '   ' + ' '*len(E_str) + 'implicit bit\n'
    s += 'S E' + ' '*len(E_str) + darrow + ' T\n'
    s += '{} {} {} {}\n'.format(S_str, E_str, ibit_str, T_str)
    s += '  ' + e_str + ' '*(len(E_str) + 3 - len(e_str)) + c_prime_str

    if not (R.isinf or R.isnan):
        Rm = FReal(c) * (FReal(2)**(1-p))
        s += '\n\n'
        if ibit == 1:
            s += ' ' + ' '*len(E_str) + 'c = {:d} + (2**{:d})\n'.format(c_prime, p-1)
            s += ' ' + ' '*len(E_str) + '  = {:d}\n'.format(c)
        else:
            s += ' ' + ' '*len(E_str) + 'c = {:d}\n'.format(c_prime)
        s += ' ' + ' '*len(E_str) + 'm = {:d} * (2**{:d})\n'.format(c, (1-p))
        s += ' ' + ' '*len(E_str) + '  ' + summarize_with(Rm, prec, spacer1=' ') + '\n'
        s += ' ' + ' '*len(E_str) + '  = 0b{}.{}'.format(ibit_str, T_str)

    return s

def explain_class(R, class_str):
    if R.sign < 0:
        sign_str = 'negative'
    else:
        sign_str = 'positive'

    if class_str == 'nan':
        return 'signalingNaN OR quietNaN - at present, Titanic does not distinguish'
    elif class_str == 'inf':
        return sign_str + 'Infinity'
    elif class_str == 'zero':
        return sign_str + 'Zero'
    elif class_str == 'subnormal':
        return sign_str + 'Subnormal'
    elif class_str == 'normal':
        return sign_str + 'Normal'
    else:
        return 'unknown class {} - this is probably a bug'.format(repr(class_str))

topc = u'\u252c'
botc = u'\u2534'
topo = u'\u2564'
boto = u'\u2567'
vert = u'\u2502'
tic = u'\u253c'
bullet = u'\u2022'
larrow = u'\u2190'
hori = u'\u2500'

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
            if start_idx == end_idx:
                rl = hori
            else:
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
uarrow = u'\u2191'
def unicode_horizontal_nl(left, R, right, width,
                          note = '', prec = 12, mirror = False):
    assert isinstance(left, FReal) or isinstance(left, tuple)
    assert isinstance(right, FReal) or isinstance(right, tuple)

    if isinstance(left, tuple):
        left, left_s, left_w, left_p = left
        link_left = True
    else:
        link_left = False

    if isinstance(right, tuple):
        right, right_s, right_w, right_p = right
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
    if left_offset.isnan:
        int_offset = effw
    else:
        int_offset = int((left_offset * effw).numeric_value(conv.ndig(effw) + 2))
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
            left_label = permalink(left_s, left_w, left_p, text=left_label)
        mid_label = note + conv.real_to_string(R, exact=True)

    if right == R:
        right_label = note + conv.real_to_string(R, exact=True)
        len_right_label = len(right_label)
        mid_at_left = False
    else:
        right_label = conv.real_to_string(right, prec=prec, exact=False)
        len_right_label = len(right_label)
        if link_right:
            right_label = permalink(right_s, right_w, right_p, text=right_label)
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
            prec, lowest_ce, midlo_ce, midhi_ce, highest_ce = conv.shortest_dec(R, S, E, T, rm, round_correctly=False)
            lowest_c, lowest_e = lowest_ce
            midlo_c, midlo_e = midlo_ce
            midhi_c, midhi_e = midhi_ce
            highest_c, highest_e = highest_ce
            rounding_info[str(rm)] = {
                'rm' : rm,
                'lower'           : lower,
                'lower_inclusive' : lower_inclusive,
                'upper'           : upper,
                'upper_inclusive' : upper_inclusive,
                'prec'    : prec,
                'lowest_c'  : lowest_c,
                'lowest_e'  : lowest_e,
                'midlo_c'   : midlo_c,
                'midlo_e'   : midlo_e,
                'midhi_c'   : midhi_c,
                'midhi_e'   : midhi_e,
                'highest_c' : highest_c,
                'highest_e' : highest_e,
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

def describe_real(R, w, p):
    assert isinstance(R, FReal)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2

    if R.isnan:
        i_below, i_above = None, None
    else:
        i_below, i_above = core.binsearch_nearest_ordinals(R, w, p)

    if i_below is None or i_below == i_above:
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
            R_center = core.implicit_to_real(S, E, T)
            i_prev = max(i-1, -umax)
            R_prev = core.implicit_to_real(*core.ordinal_to_implicit(i_prev, w, p))
            i_next = min(i+1, umax)
            R_next = core.implicit_to_real(*core.ordinal_to_implicit(i_next, w, p))
            # -0 compliant nextafter behavior
            if R_next.iszero:
                R_next = -R_next
            lower, lower_inclusive, upper, upper_inclusive = conv.implicit_to_rounding_envelope(S, E, T, rm)
            # slow, duplicates the work of building the envelope, but meh
            prec, lowest_ce, midlo_ce, midhi_ce, highest_ce = conv.shortest_dec(R, S, E, T, rm, round_correctly=False)
            lowest_c, lowest_e = lowest_ce
            midlo_c, midlo_e = midlo_ce
            midhi_c, midhi_e = midhi_ce
            highest_c, highest_e = highest_ce
            rounding_info[str(rm)] = {
                'rm' : rm,
                'lower'           : lower,
                'lower_inclusive' : lower_inclusive,
                'upper'           : upper,
                'upper_inclusive' : upper_inclusive,
                'prec'    : prec,
                'lowest_c'  : lowest_c,
                'lowest_e'  : lowest_e,
                'midlo_c'   : midlo_c,
                'midlo_e'   : midlo_e,
                'midhi_c'   : midhi_c,
                'midhi_e'   : midhi_e,
                'highest_c' : highest_c,
                'highest_e' : highest_e,
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
            'R'          : R,
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
            'R'          : R,
            'exact'      : exact,
            'i_below' : i_below,
            'i_above' : i_above,
            'R_below' : R_below,
            'R_above' : R_above,
            'difference_below' : difference_below,
            'difference_above' : difference_above,
            'rounding_info'    : rounding_info,
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

def explain_format(d, link = False):
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

def explain_rm(d, link = False):
    prec = d['prec']
    if 'R' in d:
        R = d['R']
        R_label = 'R'
        R_spacer = ' '
        R_exact = '='
    else:
        # TODO might want to improve this and always label R for consistency
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
    lowest_ce = (d['lowest_c'], d['lowest_e'],)
    midlo_ce = (d['midlo_c'], d['midlo_e'],)
    midhi_ce = (d['midhi_c'], d['midhi_e'],)
    highest_ce = (d['highest_c'], d['highest_e'],)
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
        (R_next,   '',      '', summarize_with(R_next, prec+1, spacer1='', exact_str='',
                                               linkdata=(stup(Sn, En, Tn), w, p)),         1,),
        (upper,    '',      '', summarize_with(upper, prec+1, spacer1='', exact_str=''),   0,),
        (R_center, larrow,  '', summarize_with(R_center, prec+1, spacer1='', exact_str='',
                                              linkdata=(stup(S, E, T), w, p)),             1,),
        (lower,    '',      '', summarize_with(lower, prec+1, spacer1='', exact_str=''),   0,),
        (R_prev,   '',      '', summarize_with(R_prev, prec+1, spacer1='', exact_str='',
                                               linkdata=(stup(Sp, Ep, Tp), w, p)),         1,),
    ]

    # right number line: (could all be equal, R could be anywhere in this ordering)
    # - highest
    # - midhi
    # - midlo
    # - lowest
    #   ?? R
    rnl = [(R, '', R_label + R_spacer, summarize_with(R, prec+1, spacer1=R_spacer, exact_str=R_exact), 2,),]
    # trim identical decimal tags using (hopefully) fast integer compare
    old_c_e = None
    for c, e in (highest_ce, midhi_ce, midlo_ce, lowest_ce,):
        if (old_c_e is None or (c, e,) != old_c_e) and c is not None and e is not None:
            old_c_e = (c, e,)
            R_c = FReal(c) * (FReal(10)**e)
            if (c, e,) == midhi_ce or (c, e,) == midlo_ce:
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

def explain_nl(R, S, E, w, p, fwidth = 100, ewidth = 80, enote = '', fprec = 12, link = False):
    emax = (2 ** (w - 1)) - 1
    emin = 1 - emax
    e = max(E.uint - emax, emin)
    eprec = conv.ndig(emax) + 1

    if R.iszero:
        Sl, El, Tl = BV(1,1), E, BV(-1,p-1)
        Sr, Er, Tr = BV(0,1), E, BV(-1,p-1)
        R_left = core.implicit_to_real(Sl, El, Tl)
        R_right = core.implicit_to_real(Sr, Er, Tr)
        if link:
            R_left = (R_left, stup(Sl, El, Tl), w, p,)
            R_right = (R_right, stup(Sr, Er, Tr), w, p,)
        mirror = False
        mirror_str = ''
    elif R.sign > 0:
        if e > emax:
            fmax_scale =  (FReal(2) - ((FReal(2) ** (1 - p)) / FReal(2)))
            R_left = fmax_scale * (FReal(2) ** emax)
            R_right = FReal(infinite=True, negative=False)
            if link:
                R_left = (R_left, conv.real_to_string(fmax_scale, exact=True) + ' * 2**{:d}'.format(emax), w, p,)
                R_right = (R_right, conv.real_to_string(R_right, exact=True), w, p,)
        else:
            Sl, El, Tl = S, E, BV(0,p-1)
            Sr, Er, Tr = S, E, BV(-1,p-1)
            R_left = core.implicit_to_real(Sl, El, Tl)
            R_right = core.implicit_to_real(Sr, Er, Tr)
            if link:
                R_left = (R_left, stup(Sl, El, Tl), w, p,)
                R_right = (R_right, stup(Sr, Er, Tr), w, p,)
        mirror = False
        mirror_str = ''
    else: # R.sign < 0
        if e > emax:
            fmax_scale = (FReal(2) - ((FReal(2) ** (1 - p)) / FReal(2)))
            R_left = FReal(infinite=True, negative=True)
            R_right = (-fmax_scale) * (FReal(2) ** emax)
            if link:
                R_left = (R_left, conv.real_to_string(R_left, exact=True), w, p,)
                R_right = (R_right, conv.real_to_string(-fmax_scale, exact=True) + ' * 2**{:d}'.format(emax), w, p,)
        else:
            Sl, El, Tl = S, E, BV(-1,p-1)
            Sr, Er, Tr = S, E, BV(0,p-1)
            R_left = core.implicit_to_real(Sl, El, Tl)
            R_right = core.implicit_to_real(Sr, Er, Tr)
            if link:
                R_left = (R_left, stup(Sl, El, Tl), w, p,)
                R_right = (R_right, stup(Sr, Er, Tr), w, p,)

        mirror = True
        mirror_str = ', mirrored'

    s = 'fractional position (linear scale)\n'
    s += unicode_horizontal_nl(R_left, R, R_right, fwidth,
                               note='R=', prec=fprec)
    s += '\n\nexponential position (log scale' + mirror_str + ')' + enote + '\n'
    Re = FReal(e)
    s += unicode_horizontal_nl(min(Re, FReal(emin)), Re, max(FReal(emax), Re), ewidth,
                               note='e=', prec=eprec, mirror=mirror)
    return s

def explain_float(d, link = False):
    w = d['w']
    p = d['p']
    R = d['R']
    prec = conv.bdb_round_trip_prec(p) + 1

    s = ''

    if link and not R.isnan:
        # TODO: use a nextUp implementation from core
        ordinal = d['i']
        umax = ((2 ** w) - 1) * (2 ** (p - 1))
        Sp, Ep, Tp = core.ordinal_to_implicit(max(-umax, ordinal-1), w, p)
        Sn, En, Tn = core.ordinal_to_implicit(min(umax, ordinal+1), w, p)
        # -0 compliant nextafter behavior
        if ordinal+1 == 0:
            Sn = BV(1, 1)
        s += '  ' + permalink(stup(Sp, Ep, Tp), w, p, text='nextDown')
        s += ' : ' + permalink(stup(d['S'], d['E'], d['T']), w, p, text='this number') + ' : '
        s += permalink(stup(Sn, En, Tn), w, p, text='nextUp') + '\n\n'

    s += 'binary floating point representation:\n\n'
    s += webcontent.indent(unicode_fbits(R, d['S'], d['E'], d['T'], d['C'], d['implicit_bit'], prec=prec), '  ')

    s += '\n\nIEEE 754 class:\n'
    s += webcontent.indent(explain_class(R, d['ieee_class']), '  ')

    s += '\n\nIEEE 754 binary representation:'
    B = d['B']
    s += '\n  '
    if link:
        s += '{} '.format(permalink(str(B), w, p))
    s += str(B)
    k = w + p
    if conv.ieee_split_w_p(k) == (w, p):
        hex_str = ('{:#0' + str((k//4)+2) + 'x}').format(B.uint)
        s += ('\n  ')
        if link:
            s += '{} '.format(permalink(hex_str, w, p))
            s += hex_str

    s += '\n\nSMTLIB2 notation (compatible with Z3):'
    smtlib2_str = '(fp #b{} #b{} #b{})'.format(
        str(d['S']).lower().replace('0b',''),
        str(d['E']).lower().replace('0b',''),
        str(d['T']).lower().replace('0b','')
    )
    s += '\n  '
    if link:
        s += '{} '.format(permalink(smtlib2_str, w, p))
    s += smtlib2_str

    if not R.isnan:
        s += '\n\nordinal (ulps away from zero):'
        ord_str = str(d['i'])
        s += '\n  '
        if link:
            s += '{} '.format(permalink('0n' + ord_str, w, p))
        s += ord_str

    s += '\n\nreal value:'
    s += '\n  '
    if link:
        indent = '     '
        R_str = str(R)
        # TODO: "default" nan payload is hardcoded
        if R.isnan and R.nan_payload != 1:
            R_str += str(R.nan_payload)
        R_link = '{} '.format(permalink(R_str, w, p))
    else:
        indent = ''
        R_link = ''

    if R.isnan:
        s += R_link
        s += 'R = {} (with payload: {})'.format(conv.real_to_pretty_string(R),
                                                    conv.real_to_string(R, show_payload=True))
    elif R.isinf:
        s += R_link
        s += 'R = {} ({})'.format(conv.real_to_pretty_string(R),
                                      conv.real_to_string(R))
    elif R.iszero:
        s += R_link
        s += 'R = ' + conv.real_to_string(R)
    else:
        s += indent + 'R = (-1)**{:d} * 2**{:d} * ({:d} * 2**{:d})'.format(d['s'], d['e'], d['c'], 1-d['p'])
        s += '\n  '
        if link:
            dec_str = conv.real_to_string(R, exact=True)
            s += '{} '.format(permalink(dec_str, w, p))
        s += '  ' + summarize_with(R, prec=prec)
        if R.isrational and R.rational_denominator != 1:
            s += '\n  ' + R_link + '  = ' + str(R)

        # nearest decimal approximations
        if 'rounding_info' in d:
            s += '\n\nclosest, shortest decimal approximations:'
            for k, r_info in d['rounding_info'].items():
                rm = ieee_rm_names[r_info['rm']]
                midlo_c = r_info['midlo_c']
                midlo_e = r_info['midlo_e']
                midhi_c = r_info['midhi_c']
                midhi_e = r_info['midhi_e']

                label = rm + ':' + ' '*(ieee_rm_longest - len(rm) + 1)

                midhi_str = conv.pow10_to_str(midhi_c, midhi_e)
                s += '\n  ' + label
                if link:
                    s += '{} '.format(permalink(midhi_str, w, p))
                s += midhi_str
                if midlo_c != midhi_c or midlo_e != midhi_e:
                    midlo_str = conv.pow10_to_str(midlo_c, midlo_e)
                    s += '\n  ' + ' '*len(label)
                    if link:
                        s += '{} '.format(permalink(midlo_str, w, p))
                        s += midlo_str

    # number lines
    if not R.isnan or R.isinf:
        s += '\n\n\n' + explain_nl(R, d['S'], d['E'], d['w'], d['p'], fprec=prec, link=link)

    # rounding envelopes
    if 'rounding_info' in d:
        s += '\n'
        for k, r_info in d['rounding_info'].items():
            s += '\n\n' + explain_rm(r_info, link=link)

    return s

def explain_real(d, link = False):
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

        prec = conv.bdb_round_trip_prec(p) + 1

        Sb, Eb, Tb = core.ordinal_to_implicit(i_below, w, p)
        if R_below.iszero and R_below.sign == -1:
            Sb = BV(1, 1)
        Sa, Ea, Ta = core.ordinal_to_implicit(i_above, w, p)
        if R_above.iszero and R_above.sign == -1:
            Sa = BV(1, 1)

        s = 'nearby floating point values:'

        s += '\n     '
        if link:
            s += permalink('0n'+str(i_above), w, p, text='ordinal')
        else:
            s += 'ordinal'
        s += ' ' + str(i_above)

        s += '\n       '
        if link:
            s += permalink(stup(Sa, Ea, Ta), w, p, text='above')
        else:
            s += 'above'
        s += ' ' + summarize_with(R_above, prec)

        s += '\n  difference ' + summarize_with(diff_above, prec)
        s += '\n           R ' + summarize_with(R, prec)
        s += '\n  difference ' + summarize_with(diff_below, prec)

        s += '\n       '
        if link:
            s += permalink(stup(Sb, Eb, Tb), w, p, text='below')
        else:
            s += 'below'
        s += ' ' + summarize_with(R_below, prec)

        s += '\n     '
        if link:
            s += permalink('0n'+str(i_below), w, p, text='ordinal')
        else:
            s += 'ordinal'
        s += ' ' + str(i_below)

        if link:
            Rswp_below = (R_below, stup(Sb, Eb, Tb), w, p,)
            Rswp_above = (R_above, stup(Sa, Ea, Ta), w, p,)
        else:
            Rswp_below = R_below
            Rswp_above = R_above

        if diff_above < diff_below:
            s += '\n\nrelative position: (linear scale, R is closer to above)\n'
        elif diff_above == diff_below:
            s += '\n\nrelative position: (linear scale, R is exactly between above and below)\n'
        else:
            s += '\n\nrelative position: (linear scale, R is closer to below)\n'
        s += unicode_horizontal_nl(Rswp_below, R, Rswp_above, 100, note='R=', prec=prec)

        # rounding envelopes
        if 'rounding_info' in d:
            s += '\n'
            for k, r_info in d['rounding_info'].items():
                s += '\n\n' + explain_rm(r_info, link=link)

    return s

def parse_input(s, w, p, limit_exp = None, allow_exprs = True):
    assert isinstance(s, str)
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2
    assert isinstance(limit_exp, int) or limit_exp is None
    assert limit_exp is None or limit_exp >= 0
    assert allow_exprs is True or allow_exprs is False

    discrete, parsed = conv.str_to_real_or_implicit(s, w, p, limit_exp=limit_exp)

    if parsed is None:
        if allow_exprs:
            discrete = False
            parsed = FReal(s)
        else:
            discrete = None
            parsed = ('Unable to parse {} as real number literal or binary floating point representation.'
                      .format(repr(s)))

    return discrete, parsed

# call after parse_input
def explain_input(x, w, p, discrete, parsed, hit = False, link = False):
    if discrete:
        S, E, T = parsed
        w_prime = E.n
        p_prime = T.n + 1
    else:
        R = parsed
        w_prime = w
        p_prime = p

    s = 'received input (w={:d}, p={:d}):'.format(w, p)
    s += '\n  '
    if link:
        s += '{} '.format(permalink(x, w_prime, p_prime))
    s += repr(x)
    if discrete and (w != w_prime or p != p_prime):
        s += '\nNOTE: changed format to (w={:d}, p={:d})'.format(w_prime, p_prime)

    s += '\n\ninterpreted as'
    if discrete:
        s += ' implicit triple'
        if hit:
            s += ' (cached)'
        s += ':\n  '
        s_tup = stup(S, E, T)
        if link:
            s += '{} '.format(permalink(s_tup, w_prime, p_prime))
        s += s_tup
    else:
        s += ' extended real number'
        if hit:
            s += ' (cached)'
        s += ':\n  '
        if link:
            R_str = str(R)
            if R.isnan:
                R_str += str(R.nan_payload)
            s += '{} '.format(permalink(R_str, w_prime, p_prime))
            s += R_str

    if not discrete:
        pretty_repr = webcontent.indent(conv.real_to_pretty_string(R), '  ')
        s += '\n\n' + pretty_repr

    return s

# call after parse_input
def process_parsed_input(s, w, p, discrete, parsed, link = False):
    if discrete:
        S, E, T = parsed
        f_descr = describe_float(S, E, T)
        return explain_float(f_descr, link=link)
    else:
        R = parsed
        r_descr = describe_real(R, w, p)
        if r_descr.get('exact', False):
            S, E, T = r_descr['S'], r_descr['E'], r_descr['T']
            f_descr = describe_float(S, E, T)
            return explain_float(f_descr, link=link)
        else:
            return explain_real(r_descr, link=link)

def process_format(w, p, link = False):
    assert isinstance(w, int)
    assert w >= 2
    assert isinstance(p, int)
    assert p >= 2

    fmt_descr = describe_format(w, p)
    return explain_format(fmt_descr, link=link)

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
        if args.f:
            print(process_format(args.w, args.p))
        else:
            discrete, parsed = parse_input(args.x, args.w, args.p)
            if discrete is None:
                # parsing failed, so the value is an error string
                print(parsed)
            else:
                print(explain_input(args.x, args.w, args.p, discrete, parsed))
                print()
                print(process_parsed_input(args.x, args.w, args.p, discrete, parsed))