import re
from enum import Enum

# helpers and constants

cf = re.IGNORECASE
ws = r'\s'
wss = ws + r'*'
bar = r'|'

def group(s, name=None):
    if name is None:
        return '(' + s + ')'
    else:
        return '(?P<' + name + '>' + s + ')'

def opt(s):
    return s + '?'

def cre_ws(*strs):
    return wss.join(strs)

def cre_parens(s, ws=True):
    if ws:
        return cre_ws(re.escape('('), s, re.escape(')'))
    else:
        return re.escape('(') + s + re.escape(')')

def cre_any(*strs, esc=False):
    if esc:
        return bar.join(re.escape(s) for s in strs)
    else:
        return bar.join(strs)

# common expressions
cre_pm = r'[-+]'

cre_binpre = r'0b|#b'
cre_binnz = r'[0-1]*[1][0-1]*'
cre_binopt = r'[0-1]*'
cre_binnat = r'[0-1]+'
cre_binint = opt(cre_pm) + cre_binnat

cre_hexpre = r'0x|#x'
cre_hexnz = r'[0-9a-f]*[1-9a-f][0-9a-f]*'
cre_hexopt = r'[0-9a-f]*'
cre_hexnat = r'[0-9a-f]+'
cre_hexint = opt(cre_pm) + cre_hexnat

cre_decpre = r'0d|#d'
cre_decnz = r'[0-9]*[1-9][0-9]*'
cre_decopt = r'[0-9]*'
cre_decnat = r'[0-9]+'
cre_decint = opt(cre_pm) + cre_decnat

cre_ordpre = r'0n|#n'
cre_dot = re.escape('.')
cre_e = re.escape('e')
cre_p = re.escape('p')
cre_ep = r'[ep]'

def cre_frac(r_opt, r_nat):
    return cre_any(r_opt + cre_dot + r_nat,
                   r_nat + opt(cre_dot))

cre_binfrac = cre_frac(cre_binopt, cre_binnat)
cre_hexfrac = cre_frac(cre_hexopt, cre_hexnat)
cre_decfrac = cre_frac(cre_decopt, cre_decnat)

cre_pypow = re.escape('**')
cre_carat = re.escape('^')
cre_pow = cre_any(cre_pypow, cre_carat)
cre_mul = re.escape('*')
cre_div = re.escape('/')

# named values

inf_strs = (
    'inf',
    'infinity',
    'oo',
    u'\u221e',
)
preferred_inf_str = 'inf'
re_inf = re.compile(opt(group(cre_pm, name='inf_pm')) +
                    group(cre_any(*inf_strs, esc=True), name='inf'),
                    flags=cf)

nan_strs = (
    'nan',
    'snan',
)
preferred_nan_str = 'nan'
re_nan = re.compile(opt(group(cre_pm, name='nan_pm')) +
                    group(cre_any(*nan_strs, esc=True), name='nan') +
                    opt(group(cre_any(group(cre_decint, name='nan_payload'),
                                      cre_parens(group(cre_decint, name='nan_ppayload'))))),
                    flags=cf)

fpc_constants = {
    'e'        : 'E',
    'log2e'    : '1/ln(2)',
    'log10e'   : '1/ln(10)',
    'ln2'      : 'ln(2)',
    'ln10'     : 'ln(10)',
    'pi'       : 'pi',
    'pi_2'     : 'pi/2',
    'pi_4'     : 'pi/4',
    '2_sqrtpi' : '2/sqrt(pi)',
    'sqrt2'    : 'sqrt(2)',
    'sqrt1_2'  : '1/sqrt(2)',
}
re_fpc = re.compile(opt(group(cre_pm, name='fpc_pm')) +
                    group(cre_any(*fpc_constants.keys(), esc=True), name='fpc'),
                    flags=cf)

# numbers and ordinals

re_bin = re.compile(opt(group(cre_pm, name='bin_pm')) +
                    group(cre_binpre) +
                    group(cre_binfrac, name='bin_frac') +
                    opt(group(cre_ep + group(cre_decint, name='bin_e'))),
                    flags=cf)

re_hex = re.compile(opt(group(cre_pm, name='hex_pm')) +
                    group(cre_hexpre) +
                    group(cre_hexfrac, name='hex_frac') +
                    opt(group(cre_p + group(cre_decint, name='hex_e'))),
                    flags=cf)

re_dec = re.compile(opt(group(cre_pm, name='dec_pm')) +
                    opt(group(cre_decpre)) +
                    group(cre_decfrac, name='dec_frac') +
                    opt(group(cre_e + group(cre_decint, name='dec_e'))),
                    flags=cf)

re_ord = re.compile(opt(group(cre_pm, name='ord_pm')) +
                    group(cre_ordpre) +
                    group(cre_decnat, name='ord'),
                    flags=cf)

# rationals

cre_z3pow = cre_ws(cre_decnz, group(cre_pow), cre_decint)

re_exp = re.compile(cre_ws(group(opt(group(cre_pm, name='exp_pm')) +
                                 group(cre_decfrac, name='exp_frac')),
                           cre_mul,
                           group(cre_any(cre_ws(group(cre_decnz, name='exp_base'),
                                                group(cre_pow),
                                                group(cre_decint, name='exp_e')),
                                         cre_parens(cre_ws(group(cre_decnz, name='exp_pbase'),
                                                           group(cre_pow),
                                                           group(cre_decint, name='exp_pe')))))),
                    flags=cf)

re_rat = re.compile(cre_ws(group(cre_decint, name='rat_top'),
                           cre_div,
                           group(opt(cre_pm) + cre_decnz, name='rat_bot')),
                    flags=cf)

# bitvectors and structured triples

cre_0 = re.escape('0')
cre_1 = re.escape('1')
cre_01 = cre_any(cre_0, cre_1)
cre_bin01 = cre_any(group(cre_binpre) + cre_0, group(cre_binpre) + cre_1)
cre_1bit = cre_any(cre_01, cre_bin01)
cre_csep = r'[,\s]'
cre_bv = cre_any(group(cre_binpre) + cre_binnat, group(cre_hexpre) + cre_hexnat)

re_bv  = re.compile(group(cre_bv, name='bv'), flags=cf)

re_tup3 = re.compile(cre_any(cre_ws(opt(group(cre_ws('fp',
                                                     cre_csep))),
                                    group(cre_1bit, name='tup3_0'),
                                    cre_csep,
                                    group(cre_bv, name='tup3_1'),
                                    cre_csep,
                                    group(cre_bv, name='tup3_2')),
                             cre_parens(cre_ws(opt(group(cre_ws('fp',
                                                                cre_csep))),
                                               group(cre_1bit, name='ptup3_0'),
                                               cre_csep,
                                               group(cre_bv, name='ptup3_1'),
                                               cre_csep,
                                               group(cre_bv, name='ptup3_2')))),
                     flags=cf)

re_tup4 = re.compile(cre_any(cre_ws(opt(group(cre_ws('fp',
                                                     cre_csep))),
                                    group(cre_1bit, name='tup4_0'),
                                    cre_csep,
                                    group(cre_bv, name='tup4_1'),
                                    cre_csep,
                                    group(cre_1bit, name='tup4_2'),
                                    cre_csep,
                                    group(cre_bv, name='tup4_3')),
                             cre_parens(cre_ws(opt(group(cre_ws('fp',
                                                                cre_csep))),
                                               group(cre_1bit, name='ptup4_0'),
                                               cre_csep,
                                               group(cre_bv, name='ptup4_1'),
                                               cre_csep,
                                               group(cre_1bit, name='ptup4_2'),
                                               cre_csep,
                                               group(cre_bv, name='ptup4_3')))),
                     flags=cf)


# enum to define results

class Result(Enum):
    NAN = 0
    INF = 1
    FPC = 2
    NUM = 3
    ORD = 4
    BV = 5
    ITUP = 6
    ETUP = 7

# parsing

frac_re = re.compile(group(r'[^.]*', name='left') +
                     opt(r'[.]') +
                     group(r'[^.]*', name='right'),
                     flags=cf)

nodot_re = re.compile(r'[^.]*', flags=cf)

bv_re = re.compile(cre_any(group(cre_01, name='bv1'),
                           group(cre_binpre) + group(cre_binnat, name='bbv'),
                           group(cre_hexpre) + group(cre_hexnat, name='hbv')),
                   flags=cf)

# this is intended to be called only on something that already matched a cre_xxxfrac
def parse_frac(s, base):
    m = frac_re.fullmatch(s)
    left = m.group('left').lstrip('0')
    right = m.group('right').rstrip('0')
    x = left + right
    if len(x) == 0:
        return 0, 1
    else:
        return int(x, base), base ** len(right)

def parse_pm(s):
    if s == '-':
        return -1
    elif s is None or s == '+' or s == '':
        return 1

# this is intended to be called only on something that already matched re_bv
def parse_bv(s):
    m = bv_re.fullmatch(s)
    if m.group('bv1'):
        return int(m.group('bv1'), 2), 1
    elif m.group('bbv'):
        return int(m.group('bbv'), 2), len(m.group('bbv'))
    else:
        return int(m.group('hbv'), 16), len(m.group('hbv')) * 4

re_any = re.compile(cre_any(re_inf.pattern,
                            re_nan.pattern,
                            re_fpc.pattern,
                            re_bin.pattern,
                            re_hex.pattern,
                            re_dec.pattern,
                            re_ord.pattern,
                            re_exp.pattern,
                            re_rat.pattern,
                            re_tup3.pattern,
                            re_tup4.pattern,),
                    flags=cf)

# the main parsing interface:
#   if we cannot parse this string, return (None, (),)
#   otherwise, return one of:
#     Result.INF, (the sign of the infinity,)
#     Result.NAN, (the sign of the NaN, the payload or None if there wasn't one,)
#     Result.FPC, (the sign of the constant, a sympifyable string representing the constant,)
#     Result.NUM, (the sign of the number, the unsigned numerator, the nonzero unsigned denominator,
#                  the base of the exponent or None if this was a num/denom fraction,
#                  the exponent or None if it was not given explicitly and should be 0,)
#     Result.BV,  (the unsigned integer value of the BV, the number of bits,)
#     Result.ORD, (the signed integer ordinal,)
#     Result.TUP, (as for bitvectors; one pair for each of of S E T or S E C)
def reparse(s):
    assert isinstance(s, str)

    m = re_any.fullmatch(s.strip())

    if not m:
        return None, ()

    elif m.group('inf'):
        sign = parse_pm(m.group('inf_pm'))
        return Result.INF, (sign,)

    elif m.group('nan'):
        sign = parse_pm(m.group('nan_pm'))
        if m.group('nan_payload'):
            payload = int(m.group('nan_payload'))
        elif m.group('nan_ppayload'):
            payload = int(m.group('nan_ppayload'))
        else:
            payload = None
        return Result.NAN, (sign, payload,)

    elif m.group('fpc'):
        sign = parse_pm(m.group('fpc_pm'))
        expr = fpc_constants[m.group('fpc').lower()]
        return Result.FPC, (sign, expr,)

    elif m.group('bin_frac'):
        sign = parse_pm(m.group('bin_pm'))
        top, bot = parse_frac(m.group('bin_frac'), 2)
        if m.group('bin_e'):
            base, exp = 2, int(m.group('bin_e'))
        else:
            base, exp = 2, None
        if sign == 1 and exp is None and nodot_re.fullmatch(m.group('bin_frac')):
            return Result.BV, (top, len(m.group('bin_frac')),)
        else:
            return Result.NUM, (sign, top, bot, base, exp,)

    elif m.group('hex_frac'):
        sign = parse_pm(m.group('hex_pm'))
        top, bot = parse_frac(m.group('hex_frac'), 16)
        if m.group('hex_e'):
            base, exp = 16, int(m.group('hex_e'))
        else:
            base, exp = 16, None
        if sign == 1 and exp is None and nodot_re.fullmatch(m.group('hex_frac')):
            return Result.BV, (top, len(m.group('hex_frac')) * 4,)
        else:
            return Result.NUM, (sign, top, bot, base, exp,)

    elif m.group('dec_frac'):
        sign = parse_pm(m.group('dec_pm'))
        top, bot = parse_frac(m.group('dec_frac'), 10)
        if m.group('dec_e'):
            base, exp = 10, int(m.group('dec_e'))
        else:
            base, exp = 10, None
        return Result.NUM, (sign, top, bot, base, exp,)

    elif m.group('exp_frac'):
        sign = parse_pm(m.group('exp_pm'))
        top, bot = parse_frac(m.group('exp_frac'), 10)
        if m.group('exp_base'):
            base, exp = int(m.group('exp_base')), int(m.group('exp_e'))
        else:
            base, exp = int(m.group('exp_pbase')), int(m.group('exp_pe'))
        return Result.NUM, (sign, top, bot, base, exp,)

    elif m.group('rat_top'):
        top, bot = int(m.group('rat_top')), int(m.group('rat_bot'))
        if (top < 0) != (bot < 0):
            sign = -1
        else:
            sign = 1
        return Result.NUM, (sign, abs(top), abs(bot), None, None,)

    elif m.group('ord'):
        sign = parse_pm(m.group('ord_pm'))
        return Result.ORD, (sign * int(m.group('ord')),)

    elif m.group('tup3_0'):
        return Result.ITUP, (parse_bv(m.group('tup3_0')),
                             parse_bv(m.group('tup3_1')),
                             parse_bv(m.group('tup3_2')),)

    elif m.group('ptup3_0'):
        return Result.ITUP, (parse_bv(m.group('ptup3_0')),
                             parse_bv(m.group('ptup3_1')),
                             parse_bv(m.group('ptup3_2')),)

    elif m.group('tup4_0'):
        return Result.ETUP, (parse_bv(m.group('tup4_0')),
                             parse_bv(m.group('tup4_1')),
                             parse_bv(m.group('tup4_2')),
                             parse_bv(m.group('tup4_3')),)

    elif m.group('ptup4_0'):
        return Result.ETUP, (parse_bv(m.group('ptup4_0')),
                             parse_bv(m.group('ptup4_1')),
                             parse_bv(m.group('ptup4_2')),
                             parse_bv(m.group('ptup4_3')),)

    else:
        raise ValueError('failed to parse matching input {}'.format(repr(s)))
