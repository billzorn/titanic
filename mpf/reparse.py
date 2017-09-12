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
cre_ep = re.escape('[ep]')

def cre_frac(r_opt, r_nat):
    return cre_any(r_nat + opt(cre_dot),
                   r_opt + cre_dot + r_nat)

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
                    opt(group(cre_ep + group(cre_decint, name='hex_e'))),
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

cre_bin1 = r'0b0|0b1|#b0|#b1'
cre_csep = r'[,\s]'
cre_bv = cre_any(group(cre_binpre) + cre_binnat, group(cre_hexpre) + cre_hexnat)

re_bv  = re.compile(group(cre_bv, name='bv'), flags=cf)

re_tup = re.compile(cre_any(cre_ws(opt(group(cre_ws('fp',
                                                    cre_csep))),
                                   group(cre_bin1, name='tup0'),
                                   cre_csep,
                                   group(cre_bv, name='tup1'),
                                   cre_csep,
                                   group(cre_bv, name='tup2')),
                            cre_parens(cre_ws(opt(group(cre_ws('fp',
                                                               cre_csep))),
                                              group(cre_bin1, name='ptup0'),
                                              cre_csep,
                                              group(cre_bv, name='ptup1'),
                                              cre_csep,
                                              group(cre_bv, name='ptup2')))),
                    flags=cf)

# enum to define results

class Result(Enum):
    NAN = 0
    INF = 1
    FPC = 2
    NUM = 4
    ORD = 5
    BV = 6
    TUP = 7

# parsing

frac_re = re.compile(group(r'[^.]*', name='left') +
                     opt(r'[.]') +
                     group(r'[^.]*', name='right'),
                     flags=cf)

nodot_re = re.compile(r'[^.]*', flags=cf)

bv_re = re.compile(cre_any(group(group(cre_binpre) + group(cre_binnat, name='bbv')),
                           group(group(cre_hexpre) + group(cre_hexnat, name='hbv'))),
                   flags=cf)

# this is intended to be called only on something that already matched a cre_xxxfrac
def parse_frac(s, base):
    m = frac_re.fullmatch(s.strip('0'))
    left = m.group('left')
    right = m.group('right')
    return int(left + right, base), base ** len(right)

def parse_pm(s):
    if s == '-':
        return -1
    elif s is None or s == '+' or s == '':
        return 1

# this is intended to be called only on something that already matched re_bv
def parse_bv(s):
    m = bv_re.fullmatch(s)
    if m.group('bbv'):
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
                            re_tup.pattern),
                    flags=cf)

# the main parsing interface
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
            return Result.NUM, (sign * top, bot, base, exp,)

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
            return Result.NUM, (sign * top, bot, base, exp,)

    elif m.group('dec_frac'):
        sign = parse_pm(m.group('dec_pm'))
        top, bot = parse_frac(m.group('dec_frac'), 10)
        if m.group('dec_e'):
            base, exp = 10, int(m.group('dec_e'))
        else:
            base, exp = 10, None
        return Result.NUM, (sign * top, bot, base, exp,)

    elif m.group('exp_frac'):
        sign = parse_pm(m.group('exp_pm'))
        top, bot = parse_frac(m.group('dec_frac'), 10)
        if m.group('exp_base'):
            base, exp = int(m.group('exp_base')), int(m.group('exp_e'))
        else:
            base, exp = int(m.group('exp_pbase')), int(m.group('exp_pe'))
        return Result.NUM, (sign * top, bot, base, exp,)

    elif m.group('rat_top'):
        top, bot = int(m.group('rat_top')), int(m.group('rat_bot'))
        if (top < 0) != (bot < 0):
            sign = -1
        else:
            sign = 1
        return Result.NUM, (sign * abs(top), abs(bot), None, None,)

    elif m.group('ord'):
        sign = parse_pm(m.group('ord_pm'))
        return Result.ORD, (sign * int(m.group('ord')),)

    elif m.group('tup0'):
        return Result.TUP, (parse_bv(m.group('tup0')),
                            parse_bv(m.group('tup1')),
                            parse_bv(m.group('tup2')),)

    elif m.group('ptup0'):
        return Result.TUP, (parse_bv(m.group('ptup0')),
                            parse_bv(m.group('ptup1')),
                            parse_bv(m.group('ptup2')),)

    else:
        raise ValueError('failed to parse matching input {}'.format(repr(s)))
