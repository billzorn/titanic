import re

# helpers and constants

cf = re.IGNORECASE
ws = r'\s*'
bar = r'|'

def group(s, name=None):
    if name is None:
        return '(' + s + ')'
    else:
        return '(P<' + name + '>' + s + ')'

def cre_ws(*strs):
    return ws.join(strs)

def cre_parens(s, ws=True):
    if ws:
        return cre_sw(re.escape('('), s, re.escape(')'))
    else:
        return re.escape('(') + s + re.escape(')')

def cre_any(*strs, esc=False):
    if esc:
        return bar.join(re.escape(s) for s in strs)
    else:
        return bar.join(strs)

# common expressions
cre_pm = r'[-+]?'

cre_binpre = r'0b|#b'
cre_binnz = r'[0-1]*[1][0-1]*'
cre_binopt = r'[0-1]*'
cre_binnat = r'[0-1]+'
cre_binint = cre_pm + cre_binnat

cre_hexpre = r'0x|#x'
cre_hexnz = r'[0-9a-f]*[1-9a-f][0-9a-f]*'
cre_hexopt = r'[0-9a-f]*'
cre_hexnat = r'[0-9a-f]+'
cre_hexint = cre_pm + cre_hexnat

cre_ordpre = r'0n|#n'

cre_decpre = r'0d|#d'
cre_decnz = r'[0-9]*[1-9][0-9]*'
cre_decopt = r'[0-9]*'
cre_decnat = r'[0-9]+'
cre_decint = cre_pm + cre_decnat

cre_dot = re.escape('.')

def cre_frac(r_opt, r_nat, r_int):
    return cre_any(r_int + cre_dot + '?',
                   cre_pm + r_opt + cre_dot + r_nat)

cre_binfrac = cre_frac(cre_binopt, cre_binnat, cre_binint)
cre_hexfrac = cre_frac(cre_hexopt, cre_hexnat, cre_hexint)
cre_decfrac = cre_frac(cre_decopt, cre_decnat, cre_decint)

cre_e = re.escape('e')

def cre_exp(r_int, r_frac):
    return r_frac + cre_e + r_int

cre_binexp = cre_exp(cre_binint, cre_binfrac)
cre_hexexp = cre_exp(cre_hexint, cre_hexfrac)
cre_decexp = cre_exp(cre_decint, cre_decfrac)

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
re_inf = re.compile(group(cre_any(*inf_strs, esc=True), name='inf'),
                    flags=cf)

nan_strs = (
    'nan',
    'snan',
)
preferred_nan_str = 'nan'
re_nan = re.compile(group(cre_any(*nan_strs, esc=True), name='nan') +
                    group(cre_any(cre_decint, parens(cre_decint)), name='nan_payload') + r'?',
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
re_fpc = re.compile(group(cre_any(*fpc_constants.keys(), esc=True), name='fpc'),
                    flags=cf)

# real numbers

re_dec = re.compile(group(cre_decfrac, name='dec'), flags=cf)
re_idec = re.compile(group(cre_decint, name='idec'), flags=cf)
re_edec = re.compile(group(cre_decexp, name='edec'), flags=cf)

cre_z3pow = cre_ws(cre_decnz, cre_pow, cre_decint)
re_exp = re.compile(cre_ws(group(cre_decfrac, name='exp_left'),
                           cre_mul,
                           group(cre_any(cre_z3pow, parens(cre_z3pow)), name='exp_right')),
                    flags=cf)

re_rat = re.compile(group(cre_ws(cre_decint,
                                 cre_div,
                                 cre_pm + cre_decnz), name='rat'),
                    flags=cf)

# bitvectors and ordinals

re_bin = re.compile(group(cre_pm + cre_binpre + cre_binnat, name='bin'), flags=cf)
re_hex = re.compile(group(cre_pm + cre_hexpre + cre_hexnat, name='hex'), flags=cf)
re_ord = re.compile(group(cre_pm + cre_ordpre + cre_decnat, name='ord'), flags=cf)

# structured triples

cre_bin1 = r'0b0|0b1'
cre_csopt = re.escape(',') + r'?'

re_tup =
