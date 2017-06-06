from sexpdata import sexpdata
import ast

# from WolframAlpha, to 100 decimal places
constants = {
    'E'        : '2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274',
    'LOG2E'    : '1.4426950408889634073599246810018921374266459541529859341354494069311092191811850798855266228935063445',
    'LOG10E'   : '0.434294481903251827651128918916605082294397005803666566114453783165864649208870774729224949338431748319',
    'LN2'      : '0.69314718055994530941723212145817656807550013436025525412068000949339362196969471560586332699641868754',
    'LN10'     : '2.3025850929940456840179914546843642076011014886287729760333279009675726096773524802359972050895982983',
    'PI'       : '3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170680',
    'PI_2'     : '1.5707963267948966192313216916397514420985846996875529104874722961539082031431044993140174126710585340',
    'PI_4'     : '0.7853981633974483096156608458198757210492923498437764552437361480769541015715522496570087063355292670',
    '1_PI'     : '0.31830988618379067153776752674502872406891929148091289749533468811779359526845307018022760553250617191',
    '2_PI'     : '0.63661977236758134307553505349005744813783858296182579499066937623558719053690614036045521106501234382',
    '2_SQRTPI' : '1.1283791670955125738961589031215451716881012586579977136881714434212849368829868289734873204042147269',
    'SQRT2'    : '1.4142135623730950488016887242096980785696718753769480731766797379907324784621070388503875343276415727',
    'SQRT1_2'  : '0.70710678118654752440084436210484903928483593768847403658833986899536623923105351942519376716382078637',
    'INFINITY' : '+inf',
    'NAN'      : 'nan',
}

def impl_by_pairs(op, conj):
    def impl(*args):
        if len(args) > 2:
            return conj(*(op(args[i], args[i+1]) for i in range(len(args)-1)))
        else:
            return op(*args)
    return impl

def impl_all_pairs(op, conj):
    def impl(*args):
        if len(args) > 2:
            return conj(*(op(args[i], args[j]) for i in range(len(args)-1) for j in range(i+1,len(args))))
        else:
            return op(*args)    
    return impl

def is_number(e):
    return isinstance(e, sexpdata.Number)
def get_number(e):
    return e.value()

def is_constant(e):
    return isinstance(e, sexpdata.Symbol) and e.value() in constants
def get_constant(e):
    return constants[e.value()]

def is_symbol(e):
    return isinstance(e, sexpdata.Symbol)
def get_symbol(e):
    return e.value()

def is_parens(e):
    return isinstance(e, list)
def get_parens(e):
    return e

def is_brackets(e):
    return isinstance(e, sexpdata.Bracket)
def get_brackets(e):
    return e.value()

operations = {
    ast.Add.name : ast.Add,
    ast.Sub.name : ast.Sub,
    ast.Mul.name : ast.Mul,
    ast.Div.name : ast.Div,
    ast.Sqrt.name : ast.Sqrt,
    ast.Neg.name : ast.Neg,
    ast.LT.name : impl_by_pairs(ast.LT, ast.And),
    ast.GT.name : impl_by_pairs(ast.GT, ast.And),
    ast.LEQ.name : impl_by_pairs(ast.LEQ, ast.And),
    ast.GEQ.name : impl_by_pairs(ast.GEQ, ast.And),
    ast.EQ.name : impl_by_pairs(ast.EQ, ast.And),
    ast.NEQ.name : impl_all_pairs(ast.NEQ, ast.And),
    ast.And.name : ast.And,
    ast.Or.name : ast.Or,
    ast.Not.name : ast.Not,
}

def parse_expr(e):
    if is_parens(e):
        op = get_symbol(e[0])
        es = e[1:]
        # special case for unary negation
        if op == '-' and len(es) == 1:
            op = 'neg'
        # unimplemented
        if op in {'if', 'let', 'while'}:
            raise ValueError('op {} is unimplemented'.format(op))
        # build ast
        elif op in operations:
            return operations[op](*(parse_expr(child) for child in es))
        else:
            raise ValueError('unknown operation {}'.format(op))
    else:
        if is_number(e):
            return ast.Val(get_number(e))
        elif is_constant(e):
            return ast.Val(get_constant(e))
        elif is_symbol(e):
            return ast.Var(get_symbol(e))
        else:
            raise ValueError('what is this? {}'.format(repr(e)))

class FPCore(object):
    def __init__(self, s):
        core = sexpdata.loads(s)
        if len(core) < 3 or core[0] != sexpdata.Symbol('FPCore'):
            raise ValueError('invalid FPCore: expected ( FPCore (symbol*) property* expr )')
        core_args = core[1]
        core_expr = core[-1]
        core_properties = core[2:-1]
        self.core = core

        if len(core_properties) % 2 != 0:
            raise ValueError('invalid properties: missing key or value')
        self.properties = {}
        for i in range(0, len(core_properties), 2):
            if not is_symbol(core_properties[i]):
                raise ValueError('invalid properties: key {} is not a symbol'.format(core_properties[i]))
            k = get_symbol(core_properties[i])
            if not k.startswith(':'):
                raise ValueError('invalid properties: key {} does not start with ":"'.format(k))
            self.properties[k[1:]] = core_properties[i+1]

        self.args = tuple(get_symbol(e) for e in core_args)
        self.expr = parse_expr(core_expr)
        self.name = self.properties['name'] if 'name' in self.properties else None
        self.pre = parse_expr(self.properties['pre']) if 'pre' in self.properties else None

    def __str__(self):
        return 'FPCore( {}\n{}{}{}\n)'.format(
            self.args,
            ':name "' + self.name + '"\n' if self.name else '',
            ':pre ' + str(self.pre) + '\n' if self.pre else '',
            self.expr)

    def __repr__(self):
        return 'FPCore(' + repr(sexpdata.dumps(self.core)) + ')'


if __name__ == '__main__':
    examples = [
'''(FPCore (x)
 :name "NMSE example 3.1"
 :cite (hamming-1987)
 :pre (>= x 0)
 (- (sqrt (+ x 1)) (sqrt x)))
''',
'''(FPCore (x)
 :name "NMSE example 3.6"
 :cite (hamming-1987)
 :pre (>= x 0)
 (- (/ 1 (sqrt x)) (/ 1 (sqrt (+ x 1)))))
''',
'''(FPCore (x)
 :name "NMSE problem 3.3.1"
 :cite (hamming-1987)
 :pre (!= x 0)
 (- (/ 1 (+ x 1)) (/ 1 x)))
''',
'''(FPCore (x)
  :name "NMSE problem 3.3.3"
  :cite (hamming-1987)
  :pre (!= x 0)
  (+ (- (/ 1 (+ x 1)) (/ 2 x)) (/ 1 (- x 1))))
''',
'''(FPCore (a b c)
 :name "NMSE p42, positive"
 :cite (hamming-1987)
 :pre (and (>= (* b b) (* 4 (* a c))) (!= a 0))
 (/ (+ (- b) (sqrt (- (* b b) (* 4 (* a c))))) (* 2 a)))
''',
'''(FPCore (a b c)
 :name "NMSE p42, negative"
 :cite (hamming-1987)
 :pre (and (>= (* b b) (* 4 (* a c))) (!= a 0))
 (/ (- (- b) (sqrt (- (* b b) (* 4 (* a c))))) (* 2 a)))
''',
'''(FPCore (a b2 c)
 :name "NMSE problem 3.2.1, positive"
 :cite (hamming-1987)
 :pre (and (>= (* b2 b2) (* a c)) (!= a 0))
 (/ (+ (- b2) (sqrt (- (* b2 b2) (* a c)))) a))
''',
'''(FPCore (a b2 c)
  :name "NMSE problem 3.2.1, negative"
  :cite (hamming-1987)
  :pre (and (>= (* b2 b2) (* a c)) (!= a 0))
  (/ (- (- b2) (sqrt (- (* b2 b2) (* a c)))) a))
''',
]
    import z3
    import random
    sort = 32
    z3sort = z3.FPSort(8, 24)
    for s in examples:
        core = FPCore(s)
        print(core)
        rargs = {a : str(random.random()) for a in core.args}
        print(rargs)
        native_ok = core.pre(rargs)
        native = core.expr(rargs) if native_ok else None
        print('  native : {} : {}'.format(native_ok, native))
        np_ok = core.pre.apply_np(rargs, sort)
        np_output = core.expr.apply_np(rargs, sort)
        print('  numpy({}) : {} : {}'.format(sort, np_ok, np_output))
        mp_ok = core.pre.apply_mp(rargs, sort)
        mp_output = core.expr.apply_mp(rargs, sort)
        print('  mpfr({}) : {} : {}'.format(sort, mp_ok, mp_output))

        z3_pre_r = core.pre.apply_z3(rargs, sort)
        print(z3_pre_r)
        z3_expr_r = core.expr.apply_z3(rargs, sort)
        print(z3_expr_r)

        z3args = {a : z3.FP(a, z3sort) for a in core.args}
        z3_pre = core.pre.apply_z3(z3args, sort)
        print(z3_pre)
        z3_expr = core.expr.apply_z3(z3args, sort)
        print(z3_expr)

        print()
