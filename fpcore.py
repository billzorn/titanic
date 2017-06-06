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
    '+' : ast.Add,
    '-' : ast.Sub,
    '*' : ast.Mul,
    '/' : ast.Div,
    
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
    



s = '''(FPCore 
 (x)
 :name "Rosa's Benchmark"
 :cite (darulova-kuncak-2014)
 (- (* 0.954929658551372 x) (* 0.12900613773279798 (* (* [x x]) x))))
'''
e = sexpdata.loads(s)

print(e)

x = sexpdata.dumps(e)

print(x)

print(' '.join(x.strip().split()) == ' '.join(s.strip().split()))








print('\n\n\n\n')



class Foo(object):
    name = 'foofoo'

    def __init__(self):
        print(self.__class__)
        self.iname = self.__class__.name

class Bar(Foo):
    name = 'barbar'
