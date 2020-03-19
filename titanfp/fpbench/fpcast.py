"""A reusable AST for manipulating or executing FPCores in python."""

import typing


def sexp_to_string(e):
    if isinstance(e, list) or isinstance(e, tuple):
        return '(' + ' '.join((sexp_to_string(x) for x in e)) + ')'
    else:
        return str(e)

def annotation_to_string(e, props, shape=None):
    if props:
        if shape:
            return ('(! ' + ''.join((':' + k + ' ' + sexp_to_string(v) + ' ' for k, v in props.items()))
                    + str(e) + ' '.join((sexp_to_string(dim) for dim in shape)) + ')')
        else:
            return '(! ' + ''.join((':' + k + ' ' + sexp_to_string(v) + ' ' for k, v in props.items())) + str(e) + ')'
    else:
        if shape:
            return '(' + str(e) + ' '.join((sexp_to_string(dim) for dim in shape)) + ')'
        else:
            return str(e)

def diff_props(global_props, local_props):
    if global_props is None:
        if local_props is None:
            return {}, {}
        else:
            return local_props, local_props
    else:
        if local_props is None:
            return global_props, {}
        else:
            all_props = {}
            all_props.update(global_props)
            all_props.update(local_props)
            new_props = {k:v for k, v in local_props.items() if k not in global_props or global_props[k] != v}
            return all_props, new_props

def update_props(old_props, new_props):
    if old_props:
        if new_props:
            updated_props = {}
            updated_props.update(old_props)
            updated_props.update(new_props)
            return updated_props
        else:
            return old_props
    else:
        return new_props


# base ast class

class Expr(object):
    name: str = 'Expr'

    def subexprs(self):
        raise NotImplementedError()

    def replace_subexprs(self, exprs):
        raise NotImplementedError()

    def copy(self):
        exprs = [[e.copy() for e in es] for es in self.subexprs()]
        return self.replace_subexprs(exprs)

    def remove_annotations(self):
        exprs = [[e.remove_annotations() for e in es] for es in self.subexprs()]
        return self.replace_subexprs(exprs)

    def condense_annotations(self, global_props=None, local_props=None):
        all_props, new_props = diff_props(global_props, local_props)
        exprs = [[e.condense_annotations(all_props, None) for e in es] for es in self.subexprs()]
        if new_props:
            return Ctx(new_props, self.replace_subexprs(exprs))
        else:
            return self.replace_subexprs(exprs)

    def canonicalize_annotations(self, global_props=None):
        exprs = [[e.canonicalize_annotations(global_props) for e in es] for es in self.subexprs()]
        return self.replace_subexprs(exprs)

    def merge_annotations(self, annotations, local_props=None):
        new_props = update_props(local_props, annotations.get(id(self)))
        exprs = [[e.merge_annotations(annotations, None) for e in es] for es in self.subexprs()]
        if new_props:
            return Ctx(new_props, self.replace_subexprs(exprs))
        else:
            return self.replace_subexprs(exprs)


# arbitrary s-expression data (usually from properties)

class Data(Expr):
    name: str = 'Data'

    def __init__(self, value) -> None:
        self.value = value

    def __str__(self):
        return sexp_to_string(self.value)

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.value) + ')'

    def __eq__(self, other):
        try:
            return self.value == other.value
        except AttributeError:
            return self.value == other

    def __hash__(self):
        return hash(self.value)

    def subexprs(self):
        return []

    def replace_subexprs(self, exprs):
        return type(self)(self.value)

    def is_number(self):
        return isinstance(self.value, Val) and not isinstance(self.value, Constant)

    def as_number(self, strict=False):
        if isinstance(self.value, Val) and not isinstance(self.value, Constant):
            return self.value
        elif strict:
            raise TypeError('data is not a number')
        else:
            return None

    def is_symbol(self):
        return isinstance(self.value, Var) or isinstance(self.value, Constant)

    def as_symbol(self, strict=False):
        if isinstance(self.value, Var) or isinstance(self.value, Constant):
            return self.value.value
        elif strict:
            raise TypeError('data is not a symbol')
        else:
            return None

    def is_string(self):
        return isinstance(self.value, String)

    def as_string(self, strict=False):
        if isinstance(self.value, String):
            return self.value.value
        elif strict:
            raise TypeError('data is not a string')
        else:
            return None

    def is_list(self):
        return isinstance(self.value, tuple)

    def as_list(self, strict=False):
        if isinstance(self.value, tuple):
            return self.value
        elif strict:
            raise TypeError('data is not a list')
        else:
            return None


# operations

class NaryExpr(Expr):
    name: str = 'NaryExpr'

    def __init__(self, *children: Expr) -> None:
        self.children: typing.List[Expr] = children

    def __str__(self):
        return '(' + self.name + ''.join((' ' + str(child) for child in self.children)) + ')'

    def __repr__(self):
        return type(self).__name__ + '(' + ', '.join((repr(child) for child in self.children)) + ')'

    def subexprs(self):
        return [self.children]

    def replace_subexprs(self, exprs):
        (children,) = exprs
        return type(self)(*children)

    def canonicalize_annotations(self, global_props=None):
        result = super().canonicalize_annotations(global_props)
        if global_props:
            return Ctx(global_props, result)
        else:
            return result

class UnknownOperator(NaryExpr):
    name: str = 'UnknownOperator'

    def __init__(self, *children: Expr, name='UnknownOperator') -> None:
        super().__init__(*children)
        self.name = name

    def __repr__(self):
        return type(self).__name__ + '(' + ''.join((repr(child) + ', ' for child in self.children)) + 'name=' + repr(self.name) + ')'

class UnaryExpr(NaryExpr):
    name: str = 'UnaryExpr'

    def __init__(self, child0: Expr) -> None:
        super().__init__(child0)

class BinaryExpr(NaryExpr):
    name: str = 'BinaryExpr'

    def __init__(self, child0: Expr, child1: Expr) -> None:
        super().__init__(child0, child1)

class TernaryExpr(NaryExpr):
    name: str = 'TernaryExpr'

    def __init__(self, child0: Expr, child1: Expr, child2: Expr) -> None:
        super().__init__(child0, child1, child2)


# values

class ValueExpr(Expr):
    name: str = 'ValueExpr'

    # Except for integers, all values (variables, constants, or numbers)
    # are represented as strings in the AST.
    def __init__(self, value: str) -> None:
        self.value: str = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.value) + ')'

    def subexprs(self):
        return []

    def replace_subexprs(self, exprs):
        return type(self)(self.value)

class Var(ValueExpr):
    name: str = 'Var'

    def __eq__(self, other):
        try:
            return self.value == other.value
        except AttributeError:
            return self.value == other

    def __hash__(self):
        return hash(self.value)

class Val(ValueExpr):
    name: str = 'Val'

    def __eq__(self, other):
        try:
            return self.value == other.value
        except AttributeError:
            return self.value == other

    def __hash__(self):
        return hash(self.value)

    def canonicalize_annotations(self, global_props=None):
        result = super().canonicalize_annotations(global_props)
        if global_props:
            return Ctx(global_props, result)
        else:
            return result

class Constant(Val):
    name: str = 'Constant'

class Decnum(Val):
    name: str = 'Decnum'

class Hexnum(Val):
    name: str = 'Hexnum'

class Integer(Val):
    name: str = 'Integer'

    def __init__(self, i: int) -> None:
        super().__init__(str(i))
        self.i = i

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.i) + ')'

    def replace_subexprs(self, exprs):
        return type(self)(self.i)

class Rational(Val):
    name: str = 'Rational'

    def __init__(self, p: int, q: int) -> None:
        super().__init__(str(p) + '/' + str(q))
        self.p : int = p
        self.q : int = q

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.p) + ', ' + repr(self.q) + ')'

    def replace_subexprs(self, exprs):
        return type(self)(self.p, self.q)

class Digits(Val):
    name: str = 'digits'

    def __init__(self, m: int, e: int, b: int) -> None:
        super().__init__('(' + self.name + ' ' + str(m) + ' ' + str(e) + ' ' + str(b) + ')')
        self.m: int = m
        self.e: int = e
        self.b: int = b

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.m) + ', ' + repr(self.e) + ', ' + repr(self.b) + ')'

    def replace_subexprs(self, exprs):
        return type(self)(self.m, self.e, self.b)

class TensorLit(Val):
    name: str = 'data'

    def __init__(self, value) -> None:
        self.value = value

    def __str__(self):
        return '(' + self.name + sexp_to_string(self.value) + ')'

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.value) + ')'

    def is_list(self):
        return isinstance(self.value, tuple)

    def as_list(self, strict=False):
        if isinstance(self.value, tuple):
            return self.value
        elif strict:
            raise TypeError('data is not a list')
        else:
            return None

class String(ValueExpr):
    name: str = 'String'

    def __eq__(self, other):
        try:
            return self.value == other.value
        except AttributeError:
            return self.value == other

    def __hash__(self):
        return hash(self.value)


# rounding contexts

class Ctx(Expr):
    name: str = '!'

    def __init__(self, props: dict, body: Expr) -> None:
        self.props = props
        self.body = body

    def __str__(self):
        return ('(' + self.name + ' '
                + ''.join((':' + k + ' ' + sexp_to_string(v) + ' ' for k, v in self.props.items()))
                + str(self.body) + ')')

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.props) + ', ' + repr(self.body) + ')'

    def subexprs(self):
        return [[self.body]]

    def replace_subexprs(self, exprs):
        ((body,),) = exprs
        return type(self)(self.props, body)

    def remove_annotations(self):
        return self.body.remove_annotations()

    def condense_annotations(self, global_props=None, local_props=None):
        new_props = update_props(local_props, self.props)
        return self.body.condense_annotations(global_props, new_props)

    def canonicalize_annotations(self, global_props=None):
        all_props = update_props(global_props, self.props)
        return self.body.canonicalize_annotations(all_props)

    def merge_annotations(self, annotations, local_props=None):
        new_props = update_props(local_props, self.props)
        return self.body.merge_annotations(annotations, new_props)


# control flow and tensors

class ControlExpr(Expr):
    name: str = 'ControlExpr'

class If(ControlExpr):
    name: str = 'if'

    def __init__(self, cond: Expr, then_body: Expr, else_body: Expr) -> None:
        self.cond: Expr = cond
        self.then_body: Expr = then_body
        self.else_body: Expr = else_body

    def __str__(self):
        return '(' + self.name + ' ' + str(self.cond) + ' ' + str(self.then_body) + ' ' + str(self.else_body) + ')'

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.cond) + ', ' + repr(self.then_body) + ', ' + repr(self.else_body) + ')'

    def subexprs(self):
        return [[self.cond, self.then_body, self.else_body]]

    def replace_subexprs(self, exprs):
        ((cond, then_body, else_body,),) = exprs
        return type(self)(cond, then_body, else_body)

class Let(ControlExpr):
    name: str = 'let'

    def __init__(self, let_bindings: typing.List[typing.Tuple[str, Expr]], body: Expr) -> None:
        self.let_bindings: typing.List[typing.Tuple[str, Expr]] = let_bindings
        self.body: Expr = body

    def __str__(self):
        return ('(' + self.name
                + ' (' + ' '.join(('[' + x + ' ' + str(e) + ']' for x, e in self.let_bindings)) + ') '
                + str(self.body) + ')')

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.let_bindings) + ', ' + repr(self.body) + ')'

    def subexprs(self):
        let_vars, let_exprs = zip(*self.let_bindings)
        return [let_exprs, [self.body]]

    def replace_subexprs(self, exprs):
        (let_exprs, (body,),) = exprs
        let_bindings = [(x, e,) for ((x, _,), e,) in zip(self.let_bindings, let_exprs)]
        return type(self)(let_bindings, body)

class LetStar(Let):
    name: str = 'let*'

class While(ControlExpr):
    name: str = 'while'

    def __init__(self, cond: Expr, while_bindings: typing.List[typing.Tuple[str, Expr, Expr]], body: Expr) -> None:
        self.cond: Expr = cond
        self.while_bindings: typing.List[typing.Tuple[str, Expr, Expr]] = while_bindings
        self.body: Expr = body

    def __str__(self):
        return ('(' + self.name + ' ' + str(self.cond)
                + ' (' + ' '.join(('[' + x + ' ' + str(e0) + ' ' + str(e) + ']' for x, e0, e in self.while_bindings)) + ') '
                + str(self.body) + ')')

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.cond) + ', ' + repr(self.while_bindings) + ', ' + repr(self.body) + ')'

    def subexprs(self):
        while_vars, while_inits, while_updates = zip(*self.while_bindings)
        return [[self.cond], while_inits, while_updates, [self.body]]

    def replace_subexprs(self, exprs):
        ((cond,), while_inits, while_updates, (body,),) = exprs
        while_bindings = [(x, e0, e,) for ((x, _, _,), e0, e,) in zip(self.while_bindings, while_inits, while_updates)]
        return type(self)(cond, while_bindings, body)

class WhileStar(While):
    name: str = 'while*'

class For(ControlExpr):
    name: str = 'for'

    def __init__(self,
                 dim_bindings: typing.List[typing.Tuple[str, Expr]],
                 while_bindings: typing.List[typing.Tuple[str, Expr, Expr]],
                 body: Expr) -> None:
        self.dim_bindings: typing.List[typing.Tuple[str, Expr]] = dim_bindings
        self.while_bindings: typing.List[typing.Tuple[str, Expr, Expr]] = while_bindings
        self.body: Expr = body

    def __str__(self):
        return ('(' + self.name
                + ' (' + ' '.join(('[' + x + ' ' + str(e) + ']' for x, e in self.dim_bindings)) + ')'
                + ' (' + ' '.join(('[' + x + ' ' + str(e0) + ' ' + str(e) + ']' for x, e0, e in self.while_bindings)) + ') '
                + str(self.body) + ')')

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.dim_bindings) + ', ' + repr(self.while_bindings) + ', ' + repr(self.body) + ')'

    def subexprs(self):
        dim_vars, dim_exprs = zip(*self.dim_bindings)
        while_vars, while_inits, while_updates = zip(*self.while_bindings)
        return [dim_exprs, while_inits, while_updates, [self.body]]

    def replace_subexprs(self, exprs):
        (dim_exprs, while_inits, while_updates, (body,),) = exprs
        dim_bindings = [(x, e,) for ((x, _,), e,) in zip(self.dim_bindings, dim_exprs)]
        while_bindings = [(x, e0, e,) for ((x, _, _,), e0, e,) in zip(self.while_bindings, while_inits, while_updates)]
        return type(self)(dim_bindings, while_bindings, body)

class ForStar(For):
    name: str = 'for*'

class Tensor(ControlExpr):
    name: str = 'tensor'

    def __init__(self, dim_bindings: typing.List[typing.Tuple[str, Expr]], body: Expr) -> None:
        self.dim_bindings: typing.List[typing.Tuple[str, Expr]] = dim_bindings
        self.body: Expr = body

    def __str__(self):
        return ('(' + self.name
                + ' (' + ' '.join(('[' + x + ' ' + str(e) + ']' for x, e in self.dim_bindings)) + ') '
                + str(self.body) + ')')

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.dim_bindings) + ', ' + repr(self.body) + ')'

    def subexprs(self):
        dim_vars, dim_exprs = zip(*self.dim_bindings)
        return [dim_exprs, [self.body]]

    def replace_subexprs(self, exprs):
        (dim_exprs, (body,),) = exprs
        dim_bindings = [(x, e,) for ((x, _,), e,) in zip(self.dim_bindings, dim_exprs)]
        return type(self)(dim_bindings, body)


class TensorStar(Tensor):
    name: str = 'tensor*'

    def __init__(self,
                 ident: str,
                 dim_bindings: typing.List[typing.Tuple[str, Expr]],
                 while_bindings: typing.List[typing.Tuple[str, Expr, Expr]],
                 body: Expr) -> None:
        self.ident: str = ident
        self.dim_bindings: typing.List[typing.Tuple[str, Expr]] = dim_bindings
        self.while_bindings: typing.List[typing.Tuple[str, Expr, Expr]] = while_bindings
        self.body: Expr = body

    def __str__(self):
        if self.ident:
            ident_str = ' ' + self.ident
        else:
            ident_str = ''

        if self.while_bindings:
            while_str = ' (' + ' '.join(('[' + x + ' ' + str(e0) + ' ' + str(e) + ']' for x, e0, e in self.while_bindings)) + ')'
        else:
            while_str = ''

        return ('(' + self.name
                + ident_str
                + ' (' + ' '.join(('[' + x + ' ' + str(e) + ']' for x, e in self.dim_bindings)) + ')'
                + while_str
                + ' ' + str(self.body) + ')')

    def __repr__(self):
        return (type(self).__name__ + '('
                + repr(self.ident) + ', '
                + repr(self.dim_bindings) + ', '
                + repr(self.while_bindings) + ', '
                + repr(self.body) + ')')

    def subexprs(self):
        dim_vars, dim_exprs = zip(*self.dim_bindings)
        while_vars, while_inits, while_updates = zip(*self.while_bindings)
        return [dim_exprs, while_inits, while_updates, [self.body]]

    def replace_subexprs(self, exprs):
        (dim_exprs, while_inits, while_updates, (body,),) = exprs
        dim_bindings = [(x, e,) for ((x, _,), e,) in zip(self.dim_bindings, dim_exprs)]
        while_bindings = [(x, e0, e,) for ((x, _, _,), e0, e,) in zip(self.while_bindings, while_inits, while_updates)]
        return type(self)(self.ident, dim_bindings, while_bindings, body)


# cast is the identity function, used for repeated rounding

class Cast(UnaryExpr):
    name: str = 'cast'

# tensor operations

class Dim(UnaryExpr):
    name: str = 'dim'

class Size(NaryExpr):
    name: str = 'size'

class Ref(NaryExpr):
    name: str = 'ref'

# IEEE 754 required arithmetic

class Add(BinaryExpr):
    name: str = '+'

class Sub(BinaryExpr):
    name: str = '-'

class Mul(BinaryExpr):
    name: str = '*'

class Div(BinaryExpr):
    name: str = '/'

class Sqrt(UnaryExpr):
    name: str = 'sqrt'

class Fma(TernaryExpr):
    name: str = 'fma'

# discrete operations

class Neg(UnaryExpr):
    # note that unary negation has the same "name" as subtraction
    name: str = '-'

class Copysign(BinaryExpr):
    name: str = 'copysign'

class Fabs(UnaryExpr):
    name: str = 'fabs'

# composite arithmetic

class Fdim(BinaryExpr):
    name: str = 'fdim'

class Fmax(BinaryExpr):
    name: str = 'fmax'

class Fmin(BinaryExpr):
    name: str = 'fmin'

class Fmod(BinaryExpr):
    name: str = 'fmod'

class Remainder(BinaryExpr):
    name: str = 'remainder'

# rounding and truncation

class Ceil(UnaryExpr):
    name: str = 'ceil'

class Floor(UnaryExpr):
    name: str = 'floor'

class Nearbyint(UnaryExpr):
    name: str = 'nearbyint'

class Round(UnaryExpr):
    name: str = 'round'

class Trunc(UnaryExpr):
    name: str = 'trunc'

# trig

class Acos(UnaryExpr):
    name: str = 'acos'

class Acosh(UnaryExpr):
    name: str = 'acosh'

class Asin(UnaryExpr):
    name: str = 'asin'

class Asinh(UnaryExpr):
    name: str = 'asinh'

class Atan(UnaryExpr):
    name: str = 'atan'

class Atan2(BinaryExpr):
    name: str = 'atan2'

class Atanh(UnaryExpr):
    name: str = 'atanh'

class Cos(UnaryExpr):
    name: str = 'cos'

class Cosh(UnaryExpr):
    name: str = 'cosh'

class Sin(UnaryExpr):
    name: str = 'sin'

class Sinh(UnaryExpr):
    name: str = 'sinh'

class Tan(UnaryExpr):
    name: str = 'tan'

class Tanh(UnaryExpr):
    name: str = 'tanh'

# exponentials

class Exp(UnaryExpr):
    name: str = 'exp'

class Exp2(UnaryExpr):
    name: str = 'exp2'

class Expm1(UnaryExpr):
    name: str = 'expm1'

class Log(UnaryExpr):
    name: str = 'log'

class Log10(UnaryExpr):
    name: str = 'log10'

class Log1p(UnaryExpr):
    name: str = 'log1p'

class Log2(UnaryExpr):
    name: str = 'log2'

# powers

class Cbrt(UnaryExpr):
    name: str = 'cbrt'

class Hypot(BinaryExpr):
    name: str = 'hypot'

class Pow(BinaryExpr):
    name: str = 'pow'

# other

class Erf(UnaryExpr):
    name: str = 'erf'

class Erfc(UnaryExpr):
    name: str = 'erfc'

class Lgamma(UnaryExpr):
    name: str = 'lgamma'

class Tgamma(UnaryExpr):
    name: str = 'tgamma'


# comparison

class LT(NaryExpr):
    name: str = '<'

class GT(NaryExpr):
    name: str = '>'

class LEQ(NaryExpr):
    name: str = '<='

class GEQ(NaryExpr):
    name: str = '>='

class EQ(NaryExpr):
    name: str = '=='

class NEQ(NaryExpr):
    name: str = '!='

# classification

class Isfinite(UnaryExpr):
    name: str = 'isfinite'

class Isinf(UnaryExpr):
    name: str = 'isinf'

class Isnan(UnaryExpr):
    name: str = 'isnan'

class Isnormal(UnaryExpr):
    name: str = 'isnormal'

class Signbit(UnaryExpr):
    name: str = 'signbit'

# logic

class And(NaryExpr):
    name: str = 'and'

class Or(NaryExpr):
    name: str = 'or'

class Not(UnaryExpr):
    name: str = 'not'


# fpcore objects and helpers

class FPCore(object):
    def __init__(self, inputs, e, props=None, ident=None, name=None, pre=None, spec=None):
        self.inputs = inputs
        self.e = e
        if props is None:
            self.props = {}
        else:
            self.props = props
        self.ident = ident
        self.name = name
        self.pre = pre
        self.spec = spec

    def __str__(self):
        return 'FPCore ({})\n  ident: {}\n  name: {}\n   pre: {}\n  spec: {}\n  {}'.format(
            ' '.join((annotation_to_string(*arg) for arg in self.inputs)),
            str(self.ident), str(self.name), str(self.pre), str(self.spec), str(self.e))

    def __repr__(self):
        return 'FPCore(\n  {},\n  {},\n  ident={}\n  props={}\n)'.format(
            repr(self.inputs), repr(self.e), repr(self.ident), repr(self.props))

    def __eq__(self, other):
        if not isinstance(other, FPCore):
            return False
        return self.inputs == other.inputs and self.e == other.e and self.props == other.props

    @property
    def sexp(self):
        return '(FPCore ({}) {}{})'.format(
            ' '.join((annotation_to_string(*arg) for arg in self.inputs)),
            ''.join(':' + name + ' ' + sexp_to_string(prop) + ' ' for name, prop in self.props.items()),
            str(self.e))
