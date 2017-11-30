import random

import fpcast as ast
import fpcparser

var_str = 'x'
const_strs = [
    '1',
]
unops = [
    ast.Neg,
    ast.Sqrt,
]
binops = [
    ast.Add,
    ast.Sub,
    ast.Mul,
    ast.Div,    
]


def gen_up_to_depth(n, nvars = 1):
    # don't redefine at every level?
    
    variables = [
        ast.Var(var_str + str(i)) for i in range(nvars)
    ]
    constants = [
        ast.Val(cstr) for cstr in const_strs
    ]

    for x in variables:
        yield x
    for c in constants:
        yield c

    if n > 0:
        for op in unops:
            for t in gen_up_to_depth(n-1, nvars=nvars):
                yield op(t)
        for op in binops:
            for t1 in gen_up_to_depth(n-1, nvars=nvars):
                for t2 in gen_up_to_depth(n-1, nvars=nvars):
                    yield op(t1, t2)

def gen_at_depth(n, nvars = 1):
    variables = [
        ast.Var(var_str + str(i)) for i in range(nvars)
    ]
    constants = [
        ast.Val(cstr) for cstr in const_strs
    ]

    if n == 0:
        for x in variables:
            yield x
        for c in constants:
            yield c

    else:
        for op in unops:
            # subtree must be depth n-1
            for t in gen_at_depth(n-1, nvars=nvars):
                yield op(t)
        for op in binops:
            # left subtree is depth n-1, right subtree is depth 0 to n-1
            for t1 in gen_at_depth(n-1, nvars=nvars):
                for i in range(0, n):
                    for t2 in gen_at_depth(i, nvars=nvars):
                        yield op(t1, t2)
            # left subtree is depth 0 to n-2, right subtree is depth n-1
            for i in range(0, n-1):
                for t1 in gen_at_depth(i):
                    for t2 in gen_at_depth(n-1):
                        yield op(t1, t2)

def gen_random_depth(n, nvars = 1):
    variables = [
        ast.Var(var_str + str(i)) for i in range(nvars)
    ]
    constants = [
        ast.Val(cstr) for cstr in const_strs
    ]

    if n == 0:
        return random.choice(variables + constants)

    else:
        op = random.choice(unops + binops)
        if op.nargs == 1:
            t = gen_random_depth(n-1, nvars=nvars)
            return op(t)
        else:
            t1 = gen_random_depth(n-1, nvars=nvars)
            t2 = gen_random_depth(random.choice(range(n)), nvars=nvars)
            args = [t1, t2]
            random.shuffle(args)
            return op(*args)

def mkcore(e, nvars = 1):
    # could inspect core, but this is easier
    variables = [
        var_str + str(i) for i in range(nvars)
    ]
    return fpcparser.FPCoreObject(variables, {}, e)
    
        
def gen_from_file(fname):
    with open(fname, 'rt') as f:
        cores = fpcparser.compile(f.read())
        for core in cores:
            yield core

    
    
if __name__ == '__main__':
    i = 0
    for tree in gen_at_depth(1):
        i += 1
        print(tree)
    print(i)
    i = 0
    for tree in gen_at_depth(2):
        i += 1
        #print(tree)
    print(i)
    i = 0
    # too deep
    # for tree in gen_at_depth(3):
    #     i += 1
    #     #print(tree)
    # print(i)

    for i in range(10):
        print(gen_random_depth(random.choice(range(3,10)), nvars=random.choice(range(1,4))))

    for e in gen_from_file('allcores.fpcore'):
        print(e)
