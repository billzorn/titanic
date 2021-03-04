from ..arithmetic import ieee754
from ..titanic import ndarray

es = 2
nbits = 4
ctx = ieee754.ieee_ctx(es, nbits)

def render_element(e):
    return str(e)


# lets just print every number in the representation to see if they look right
for i in range(2**nbits):
    fp = ieee754.bits_to_digital(i, ctx)
    print(f'{i!s:<8}{render_element(fp)!s:<20}{ieee754.show_bitpattern(fp)}')
print('\n\n')


# 1d (unary op)

# for the list of function names, see arithmetic/mpnum.py
def f1(x):
    return x.sqrt()

lut = []
for i in range(2**nbits):
    fp = ieee754.bits_to_digital(i, ctx)
    #print(f'{i!s:<8}{fp!s:<20}{ieee754.show_bitpattern(fp)}')
    lut.append(f1(fp))

lut_nd = ndarray.NDSeq(lut)
print(lut_nd.shape)
print()
print(ndarray.describe(lut_nd, descr=render_element, lparen='[', rparen=']'))
print('\n\n')

# now 2 dimensions

def f2(x, y):
    return x.mul(y)

lut2d = []
for i in range(2**nbits):
    row = []
    fp_x = ieee754.bits_to_digital(i, ctx)
    for j in range(2**nbits):
        fp_y = ieee754.bits_to_digital(j, ctx)
        row.append(f2(fp_x, fp_y))
    lut2d.append(row)

lut2d_nd  = ndarray.NDSeq(lut2d)
print(lut2d_nd.shape)
print()
print(ndarray.describe(lut2d_nd, descr=render_element, lparen='[', rparen=']'))
print('\n\n')



# or you can print it as a matrix if you want, because my NDSeq library
print(lut2d_nd)
