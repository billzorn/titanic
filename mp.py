import math
import gmpy2
from gmpy2 import mpfr

with gmpy2.ieee(64) as ctx:
    arg = -1.000000000000003

    x = mpfr(arg)
    r = gmpy2.sqrt(gmpy2.sub(gmpy2.mul(x, x), mpfr(1.0)))
    r2 = math.sqrt((x * x) - 1)
    
    print(r, r2, 1.4197270870208740234375*(2**-3))
    
    
