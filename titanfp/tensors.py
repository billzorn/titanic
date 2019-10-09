from .fpbench import fpcparser
from .arithmetic import ieee754, sinking
from .arithmetic import posit
from .arithmetic import mpmf
from .arithmetic import core2math
from .arithmetic import evalctx
from .titanic import digital

from .titanic import gmpmath

text = """(FPCore foo (x)
(let ([arr (tensor ([a 3] (b 3])
            (+ x a))])
  (dim arr)))
"""

core = fpcparser.compile1(text)

