"""Direct softposit arithmetic. No emulation with MPFR.
"""

from sfpy import Posit8, Posit16

from ..fpbench import fpcast as ast

from .evalctx import PositCtx
