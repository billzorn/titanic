# Titanic

Titanic is a tool for designing and experimenting
with novel computer arithmetic formats.
It builds on the GNU MPFR library for arbitrary-precision arithmetic,
providing additional support for low-level floating-point behaviors
such as precise control of rounding and precision tracking.

Titanic uses the [FPCore](http://fpbench.org/spec/)
benchmark format for floating-point computations.
Reference interpereters are provided for the FPCore language
using both [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754)
and [posit](https://posithub.org/index) arithmetics,
as well as fixed-point.
These interpreters can also interface with each other
(and with custom user-defined arithmetics)
to perform multiple precision, multiple format computations.
