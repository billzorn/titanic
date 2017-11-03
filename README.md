# ![Image](../master/www/piceberg.png?raw=true)

# Titanic

Titanic is a floating point analysis tool that is "guaranteed to float correctly."
It features an implementation of the [IEEE 754 (2008)](http://ieeexplore.ieee.org/document/4610935/) standard for floating point
in terms of real numbers (provided by the [sympy](http://www.sympy.org/en/index.html) Python library), and a web
interface for examining the representation of numbers or the behavior of
computations in any IEEE 754-like floating point format.

Titanic is designed to interface with other floating point libraries, to determine
if their behavior is correct, and with solvers such as [Z3](https://github.com/Z3Prover/z3), to quickly find inputs
that cause floating point computations to misbehave.
