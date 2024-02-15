from sympy import *
from sympy.abc import x

init_printing()

# Coefficients
a0 = 2.0/pi
a2 = 4.0/(3*pi)
a4 = -4.0/(15*pi)

# Chebyschev polys
T0 = 1
T2 = 2*x**2 - 1
T4 = 2*T2**2 - 1

# Fit to |x|
tau = a0*T0 + a2*T2 + a4*T4

resid = x - tau

sols = roots(diff(resid,x),x)

for k in sols:
    pprint(resid.subs(x,re(k)).evalf(20))
