from sympy import *

# Constants
g = Symbol('g',Positive=True)
J = Symbol('J',Positive=True)

# conservative vars
jr = Symbol('JRho',Positive=True)
jru = Symbol('JRhoU')
jrv = Symbol('JRhoV')
jrw = Symbol('JRhoW')
jE = Symbol('JE',Positive=True)
cvs = Matrix([jr,jru,jrv,jrw,jE]).T

# cons -> prim
c2pR = jr/J
c2pU = jru/jr
c2pV = jrv/jr
c2pW = jrw/jr
c2pP = (g - 1)*(jE/J - Rational(1,2)*c2pR*(c2pU**2 + c2pV**2 + c2pW**2))

# derivatives
print('Rho derivatives')
pprint(diff(c2pR,cvs))
print('u derivatives')
pprint(diff(c2pU,cvs))
print('v derivatives')
pprint(diff(c2pV,cvs))
print('w derivatives')
pprint(diff(c2pW,cvs))
print('p derivatives')
pprint(simplify(diff(c2pP,cvs)))

# primitive vars
rho = Symbol('rho',Positive=True)
u = Symbol('u')
v = Symbol('v')
w = Symbol('w')
p = Symbol('p',Positive=True)

# prim -> cons
p2cR = J*rho
p2cRU = J*rho*u
p2cRV = J*rho*v
p2cRW = J*rho*w
p2cE = J*(p/(g-1) + Rational(1,2)*rho*(u**2 + v**2 + w**2))
pvs = Matrix([rho,u,v,w,p]).T

# derivatives
print('JRho derivatives')
pprint(diff(p2cR,pvs))
print('JRhoU derivatives')
pprint(diff(p2cRU,pvs))
print('JRhoV derivatives')
pprint(diff(p2cRV,pvs))
print('JRhoW derivatives')
pprint(diff(p2cRW,pvs))
print('JE derivatives')
pprint(diff(p2cE,pvs))
