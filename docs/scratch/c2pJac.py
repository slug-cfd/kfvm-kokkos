from sympy import *

init_printing()

# Conservative variables
D = Symbol(r'D',positive=True)
Sx = Symbol(r'S_x')
Sy = Symbol(r'S_y')
Sz = Symbol(r'S_z')
#tau = Symbol(r'\tau',positive=True)
tau = Symbol(r'T',positive=True)

# Gamma ratio
#gam = Symbol(r'\widetilde{\Gamma}',positive=True)
gam = Symbol(r'G',positive=True)

# Primitive variables are functions of conservative ones
rho = Function(r'\rho',positive=True)(D,Sx,Sy,Sz,tau)
Vx = Function(r'Vx')(D,Sx,Sy,Sz,tau)
Vy = Function(r'Vy')(D,Sx,Sy,Sz,tau)
Vz = Function(r'Vz')(D,Sx,Sy,Sz,tau)
p = Function(r'p',positive=True)(D,Sx,Sy,Sz,tau)

# Lorenz and enthalpy 
W = 1/sqrt(1 - (Sx**2 + Sy**2 + Sz**2)/(D + tau + p)**2)
h = 1 + gam*W*p/D
hW = h*W

# Start building up derivatives
dhWdD = diff(hW,D)
dhWdSx = diff(hW,Sx)
dhWdT = diff(hW,tau)

dPdD = D*dhWdD + (hW - 1)
dPdSx = D*dhWdSx
dPdT = D*dhWdT - 1

# Create set of substitutions
S = Symbol(r'S',positive=True)
A = Symbol(r'A',positive=True)
B = Symbol(r'B',positive=True)
subList = [(Sx**2 + Sy**2 + Sz**2,S**2),(D + tau + p,A),(A**2 - S**2,B**2)]

print('======= Helper terms ===========')
print('d(Winv)dD')
print(latex(simplify(diff(1/W,D).subs(subList))))
print('d(Winv)dSx')
print(latex(simplify(diff(1/W,Sx).subs(subList))))
print('d(Winv)dT')
print(latex(simplify(diff(1/W,tau).subs(subList))))
print('d(hW)dD')
#print(latex(collect(factor(simplify(dhWdD).subs(subList)),A)))
pprint(collect(factor(simplify(dhWdD).subs(subList)),A))
print('d(hW)dSx')
print(latex(collect(factor(simplify(dhWdSx).subs(subList)),A)))
print('d(hW)dT')
print(latex(collect(factor(simplify(dhWdT).subs(subList)),A)))
