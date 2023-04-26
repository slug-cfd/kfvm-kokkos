from sympy import *

init_printing()

# Primitive variables
#rho = Symbol(r'\rho',positive=True)
rho = Symbol(r'r',positive=True)
Vx = Symbol(r'V_x')
Vy = Symbol(r'V_y')
Vz = Symbol(r'V_z')
p = Symbol(r'p',positive=True)

# Gamma ratio
#gam = Symbol(r'\widetilde{\Gamma}',positive=True)
gam = Symbol(r'G',positive=True)

# Intermediate quantities
Vsq = Vx**2 + Vy**2 + Vz**2
W = 1/sqrt(1 - Vsq)
h = 1 + gam*p/rho

# Conservative variables in terms of primitives
D = rho*W
Sx = rho*W**2*h*Vx
Sy = rho*W**2*h*Vy
Sz = rho*W**2*h*Vz
tau = rho*W*(h*W - 1) - p

# List of useful substitutions
V = Symbol(r'V',positive=True)
Ws = Symbol(r'W',positive=True)
subList = [(Vx**2 + Vy**2 + Vz**2,V**2),(1/sqrt(1 - V**2),Ws),(1/((V**2 - 1)**2),Ws**4)]

# Derivatives of conservative variables
U = Matrix([D,Sx,Sy,Sz,tau])
dUdV = simplify(U.jacobian([rho,Vx,Vy,Vz,p]).subs(subList))

print('======== Prim -> Cons Jacobian ========')
pprint(dUdV)
#print(latex(dUdV))
