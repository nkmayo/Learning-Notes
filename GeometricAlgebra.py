""" Interested in geometric algebra providing an 'extended' viewpoint of some physical problems 
(see https://www.youtube.com/watch?v=60z_hpEAtD8). Turns out, they've got a Python package for 
that! Let's run through the documentation tutorials...
"""
# %%
import clifford as cf
from math import e, pi

layout, blades = cf.Cl(2) # creates a 2-dimensional clifford algebra

print(blades) # inspect the blades

# assign the blades to variables
e1 = blades['e1']
e2 = blades['e2']
e12 = blades['e12']

# basics
print(e1*e2)  # geometric product

print(e1|e2)  # inner product

print(e1^e2)  # outer product

# reflection
a = e1+e2     # the vector
n = e1        # the reflector
print(-n*a*n.inv())  # reflect `a` in hyperplane normal to `n`

# rotation
R = e**(pi/4*e12)  # enacts rotation by pi/2
print(R)

print(R*e1*~R)    # rotate e1 by pi/2 in the e12-plane

# %% The Algebra of Space (G3)
