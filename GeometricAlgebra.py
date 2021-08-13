""" Interested in geometric algebra providing an 'extended' viewpoint of some physical problems 
(see https://www.youtube.com/watch?v=60z_hpEAtD8). Turns out, they've got a Python package for 
that! Let's run through the documentation tutorials...
"""
# %%
import clifford as cf
import math

# setup
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
R = math.e**(math.pi/4*e12)  # enacts rotation by pi/2
print(R)

print(R*e1*~R)    # rotate e1 by pi/2 in the e12-plane

# %% The Algebra of Space (G3)
layout, blades = cf.Cl(3)  # creates a 3-dimensional clifford algebra

print(blades)

# You may wish to explicitly assign the blades to variables like so,
e1 = blades['e1']
e2 = blades['e2']
# etc...

# Or, if you’re lazy and just working in an interactive session you 
# can use locals() to update your namespace with all of the blades at once.
locals().update(blades)

# basics
print(e1*e2)  # geometric product

print(e1|e2)  # inner product

print(e1^e2)  # outer product

print(e1^e2^e3)  # even more outer products

# Defects in Precedence

# Python’s operator precedence makes the outer product evaluate after addition. 
# This requires the use of parentheses when using outer products. For example
print(e1^e2 + e2^e3)  # fail, evaluates as (2^e123)

print((e1^e2) + (e2^e3))  # correct: (1^e12) + (1^e23)

# Also the inner product of a scalar and a Multivector is 0,
print(4|e1) # 0

# So for scalars, use the outer product or geometric product instead

print(4*e1) # (4^e1)

# Multivectors
# Multivectors can be defined in terms of the basis blades. For example you can 
# construct a rotor as a sum of a scalar and bivector, like so
theta = math.pi/4
R = math.cos(theta) - math.sin(theta)*e23
print(R)

# You can also mix grades without any reason
A = 1 + 2*e1 + 3*e12 + 4*e123
print(A)

# Reversion
# The reversion operator is accomplished with the tilde ~ in front of the 
# Multivector on which it acts
print(~A)
# %%
