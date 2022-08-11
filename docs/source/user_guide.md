# A Quick Guide

## Installation
This is optional, but we suggest you to create a dedicated virtual enviroment at all times to avoid conflicts with your other projects. Create a folder, open a command shell in that folder and use the following command

```console
>>> python -m venv venv_name
```

Once the enviroment is created, activate it via typing

```console
>>> .\venv_name\Scripts\activate
```

`dewloosh.math` can be installed (either in a virtual enviroment or globally) from PyPI using `pip` on Python >= 3.6:

```console
>>> pip install dewloosh.math
```

## Linear Algebra

Define a reference frame (B) relative to the ambient frame (A):
```python
>>> from dewloosh.math.linalg import ReferenceFrame
>>> A = ReferenceFrame(name='A', axes=np.eye(3))
>>> B = A.orient_new('Body', [0, 0, 90*np.pi/180], 'XYZ', name='B')
```
Get the DCM matrix of the transformation between two frames:
```python
>>> B.dcm(target=A)
```
Define a vector in frame A and view the components of it in frame B:
```python
>>> v = Vector([0.0, 1.0, 0.0], frame=A)
>>> v.view(B)
```
Define the same vector in frame B:
```python
>>> v = Vector(v.show(B), frame=B)
>>> v.show(A)
```

## Linear Programming

Solve a following Linear Programming Problem (LPP) with one 
unique solution:

$$
\begin{eqnarray}
    & minimize&  \quad  3 x_1 + x_2 + 9 x_3 + x_4  \\
    & subject \, to& & \\
    & & x_1 + 2 x_3 + x_4 \,=\, 4, \\
    & & x_2 + x_3 - x_4 \,=\, 2, \\
    & & x_i \,\geq\, \, 0, \qquad i=1, \ldots, 4.
\end{eqnarray}
$$

```python
>>> from dewloosh.math.optimize import LinearProgrammingProblem as LPP
>>> import sympy as sy
>>> variables = ['x1', 'x2', 'x3', 'x4']
>>> x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
>>> obj1 = Function(3*x1 + 9*x3 + x2 + x4, variables=syms)
>>> eq11 = Equality(x1 + 2*x3 + x4 - 4, variables=syms)
>>> eq12 = Equality(x2 + x3 - x4 - 2, variables=syms)
>>> problem = LPP(cost=obj1, constraints=[eq11, eq12], variables=syms)
>>> problem.solve()
array([0., 6., 0., 4.])
```

## NonLinear Programming

Find the minimizer of the Rosenbrock function:

$$
minimize  \quad  (a-x)^2 + b (y-x^2)^2
$$

```python
>>> from dewloosh.math.optimize import BinaryGeneticAlgorithm
>>> def Rosenbrock(x, y):
>>>     a = 1, b = 100
>>>     return (a-x)**2 + b*(y-x**2)**2
>>> ranges = [[-10, 10],[-10, 10]]
>>> BGA = BinaryGeneticAlgorithm(Rosenbrock, ranges, length=12, nPop=200)
>>> BGA.solve()
array([0.99389553, 0.98901176]) 
```
