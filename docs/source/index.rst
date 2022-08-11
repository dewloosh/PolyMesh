==============================================
DewLoosh.Geom - Numerical and Symbolic Methods
==============================================

`DewLoosh` is a rapid prototyping platform focused on numerical calculations 
mainly corcerned with simulations of natural phenomena. It provides a set of common 
functionalities and interfaces with a number of state-of-the-art open source 
packages to combine their power seamlessly under a single development
environment.

Requirements
------------

The implementations in this module are created on top of 

* | `NumPy`, `SciPy` and `Numba` to speed up computationally sensitive parts,

* | `SymPy` for symbolic operations and some vector algebra,

* | the `awkward` library for its high performance data structures, gpu support
  | and general `Numba` compliance.


Features
--------

* | Numba-jitted classes and an extendible factory to define and manipulate 
  | vectors and tensors.

* | Classes to define and solve linear and nonlinear optimization
  | problems.

* | A set of array routines for fast prorotyping, including random data creation
  | to assure well posedness, or other properties of test problems.


Gallery
-------

.. nbgallery::
    :caption: Gallery
    :name: rst-gallery
    :glob:
    :reversed:

    _notebooks/*
    
        
Contents
--------

.. toctree::
    :caption: Contents
    :maxdepth: 3
   
    user_guide
    notebooks

API
---

.. toctree::
    :caption: API
    :maxdepth: 3
   
    api
   
Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



