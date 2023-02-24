import numpy as np
from typing import Union
from numpy import ndarray
from copy import deepcopy as dcopy

from neumann.linalg import CartesianFrame as Frame, FrameLike, Vector
from neumann.linalg.vector import Vector


__all__ = ["CartesianFrame"]


VectorLike = Union[Vector, ndarray]


class CartesianFrame(Frame):
    """
    A field-specific reference frame to be used in problems related to
    Euclidean geometry.

    It builds on top of :class:`FrameLike` from `neumann`, but adds
    the contept of 'origo', and some other applications related to the field.

    See Also
    --------
    :class:`~neumann.linalg.frame.frame.FrameLike`

    Parameters
    ----------
    axes : ndarray, Optional.
        2d numpy array of floats specifying cartesian reference frames.
        Dafault is None.
    dim : int, Optional
        Dimension of the mesh. Deafult is 3.
    origo : ndarray, Optional.
        The origo of the mesh. Default is the zero vector.

    Note
    ----
    See the documentation of :class:`neumann.FrameLike` for more control over
    object creation. However, if your problem not very extreme in some sense,
    you are probably good to goo only by following the examples.

    Example
    -------
    Define a standard Cartesian frame and rotate it around axis 'Z'
    with an amount of 180 degrees:

    >>> A = CartesianFrame(dim=3)
    >>> B = A.orient_new('Body', [0, 0, np.pi], 'XYZ')

    To create a third frame that rotates from B the way B rotates from A, we
    can do

    >>> A = CartesianFrame(dim=3)
    >>> C = A.orient_new('Body', [0, 0, 2*np.pi], 'XYZ')

    or we can define it relative to B (this literally makes C to looke
    in B like B looks in A)

    >>> C = CartesianFrame(B.axes, parent=B)

    Then, the *DCM from A to B* , that is :math:`^{A}\mathbf{R}^{B}` would be

    >>> A_R_B = B.dcm(source=A)

    or equivalently

    >>> A_R_B = A.dcm(target=A)
    """

    def __init__(
        self, axes: ndarray = None, *args, dim: int = 3, origo: ndarray = None, **kwargs
    ):
        axes = np.eye(dim) if axes is None else axes
        super().__init__(axes, *args, **kwargs)
        self._origo = origo

    def origo(self, target: FrameLike = None) -> Vector:
        """
        Returns the origo of the current frame in ambient space
        or with respect to another frame.

        Parameters
        ----------
        target : FrameLike, Optional
            A frame in which we want to get the origo of the current frame.
            A None value returns the origo of the current frame with respect
            to the root. Default is None.

        Returns
        -------
        Vector
            A vector defined in ambient space, the parent frame,
            or the specified frame.

        Examples
        --------
        Define a standard Cartesian frame and rotate it around axis 'Z'
        with an amount of 180 degrees:

        >>> A = CartesianFrame()
        >>> B = A.orient_new('Body', [0, 0, 45*np.pi/180],  'XYZ')

        To get the origin of frame B:

        >>> B.origo()
        [0., 0., 0.]

        Move frame B (the motion is defined locally) and print the
        new point of origin with respect to A:

        >>> B.move(Vector([1, 0, 0], frame=B))
        >>> B.origo(A)
        [0.7071, 0.7071, 0.]

        Of course, the point of origin of a frame with respect to itself
        must be a zero vector:

        >>> B.origo(B)
        [0., 0., 0.]

        Providing with no arguments returns the distance of origin with
        respect to the root frame:

        >>> B.origo()  # same as B.origo(B.root())
        [0.7071, 0.7071, 0.]
        """
        if not isinstance(self._origo, ndarray):
            self._origo = np.zeros(len(self.axes))

        if target is None:
            return Vector(self._origo).show()
        else:
            t = target.origo()
            s = self.origo()
            return Vector(s - t).show(target)

    def move(self, d: VectorLike, frame: FrameLike = None) -> "CartesianFrame":
        """
        Moves the frame by shifting its origo.

        Parameters
        ----------
        d : VectorLike
            :class:`Vector` or :class:`Array`, the amount of the motion.
        frame : FrameLike, Optional
            A frame in which the input is defined if it is not a Vector.
            Default is None, which assumes the root frame.

        Returns
        -------
        CartesianFrame
            The object the function is called on.

        Examples
        --------
        >>> A = CartesianFrame()
        >>> v = Vector([1., 0., 0.], frame=A)
        >>> B = A.fork('Body', [0, 0, 45*np.pi/180], 'XYZ').move(v)

        Move the frame locally with the same amount

        >>> B.move(v.array, frame=B)
        """
        if not isinstance(d, Vector):
            d = Vector(d, frame=frame)
        if self._origo is None:
            self._origo = np.zeros(len(self.axes))
        self._origo += d.show()
        return self

    def fork(self, *args, **kwargs) -> "CartesianFrame":
        """
        Alias for `orient_new`.
        """
        return self.orient_new(*args, **kwargs)

    def copy(self, deep: bool = False, name: str = None) -> "CartesianFrame":
        """
        Returns a shallow or deep copy of this object, depending of the
        argument `deepcopy` (default is False).
        """
        if deep:
            return self.__class__(
                dcopy(self.axes), origo=dcopy(self.origo()), name=name
            )
        else:
            return self.__class__(self.axes, origo=self.origo(), name=name)
