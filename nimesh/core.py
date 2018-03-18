import numpy as np
from enum import IntEnum
from typing import Sequence


class AffineTransform(object):

    def __init__(self, transform_coord_sys: 'CoordinateSystem',
                 affine: Sequence):
        """Affine transformation from one coordinate system to another.

        The AffineTransform represents an affine transformation to a new
        stereotaxic space.

        Args:
            transform_coord_sys: The coordinate system of the data after the
                application of the affine transformation.
            affine: The affine transform as a numpy array with a shape of
                (4, 4).

        Raises:
            TypeError: If the affine is none or not convertible to a a numpy
                array of floats.
            ValueError: If the affine does not have a shape of (4, 4).

        Examples:

            Create an affine transform to scanner space.

            >>> import numpy as np
            >>> from nimesh import AffineTransform, CoordinateSystem
            >>> affine = np.eye(4)
            >>> transform = AffineTransform(CoordinateSystem.SCANNER, affine)

        """

        # The affine must be convertible to a numpy array with a shape of
        # (4, 4).
        if affine is None:
            raise TypeError('\'affine\' cannot be None.')

        try:
            affine = np.array(affine, dtype=np.float64)
        except Exception:
            raise TypeError('\'affine\' must be convertible to a numpy array '
                            'of floats.')

        if affine.ndim != 2 or affine.shape != (4, 4):
            raise ValueError('\'affine\' must have a shape of (4, 4), '
                             'not {}.'.format(affine.shape))

        self._affine = affine
        self._transform_coord_sys = transform_coord_sys

    @property
    def affine(self) -> np.array:
        """Returns the affine transform as a numpy array."""
        return self._affine.copy()

    @property
    def transform_coord_sys(self) -> 'CoordinateSystem':
        """Returns the coord. system after the application of the affine."""
        return self._transform_coord_sys


class CoordinateSystem(IntEnum):
    """The possible coordinate systems of meshes."""
    SCANNER = 0
    VOXEL = 1


class Mesh(object):

    def __init__(self, vertices: Sequence, triangles: Sequence,
                 coordinate_system: CoordinateSystem = CoordinateSystem.VOXEL):
        """Triangle mesh of the cortical surface.

        The Mesh class represents a polygon mesh of the cortical surface
        defined by vertices and triangles.

        Args:
            vertices: The vertices of the mesh. Must a sequence that can be
                converted to a numpy array of floats with a shape of (N, 3)
                where N is the number of vertices.
            triangles: The triangles of the mesh. Must be a sequence that
                can be converted to a numpy array of integers with a shape of
                (M, 3) where M is the number of triangles.
            coordinate_system (optional): The coordinate system of the
                vertices (see CoordinateSystem). Defaults to VOXEL.

        Raises:
            TypeError: If vertices or triangles cannot be converted to numpy
                arrays.
            TypeError: If coordinate_system is not a CoordinateSystem member
                value.
            ValueError: If vertices or triangles do not have a shape of (N, 3)
                and (M, 3), respectively.

        Examples:
            Create a mesh with a single triangle.

            >>> from nimesh import Mesh
            >>> vertices = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            >>> triangles = [[0, 1, 2]]
            >>> mesh = Mesh(vertices, triangles)
            >>> print(mesh)
            Mesh: 3 vertices, 1 triangles

        """

        # Special case of vertices or triangles set to None.
        if vertices is None or triangles is None:
            raise TypeError('\'vertices\' and \'triangles\' cannot be None')

        # Try to convert the vertices and triangles to numpy arrays.
        try:
            vertices = np.array(vertices, dtype=np.float64)
        except Exception:
            raise TypeError('\'vertices\' must be convertible to a numpy '
                            'array of floats.')

        try:
            triangles = np.array(triangles, dtype=np.int64)
        except Exception:
            raise TypeError('\'triangles\' must be convertible to a numpy '
                            'array of integers.')

        # The shape of vertices and triangles must be (?, 3).
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError('\'vertices\' must have a shape of (N, 3) not '
                             '{}.'.format(vertices.shape))

        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError('\'triangles\' must have a shape of (M, 3) not '
                             '{}.'.format(triangles.shape))

        if not isinstance(coordinate_system, CoordinateSystem):
            raise TypeError('\'coordinate_system\' must be a member value of '
                            'CoordinateSystem, not a {}.'
                            .format(type(coordinate_system)))

        self._vertices = vertices
        self._triangles = triangles
        self._coordinate_system = coordinate_system

    def __repr__(self):
        return 'Mesh: {} vertices, {} triangles'.format(self.nb_vertices,
                                                        self.nb_triangles)

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Returns the coordinate system of the mesh."""
        return self._coordinate_system

    @property
    def nb_triangles(self):
        """Returns the number of triangles of the mesh."""
        return len(self._triangles)

    @property
    def nb_vertices(self):
        """Returns the number of vertices of the mesh."""
        return len(self._vertices)

    @property
    def triangles(self):
        """Returns a copy of the mesh's triangles."""
        return self._triangles.copy()

    @property
    def vertices(self):
        """Returns a copy of the mesh's vertices."""
        return self._vertices.copy()
