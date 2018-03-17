import numpy as np
from typing import Sequence


class Mesh(object):

    def __init__(self, vertices: Sequence, triangles: Sequence):
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

        Raises:
            TypeError: If vertices or triangles cannot be converted to numpy
                arrays.
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

        self._vertices = vertices
        self._triangles = triangles

    def __repr__(self):
        return 'Mesh: {} vertices, {} triangles'.format(self.nb_vertices,
                                                        self.nb_triangles)

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
