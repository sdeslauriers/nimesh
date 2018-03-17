import unittest

import numpy as np

from nimesh import Mesh


class TestMesh(unittest.TestCase):
    """Test the nimesh.Mesh class."""

    def test_init(self):
        """Test the __init__ method."""

        # Create a simple mesh with two triangles.
        vertices = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ]

        triangles = [
            [0, 1, 2],
            [1, 2, 3],
        ]

        mesh = Mesh(vertices, triangles)
        self.assertEqual(mesh.nb_vertices, 4)
        self.assertEqual(mesh.nb_triangles, 2)
        np.testing.assert_array_almost_equal(mesh.vertices, vertices)
        np.testing.assert_array_almost_equal(mesh.triangles, triangles)

        # The vertices and triangles must be convertible to numpy arrays.
        self.assertRaises(TypeError, Mesh, None, triangles)
        self.assertRaises(TypeError, Mesh, vertices, None)

        # The vertices and triangles must have a shape of (?, 3).
        incorrect_vertices = np.array(vertices)[:, :2]
        self.assertRaises(ValueError, Mesh, incorrect_vertices, triangles)
        incorrect_triangles = np.array(triangles)[:, :2]
        self.assertRaises(ValueError, Mesh, vertices, incorrect_triangles)
