from unittest import TestCase

import numpy as np

from nimesh.asarray import adjacency_matrix
from nimesh.asarray import compute_normals


class TestComputeNormals(TestCase):
    """Test the nimesh.asarray.compute_normals function"""

    def test_simple(self):
        """Test using a simple mesh"""

        vertices = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ])
        triangles = np.array([
            [0, 1, 2]
        ], dtype=np.uint8)

        normals = compute_normals(vertices, triangles)
        expected_normals = np.array([[0, 0, -1], [0, 0, -1], [0, 0, -1]])
        np.testing.assert_array_almost_equal(normals, expected_normals)

    def test_orphan_vertex(self):
        """Test using a vertex with no triangles"""

        vertices = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
        ])
        triangles = np.array([
            [0, 1, 2]
        ], dtype=np.uint8)

        normals = compute_normals(vertices, triangles)
        expected_normals = np.array([
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(normals, expected_normals)

    def test_zero_norm(self):
        """Test with a vertex with a zero normal"""

        vertices = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ])
        triangles = np.array([
            [0, 1, 2]
        ], dtype=np.uint8)

        self.assertRaises(ValueError, compute_normals, vertices, triangles)


class TestConstructAdjacencyMatrix(TestCase):
    """Test the nimesh.asarray.construct_adjacency_matrix function"""

    def test_simple(self):
        """Test using a very simple mesh"""

        triangles = np.array([
            [0, 1, 2]
        ], dtype=np.uint8)

        adjacency = adjacency_matrix(triangles)
        expected = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=np.bool)

        np.testing.assert_array_equal(adjacency.todense(), expected)

    def test_missing_vertices(self):
        """Test using a very simple mesh"""

        triangles = np.array([
            [0, 1, 2]
        ], dtype=np.uint8)

        # The last vertex is not adjacent to anyone.
        adjacency = adjacency_matrix(triangles, 4)
        expected = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.bool)

        np.testing.assert_array_equal(adjacency.todense(), expected)
