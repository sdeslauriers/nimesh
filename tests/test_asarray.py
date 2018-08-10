from unittest import TestCase

import numpy as np

from nimesh.asarray import adjacency_matrix


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
