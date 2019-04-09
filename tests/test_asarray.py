from collections import defaultdict
from unittest import TestCase

import numpy as np

from nimesh.asarray import adjacency_matrix
from nimesh.asarray import compute_normals
from nimesh.asarray import icosahedron
from nimesh.asarray import project_to_sphere
from nimesh.asarray import upsample
from nimesh.core import icosphere


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


class TestIcosahedron(TestCase):
    """Test the nimesh.asarray.icosahedron method."""

    def test_polyhedron(self):
        vertices, triangles = icosahedron()

        self.assertEqual(vertices.shape[0], 12)
        self.assertEqual(triangles.shape[0], 20)

        edges = defaultdict()
        for t in triangles:
            e1 = tuple(sorted([t[0], t[1]]))
            e2 = tuple(sorted([t[0], t[2]]))
            e3 = tuple(sorted([t[1], t[2]]))
            if e1 not in edges:
                edges[e1] = True
            if e2 not in edges:
                edges[e2] = True
            if e3 not in edges:
                edges[e3] = True

        # Check Euler's formula
        self.assertEqual(len(vertices) - len(edges.keys()) + len(triangles), 2)

        # Check regularity on edges
        edge_length = []
        for e in edges.keys():
            edge_length.append(np.linalg.norm(vertices[e[0]] - vertices[e[1]]))

        self.assertEqual(len(np.unique(edge_length)), 1)


class TestProjectOnSphere(TestCase):
    def test_warning(self):
        with self.assertWarns(RuntimeWarning):
            project_to_sphere(np.array([[0, 0, 0]]))

    def test_on_random_points(self):
        vertices = np.random.rand(20, 3).astype(np.float32)
        pv = project_to_sphere(vertices)

        norms = np.linalg.norm(pv, axis=1)
        np.testing.assert_almost_equal(norms, np.ones(vertices.shape[0]))

    def test_on_upsampled_icosahedron(self):
        vv, tt = upsample(*icosahedron())
        pv = project_to_sphere(vv)

        norms = np.linalg.norm(pv, axis=1)
        np.testing.assert_almost_equal(norms, np.ones(vv.shape[0]))


class TestSphere(TestCase):
    def test_default(self):
        my_sphere = icosphere()
        ico_v, ico_t = icosahedron()

        np.testing.assert_almost_equal(my_sphere.vertices, ico_v)
        np.testing.assert_almost_equal(my_sphere.triangles, ico_t)

    def test_upsampled(self):
        n = np.random.randint(1, 5)
        my_sphere = icosphere(n)

        self.assertEqual(20 * (4 ** n), my_sphere.triangles.shape[0])

        norms = np.linalg.norm(my_sphere.vertices, axis=1)
        np.testing.assert_almost_equal(norms,
                                       np.ones(my_sphere.vertices.shape[0]))

    def test_error(self):
        with self.assertRaises(ValueError):
            icosphere(-1)
