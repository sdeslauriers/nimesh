import unittest

import numpy as np

from collections import defaultdict
from nimesh.asarray import icosahedron, project_to_sphere, upsample
from nimesh.core import icosphere


class TestIcosahedron(unittest.TestCase):
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


class TestProjectOnSphere(unittest.TestCase):
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


class TestSphere(unittest.TestCase):
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
