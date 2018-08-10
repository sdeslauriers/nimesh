import unittest

import numpy as np

from nimesh import AffineTransform, CoordinateSystem, Mesh
from nimesh.core import VertexData


class TestAffineTransform(unittest.TestCase):
    """Test the nimesh.AffineTransform class."""

    def test_init(self):
        """Test the __init__ method."""

        # Create a basic transform to scanner.
        affine = np.eye(4)
        transform = AffineTransform(CoordinateSystem.SCANNER, affine)

        np.testing.assert_array_equal(transform.affine, affine)
        self.assertEqual(transform.transform_coord_sys,
                         CoordinateSystem.SCANNER)

        # The affine cannot be None.
        self.assertRaises(TypeError, AffineTransform,
                          CoordinateSystem.SCANNER, None)

        # The affine must be (4, 4).
        self.assertRaises(ValueError, AffineTransform,
                          CoordinateSystem.SCANNER, [[0, 0], [0, 0]])


class TestMesh(unittest.TestCase):
    """Test the nimesh.Mesh class."""

    def test_add_vertex_data(self):
        """Test the add_vertex_data method"""

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

        # Normal addition of vertex data.
        vertex_data = VertexData('test', np.ones((4, 1)))
        mesh.add_vertex_data(vertex_data)
        self.assertEqual(mesh.vertex_data['test'], vertex_data)

        # Two vertex data with the same name is not permitted.
        def same_name():
            bad_data = VertexData('test', np.ones((4, 1)))
            mesh.add_vertex_data(bad_data)
        self.assertRaises(ValueError, same_name)

        # Vertex data with the wrong number of vertices fails.
        def wrong_data():
            bad_data = VertexData('other-test', np.ones((3, 1)))
            mesh.add_vertex_data(bad_data)
        self.assertRaises(ValueError, wrong_data)

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

        # Changing the vertices is permitted.
        mesh.vertices = np.zeros((4, 3))
        np.testing.assert_array_almost_equal(mesh.vertices, np.zeros((4, 3)))

        # Changing the number of vertices is not permitted.
        def set_vertices():
            mesh.vertices = np.zeros((3, 3))
        self.assertRaises(ValueError, set_vertices)

        # The vertices and triangles must be convertible to numpy arrays.
        self.assertRaises(TypeError, Mesh, None, triangles)
        self.assertRaises(TypeError, Mesh, vertices, None)

        # The vertices and triangles must have a shape of (?, 3).
        incorrect_vertices = np.array(vertices)[:, :2]
        self.assertRaises(ValueError, Mesh, incorrect_vertices, triangles)
        incorrect_triangles = np.array(triangles)[:, :2]
        self.assertRaises(ValueError, Mesh, vertices, incorrect_triangles)

    def test_transform_to(self):
        """Test the transform_to method."""

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

        mesh = Mesh(vertices, triangles, CoordinateSystem.VOXEL)

        # Add a transform to scanner space.
        affine = [
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ]
        transform = AffineTransform(CoordinateSystem.SCANNER, affine)
        mesh.add_transform(transform)

        # Change the coordinate system.
        mesh.transform_to(CoordinateSystem.SCANNER)
        expected_vertices = [
            [1, 2, 3],
            [2, 2, 3],
            [2, 3, 3],
            [1, 3, 3],
        ]

        self.assertEqual(mesh.coordinate_system, CoordinateSystem.SCANNER)
        np.testing.assert_array_almost_equal(mesh.vertices, expected_vertices)

        # Going back to the original coordinate system should yield the
        # original mesh.
        mesh.transform_to(CoordinateSystem.VOXEL)
        self.assertEqual(mesh.coordinate_system, CoordinateSystem.VOXEL)
        np.testing.assert_array_almost_equal(mesh.vertices, vertices)

        # Transforming to the current coordinate system does nothing.
        mesh.transform_to(CoordinateSystem.VOXEL)
        self.assertEqual(mesh.coordinate_system, CoordinateSystem.VOXEL)
        np.testing.assert_array_almost_equal(mesh.vertices, vertices)


class TestVertexData(unittest.TestCase):
    """Test the nimesh.core.VertexData class"""

    def test_init(self):
        """Test the __init__ method"""

        # Normal use of init.
        vertex_data = VertexData('test', np.zeros((10, 1)))
        self.assertEqual(vertex_data.name, 'test')
        self.assertEqual(vertex_data.nb_vertices, 10)

        # Modifying the data is permitted if we do not change the length.
        vertex_data.data = np.ones((10, 2))
        np.testing.assert_array_almost_equal(vertex_data.data,
                                             np.ones((10, 2)))

        # Changing the length fails.
        def wrong_data():
            vertex_data.data = np.ones((5, 2))
        self.assertRaises(ValueError, wrong_data)

        # The data must be convertible to an array of floats.
        self.assertRaises(TypeError, VertexData, 'test', {})
