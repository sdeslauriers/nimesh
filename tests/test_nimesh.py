import unittest

import numpy as np

from nimesh import AffineTransform, CoordinateSystem, Mesh


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

        mesh = Mesh(vertices, triangles)

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
