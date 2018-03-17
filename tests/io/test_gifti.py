import os
import tempfile
import unittest

import numpy as np

import nimesh.io
from nimesh import CoordinateSystem, Mesh


class TestGifTI(unittest.TestCase):
    """Test the nimesh.io.gifti module."""

    def test_minimal_save_load(self):
        """Test saving and loading with a minimal mesh."""

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

        mesh = Mesh(vertices, triangles, CoordinateSystem.SCANNER)

        # Work in a temporary directory. This guarantees cleanup even on error.
        with tempfile.TemporaryDirectory() as directory:

            # Save the mesh and reload it.
            filename = os.path.join(directory, 'mesh.gii')
            nimesh.io.save(filename, mesh)
            loaded = nimesh.io.load(filename)

            # The coordinate system should not have changed.
            self.assertEqual(mesh.coordinate_system, loaded.coordinate_system)

            # The loaded data should match the saved data. Because there is no
            # data manipulation, this should be bit perfect.
            np.testing.assert_array_equal(mesh.vertices, loaded.vertices)
            np.testing.assert_array_equal(mesh.triangles, loaded.triangles)
