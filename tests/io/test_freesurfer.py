import os
import tempfile
import unittest

import numpy as np

import nimesh
from nimesh import AffineTransform, CoordinateSystem, Segmentation

from .test_gifti import minimal_mesh


class TestFreeSurfer(unittest.TestCase):
    """Test the nimesh.io.freesurfer module."""

    def test_surface_load(self):
        """Test minimal surface loading."""

        mesh = minimal_mesh()

        # Add a RAS transform, required for FreeSurfer.
        transform = AffineTransform(CoordinateSystem.RAS, np.eye(4))
        mesh.add_transform(transform)
        mesh.transform_to(CoordinateSystem.RAS)

        # Work in a temporary directory. This guarantees cleanup even on error.
        with tempfile.TemporaryDirectory() as directory:

            # Save and reload the test data.
            filename = os.path.join(directory, 'lh.pial')
            nimesh.io.freesurfer.save(filename, mesh)
            loaded = nimesh.io.freesurfer.load(filename)

            # Verify the general shape of the mesh.
            self.assertEqual(mesh.nb_vertices, loaded.nb_vertices)
            self.assertEqual(mesh.nb_triangles, loaded.nb_triangles)
            self.assertEqual(mesh.coordinate_system, loaded.coordinate_system)

    def test_segmentation_save_load(self):
        """Test saving and loading a segmentation."""

        segmentation = Segmentation('test.annot', [0, 0, 1, 1])

        # Work in a temporary directory. This guarantees cleanup even on error.
        with tempfile.TemporaryDirectory() as directory:

            # Save and reload the test data.
            filename = os.path.join(directory, segmentation.name)
            nimesh.io.freesurfer.save_segmentation(filename, segmentation)
            loaded = nimesh.io.freesurfer.load_segmentation(filename)

            np.testing.assert_array_almost_equal(segmentation.keys,
                                                 loaded.keys)
            self.assertEqual(segmentation.name, loaded.name)
