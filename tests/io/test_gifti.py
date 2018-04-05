import os
import tempfile
import unittest

import numpy as np

import nimesh.io
from nimesh import AffineTransform, CoordinateSystem, Mesh, Segmentation


def minimal_mesh():
    """Returns a mesh with two triangles."""

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

    return Mesh(vertices, triangles, CoordinateSystem.SCANNER)


class TestGifTI(unittest.TestCase):
    """Test the nimesh.io.gifti module."""

    def test_minimal_save_load(self):
        """Test saving and loading with a minimal mesh."""

        mesh = minimal_mesh()

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

    def test_transform_save_load(self):
        """Test saving with transforms."""

        mesh = minimal_mesh()

        # Add a transform to voxel space.
        affine = [
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ]
        transform = AffineTransform(CoordinateSystem.VOXEL, affine)
        mesh.add_transform(transform)

        # Work in a temporary directory. This guarantees cleanup even on error.
        with tempfile.TemporaryDirectory() as directory:

            # Save the mesh and reload it.
            filename = os.path.join(directory, 'mesh-transform.gii')
            nimesh.io.save(filename, mesh)
            loaded = nimesh.io.load(filename)

            # The coordinate system should not have changed.
            self.assertEqual(mesh.coordinate_system, loaded.coordinate_system)

            # The loaded data should match the saved data. Because there is no
            # data manipulation, this should be bit perfect.
            np.testing.assert_array_equal(mesh.vertices, loaded.vertices)
            np.testing.assert_array_equal(mesh.triangles, loaded.triangles)

            # The transform should be present.
            self.assertEqual(len(loaded.transforms), 1)
            loaded_transform = loaded.transforms[0]
            self.assertEqual(loaded_transform.transform_coord_sys,
                             CoordinateSystem.VOXEL)
            np.testing.assert_array_almost_equal(loaded_transform.affine,
                                                 affine)

    def test_mesh_segmentation_save_load(self):
        """Test saving a mesh with a segmentation."""

        mesh = minimal_mesh()

        # Add a segmentation.
        segmentation = Segmentation('seg', [0, 0, 1, 1])
        mesh.add_segmentation(segmentation)

        # Work in a temporary directory. This guarantees cleanup even on error.
        with tempfile.TemporaryDirectory() as directory:

            # Save the mesh and reload it.
            filename = os.path.join(directory, 'mesh-segmentation.gii')
            nimesh.io.save(filename, mesh)
            loaded = nimesh.io.load(filename)

            # The coordinate system should not have changed.
            self.assertEqual(mesh.coordinate_system, loaded.coordinate_system)

            # The loaded data should match the saved data. Because there is no
            # data manipulation, this should be bit perfect.
            np.testing.assert_array_equal(mesh.vertices, loaded.vertices)
            np.testing.assert_array_equal(mesh.triangles, loaded.triangles)

            # The segmentation should be present.
            self.assertEqual(len(loaded.segmentations), 1)
            loaded_segmentation = loaded.segmentations[0]
            self.assertEqual(loaded_segmentation.name,
                             segmentation.name)
            np.testing.assert_array_almost_equal(loaded_segmentation.keys,
                                                 segmentation.keys)

    def test_segmentation_save_load(self):
        """Test saving a loading segmentations."""

        segmentation = Segmentation('test', [0, 0, 1, 1, 2, 2, 3, 3])

        # Work in a temporary directory. This guarantees cleanup even on error.
        with tempfile.TemporaryDirectory() as directory:

            # Save the mesh and reload it.
            filename = os.path.join(directory, 'segmentation.gii')
            nimesh.io.gifti.save_segmentation(filename, segmentation)
            loaded = nimesh.io.gifti.load_segmentation(filename)

            # The loaded data should match the saved data.
            np.testing.assert_array_almost_equal(segmentation.keys,
                                                 loaded.keys)
