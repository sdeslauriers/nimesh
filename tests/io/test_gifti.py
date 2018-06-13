import os
import tempfile
import unittest

import numpy as np

import nibabel as nib
import nimesh.io
from nimesh import AffineTransform, CoordinateSystem, Mesh, Segmentation
from nimesh.core import VertexData


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

    def test_coordinate_system_save_load(self):
        """Test saving and loading the transform coordinate system"""

        mesh = minimal_mesh()

        # Work in a temporary directory. This guarantees cleanup even on error.
        with tempfile.TemporaryDirectory() as directory:

            mesh.coordinate_system = CoordinateSystem.VOXEL
            mesh.add_transform(AffineTransform(CoordinateSystem.RAS,
                                               np.eye(4)))

            filename = os.path.join(directory, 'mesh.gii')
            nimesh.io.save(filename, mesh)
            loaded = nimesh.io.load(filename)

            self.assertEqual(mesh.coordinate_system, loaded.coordinate_system)
            self.assertTrue(len(loaded.transforms) == 1)
            self.assertEqual(mesh.transforms[0].transform_coord_sys,
                             loaded.transforms[0].transform_coord_sys)

        # A mesh with no transforms should also be loaded correctly.
        mesh = minimal_mesh()
        with tempfile.TemporaryDirectory() as directory:

            filename = os.path.join(directory, 'other-mesh.gii')
            nimesh.io.save(filename, mesh)
            loaded = nimesh.io.load(filename)

            self.assertEqual(mesh.coordinate_system, loaded.coordinate_system)
            self.assertTrue(len(loaded.transforms) == 0)

        # Meshes without transforms and saved without nimesh should also load.
        with tempfile.TemporaryDirectory() as directory:

            gii = nib.gifti.GiftiImage()

            vertices_array = nib.gifti.GiftiDataArray(
                mesh.vertices.astype('f4'),
                intent='NIFTI_INTENT_POINTSET',
                datatype='NIFTI_TYPE_FLOAT32')
            gii.add_gifti_data_array(vertices_array)

            triangles_array = nib.gifti.GiftiDataArray(
                mesh.triangles.astype('i4'),
                intent='NIFTI_INTENT_TRIANGLE',
                datatype='NIFTI_TYPE_INT32')
            gii.add_gifti_data_array(triangles_array)

            filename = os.path.join(directory, 'mesh.gii')
            nib.save(gii, filename)
            loaded = nimesh.io.load(filename)

            self.assertEqual(loaded.coordinate_system,
                             CoordinateSystem.UNKNOWN)

        # Meshes with a transforms and saved without nimesh should also load.
        with tempfile.TemporaryDirectory() as directory:

            gii = nib.gifti.GiftiImage()

            gifti_coord_sys = nib.gifti.GiftiCoordSystem(
                'NIFTI_XFORM_SCANNER_ANAT',
                'NIFTI_XFORM_MNI_152',
                np.eye(4))
            vertices_array = nib.gifti.GiftiDataArray(
                mesh.vertices.astype('f4'),
                intent='NIFTI_INTENT_POINTSET',
                datatype='NIFTI_TYPE_FLOAT32',
                coordsys=gifti_coord_sys)
            gii.add_gifti_data_array(vertices_array)

            triangles_array = nib.gifti.GiftiDataArray(
                mesh.triangles.astype('i4'),
                intent='NIFTI_INTENT_TRIANGLE',
                datatype='NIFTI_TYPE_INT32')
            gii.add_gifti_data_array(triangles_array)

            filename = os.path.join(directory, 'mesh.gii')
            nib.save(gii, filename)
            loaded = nimesh.io.load(filename)

            self.assertEqual(loaded.coordinate_system,
                             CoordinateSystem.SCANNER)
            self.assertTrue(len(loaded.transforms) == 1)
            self.assertEqual(
                loaded.transforms[0].transform_coord_sys,
                CoordinateSystem.MNI)

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

    def test_vertex_data_save_load(self):
        """Test saving and loading vertex data"""

        mesh = minimal_mesh()

        # Add some vertex data.
        vertex_data_a = VertexData('a', np.random.randn(mesh.nb_vertices, 2))
        mesh.add_vertex_data(vertex_data_a)
        vertex_data_b = VertexData('b', np.random.randn(mesh.nb_vertices, 2))
        mesh.add_vertex_data(vertex_data_b)

        # Work in a temporary directory. This guarantees cleanup even on error.
        with tempfile.TemporaryDirectory() as directory:

            # Save the mesh and reload it.
            filename = os.path.join(directory, 'data.gii')
            nimesh.io.gifti.save(filename, mesh)
            loaded = nimesh.io.gifti.load(filename)

            # The loaded data should match the saved data.
            np.testing.assert_array_almost_equal(
                loaded.vertex_data['a'].data,
                vertex_data_a.data)
            np.testing.assert_array_almost_equal(
                loaded.vertex_data['b'].data,
                vertex_data_b.data)
