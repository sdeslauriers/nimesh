import warnings

import nibabel.freesurfer as nibfs
import numpy as np

from nimesh import CoordinateSystem, Mesh


def load(surface: str):
    """Loads a FreeSurfer surface mesh."""

    # Load the vertices and remove the CRAS translation.
    vertices, triangles, meta, info = nibfs.read_geometry(surface, True, True)
    vertices += meta['cras']

    # We supposed the FreeSurfer surface is in RAS.
    mesh = Mesh(vertices, triangles, CoordinateSystem.RAS)

    return mesh


def save(filename: str, mesh: Mesh):
    """Saves the vertices and triangles of a mesh to a FreeSurfer file."""

    # Convert the mesh to RAS.
    if mesh.coordinate_system != CoordinateSystem.RAS:
        available_transforms = [t.transform_coord_sys for t in mesh.transforms]
        if CoordinateSystem.RAS not in available_transforms:
            raise ValueError('To save a mesh in the FreeSurfer format, '
                             'it must be in the RAS coordinate system or '
                             'have a transform to RAS.')

        warnings.warn('The mesh is not in RAS, transforming it to RAS.')
        mesh.transform_to(CoordinateSystem.RAS)

    # Generate fake metatdata for the geometry file.
    meta = {
        'head': np.array([2, 0, 20], dtype=np.int32),
        'valid': '1 # volume info valid',
        'filename': filename,
        'volume': np.array([256, 256, 256]),
        'voxelsize': np.array([1, 1, 1]),
        'xras': [1, 0, 0],
        'yras': [0, 1, 0],
        'zras': [0, 0, 1],
        'cras': [0, 0, 0],
    }

    nibfs.write_geometry(filename, mesh.vertices, mesh.triangles, True, meta)
