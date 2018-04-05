import os
import warnings

import nibabel.freesurfer as nibfs
import numpy as np

from nimesh import CoordinateSystem, Mesh, Segmentation


def load(surface: str, annotation: str = None) -> Mesh:
    """Loads a FreeSurfer surface mesh.

    Loads a mesh from a FreeSurfer geometry file, optionally loading
    annotations.

    Args:
        surface: The FreeSurfer geometry file, e.g. lh.pial or rh.white.
        annotation (optional): The annotation file to load, for example
            lh.aparc.DKTatlas.annot or lh.aparc.annot.

    Returns:
        mesh: The loaded mesh.

    """

    # Load the vertices and remove the CRAS translation if it exists.
    vertices, triangles, meta, info = nibfs.read_geometry(surface, True, True)
    if 'cras' in meta:
        vertices += meta['cras']

    # We supposed the FreeSurfer surface is in RAS.
    mesh = Mesh(vertices, triangles, CoordinateSystem.RAS)

    # Load the annotations, if requested.
    if annotation is not None:
        segmentation = load_segmentation(annotation)
        mesh.add_segmentation(segmentation)

    return mesh


def load_segmentation(filename: str) -> Segmentation:
    """Loads a segmentation from a FreeSurfer annot file.

    Loads the a segmentation from a FreeSurfer annotation file. Because annot
    files do not have a segmentation name, the file name is used instead.

    Args:
        filename: The name of the annot file that is loaded.

    Returns:
        segmentation: A segmentation whose name is `filename`.

    """

    labels, ctab, names = nibfs.read_annot(filename, True)

    return Segmentation(os.path.basename(filename), labels)


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


def save_segmentation(filename: str, segmentation: Segmentation):
    """Saves a segmentation to a FreeSurfer annot file.

    Args:
        filename: The name of the annotation file where the segmentation will
            be saved.
        segmentation: The segmentation to save.

    """

    keys = segmentation.keys
    ctab = np.array([(i, 0, 0, 0, i) for i in np.unique(keys)],
                    dtype=np.int16)
    names = [str(i) for i in np.unique(keys)]
    nibfs.write_annot(filename, keys, ctab, names)
