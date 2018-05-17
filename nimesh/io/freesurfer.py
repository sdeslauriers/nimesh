import os
import warnings

import nibabel.freesurfer as nibfs
import numpy as np

from nimesh import CoordinateSystem, Label, Mesh, Segmentation


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

    keys, ctab, names = nibfs.read_annot(filename)
    segmentation = Segmentation(os.path.basename(filename), keys)

    def rgbt2rgba(color):
        return color[0], color[1], color[2], 255

    # Add the label names.
    for i, (color, name) in enumerate(zip(ctab, names)):
        segmentation.add_label(i, Label(name.decode(), rgbt2rgba(color[:4])))

    return segmentation


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

    Saves a segmentation to a FreeSurfer annotation file. To be saved
    correctly, each label needs to have a distinct color. The current
    implementation relies on nibabel and does not save/load alpha values
    correctly.

    Args:
        filename: The name of the annotation file where the segmentation will
            be saved.
        segmentation: The segmentation to save.

    Raises:
        ValueError if the segmentation has labels with duplicate colors.
            This is a limitation imposed by the FreeSurfer annotation format.

    """

    # nibabel does not appear to save the alpha value and always returns
    # 255. Warn the user that alpha values will be lost if they are not all
    # 255.
    alphas = {label.color[-1] for _, label in segmentation.labels}
    if np.any(alphas != 255):
        warnings.warn('Alpha values for labels currently cannot be saved in '
                      'the FreeSurfer annotation file format. They will be '
                      'lost.',
                      RuntimeWarning)

    # To save a segmentation to a FreeSurfer annotation file, all labels must
    # have a distinct color.
    colors = {label.color for _, label in segmentation.labels}
    if len(colors) != len(segmentation.labels):
        raise ValueError('To save a segmentation to a FreeSurfer annotation '
                         'file, all labels must have a distinct color.')

    def color2annot(color):
        return color[0] + \
               color[1] * 256 + \
               color[2] * 256 ** 2 + \
               (255 - color[3]) * 256 ** 3

    def rgba2rgbt(color):
        """Change alpha to transparency in color"""
        return color[0], color[1], color[2], 255 - color[3]

    keys = segmentation.keys
    ctab = np.array([(*rgba2rgbt(label.color), color2annot(label.color))
                     for key, label in segmentation.labels])
    names = [label.name for _, label in segmentation.labels]
    nibfs.write_annot(filename, keys, ctab, names)
