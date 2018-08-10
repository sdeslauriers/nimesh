import os
import warnings
from os.path import isfile
from os.path import join
from typing import Tuple
from typing import Union

import nibabel as nib
import nibabel.freesurfer as nibfs
import numpy as np

from nimesh import AffineTransform
from nimesh import CoordinateSystem, Label, Mesh, Segmentation


def load(subject_directory: str, hemisphere: str, surface: str) -> Mesh:
    """Loads a mesh and associated data from a FreeSurfer subject directory

    Given the path to the root of a FreeSurfer subject directory, loads a mesh,
    its coordinate system, and segmentations.

    Args:
        subject_directory: The subject directory from which to load the mesh.
        hemisphere: The hemisphere to load. Must be either 'lh' or 'rh'.
        surface: The surface to load. Must be 'pial', 'white', 'inflated', or
        'smoothwm'.

    Returns:
        mesh: The requested mesh and associated data.

    Raises:
        ValueError if `hemisphere` is not 'lh` or 'rh'.
        ValueError if `surface` is not 'pial', 'white', 'inflated', or
            'smoothwm'.

    """

    # Validate the hemisphere and the surface names.
    if hemisphere not in ('rh', 'lh'):
        raise ValueError('"hemisphere" must be "lh" of "rh", not "{}".'
                         .format(hemisphere))

    if surface not in ('pial', 'white', 'inflated', 'smoothwm'):
        raise ValueError('"surface" must be "pial", "white", '
                         'or "inflated", not "{}".'
                         .format(surface))

    mesh_filename = join(subject_directory, 'surf', hemisphere + '.' + surface)
    mesh = load_mesh(mesh_filename)

    # Add the coordinate system and transform to voxel space if possible.
    coordinate_system, transform = _get_coordinate_system(subject_directory)
    mesh.coordinate_system = coordinate_system

    if transform is not None:
        mesh.add_transform(transform)

    # Load the segmentations from annotation files.
    label_dir = join(subject_directory, 'label')

    desikan_file = join(label_dir, hemisphere + '.aparc.annot')
    if isfile(desikan_file):
        mesh.add_segmentation(load_segmentation(desikan_file))

    dkt_file = join(label_dir, hemisphere + '.aparc.DKTatlas.annot')
    if isfile(dkt_file):
        mesh.add_segmentation(load_segmentation(dkt_file))

    destrieux_file = join(label_dir, hemisphere + '.aparc.a2009s.annot')
    if isfile(destrieux_file):
        mesh.add_segmentation(load_segmentation(destrieux_file))

    return mesh


def load_mesh(surface: str) -> Mesh:
    """Loads a FreeSurfer surface mesh.

    Loads a mesh from a FreeSurfer geometry file.

    Args:
        surface: The FreeSurfer geometry file, e.g. lh.pial or rh.white.

    Returns:
        mesh: The loaded mesh.

    """

    # Load the vertices and remove the CRAS translation if it exists.
    vertices, triangles, meta, info = nibfs.read_geometry(surface, True, True)
    if 'cras' in meta:
        vertices += meta['cras']

    return Mesh(vertices, triangles)


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
    alphas = {label.color[-1] for label in segmentation.labels.values()}
    if np.any(alphas != 255):
        warnings.warn('Alpha values for labels currently cannot be saved in '
                      'the FreeSurfer annotation file format. They will be '
                      'lost.',
                      RuntimeWarning)

    # To save a segmentation to a FreeSurfer annotation file, all labels must
    # have a distinct color.
    colors = {label.color for label in segmentation.labels.values()}
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
                     for key, label in segmentation.labels.items()])
    names = [label.name for label in segmentation.labels.values()]
    nibfs.write_annot(filename, keys, ctab, names)


def _get_coordinate_system(subject_directory: str) \
        -> Tuple[CoordinateSystem, Union[None, AffineTransform]]:
    """Gets the native coordinate system of a FreeSurfer subject

    Gets the native coordinate system of a FreeSurfer subject by looking at
    the affine of the MRI input. If the MRI does not exists or the
    coordinate system cannot be identified, the a warning is issued and
    CoordinateSystem.UNKNOWN is returned.

    Args:
        subject_directory: The FreeSurfer subject directory.

    Returns:
        coordinate_system: The native coordinate system of the subject.

    """

    # Verify that the MRI exists.
    mri_filename = os.path.join(subject_directory, 'mri', 'rawavg.mgz')
    if not os.path.isfile(mri_filename):
        warnings.warn('The coordinate system could not be identified because '
                      'MRI file ({}) does not exist.'
                      .format(mri_filename))
        return CoordinateSystem.UNKNOWN, None

    # Get the coordinate system information from the MRI image.
    affine = nib.load(mri_filename).affine
    codes = nib.aff2axcodes(affine)

    if codes == ('L', 'P', 'S'):
        coordinate_system = CoordinateSystem.LPS
    elif codes == ('R', 'A', 'S'):
        coordinate_system = CoordinateSystem.RAS
    else:
        coordinate_system = CoordinateSystem.UNKNOWN
        warnings.warn('The coordinate system could not be identified from '
                      'the MRI ({}).'
                      .format(mri_filename))

    transform = AffineTransform(CoordinateSystem.VOXEL, np.linalg.inv(affine))

    return coordinate_system, transform
