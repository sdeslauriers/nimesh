import os
import warnings

import nibabel as nib
import numpy as np
from nimesh.core import AffineTransform, CoordinateSystem, Mesh
from nimesh.core import Label, Segmentation
from nimesh.core import icosphere
from nimesh import io


def from_freesurfer(subject_directory: str,
                    hemisphere: str,
                    surface: str):
    """Loads a mesh from a FreeSurfer subject directory

    Args:
        subject_directory: The path to the subject's directory.
        hemisphere: The hemisphere to load. Must be either 'lh' or 'rh'.
        surface: The surface to load. Must be 'pial', 'white', or 'inflated'.

    Raises:
        ValueError: If `hemisphere` or `surface` are not valid strings.

    """

    warnings.warn('The nimesh.from_freesurfer function is deprecated. Use '
                  'nimesh.io.load instead.')

    # Validate the hemisphere and the surface names.
    if hemisphere not in ('rh', 'lh'):
        raise ValueError('\'hemisphere\' must be \'lh\' of \'rh\', not \'{}\'.'
                         .format(hemisphere))

    if surface not in ('pial', 'white', 'inflated', 'smoothwm'):
        raise ValueError('\'surface\' must be \'pial\', \'white\', '
                         'or \'inflated\', not \'{}\'.'
                         .format(surface))

    # Load the requested surface.
    surface_file = os.path.join(subject_directory, 'surf',
                                hemisphere + '.' + surface)
    mesh = io.freesurfer.load(surface_file)

    # Get the coordinate system information from the MRI image.
    nii = nib.load(os.path.join(subject_directory, 'mri', 'rawavg.mgz'))
    codes = nib.aff2axcodes(nii.affine)

    if codes == ('L', 'P', 'S'):
        coordinate_system = CoordinateSystem.LPS
    elif codes == ('R', 'A', 'S'):
        coordinate_system = CoordinateSystem.RAS
    else:
        coordinate_system = CoordinateSystem.UNKNOWN
        warnings.warn('The coordinate system could not be identified from '
                      'the MRI (rawavg.mgz). The transform will not be set.')

    mesh.coordinate_system = coordinate_system

    # Add the transform to voxel space if the coordinate system is known.
    if coordinate_system != CoordinateSystem.UNKNOWN:
        transform = AffineTransform(CoordinateSystem.VOXEL,
                                    np.linalg.inv(nii.affine))
        mesh.add_transform(transform)

    # Load the annotations.
    desikan_file = os.path.join(subject_directory, 'label',
                                hemisphere + '.aparc.annot')
    mesh.add_segmentation(io.freesurfer.load_segmentation(desikan_file))
    dkt_file = os.path.join(subject_directory, 'label',
                            hemisphere + '.aparc.DKTatlas.annot')
    mesh.add_segmentation(io.freesurfer.load_segmentation(dkt_file))
    destrieux_file = os.path.join(subject_directory, 'label',
                                  hemisphere + '.aparc.a2009s.annot')
    mesh.add_segmentation(io.freesurfer.load_segmentation(destrieux_file))

    return mesh
