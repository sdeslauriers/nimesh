import os

from nimesh.core import AffineTransform, CoordinateSystem, Mesh
from nimesh.core import Label, Segmentation
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
