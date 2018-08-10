import os.path
from os.path import isdir

from nimesh import Mesh
from nimesh.io import gifti
from nimesh.io import freesurfer


def load(filename: str, hemisphere: str = 'lh', surface: str = 'pial') -> Mesh:
    """Loads a mesh from a file.

    The file format is detected from the filename extension. The supported file
    formats are:
        - .gii : GifTI file containing a mesh and associated information
        - .white, .pial, .inflated, .smoothwm, .surf : FreeSurfer meshes
            with no associated information

    In addition to those formats, if the filename points to a directory,
    a FreeSurfer subject directory is assumed and a mesh will be loaded
    according to the `hemisphere` and `surface` parameters.

    Args:
        filename: The name of the file to load.
        hemisphere: If the filename is a FreeSurfer subject directory,
            `hemisphere` specifies which hemisphere to load. Must be 'lh' or
            'rh'. Ignored if the filename is not a FreeSurfer directory.
        surface: If the filename is a FreeSurfer subject directory,
            `surface` specifies which surface to load. Must be 'pial',
            'white', 'inflated', or 'smoothwm'. Ignored if the filename is
            not a FreeSurfer directory.

    Returns:
        mesh: The mesh contained in the supplied file.

    Raises:
        ValueError: If the file extension is not supported.

    """

    extension = os.path.splitext(filename)[1]

    if extension == '.gii':
        mesh = gifti.load(filename)

    elif extension in ('.white', '.pial', '.inflated', '.smoothwm', '.surf'):
        mesh = freesurfer.load_mesh(filename)

    elif isdir(filename):
        mesh = freesurfer.load(filename, hemisphere, surface)

    else:
        raise ValueError('Unknown file format {}.'.format(extension))

    return mesh


def save(filename: str, mesh: Mesh) -> None:
    """ Saves a mesh to a file

    The output format depends on the extension of the supplied filename. For
    now, the only supported file format is GifTI (.gii).

    Args:
        filename: The name of the file were the mesh will be saved. If it
            exists, it will be overwritten.
        mesh: The mesh to save.

    Raises:
        ValueError: If the file extension is not supported.

    """

    extension = os.path.splitext(filename)[1]

    if extension == '.gii':
        gifti.save(filename, mesh)

    else:
        raise ValueError('Unknown file format {}.'.format(extension))
