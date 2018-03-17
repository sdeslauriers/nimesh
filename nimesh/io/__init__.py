import os.path

from nimesh import Mesh
from nimesh.io import gifti


def load(filename: str) -> Mesh:
    """Loads a mesh from a file.

    The file format is detected from the filename extension. For now, the only
    supported file format is GifTI (.gii).

    Args:
        filename: The name of the file to load. Must have a .gii extension.

    Returns:
        mesh: The mesh contained in the supplied file.

    Raises:
        ValueError: If the file extension is not supported.

    """

    extension = os.path.splitext(filename)[1]

    if extension == '.gii':
        mesh = gifti.load(filename)

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
