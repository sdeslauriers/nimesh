import warnings

import nibabel as nib

from nimesh import CoordinateSystem, Mesh


def load(filename: str) -> Mesh:
    """Loads a mesh from a GifTI file

    Loads the vertices and triangles of a mesh in the GifTI file format.

    Args:
        filename: The name of the GifTI file to load.

    Returns:
        mesh: The mesh loaded from the supplied file.

    Raises:
        ValueError: If the file does not contains a vertex and triangle array.

    Warnings:
        RuntimeWarning: If the file contains more than one vertex or
            triangle array.

    """

    gii = nib.load(filename)

    # Get the vertices array. If there is more than one, warn the user and
    # use the first one.
    vertex_arrays = gii.get_arrays_from_intent('NIFTI_INTENT_POINTSET')

    if len(vertex_arrays) == 0:
        raise ValueError('The file {} does not contain any vertex data.'
                         .format(filename))

    elif len(vertex_arrays) > 1:
        warnings.warn('The file {} contains more than one vertex array. The '
                      'first one was used. Proceed with caution.'
                      .format(filename),
                      RuntimeWarning)
    vertex_array = vertex_arrays[0]
    vertices = vertex_array.data

    # Get the coordinate system from the metadata of the vertex array. If
    # there is none, assume voxel space.
    metadata = vertex_array.meta.metadata
    if 'cs' in metadata:
        coordinate_system = CoordinateSystem(int(metadata['cs']))
    else:
        warnings.warn('The file {} does not contain a reference coordinate '
                      'system. Assuming voxel space.'.format(filename),
                      RuntimeWarning)
        coordinate_system = CoordinateSystem.VOXEL

    # Get the triangles array. If there is more than one, warn the user and
    # use the first one.
    triangle_arrays = gii.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')

    if len(triangle_arrays) == 0:
        raise ValueError('The file {} does not contain any triangles.'
                         .format(filename))

    elif len(triangle_arrays) > 1:
        warnings.warn('The file {} contains more than one triangle array. '
                      'The first one was used. Proceed with caution.'
                      .format(filename),
                      RuntimeWarning)

    triangles = triangle_arrays[0].data

    return Mesh(vertices, triangles, coordinate_system)


def save(filename: str, mesh: Mesh) -> None:
    """Save a mesh to a GifTI file.

    Saves the mesh and its metadata to a GifTI file.

    Args:
        filename: The name of the file were the mesh will be saved. If it
            exists, it will be overwritten.
        mesh: The mesh to save.

    """

    gii = nib.gifti.GiftiImage()

    # The coordinate system is saved in the metadata of the vertex array.
    # The GifTI specifications indicate that multiple coordinate systems
    # could be saved using GiftiCoordSystem, but the nibabel implementation
    # does not seem to support that functionality.
    meta = {
        'cs': str(mesh.coordinate_system.value)
    }

    vertices_array = nib.gifti.GiftiDataArray(
        mesh.vertices,
        intent='NIFTI_INTENT_POINTSET',
        datatype='NIFTI_TYPE_FLOAT64',
        meta=meta
    )
    gii.add_gifti_data_array(vertices_array)

    triangles_array = nib.gifti.GiftiDataArray(
        mesh.triangles,
        intent='NIFTI_INTENT_TRIANGLE',
        datatype='NIFTI_TYPE_UINT64')
    gii.add_gifti_data_array(triangles_array)

    nib.save(gii, filename)
