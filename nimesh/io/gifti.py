import warnings

import nibabel as nib

from nimesh import AffineTransform, CoordinateSystem, Mesh


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

    # Get the coordinate system from the transform
    gifti_transform = vertex_array.coordsys
    coordinate_system = CoordinateSystem(gifti_transform.dataspace)

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

    # Create the mesh.
    mesh = Mesh(vertices, triangles, coordinate_system)

    # Add the transform if one was saved.
    if gifti_transform.dataspace != gifti_transform.xformspace:
        transform = AffineTransform(
            CoordinateSystem(gifti_transform.xformspace),
            gifti_transform.xform)
        mesh.add_transform(transform)

    return mesh


def save(filename: str, mesh: Mesh,
         anatomical_structure_primary: str = 'Cortex',
         anatomical_structure_secondary: str = 'GrayWhite',
         geometric_type: str = 'Anatomical') -> None:
    """Save a mesh to a GifTI file.

    Saves the mesh and its metadata to a GifTI file.

    Args:
        filename: The name of the file were the mesh will be saved. If it
            exists, it will be overwritten.
        mesh: The mesh to save.
        anatomical_structure_primary (optional): The tag of the primary
            anatomical structure for workbench.
        anatomical_structure_secondary (optional): The tag of the secondary
            anatomical structure for workbench.
        geometric_type (optional): The geometric type for workbench.

    """

    gii = nib.gifti.GiftiImage()

    # The coordinate system is saved in the metadata of the vertex array.
    # The GifTI specifications indicate that multiple coordinate systems
    # could be saved using GiftiCoordSystem, but the nibabel implementation
    # does not seem to support that functionality.
    meta = {
        'cs': str(mesh.coordinate_system.value)
    }

    # For now, the GifTI implementation of nibabel seems to support only a
    # single transform. If the mesh has more than one, warn the user and
    # save the first one.
    transforms = mesh.transforms
    if len(transforms) > 1:
        warnings.warn('The mesh has more than one transform but GifTI only '
                      'supports a single transform. Only the first one will '
                      'be saved.')

    if len(transforms) != 0:
        coordinate_system = nib.gifti.GiftiCoordSystem(
            mesh.coordinate_system.value,
            transforms[0].transform_coord_sys.value,
            transforms[0].affine)
    else:
        coordinate_system = nib.gifti.GiftiCoordSystem(mesh.coordinate_system,
                                                       mesh.coordinate_system)

    vertices_array = nib.gifti.GiftiDataArray(
        mesh.vertices.astype('f4'),
        intent='NIFTI_INTENT_POINTSET',
        datatype='NIFTI_TYPE_FLOAT32',
        meta=meta,
        coordsys=coordinate_system
    )

    # Add metadata for Workbench. In a surface file, the meta data of the
    # structure is in the meta of the point set array.
    vertices_array.meta.data.append(
        nib.gifti.GiftiNVPairs('AnatomicalStructurePrimary',
                               anatomical_structure_primary))
    vertices_array.meta.data.append(
        nib.gifti.GiftiNVPairs('AnatomicalStructureSecondary',
                               anatomical_structure_secondary))
    vertices_array.meta.data.append(
        nib.gifti.GiftiNVPairs('GeometricType', geometric_type))

    gii.add_gifti_data_array(vertices_array)

    triangles_array = nib.gifti.GiftiDataArray(
        mesh.triangles.astype('i4'),
        intent='NIFTI_INTENT_TRIANGLE',
        datatype='NIFTI_TYPE_INT32')
    gii.add_gifti_data_array(triangles_array)

    nib.save(gii, filename)
