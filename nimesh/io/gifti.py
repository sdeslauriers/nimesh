import warnings
from typing import Union

import nibabel as nib

from nimesh import AffineTransform, CoordinateSystem, Mesh, Segmentation
from nimesh.core import VertexData


def load(filename: str) -> Mesh:
    """Loads a mesh from a GifTI file

    Loads the vertices and triangles of a mesh in the GifTI file format. If
    a transform or segmentation is contained in the file, they will also be
    loaded.

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

    # Get the coordinate system from the metadata.
    if 'cs' in vertex_array.metadata:
        coordinate_system = CoordinateSystem(int(vertex_array.metadata['cs']))
    else:
        coordinate_system = _convert_nifti_code_to_coord_sys(
            vertex_array.coordsys.dataspace)

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

    # Get the normals array if it exists.
    normals_arrays = gii.get_arrays_from_intent('NIFTI_INTENT_VECTOR')
    if len(normals_arrays) > 0:
        if len(normals_arrays) != 1:
            warnings.warn('The file {} contains more than one vector array. '
                          'The first one was interpreted as mesh normals. '
                          'Proceed with caution.'.format(filename))

        normals = normals_arrays[0].data

    else:
        normals = None

    # Create the mesh.
    mesh = Mesh(vertices, triangles, coordinate_system, normals)

    # Add the transform if one was saved.
    transform = _get_transform_from_vertex_array(vertex_array)
    if transform is not None:
        mesh.add_transform(transform)

    # Add the segmentation if one was saved.
    segmentation = _create_segmentation_from_gii(gii)
    if segmentation is not None:
        mesh.add_segmentation(segmentation)

    # Add the vertex data if any was saved.
    vertex_data_arrays = gii.get_arrays_from_intent('NIFTI_INTENT_ESTIMATE')
    for vertex_data_array in vertex_data_arrays:

        if 'name' not in vertex_data_array.meta.metadata:
            warnings.warn('The file {} contains a vertex data array without '
                          'a name. It was ignored.'.format(filename))
            continue

        vertex_data = VertexData(vertex_data_array.meta.metadata['name'],
                                 vertex_data_array.data)
        mesh.add_vertex_data(vertex_data)

    return mesh


def load_segmentation(filename: str) -> Segmentation:
    """Loads a segmentation from a GifTI file.

    Loads the segmentation of a mesh from a GifTI file without loading the
    mesh data.

    Args:
        filename: The name of the file from which to load the segmentation.

    Returns:
        segmentation: The segmentation loaded from the GifTI file.

    Raises:
        ValueError: If the file does not contain a segmentation.

    """

    gii = nib.load(filename)

    segmentation = _create_segmentation_from_gii(gii)

    if segmentation is None:
        raise ValueError('The file {} does not contain any segmentation data.'
                         .format(filename))

    return segmentation


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

    # The GifTI file format only allows saving one segmentation. If the mesh
    # contains more that one, warn the user and save only the first one.
    segmentations = mesh.segmentations
    if len(segmentations) > 1:
        warnings.warn('The mesh has more that one segmentation, but GifTI '
                      'only supports one segmentation per file. Only the '
                      'first segmentation will be saved.')

    if len(segmentations) != 0:
        add_segmentation_to_gii(segmentations[0], gii)

    # The coordinate system is saved in the metadata of the vertex array.
    # Because it is possible to save the coordinate system of the points
    # without having a transform, we add it to the metadata.
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
        transform = transforms[0]
        coordinate_system = nib.gifti.GiftiCoordSystem(
            _convert_coord_sys_to_nifti_code(mesh.coordinate_system),
            _convert_coord_sys_to_nifti_code(transform.transform_coord_sys),
            transform.affine)
        meta['tcs'] = str(transform.transform_coord_sys.value)
    else:
        coordinate_system = None

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

    # Save the normals if they exist.
    if mesh.normals is not None:
        normals_array = nib.gifti.GiftiDataArray(
            mesh.normals.astype('f4'),
            intent='NIFTI_INTENT_VECTOR',
            datatype='NIFTI_TYPE_FLOAT32')
        gii.add_gifti_data_array(normals_array)

    # Save the vertex data if there is any.
    for vertex_data in mesh.vertex_data:

        vertex_data_array = nib.gifti.GiftiDataArray(
            vertex_data.data.astype('f4'),
            intent='NIFTI_INTENT_ESTIMATE',
            datatype='NIFTI_TYPE_FLOAT32',
            meta={'name': vertex_data.name})
        gii.add_gifti_data_array(vertex_data_array)

    nib.save(gii, filename)


def save_segmentation(filename: str,
                      segmentation: Segmentation,
                      anatomical_structure_primary: str = 'Cortex',
                      anatomical_structure_secondary: str = 'GrayWhite',
                      geometric_type: str = 'Anatomical') -> None:
    """Save a segmentation to a GifTI image.

    Save a the segmentation of a mesh into a GifTI file without saving the
    mesh itself.

    Args:
        filename: The name of the file where the segmentation is saved. If
            it exists, it will be overwritten.
        segmentation: The segmentation to save.
        anatomical_structure_primary: The name of the primary anatomical
            structure the segmentation refers to. For workbench.
        anatomical_structure_secondary: The name of the secondary anatomical
            structure the segmentation refers to. For workbench.
        geometric_type: The geometric type of the structure. For workbench.

    """

    gii = nib.gifti.GiftiImage()

    # In a label file, the meta data of the structure is in the
    # meta of the file.
    gii.meta.data.append(
        nib.gifti.GiftiNVPairs('AnatomicalStructurePrimary',
                               anatomical_structure_primary))
    gii.meta.data.append(
        nib.gifti.GiftiNVPairs('AnatomicalStructureSecondary',
                               anatomical_structure_secondary))
    gii.meta.data.append(
        nib.gifti.GiftiNVPairs('GeometricType',
                               geometric_type))

    add_segmentation_to_gii(segmentation, gii)

    nib.save(gii, filename)


def add_segmentation_to_gii(segmentation: Segmentation,
                            gii: nib.gifti.GiftiImage):
    """Adds a segmentation to a nibabel GifTI object.

    Add the segmentation data to a nibabel GifTI object by adding a new data
    array.

    Args:
        segmentation: The segmentation to add to the GifTI object.
        gii: The GifTI object where the segmentation is added.

    """

    # Save the name of the segmentation in the metadata of the array.
    meta = nib.gifti.GiftiMetaData.from_dict({'name': segmentation.name})

    label_array = nib.gifti.GiftiDataArray(
        segmentation.keys,
        intent='NIFTI_INTENT_LABEL',
        datatype='NIFTI_TYPE_INT32',
        meta=meta)

    gii.add_gifti_data_array(label_array)


def _convert_coord_sys_to_nifti_code(coord_sys: CoordinateSystem) -> int:
    """Converts coordinate systems to NIfTI xform codes"""

    convert = {
        CoordinateSystem.UNKNOWN: 0,
        CoordinateSystem.SCANNER: 1,
        CoordinateSystem.RAS: 2,
        CoordinateSystem.LPS: 2,
        CoordinateSystem.VOXEL: 0,
        CoordinateSystem.MNI: 4,
        CoordinateSystem.TALAIRACH: 3,
    }

    return convert[coord_sys]


def _convert_nifti_code_to_coord_sys(code: int) -> CoordinateSystem:
    """Converts a NIfTI xform code to a coordinate system"""

    convert = {
        0: CoordinateSystem.UNKNOWN,
        1: CoordinateSystem.SCANNER,
        2: CoordinateSystem.UNKNOWN,
        3: CoordinateSystem.TALAIRACH,
        4: CoordinateSystem.MNI,
    }

    return convert[code]


def _create_segmentation_from_gii(gii) -> Segmentation:
    """Creates a segmentation from a nibabel GifTI object.

    Creates a new segmentation from the data array of a nibabel GifTI
    object. If none of the data arrays have an intent of NIFTI_INTENT_LABEL,
    None is returned.

    Args:
        gii: The nibabel GifTI object from which to create the segmentation.

    Returns:
        segmentation: The loaded segmentation or None.

    """

    # Get the labels array. If there is more than one, warn the user and
    # use the first one.
    label_arrays = gii.get_arrays_from_intent('NIFTI_INTENT_LABEL')

    if len(label_arrays) == 0:
        return None

    elif len(label_arrays) > 1:
        warnings.warn('The file contains more than one label array. The '
                      'first one was used. Proceed with caution.',
                      RuntimeWarning)
    label_array = label_arrays[0]
    labels = label_array.data
    name = label_array.metadata['name']

    return Segmentation(name, labels)


def _get_transform_from_vertex_array(vertex_array)\
        -> Union[AffineTransform, None]:
    """Gets the affine transform from a GIfTI vertex array"""

    # GIfTI always contain a transform.
    gifti_transform = vertex_array.coordsys

    # If the metadata of the vertex array contains the key 'tcs',
    # use it to create the transform. If is doesn't, only add the
    # transform if the xform and xformspace are not both unknown.
    if 'tcs' in vertex_array.metadata:
        transform = AffineTransform(
            CoordinateSystem(int(vertex_array.metadata['tcs'])),
            gifti_transform.xform)

    elif gifti_transform.dataspace != 0 or gifti_transform.xformspace != 0:
        transform = AffineTransform(
            _convert_nifti_code_to_coord_sys(gifti_transform.xformspace),
            gifti_transform.xform)

    else:
        transform = None

    return transform
