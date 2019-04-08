import numpy as np
import warnings

from collections import defaultdict


def compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """ Computes the vertex normals of a mesh

    The normal of a vertex is the average normal of all faces that include
    the vertex.

    Args:
        vertices: A 2D numpy array with a shape of (N, 3)
            where N is the number of vertices.
        faces: A 2D numpy array with a shape of (M, 3)
            where M is the number of faces.

    Returns:
        normals: A 2D numpy array with a shape of (N, 3) that
            contains the normal for each vertex.

    Raises:
        ValueError if a vertex has a normal with a norm of 0.

    """

    normals = []
    for i in range(len(vertices)):

        # Get all faces which involve the vertex.
        contains_vertex = np.any(faces == i, axis=1)
        vertex_faces = faces[contains_vertex, :]

        normal = np.zeros((3,))
        for face in vertex_faces:
            normal += np.cross(vertices[face[1]] - vertices[face[0]],
                               vertices[face[2]] - vertices[face[0]])

        # Normalize the normal.
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError('A vertex has a 0 norm normal')

        normal /= norm
        normals.append(normal)

    return np.array(normals)


def icosahedron() -> (np.ndarray, np.ndarray):
    """ Returns the vertices and the triangles of a regular icosahedron
    inscribed in a icosphere of radius 1 centered at the origin.

    Returns:
        vertices: A 2D numpy array with a shape of (M, 3) that
            contains the vertices of the icosahedron.
        faces: A 2D numpy array with a shape of (N, 3) that
            contains the faces of the icosahedron.

    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = [[-1, t, 0],
                [1, t, 0],
                [-1, -t, 0],
                [1, -t, 0],
                [0, -1, t],
                [0, 1, t],
                [0, -1, -t],
                [0, 1, -t],
                [t, 0, -1],
                [t, 0, 1],
                [-t, 0, -1],
                [-t, 0, 1]]

    triangles = [[0, 11, 5],
                 [0, 5, 1],
                 [0, 1, 7],
                 [0, 7, 10],
                 [0, 10, 11],
                 [1, 5, 9],
                 [5, 11, 4],
                 [11, 10, 2],
                 [10, 7, 6],
                 [7, 1, 8],
                 [3, 9, 4],
                 [3, 4, 2],
                 [3, 2, 6],
                 [3, 6, 8],
                 [3, 8, 9],
                 [4, 9, 5],
                 [2, 4, 11],
                 [6, 2, 10],
                 [8, 6, 7],
                 [9, 8, 1]]

    vertices = project_to_sphere(np.asarray(vertices))
    triangles = np.asarray(triangles)
    return vertices, triangles


def project_to_sphere(vertices: np.ndarray) -> np.ndarray:
    """ Computes the projection of the vertices of a mesh onto a icosphere of
    radius 1 centered at the origin.

    Args:
        vertices: A 2D numpy array with a shape of (N, 3)
            where N is the number of vertices.

    Returns:
        projected_vertices: A 2D numpy array with a shape of (N, 3) that
            contains the coordinates of the projection of each vertex in the
            same order as the input.

    Warnings:
        RuntimeWarning if one of the vertices has norm equal to zero

    """
    projected_vertices = np.zeros((len(vertices), 3))
    for k, v in enumerate(vertices):
        norm = np.linalg.norm(v)
        if norm == 0:
            warnings.warn('Trying to project a point with zero norm.',
                          RuntimeWarning)
            projected_vertices[k, :] = np.asarray(v)
        else:
            projected_vertices[k, :] = np.asarray(v) / norm
    return projected_vertices


def upsample(vertices: np.ndarray, triangles: np.ndarray) -> (
        np.ndarray, np.ndarray):
    """ Upsamples each triangle of a mesh

    |\                |\
    |  \              |  \
    |    \      -->   |----\
    |      \          | \  | \
    |________\        |__ \|__ \

    The upsampling strategy defines three new vertices for every face of the
    initial mesh. Each of these new vertices correspond to the middle point of
    one edge of the original triangle. Finally, a new covering of the original
    face is determined with the 6 points.

    The upsampling factor of the triangles is four.

    Args:
        vertices: A 2D numpy array with a shape of (N, 3)
            where N is the number of vertices.
        triangles: A 2D numpy array with a shape of (M, 3)
            where M is the number of faces.

    Returns:
        vertices: A 2D numpy array with a shape of (K, 3)
            where K is the number of vertices after upsampling.
        faces: A 2D numpy array with a shape of (4*M, 3)
            where M is the number of faces before upsampling.

    """
    upsampled_vertices = [list(v) for v in vertices]

    processed_points = defaultdict()

    def index_middle_point(point_1, point_2):
        # Given the indices of two points, it returns the index of the middle
        # point
        # Check if the edge has already been cut in two
        key = tuple(sorted([point_1, point_2]))
        if key in processed_points:
            return processed_points[key]
        else:
            vertex1 = upsampled_vertices[point_1]
            vertex2 = upsampled_vertices[point_2]

            middle = (np.asarray(vertex1) + np.asarray(vertex2)) / 2.0
            upsampled_vertices.append(list(middle))

            index = len(upsampled_vertices) - 1
            processed_points[key] = index
            return index

    upsampled_triangles = []
    for face in triangles:
        idx1, idx2, idx3 = face

        mid12 = index_middle_point(idx1, idx2)
        mid13 = index_middle_point(idx1, idx3)
        mid23 = index_middle_point(idx2, idx3)

        upsampled_triangles.append([idx1, mid12, mid13])
        upsampled_triangles.append([idx2, mid12, mid23])
        upsampled_triangles.append([idx3, mid13, mid23])
        upsampled_triangles.append([mid12, mid13, mid23])

    return np.array(upsampled_vertices), np.array(upsampled_triangles)
