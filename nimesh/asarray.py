from itertools import permutations

import numpy as np
from scipy.sparse import csr_matrix


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


def adjacency_matrix(
        triangles: np.array,
        nb_vertices: int = None) -> csr_matrix:
    """Constructs an adjacency matrix

    Constructs the adjacency matrix of a mesh given its triangles. To be
    able to handle very large meshes, the matrix is return as a
    scipy.sparse.csr_matrix.

    Args:
        triangles: A numpy array with a shape of (n, 3) where each row contains
            the vertex indices of a triangle of a mesh.
        nb_vertices: The total number of the mesh. Specifying the number of
            vertices is essential if the mesh contains vertices that are not
            part of any triangle. If not specified, then
            np.max(triangles) + 1 is used.

    Returns:
        adjacency_matrix: The adjacency matrix of the list of triangles as a
            csr_matrix. Use `todense` to convert to a numpy array.

    """

    # Determine the number of vertices and initialize the output matrix.
    if nb_vertices is None:
        nb_vertices = np.max(triangles) + 1

    indices = zip(*((r, c) for t in triangles for r, c in permutations(t, 2)))
    data = np.ones((len(triangles) * 6,), dtype=np.bool)
    shape = (nb_vertices, nb_vertices)
    adjacency_matrix = csr_matrix((data, indices), shape)

    return adjacency_matrix
