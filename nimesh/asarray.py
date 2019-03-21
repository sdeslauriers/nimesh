from itertools import permutations
from typing import List

import numpy as np
from scipy.sparse import csr_matrix


def apply_affine(vertices, affine):
    """"""

    homogeneous_vertices = np.hstack((vertices, np.ones((len(vertices), 1))))
    new_vertices = np.dot(affine, homogeneous_vertices.T)
    return new_vertices[:3, :].T


def compute_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """ Computes the vertex normals of a mesh

    The normal of a vertex is the average normal of all faces that include
    the vertex. If a vertex is not part of at least one triangles, its normal
    is set to [0, 0, 1].

    Args:
        vertices: A 2D numpy array with a shape of (N, 3)
            where N is the number of vertices.
        triangles: A 2D numpy array with a shape of (M, 3)
            where M is the number of triangles.

    Returns:
        normals: A 2D numpy array with a shape of (N, 3) that
            contains the normal for each vertex.

    Raises:
        ValueError if a vertex has a normal with a norm of 0.

    """

    # Add the normal of a triangle to each vertex of the triangle.
    normals: List[List[np.ndarray]] = [[] for _ in range(len(vertices))]
    for triangle in triangles:
        normal = np.cross(vertices[triangle[1]] - vertices[triangle[0]],
                          vertices[triangle[2]] - vertices[triangle[0]])
        for vertex_id in triangle:
            normals[vertex_id].append(normal)

    def compute_normal(vertex_normals):

        # Give vertices with no triangles an arbitrary normal
        if len(vertex_normals) == 0:
            return np.array([0, 0, 1])

        vertex_normal = np.mean(vertex_normals, axis=0)

        # Normalize the normal.
        norm = np.linalg.norm(vertex_normal)
        if norm == 0:
            raise ValueError('A vertex has a 0 norm normal')

        return vertex_normal / norm

    return np.array([compute_normal(n) for n in normals])


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
