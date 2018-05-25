import numpy as np


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
