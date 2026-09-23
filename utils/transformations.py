import numpy as np
import cv2

def get_affine_matrix(angle, hx, hy, sx, sy, tx, ty):
    # convert degrees to radians
    angle = np.radians(angle)

    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ], dtype=np.float32)

    H = np.array([
        [1,  hx, 0],
        [hy, 1,  0],
        [0,  0,  1]
    ], dtype=np.float32)

    S = np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ], dtype=np.float32)

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1 ]
    ], dtype=np.float32)

    affine_matrix = T @ R @ H @ S
    return affine_matrix

def get_affine_matrix_from_cfg(cfg):
    affine_matrix = get_affine_matrix(
        cfg.angle, 
        cfg.hx, 
        cfg.hy, 
        cfg.sx, 
        cfg.sy, 
        cfg.tx, 
        cfg.ty,
    )
    return affine_matrix

def get_distortion_coefficients_from_cfg(cfg):
    dist_coeffs = np.array([cfg.k1, cfg.k2, cfg.p1, cfg.p2, cfg.k3], dtype=np.float32)
    return dist_coeffs

# def get_affine_dimensions(height, width, affine_matrix):
#     """
#     Calculates the output dimensions required to contain the entire
#     affine-transformed image.

#     Args:
#         height (int): Original image height.
#         width (int): Original image width.
#         affine_matrix (numpy.ndarray): 3x3 affine transformation matrix.

#     Returns:
#         new_height (int): Required output height.
#         new_width (int): Required output width.
#         adjusted_matrix (numpy.ndarray): Affine matrix shifted so the
#                                          transformed image begins at (0, 0).
#     """

#     # Image boundary corners in homogeneous coordinates
#     corners = np.array([
#         [0,     0,      1],
#         [width, 0,      1],
#         [width, height, 1],
#         [0,     height, 1]
#     ], dtype=np.float64)

#     # Transform all four corners
#     transformed_corners = (affine_matrix @ corners.T).T

#     # Find the transformed bounding box
#     min_x = np.floor(transformed_corners[:, 0].min())
#     min_y = np.floor(transformed_corners[:, 1].min())
#     max_x = np.ceil(transformed_corners[:, 0].max())
#     max_y = np.ceil(transformed_corners[:, 1].max())

#     new_width = int(max_x - min_x)
#     new_height = int(max_y - min_y)

#     # Shift the bounding box to start at output coordinate (0, 0)
#     adjusted_matrix = affine_matrix.copy()
#     adjusted_matrix[0, 2] -= min_x
#     adjusted_matrix[1, 2] -= min_y

#     return new_height, new_width, adjusted_matrix