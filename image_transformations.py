import numpy as np
import cv2
import argparse

def get_affine_matrix(angle, hx, hy, sx, sy, tx, ty):
    """
    Calculates the affine matrix

    Args:
        angle (float): angle of rotation in degrees
        hx, hy (float): shear factors 
        sx, sy (float): scale factors 
        tx, ty (float): translation factors
        
    Returns:
        numpy.ndarray: 3x3 affine matrix
    """

    # convert degrees to radians
    angle = np.deg2rad(angle)

    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])

    H = np.array([
        [1,  hx, 0],
        [hy, 1,  0],
        [0,  0,  1]
    ])

    S = np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ])

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1 ]
    ])

    affine_matrix = T @ R @ H @ S
    return affine_matrix

def get_affine_dimensions(height, width, affine_matrix):
    """
    Calculates the output dimensions required to contain the entire
    affine-transformed image.

    Args:
        height (int): Original image height.
        width (int): Original image width.
        affine_matrix (numpy.ndarray): 3x3 affine transformation matrix.

    Returns:
        new_height (int): Required output height.
        new_width (int): Required output width.
        adjusted_matrix (numpy.ndarray): Affine matrix shifted so the
                                         transformed image begins at (0, 0).
    """

    # Image boundary corners in homogeneous coordinates
    corners = np.array([
        [0,     0,      1],
        [width, 0,      1],
        [width, height, 1],
        [0,     height, 1]
    ], dtype=np.float64)

    # Transform all four corners
    transformed_corners = (affine_matrix @ corners.T).T

    # Find the transformed bounding box
    min_x = np.floor(transformed_corners[:, 0].min())
    min_y = np.floor(transformed_corners[:, 1].min())
    max_x = np.ceil(transformed_corners[:, 0].max())
    max_y = np.ceil(transformed_corners[:, 1].max())

    new_width = int(max_x - min_x)
    new_height = int(max_y - min_y)

    # Shift the bounding box to start at output coordinate (0, 0)
    adjusted_matrix = affine_matrix.copy()
    adjusted_matrix[0, 2] -= min_x
    adjusted_matrix[1, 2] -= min_y

    return new_height, new_width, adjusted_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=float, default=0)
    parser.add_argument("--hx", type=float, default=0)
    parser.add_argument("--hy", type=float, default=0)
    parser.add_argument("--sx", type=float, default=1)
    parser.add_argument("--sy", type=float, default=1)
    parser.add_argument("--tx", type=float, default=0)
    parser.add_argument("--ty", type=float, default=0)
    args = parser.parse_args()

    affine_matrix = get_affine_matrix(args.angle, args.hx, args.hy, args.sx, args.sy, args.tx, args.ty)

    image = cv2.imread("checkerboard.png")
    H, W = image.shape[:2]

    new_H, new_W, adjusted_affine_matrix = get_affine_dimensions(
        H,
        W,
        affine_matrix
    )

    transformed_image = cv2.warpAffine(
        image,
        adjusted_affine_matrix[:2, :], # OpenCV warpAffine expects a 2x3 matrix
        (new_W, new_H)
    )

    cv2.imshow('Affine transformed', transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()