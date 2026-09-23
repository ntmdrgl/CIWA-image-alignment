import numpy as np
import cv2

from utils.config import load_config
from utils.camera_intrinsics import get_camera_matrix_from_camera
from utils.transformations import get_affine_matrix_from_cfg, get_distortion_coefficients_from_cfg

cfg = load_config()

image = cv2.imread("checkerboard.png")
H, W = image.shape[:2]

camera_matrix = get_camera_matrix_from_camera(cfg.visible_camera)
affine_matrix = get_affine_matrix_from_cfg(cfg)
dist_coeffs = get_distortion_coefficients_from_cfg(cfg)

print("Camera matrix:") 
print(camera_matrix, camera_matrix.shape, camera_matrix[0][0].dtype, "\n")

print("Affine matrix:")
print(affine_matrix, affine_matrix.shape, affine_matrix[0][0].dtype, "\n")

print("Distortion Coefficients:")
print(dist_coeffs, dist_coeffs.shape, dist_coeffs[0].dtype, "\n")

transformed_image = cv2.warpAffine(
    image,
    affine_matrix[:2, :], # OpenCV warpAffine expects a 2x3 matrix
    (W, H)
)

cv2.imshow('Affine transformed', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()