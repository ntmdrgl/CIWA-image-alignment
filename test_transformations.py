import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.config import load_config
from utils.camera_intrinsics import get_camera_matrix_from_camera
from utils.transformations import get_affine_matrix_from_cfg, get_distortion_coefficients_from_cfg

cfg = load_config()

image = cv2.imread("visible.png")
checkboard = cv2.imread("checkerboard_large.png")
H, W = image.shape[:2]

camera = cfg.visible_camera

camera_matrix = get_camera_matrix_from_camera(camera)
affine_matrix = get_affine_matrix_from_cfg(cfg, camera)
dist_coeffs = get_distortion_coefficients_from_cfg(cfg)

print("Camera matrix:") 
print(camera_matrix, "\n")

print("Affine matrix:")
print(affine_matrix, "\n")

print("Distortion Coefficients:")
print(dist_coeffs, "\n")

undistorted_image = cv2.undistort(
    image, 
    camera_matrix, 
    dist_coeffs
)

transformed_image = cv2.warpAffine(
    undistorted_image,
    affine_matrix[:2, :], # OpenCV warpAffine expects a 2x3 matrix
    (W, H)
)

# apply transformations to checkboard for side-by-side
undistorted_checkerboard = cv2.undistort(checkboard, camera_matrix, dist_coeffs)
transformed_checkerboard = cv2.warpAffine(undistorted_checkerboard, affine_matrix[:2, :], (W, H))

# Show image transformation + checkerboard side-by-side
fig, axes = plt.subplots(1, 2, figsize=(8, 6.5))

axes[0].imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Transformed Image')
axes[0].axis('off') 

axes[1].imshow(transformed_checkerboard)
axes[1].set_title('Checkerboard')
axes[1].axis('off') 

parameter_text = (
    f"angle={cfg.angle}°    "
    f"translation_x={cfg.tx}    "
    f"translation_y={cfg.ty}\n"
    f"scale_x={cfg.sx}    "
    f"scale_y={cfg.sy}    "
    f"shear_x={cfg.hx}    "
    f"shear_y={cfg.hy}\n"
    f"k1={cfg.k1}    "
    f"k2={cfg.k2}    "
    f"k3={cfg.k3}    "
    f"p1={cfg.p1}    "
    f"p2={cfg.p2}"
)

fig.text(
    0.5,
    0.02,
    parameter_text,
    ha="center",
    va="bottom",
    fontsize=9,
    family="monospace",
)

plt.tight_layout()
plt.show()