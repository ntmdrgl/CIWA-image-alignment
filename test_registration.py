import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.config import load_config
from utils.camera_intrinsics import get_camera_matrix_from_camera
from utils.transformations import get_affine_matrix_from_cfg, get_distortion_coefficients_from_cfg

cfg = load_config()

src_image = np.load('visible.npy')
src_H, src_W = src_image.shape[:2]

tgt_image = np.load('thermal.npy')
tgt_H, tgt_W = tgt_image.shape[:2]

checkboard = cv2.imread("checkerboard_small.png")

src_camera = cfg.visible_camera
tgt_camera = cfg.thermal_camera

# resize source image to target dimensions
src_image =  cv2.resize(src_image, (tgt_W, tgt_H), interpolation=cv2.INTER_AREA)

print("source:", src_image.shape, np.min(src_image), np.max(src_image))
print("target:", tgt_image.shape, np.min(tgt_image), np.max(tgt_image), "\n")

camera_matrix = get_camera_matrix_from_camera(tgt_camera)
affine_matrix = get_affine_matrix_from_cfg(cfg, tgt_camera)
dist_coeffs = get_distortion_coefficients_from_cfg(cfg)

print("Camera matrix:") 
print(camera_matrix, "\n")

print("Affine matrix:")
print(affine_matrix, "\n")

print("Distortion Coefficients:")
print(dist_coeffs, "\n")

undistorted_image = cv2.undistort(
    src_image, 
    camera_matrix, 
    dist_coeffs
)

transformed_image = cv2.warpAffine(
    undistorted_image,
    affine_matrix[:2, :], # OpenCV warpAffine expects a 2x3 matrix
    (tgt_W, tgt_H)
)

# apply transformations to checkboard for side-by-side
undistorted_checkerboard = cv2.undistort(checkboard, camera_matrix, dist_coeffs)
transformed_checkerboard = cv2.warpAffine(undistorted_checkerboard, affine_matrix[:2, :], (tgt_W, tgt_H))

def normalize_to_uint8(img):
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)

transformed_norm = normalize_to_uint8(transformed_image)
target_norm = normalize_to_uint8(tgt_image)
target_color = cv2.applyColorMap(target_norm, cv2.COLORMAP_INFERNO)
target_color = cv2.cvtColor(target_color, cv2.COLOR_BGR2RGB)

alpha = 0.6

overlay = cv2.addWeighted(
    transformed_norm, 1 - alpha,
    target_color, alpha,
    0
)

# Show image transformation + checkerboard side-by-side
fig, axes = plt.subplots(1, 4, figsize=(16, 6.5))

axes[0].imshow(transformed_checkerboard)
axes[0].set_title('Checkerboard')
axes[0].axis('off') 

axes[1].imshow(transformed_image)
axes[1].set_title('Transformed Source Image')
axes[1].axis('off')

axes[2].imshow(overlay)
axes[2].set_title('Overlay')
axes[2].axis('off')

axes[3].imshow(target_color)
axes[3].set_title('Target Image')
axes[3].axis('off')

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



# cv2.imshow("visible", normalize_to_uint8(src_image))
# cv2.imshow("thermal", normalize_to_uint8(tgt_image))
# cv2.waitKey(0)
# cv2.destroyAllWindows()