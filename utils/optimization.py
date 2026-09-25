import numpy as np
import cv2
from mealpy import FloatVar, PSO

from utils.transformations import get_affine_matrix
from utils.metrics import image_mutual_information

def run_optimization(
    objective_function,
    lower_bounds, 
    upper_bounds, 
    parameter_names,
    epoch=200,
    pop_size=50,
    c1=2.0,
    c2=2.0,
    w=0.8,
    seed=42
):
    problem = {
        "bounds": FloatVar(
            lb=lower_bounds,
            ub=upper_bounds,
            name="registration_parameters",
        ),
        "obj_func": objective_function,
        "minmax": "max",
    }

    optimizer = PSO.OriginalPSO(
        epoch=epoch,
        pop_size=pop_size,
        c1=c1,
        c2=c2,
        w=w,
    )

    best = optimizer.solve(
        problem,
        seed=seed,
    )

    best_parameters = dict(
        zip(parameter_names, best.solution)
    )

    return best_parameters, best.target.fitness


def create_objective_function(
    src_image,
    tgt_image,
    camera_matrix,
    bins=256,
):
    tgt_H, tgt_W = tgt_image.shape[:2]

    def objective_function(solution):
        (
            angle,
            hx, hy,
            sx, sy,
            tx, ty,
            k1, k2, k3,
            p1, p2,
        ) = solution

        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]

        affine_matrix = get_affine_matrix(
            angle=angle,
            hx=hx,
            hy=hy,
            sx=sx,
            sy=sy,
            tx=tx,
            ty=ty,
            cx=cx,
            cy=cy
        )

        distortion_coefficients = np.array(
            [k1, k2, p1, p2, k3],
            dtype=np.float64,
        )

        undistorted_image = cv2.undistort(
            src_image,
            camera_matrix,
            distortion_coefficients,
        )

        transformed_image = cv2.warpAffine(
            undistorted_image,
            affine_matrix[:2, :],
            (tgt_W, tgt_H),
        )

        return image_mutual_information(
            transformed_image,
            tgt_image,
            bins=bins,
        )

    return objective_function

def create_dataset_objective_function(
    src_images,
    tgt_images,
    camera_matrix,
    bins=64,
):
    """
    Creates an objective function that evaluates one shared registration
    parameter set across an entire dataset.
    """

    if len(src_images) != len(tgt_images):
        raise ValueError(
            "src_images and tgt_images must contain the same "
            f"number of images, but received {len(src_images)} "
            f"and {len(tgt_images)}."
        )

    if len(src_images) == 0:
        raise ValueError("The dataset cannot be empty.")

    # All images should use consistent dimensions when one shared
    # parameter set is applied.
    src_shape = src_images[0].shape[:2]
    tgt_shape = tgt_images[0].shape[:2]

    for index, (src_image, tgt_image) in enumerate(
        zip(src_images, tgt_images)
    ):
        if src_image.shape[:2] != src_shape:
            raise ValueError(
                f"Source image {index} has shape "
                f"{src_image.shape[:2]}, expected {src_shape}."
            )

        if tgt_image.shape[:2] != tgt_shape:
            raise ValueError(
                f"Target image {index} has shape "
                f"{tgt_image.shape[:2]}, expected {tgt_shape}."
            )

    src_H, src_W = src_shape
    tgt_H, tgt_W = tgt_shape

    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    def objective_function(solution):
        (
            angle,
            hx, hy,
            sx, sy,
            tx, ty,
            k1, k2, k3,
            p1, p2,
        ) = solution

        affine_matrix = get_affine_matrix(
            angle=angle,
            hx=hx,
            hy=hy,
            sx=sx,
            sy=sy,
            tx=tx,
            ty=ty,
            cx=cx,
            cy=cy,
        )

        # OpenCV order: k1, k2, p1, p2, k3
        distortion_coefficients = np.array(
            [k1, k2, p1, p2, k3],
            dtype=np.float64,
        )

        # Calculate the undistortion maps once for this candidate
        # solution, then reuse them for every source image.
        map_x, map_y = cv2.initUndistortRectifyMap(
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coefficients,
            R=None,
            newCameraMatrix=camera_matrix,
            size=(src_W, src_H),
            m1type=cv2.CV_32FC1,
        )

        scores = []

        for src_image, tgt_image in zip(
            src_images,
            tgt_images,
        ):
            undistorted_image = cv2.remap(
                src_image,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            transformed_image = cv2.warpAffine(
                undistorted_image,
                affine_matrix[:2, :],
                (tgt_W, tgt_H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            score = image_mutual_information(
                transformed_image,
                tgt_image,
                bins=bins,
            )

            scores.append(score)

        # One fitness value representing the whole dataset.
        return float(np.mean(scores))

    return objective_function