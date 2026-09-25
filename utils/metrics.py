import numpy as np
from sklearn.metrics import mutual_info_score

def image_mutual_information(image1, image2, bins=256):
    values1 = image1.ravel()
    values2 = image2.ravel()

    # valid = np.isfinite(values1) & np.isfinite(values2)
    # values1 = values1[valid]
    # values2 = values2[valid]

    joint_hist, _, _ = np.histogram2d(
        values1,
        values2,
        bins=bins,
    )

    return mutual_info_score(
        None,
        None,
        contingency=joint_hist,
    )