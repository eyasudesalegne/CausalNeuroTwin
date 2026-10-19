"""Small generated geometry only; no old-release imports or participant configuration."""

import nibabel as nib
import numpy as np

from causalneurotwin.validation.anatomy.geometry import Volume

REGULAR = np.array(
    [[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)]], float
)


def volume(labels, affine=None):
    affine = np.eye(4) if affine is None else affine
    image = nib.Nifti1Image(np.asarray(labels, dtype=np.float32), affine)
    image.header.set_xyzt_units("mm")
    return Volume(image, affine, "synthetic", "native", {})
