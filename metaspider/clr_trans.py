# The functions in the code are inspired by 'skbio.stats.composition' library.
# https://github.com/biocore/scikit-bio/blob/master/skbio/stats/composition.py

import numpy as np
import pandas as pd

def closure(mat):
    mat = np.atleast_2d(mat)
    if np.any(mat < 0):
        raise ValueError("Cannot have negative proportions")
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    if np.all(mat == 0, axis=1).sum() > 0:
        raise ValueError("Input matrix cannot have rows with all zeros")
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat.squeeze()

def clr(mat):
    mat = closure(mat)
    lmat = np.log(mat)
    gm = lmat.mean(axis=-1, keepdims=True)
    return (lmat - gm).squeeze()


def clr_df(data_all):

    mat = data_all.values
    clr_mat = clr(mat)
    clr_df = pd.DataFrame(clr_mat, index=data_all.index, columns=data_all.columns)

    return clr_df