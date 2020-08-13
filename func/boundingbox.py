"""Finds the bounding box for D in x and y
   (The space in D with material)"""
import numpy as np

def boundingbox(D):
    D_box = np.nonzero(np.sum(D,axis=0))
    D_ybox = np.nonzero(np.sum(D, axis=1))
    """ Below example could make it more readable? """
    # Creates a region of interest as a slice tuple - easy to pass
    # RegOfInt = (slice(500, 1100), slice(750, 1250), slice(None))
    D = D[D_ybox[0][0]:D_ybox[0][-1]+1, D_box[0][0]:D_box[0][-1]+1,np.min(D_box[1]):np.max(D_box[1]+1)]
    return D