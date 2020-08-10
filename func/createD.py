#%%

import numpy as np
import matplotlib.pyplot as plt

def drawBox(D,position_0, position_1, material, mat_name):
    """ Draws a 3D box in D from position_0 to position_1 of the given material """

    mat_val = [s for s in enumerate(mat_name) if material in s][0][0]
    D[position_0[0]:position_1[0], position_0[1]:position_1[1], position_0[2]:position_1[2]] = mat_val

    return D


def drawSphere(D, radius, position, material, mat_name):
    """ from: https://stackoverflow.com/questions/46626267/how-to-generate-a-sphere-in-3d-numpy-array/46626448"""
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    shape = D.shape
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (np.abs(x_i / semisize) ** 2)
    # the inner part of the sphere will have distance below 1
    arr = arr <= 1.0
    # this will save a sphere in a boolean array
    mat_val = [s for s in enumerate(mat_name) if material in s][0][0]
    D[arr] = mat_val
    return D

def generateD(size, kind, mat1, mat2, mat_name):
    """ Generates a pre defined version of D with two types of material
    size is a tuple, rest is strings
    """
    y, x, z = size
    D  = np.zeros(size)
    if kind == 'boxWithBox':
        drawBox(D, (y//2, x//4, z//4), (y, x//4*3, z//4*3), mat1, mat_name)
        drawBox(D, (y//16*7, x//16*7, z//16*7), (y//16*9, x//16*9, z//16*9), mat2, mat_name)
    elif kind == "boxWithSphere":
        drawBox(D, (y//2, x//4, z//4), (y, x//4*3, z//4*3), mat1, mat_name)
        drawSphere(D, y//16, (y//16*11, x//2, z//2), mat2, mat_name)
    elif kind == "boxWithSmallSphere":
        drawBox(D, (y//2, x//4, z//4), (y, x//4*3, z//4*3), mat1, mat_name)
        drawSphere(D, y//32, (y//32*27, x//2, z//2), mat2, mat_name)
    elif kind == "Sphere":
        mat_val = [s for s in enumerate(mat_name) if mat2 in s][0][0]
        D[:,:,:] = mat_val
        drawSphere(D, y//2, (y//2, x//2, z//2), mat1, mat_name)
    elif kind == "oneMat":
        mat_val = [s for s in enumerate(mat_name) if mat1 in s][0][0]
        D[:,:,:] = mat_val
    else:
        print('ERROR: That kind of D does not exits')
    return D
# %%
