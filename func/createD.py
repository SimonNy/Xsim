#%%

import numpy as np
import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as axes3d

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

def drawCylinder(D, radius, position, material, mat_name):
    """ based on sphere """

    # shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    shape = D.shape
    semisizes = (radius,) * 3
    point = position
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
    for i in range(shape[2]):
        arr[:,:,i] = arr[:,:,point[2]]
    # this will save a sphere in a boolean array
    mat_val = [s for s in enumerate(mat_name) if material in s][0][0]
    D[arr] = mat_val
    return D


def generateD(size, kind, mat_name, mat1, mat2, mat3 = "air"):
    """ Generates a pre defined version of D with two types of material
    size is a tuple, rest is strings
    """
    y, x, z = size
    D  = np.zeros(size)
    if kind == 'boxWithBox':
        drawBox(D, (y//2, x//4, z//4), (y, x//4*3, z//4*3), mat1, mat_name)
        drawBox(D, (y//16*7, x//16*7, z//16*7), (y//16*9, x//16*9, z//16*9), mat2, mat_name)
    elif kind == "boardWithMarks":
        drawBox(D, (0, 0, 0), (y, x, z), mat1, mat_name)
        # Draw the different marks as 1/10 of object thickness

        # draw two vertical lines
        drawBox(D, (y-y//10, x//10*4, z//10*4),
                (y, x//10*5, z//10*4+1), mat2, mat_name)
        drawBox(D, (y-y//10, x//10*4, z//10*4+4),
                (y, x//10*5, z//10*4+5), mat2, mat_name)
        # draw horizontal lines between them
        drawBox(D, (y-y//10, x//10*4-1, z//10*4),
                (y, x//10*4, z//10*4+5), mat2, mat_name)
        drawBox(D, (y-y//10, x//10*4+2, z//10*4),
                (y, x//10*4+3, z//10*4+5), mat2, mat_name)
        drawBox(D, (y-y//10, x//10*4+6, z//10*4),
                (y, x//10*4+7, z//10*4+5), mat2, mat_name)
        drawBox(D, (y-y//10, x//10*5, z//10*4),
                (y, x//10*5+1, z//10*4+5), mat2, mat_name)

        # Draw an L
        drawBox(D, (y-y//10, x//10*6, z//10*4),
                (y, x//10*7, z//10*4+2), mat2, mat_name)
        drawBox(D, (y-y//10, x//10*7, z//10*4),
                (y, x//10*7+2, z//10*5), mat2, mat_name)

        # Draw boxes
        drawBox(D, (y-y//10, x//10*6, z//10*7),
                (y, x//10*6+2, z//10*7+2), mat2, mat_name)
        drawBox(D, (y-y//10, x//10*6, z//10*7+4),
                (y, x//10*6+2, z//10*7+6), mat2, mat_name)
        drawBox(D, (y-y//10, x//10*7, z//10*7),
                (y, x//10*7+3, z//10*7+3), mat2, mat_name)
        drawBox(D, (y-y//10, x//10*7, z//10*7+5),
                (y, x//10*7+3, z//10*7+8), mat2, mat_name)
        drawBox(D, (y-y//10, x//10*8, z//10*7),
                (y, x//10*8+4, z//10*7+4), mat2, mat_name)
        drawBox(D, (y-y//10, x//10*8, z//10*7+7),
                (y, x//10*8+4, z//10*7+11), mat2, mat_name)
        drawBox(D, (y-y//10, x//10*9, z//10*7),
                (y, x//10*9+5, z//10*7+5), mat2, mat_name)
        drawBox(D, (y-y//10, x//10*9, z//10*7+9),
                (y, x//10*9+5, z//10*7+14), mat2, mat_name)
        # draw two horizontal lines
        drawBox(D, (y-y//10, x//10, z//10*2),
                (y, x//10+3, z//10*5), mat2, mat_name)
        drawBox(D, (y-y//10, x//10+5, z//10*2),
                (y, x//10+8, z//10*5), mat2, mat_name)
        # draw horizontal lines
        drawBox(D, (y-y//10, x//10, z//10*7),
                (y, x//10*3, z//10*7+3), mat2, mat_name)
        drawBox(D, (y-y//10, x//10, z//10*7+5),
                (y, x//10*3, z//10*7+8), mat2, mat_name)
        # draw spheres
        drawSphere(D, (y//20), (y//10+1, x//10*4, z//10*7), mat2, mat_name)
        drawSphere(D, (y//20), (y//10+1, x//10*5, z//10*8), mat2, mat_name)
        drawSphere(D, (y//10), (y//50, x//10*8, z//10*2), mat2, mat_name)

    elif kind == "boxWithSphere":
        drawBox(D, (y//2, x//4, z//4), (y, x//4*3, z//4*3), mat1, mat_name)
        drawSphere(D, y//16, (y//16*11, x//2, z//2), mat2, mat_name)
    elif kind == "boxWithSmallSphere":
        drawBox(D, (y//2, x//4, z//4), (y, x//4*3, z//4*3), mat1, mat_name)
        drawSphere(D, y//32, (y//4*3, x//2, z//2), mat2, mat_name)
    elif kind == "twoBoxesWithSmallSphere":
        drawBox(D, (y//2, 0, 0), (y, x//2-1, z//2-1), mat1, mat_name)
        drawBox(D, (y//2, x//2+1, z//2+1), (y, x, z), mat2, mat_name)
        drawSphere(D, y//32, (y//4*3, x//4, z//4), mat3, mat_name)
        drawSphere(D, y//32, (y//4*3, x//4*3, z//4*3), mat3, mat_name)
    elif kind == "boxWithCylinder":
        drawBox(D,(0,0,0),(x, y, z), mat1, mat_name )
        drawCylinder(D,x//3,(x//2, y//2, z//2), mat2, mat_name)
    elif kind == "sphere":
        mat_val = [s for s in enumerate(mat_name) if mat2 in s][0][0]
        D[:,:,:] = mat_val
        drawSphere(D, y//2, (y//2, x//2, z//2), mat1, mat_name)
    elif kind == "sphereWithSmallSphere":
        drawSphere(D, y//2-1, (y//2, x//2, z//2), mat1, mat_name)
        drawSphere(D, y//32, (y//2, x//2, z//2), mat2, mat_name)
    elif kind == "oneMat":
        mat_val = [s for s in enumerate(mat_name) if mat1 in s][0][0]
        D[:,:,:] = mat_val
    else:
        print('ERROR: That kind of D does not exits')
    return D
# %%
