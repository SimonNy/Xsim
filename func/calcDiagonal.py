"""" 
Function that calculates the diagonals trough a given voxel of 
size p_x, p_y and p_z with angles alpha_x and alpha_z

"""
#%%
import numpy as np


def calcDiagonal(size_p, alpha):

    alpha_x, alpha_z = alpha
    p_y, p_x, p_z = size_p
    #the x and z coordinates in each pixel
    p_delta_x = np.tan(alpha_x)*p_y
    p_delta_z = np.tan(alpha_z)*p_y

    delta_x, delta_z = np.meshgrid(p_delta_z, p_delta_x)
    p_d = np.sqrt(delta_x**2 + delta_z**2 + p_y**2)
    return p_d


# %%
