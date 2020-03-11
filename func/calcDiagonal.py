"""" 
Function that calculates the diagonals trough a given voxel of 
size p_x, p_y and p_z with angles alpha_x and alpha_z

"""
#%%
import numpy as np


def calcDiagonal(size_v, alpha, grid_c):

    alpha_x, alpha_z = alpha
    v_y, v_x, v_z = size_v
    #the x and z coordinates in each pixel
    v_delta_x = np.tan(alpha_x)*v_y
    v_delta_z = np.tan(alpha_z)*v_y


    delta_c = grid_c[0] - grid_c[1]
    count_start = np.arange(0, grid_c[1]*(delta_c+1), grid_c[1])
    count_end = np.cumsum(np.arange(grid_c[1], 1, -1))+grid_c[1]*delta_c
    counter = np.insert(count_end, 0, count_start)
    v_d = np.zeros(int(grid_c[0]*grid_c[1]+(grid_c[1]-grid_c[1]**2)/2))

    for i in range(delta_c):
        for j in range(grid_c[1]):
            ind = counter[i]+j
            v_d[ind] = np.sqrt(v_delta_x[i]**2 + v_delta_z[j]**2 + v_y**2)
    for i in range(delta_c, grid_c[0]):
        for j in range(i-delta_c, grid_c[1]):
            ind = counter[i]+j-i+delta_c
            v_d[ind] = np.sqrt(v_delta_x[i]**2 + v_delta_z[j]**2 + v_y**2)

    #Old way to find it
    # delta_x, delta_z = np.meshgrid(v_delta_z, v_delta_x)
    # v_d = np.sqrt(delta_x**2 = delta_z**2+ v_y**2)

    return v_d


# %%
