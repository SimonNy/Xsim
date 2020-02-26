"""
Calculates the field of view. With every trace through the material


"""
#%%
import numpy as np
from func.findIO import findIO
from func.calcDiagonal import calcDiagonal
from func.Bresenham3D import Bresenham3D


def fieldOfView(mat_shape, ccd_shape, voxel_size, d1, mat_size, alpha):
    alpha_x, alpha_z = alpha
    m_m, n_m, p_m = mat_shape
    n_c, p_c = ccd_shape
    v_y, v_x, v_z = voxel_size
    h_m, l_m, w_m = mat_size


    # fov = np.zeros([m_m, n_m, p_m])

    # Finds the x and z entry coordinates and x and z exit coordinates for a given ray
    ray_points = findIO(m_m, v_y, v_x, v_z, d1, h_m, l_m, w_m,
                        alpha_x, alpha_z)
    ray_index_list = []
    for i in range(n_c):
            for j in range(p_c):
                # Finds all the points for a given ray through the density matrix
                points = np.asarray(Bresenham3D(0, ray_points[0][i], ray_points[1][j],
                                                m_m-1, ray_points[2][i], ray_points[3][j]))

                #Masks to make sure every index is withing the density matrix
                points_mask = points >= 0
                points_mask[points[:, 0] > m_m-1] = False
                points_mask[points[:, 1] > n_m-1] = False
                points_mask[points[:, 2] > p_m-1] = False

                points_mask[points_mask[:, 0]*1+points_mask[:, 1]
                            * 1+points_mask[:, 2]*1 != 3] = False
                # Reshapes the points back to a len(points)x3 array
                points = points[points_mask].reshape(
                    np.sum(points_mask[:, 0]*1), 3)
                #Adds all indencies to a list
                ray_index_list.append([points[:, 0], points[:, 1], points[:, 2]])

    return ray_index_list
# %%

# for i in range(n_c):
#     for j in range(p_c):
#         I[i, j] = np.sum(fov[tuple(a[i*n_c+j])])
# ray_points

# %%
