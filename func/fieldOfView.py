"""
Calculates the field of view. With every trace through the material


"""
#%%
import numpy as np
from func.findIO import findIO
from func.calcDiagonal import calcDiagonal
from func.Bresenham3D import Bresenham3D


def fieldOfView(grid_m, grid_c, voxel_size, d1, fov_size, alpha):
    """ 
    Returns a vector of x and z indencies relating every ray from 
    the pointsource to the field of view. Describes the journey of rays through the FOV
    Inputs: grid_m - the shape of the object tensor D
            grid_c - the shape of the camera object
            voxel_size - the height, width and length of one voxel
            d1 - the distance from the points source to the buttom of the object
            fov_size - the physical size of the Field of View
            alpha - the angles from the points source to all points in the camera object
    """
    # Finds the x and z entry coordinates and x and z exit coordinates for a given ray
    ray_points = findIO(voxel_size, d1, fov_size, alpha)
    
    # The necessary points to evaluate
    # Counter is made to evaluate the loop properly. 
    # Only calculate the upper triangle in the FOV. 1/8 of the whole camera area
    delta_c = grid_c[0] - grid_c[1]
    count_start = np.arange(0, grid_c[1]*(delta_c+1), grid_c[1])
    count_end = np.cumsum(np.arange(grid_c[1], 1, -1))+grid_c[1]*delta_c
    counter = np.insert(count_end, 0, count_start)

    # Calculates size by number of iterations n*m+(m+m**2)/2
    ray_list_size = int(grid_c[0]*grid_c[1]+(grid_c[1]-grid_c[1]**2)/2)  
    ray_index_list = np.zeros([2, ray_list_size, grid_m[0]])
    for i in range(delta_c):
        for j in range(grid_c[1]):
            ind = counter[i]+j
            # Finds all the points for a given ray through the the field of view
            points = np.asarray(Bresenham3D(0, ray_points[0][i], ray_points[1][j],
                                            grid_m[0]-1, ray_points[2][i], ray_points[3][j]))

            """ The stuff below is unnecessary when FOV is defined from the camera size """
            #Masks to make sure every index is within the the field of view
            # points_mask = points >= 0
            # points_mask[points[:, 0] > grid_m[0]-1] = False
            # points_mask[points[:, 1] > grid_m[1]-1] = False
            # points_mask[points[:, 2] > grid_m[2]-1] = False

            # points_mask[points_mask[:, 0]*1+points_mask[:, 1]
            #             * 1+points_mask[:, 2]*1 != 3] = False
            # # Reshapes the points back to a len(points)x3 array
            # points = points[points_mask].reshape(np.sum(points_mask[:, 0]*1), 3)
            #Adds all indencies to a list
            ray_index_list[:, ind, :] = [points[:, 1], points[:, 2]]
    for i in range(delta_c, grid_c[0]):
        for j in range(i-delta_c, grid_c[1]):
            ind = counter[i]+j-i+delta_c
            points = np.asarray(Bresenham3D(0, ray_points[0][i], ray_points[1][j],
                                            grid_m[0]-1, ray_points[2][i], ray_points[3][j]))
            #Adds all indencies to a list
            ray_index_list[:, ind, :] = [points[:, 1], points[:, 2]]

    return ray_index_list.astype(int)

# %%
