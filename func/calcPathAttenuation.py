import numpy as np
from func.Bresenham3D import Bresenham3D

def calcPathAttenuation(D, mu_dict, m_m, n_m, p_m, v_d, n_c, p_c, ray_energy, ray_points):
    """ Calculates the voxel rays from all entry points to exit points. 
    Use the voxel indencies in D to find the given material it hits. 
    Takes all the attenuation coefficients from mu_dict to make an array for each ray
    with a sum of all corresponding atteenuation coefficients in its path

    """
    mu_d = np.zeros([n_c, p_c, len(ray_energy)])
    for i in range(n_c):
        for j in range(p_c):
            # Finds all the points for a given ray through the density matrix
            points = np.asarray(Bresenham3D(0, ray_points[0][i], ray_points[1][j],  \
                                        m_m-1, ray_points[2][i], ray_points[3][j]))
            
            #Masks to make sure every index is withing the density matrix
            points_mask = points >= 0
            points_mask[points[:,0] > m_m-1] = False
            points_mask[points[:,1] > n_m-1] = False
            points_mask[points[:,2] > p_m-1] = False
        
            points_mask[points_mask[:,0]*1+points_mask[:,1]*1+points_mask[:,2]*1 !=3] = False
            # Reshapes the points back to a len(points)x3 array
            points = points[points_mask].reshape(np.sum(points_mask[:,0]*1),3)
            indicies = points[:,0] * n_m + points[:,1] * p_m + points[:,2]
            # Calculates the product of the absorption coeficient and the distance for every ray and energy
            mu_d [i,j,:] = np.sum(mu_dict[:,D[points[:,0],points[:,1], points[:,2]]] * v_d[i,j], axis =1)
    return mu_d