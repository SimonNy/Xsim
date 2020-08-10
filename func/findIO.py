
#%%
import numpy as np

def findIO(voxel_size, d1, size_fov, alpha):
    """
    Return(indencies x 4, x_entry, z_entry, x_exit, z_exit) the input and out of all rays coming from a point source and hitting 
    a tensor of the given size and shape
    Inputs: voxel_size: (float x 3, y, x and z)  The physical size of all voxels in the object D
            d1: (float) the distance from the point source to the buttom of the object
            size_fov: (int val x 3, y, x and z) the shape of the fov
            alpha: (numpy tuple, for x and z) The angle for all rays from the points source to the camera object
    """
    
    # Finds the distance from the starting point of the ray to the entry point in the density matrix
    x_entry = (d1 - size_fov[1]) * np.tan(alpha[0])
    z_entry = (d1 - size_fov[2]) * np.tan(alpha[1])
    """ Should it not be size_fov[0] for both?! """
    
    # Finds the distance from the starting point of the ray to the exit point in the density matrix
    x_exit = d1 * np.tan(alpha[0])
    z_exit = d1 * np.tan(alpha[1])
    
    # Changes the reference system to be in the top left corner of the density matrix
    x_entry_Dref = x_entry + size_fov[1]/2
    z_entry_Dref = z_entry + size_fov[2]/2

    if x_entry_Dref[0] < 0 or z_entry_Dref[0] < 0:
        print("ERROR: Field of view and CCD size does not match \n")
        print("Make FOV smaller or CCD larger")

    x_exit_Dref = x_exit + size_fov[1]/2
    z_exit_Dref = z_exit + size_fov[2]/2
    
    #Finds the closest index relating to the entry point
    x_entry_index = np.round(x_entry_Dref / voxel_size[1]).astype(int)
    z_entry_index = np.round(z_entry_Dref / voxel_size[2]).astype(int)
    
    x_exit_index = np.round(x_exit_Dref / voxel_size[1]).astype(int)
    z_exit_index = np.round(z_exit_Dref / voxel_size[2]).astype(int)
    
    #makes an array of all relevant points, y starting point is 0 and end points is m_m -1
    ray_points  = (x_entry_index,  z_entry_index, x_exit_index, z_exit_index)
    return ray_points
#%%
