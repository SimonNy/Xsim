import numpy as np

def projectDtoFOV(grid_fov, grid_m, D, move):
    """ Translates D to specified place in FOV by making a reference object 
        grid_fov is the size of the field of view
        grid_m is the size of the object
        D is the object
        move is the movement in x and z
        returns an array containing D in fov
    """
    # Projects D into the FOV at the desired point with the desired size
    min_axis = np.min([grid_fov, grid_m], axis=0)
    max_axis = np.max([grid_fov, grid_m], axis=0)

    #length and width of broadcasting b_l #Broadcasting limit
    # Set to be either the minimum axis size or as the difference between the minimum and maximum axis
    b_l = np.min([min_axis[1:], min_axis[1:]//2 -
                  (np.abs(move)-max_axis[1:]//2)], axis=0)
    # Makes sure b_l isn't negative: This happens when D is out of FOV scope
    b_l = np.max([b_l, (0, 0)], axis=0)
    #Finds the difference in grid size between object and FOV
    delta_fov = np.subtract(grid_fov, grid_m)//2

    # Finds the indencies used to place D in FOV in center will a given voxel movement
    # Maximum is taken to adjust for negative diffrences.
    delta_max = np.array(
        [np.max([delta_fov[1], 0]), np.max([delta_fov[2], 0])])
    delta_min = np.array([np.abs(np.min([delta_fov[1], 0])),
                          np.abs(np.min([delta_fov[2], 0]))])
    #Instead of defining a lot of if statements fov_max and m_max define which parts should be executed
    fov_max = np.zeros(2)
    D_max = np.zeros(2)
    fov_max[delta_max != 0] = delta_max[delta_max != 0] / \
        delta_max[delta_max != 0]
    D_max[delta_min != 0] = delta_min[delta_min != 0]/delta_min[delta_min != 0]

    # Check if move is positive or negative in x
    if move[0] >= 0:
        ind_x = np.array([delta_max[0]+fov_max[0]*move[0], delta_max[0]+fov_max[0]*move[0]+b_l[0],
                          delta_min[0]+D_max[0]*move[0], delta_min[0]+D_max[0]*move[0]+b_l[0]]).astype(int)
    else:
        # If negative sets the indencies accordingly. Dependent on which is greatest FOV or D
        ind_fov = np.max([delta_max[0]+fov_max[0]*move[0], 0])
        ind_m = np.max([delta_min[0]+D_max[0]*move[0], 0])
        ind_x = np.array([(min_axis[1]-b_l[0])*D_max[0]+ind_fov*fov_max[0], min_axis[1]*D_max[0]+(ind_fov+b_l[0])*fov_max[0],
                          (min_axis[1]-b_l[0])*fov_max[0]+ind_m*D_max[0], min_axis[1]*fov_max[0]+(ind_m+b_l[0])*D_max[0]]).astype(int)
    #Check if move is posive or negative in z
    if move[1] >= 0:
        ind_z = np.array([delta_max[1]+fov_max[1]*move[1], delta_max[1]+fov_max[1]*move[1]+b_l[1],
                          delta_min[1]+D_max[1]*move[1], delta_min[1]+D_max[1]*move[1]+b_l[1]]).astype(int)
    else:
        ind_fov = np.max([delta_max[1]+fov_max[1]*move[1], 0])
        ind_m = np.max([delta_min[1]+D_max[1]*move[1], 0])
        ind_z = np.array([(min_axis[2]-b_l[1])*D_max[1]+ind_fov*fov_max[1], min_axis[1]*D_max[1]+(ind_fov+b_l[1])*fov_max[1],
                          (min_axis[2]-b_l[1])*fov_max[1]+ind_m*D_max[1], min_axis[1]*fov_max[1]+(ind_m+b_l[1])*D_max[1]]).astype(int)

    #Defines an empty field of view
    ref_fov = np.zeros(grid_fov)
    # Broadcast D to FOV
    ref_fov[0:D.shape[0], ind_x[0]:ind_x[1], ind_z[0]:ind_z[1]]\
        = D[:, ind_x[2]:ind_x[3], ind_z[2]:ind_z[3]]

    # print(f'min axis = {min_axis}')
    # print(f'max axis = {max_axis}')
    # print(f'b_l = {b_l}')
    # print(f'move = {move}')
    # print(f'grid_fov = {grid_fov}')
    # print(f'grid_m = {grid_m}')
    # print(f'delta_fov = {delta_fov}')
    # plt.imshow(np.sum(ref_fov, axis=0))

    return ref_fov.astype(int)
