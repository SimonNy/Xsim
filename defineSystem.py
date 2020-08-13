import numpy as np
""" User-defined settings for the system """
# distance from point souce to the bottom of the material
d1 = 2
# distance from the bottom of the material to the CCD array
d2 = 1
#Information about the material size _m is _material
size_v_D = (1e-3, 1e-3, 1e-3)  # volume of voxel in material

#Information about the size of the camera, _c is _camera
# physical size of one pixel
size_ccd_pixel = (1e-3, 1e-3)

#number of ccds in the camera as [n x p]
#If non symmetric, first input should be largest
grid_ccd = (500, 500)
#The "resolution" of each CCD. Makes more rays hit the same CCD
ray_ccd_ratio = 1

#Number of Photons in a single pixel direction
photons_frame = 5000
# Energies in the sampled spectrum
Emin = 10
Emax = 80
Estep = 1

# The total of Image frames
N_frames = 1
# The amount of iterations made to create one frame.
# Generating N_sub frames for every frame. Can be seen as integration time
N_sub = 1

# Movement in a single subframe in x and z dire
move_step = (0, 0)
# The start position of the object, 0 is centered
# move_start = (0, 0)
# Makes the total movement symmetrical across the center
move_start = np.multiply(int(-N_frames*N_sub/2), move_step)