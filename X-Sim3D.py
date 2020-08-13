"""
Created on Mon Jan 13 12:59:19 2020


@author: simonnyrup
X-ray simulator. Simulates the 2D image generated with x-rays 
emitted from a point source and penerating a material represented by an 3D array.

Generally the coordinate system is right-handed with center in x-ray point source. 
hat{z} points toward the camera(CCD object)
hat{x} (positive right) and hat{y}(inwards) are orthogonal and forms a plane with
hat{z} as the normal vector
In a 3D-array the direction of the grid is [hat{z}, hat{x}, hat{y}]
inputs:
    d1: The distance from point source to the bottom of the object.
    d2: The distance from the bottom of the material to the camera (CCD)
    
    D: density matrix of size [grid_D[0] X grid_D[1] x grid_D[2]] 
        - defined by the loaded target object
    grid_D: The amount of voxel in the z, x and y direction
    size_v_D: The physical size of a single voxel in the Density object
             (height, length, width)
    size_ccd_pixel = The physical size of one pixel in the ccd array
                    (lenight, width)

    
    grid_ccd: The resolution of the capturing device
            (n, p)
    ray_ccd_ratio: The amount of different angled rays hitting one ccd. 
                    1 the camera object and CCD same size if larger one pixel in
                    the image consists of 2 pixels.  
    photons_frame: the amount of photons emitted from the source in direction of
                   any pixel
    Emin:Estep:Emax: The energy range for sampling the source
    N_frames: The total amount of image frames generated
    N_sub: The amount of subframes added together to form one frame
    move_step: (move_x, move_y) the voxel movement between subframes
    move_start: (start_x, start_y) the starting position for the first subframe.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2 as cv
from datetime import datetime

from func.produceXray import produceXray
from func.calcDiagonal import calcDiagonal
from func.generateMuDict import generateMuDict
from func.normalize3D import normalize3D
from func.normalize2D import normalize2D
from func.fieldOfView import fieldOfView
from func.projectDtoFOV import projectDtoFOV

from func.createDirectories import createDirectories

import loadObject
# Imports all user defined constants from defineSystem.py
from defineSystem import d1, d2, size_v_D, size_ccd_pixel, grid_ccd, \
                ray_ccd_ratio, photons_frame, Emin, Emax, Estep, \
                N_frames, N_sub, move_step, move_start

saveImgs = True
makeMovie = True

# loads the object defineed in the loadObject file
D = loadObject.D
filename = loadObject.filename

#%%
""" Calculates all relevant constant in the system by the user-defined values """
# physical size of the whole camera object
# size_c: the real size of image capturing device. Tuple with two inputs
size_c = np.multiply(size_ccd_pixel, grid_ccd)
# The finer grid for the camera calculations.
grid_c = np.multiply(grid_ccd, ray_ccd_ratio)

#Information about the grid definition [m x n x p] # The should all match in size? or
grid_D = np.shape(D)
size_D = np.multiply(grid_D, size_v_D)

"""
Information of FOV size and grid. Length and width needs to be bigger than
material. For 1 to 1 transposability the amount of voxel per length needs 
to be the same as in the material.
"""
size_fov = (size_D[0], (size_c[0]*d1)/(d1+d2), (size_c[1]*d1)/(d1+d2))
#Increases the size of FOV by a maximum of 1. To make pretty numbers and 
# guarantee that the voxel volumes of D and fov match
size_fov = size_fov + (size_v_D-np.mod(size_fov,size_v_D))

""" 
Define FOV grid so each voxel is same size as in D
might be a rounding error
"""
grid_fov = np.round(np.divide(size_fov, size_v_D)).astype(int)
size_v_fov = np.divide(size_fov, grid_fov)


if makeMovie == False:
    N_frames = 1
    N_sub = 1
    move_step = (0,0)
    move_start = (0, 0)

#%% Creates Directories for all types of files and saves constants
""" Set input and output directories """
folder = "results/"

if saveImgs == True:
    """ Save relevant constants """
    const_names = ['d1', 'd2', 'size_D', 'grid_D','size_c', 'grid_ccd', 
                    'ray_ccd_ratio', 'move_step', 'move_start',
                    'N_frames', 'N_sub', 'photons_frame']
    const_vals = [d1, d2, size_D, grid_D, size_c, grid_ccd, 
                  ray_ccd_ratio, move_step, move_start, 
                  N_frames, N_sub, photons_frame]
    
    createDirectories(filename, folder, const_names, const_vals)

#%% Ray spectrum and intensities #The spectrum for the energies
from xraySpec.dataA import arr80

spec = np.asarray(arr80)
spec = spec[9:-1]

#For a given spectrum find the energies and the count of photons with each energy
ray_energy, ray_counts = produceXray(spec, N_points = photons_frame, 
                                        Emin = Emin, Emax = Emax, 
                                        Estep = Estep, makePlot=True)

#The Thresholdvalue(Amount of photons maximally counted) for a Single CCD
I_threshold = np.sum(ray_counts)*ray_ccd_ratio**2*N_sub

#%% Defines the material and its components
mu_dict, mat_names = generateMuDict(ray_energy)

#%% Calculates distances and angles 
# The total height of the system
h = d1 + d2 

# CCD reshaping to quarter size
grid_qc = (int(np.ceil(grid_c[0]/2)),int(np.ceil(grid_c[1]/2)))

#The distance to each center point of the CCD(centered in the middle)
# nice numbers if shape is half of grid
xc = (np.arange(grid_qc[0]) - grid_c[0]/2 + 0.5) * size_c[0] / (grid_c[0]) 
zc = (np.arange(grid_qc[1]) - grid_c[1]/2 + 0.5) * size_c[1] / (grid_c[1])   
# The angles between the center and the midpoint of a given input of the CCD 
alpha = (np.arctan(xc/h), np.arctan(zc/h))

"""Reshaped v_d to a vector and 1/4 of the original content"""
v_d = calcDiagonal(size_v_fov, alpha, grid_qc)
#%% Generates Field of View list. An list containing all indencies a ray goes 
# through in the field of view
# Only generates a Field of view list if one with those specifics dosen't ecist
import os
fov_name = f'{grid_fov[0]}x{grid_fov[1]}x{grid_fov[2]}_{grid_c[0]}x{grid_c[1]}_{size_v_fov[0]:.3f}x{size_v_fov[1]:.3f}x{size_v_fov[2]:.3f}_{size_fov[0]:.2f}x{size_fov[1]:.3f}x{size_fov[2]:.3f}_{size_c[0]:.3f}x{size_c[1]:.3f}_{d1}_{d2}'
if os.path.isfile('fov/'+fov_name+'.npy'):
    print (f"FOV File exist with name: {fov_name}")
    fov_list = np.load('fov/'+fov_name+'.npy')
else:
    t0 = time()
    print (f"FOV File with name: {fov_name} does not exist.\n Creates FOV list,")
    fov_list = fieldOfView(grid_fov, grid_qc, size_v_fov, d1, size_fov, alpha)
    np.save('fov/'+fov_name,fov_list)
    dt = time() - t0
    print(f'took {dt:02.2f} seconds\n')

#%% 
""" The main part of the loop which creates the frames"""
I_ccd = np.zeros([N_frames, grid_ccd[0], grid_ccd[1]])

mu_d = np.zeros([grid_c[0],grid_c[1], len(ray_energy)])
#The number of iterations in the loop
ind_len = int(grid_qc[0]*grid_qc[1]+(grid_qc[1]-grid_qc[1]**2)/2)

# Parameter to adjust for the unequal parts of FOV, in the reversed indencies
""" Maybe find a better expresions as it seems a little random factor to add """
rev_adjust = (grid_fov[1] - grid_fov[2])//2

y = np.arange(grid_fov[0])
x_points, z_points = fov_list[0, :, :], fov_list[1, :, :]

v_d_broad = np.broadcast_to(v_d, (grid_fov[0], ind_len))

# The necessary points to evaluate
# Counter is made to evaluate the loop properly.
# Only calculate the upper triangle in the FOV. 1/8 of the whole camera area
delta_c = grid_qc[0] - grid_qc[1]
count_start = np.arange(0, grid_qc[1]*(delta_c+1), grid_qc[1])
count_end = np.cumsum(np.arange(grid_qc[1], 1, -1))+grid_qc[1]*delta_c
counter = np.insert(count_end, 0, count_start)

move = move_start

for i in range(N_frames):
    t0 = time()
    I_photon = np.zeros(tuple(grid_c))
    for j in range(N_sub):
        t1 = time()
        #Finds photons from a new distribution. Needs to sample everytime?
        ray_energy, ray_counts = produceXray(spec, N_points=photons_frame, 
                                Emin=Emin, Emax=Emax, Estep=Estep)
        # Calculates all entry and exit points of the rays in the density matrix
        ref_fov = projectDtoFOV(grid_fov, grid_D, D, move)
        move = np.add(move, move_step)
    
        #Loops over the part where the indencies cannot be reversed 
        for i2 in range(delta_c):
            for j2 in range(grid_qc[1]):
                ind = counter[i2]+j2
                x = x_points[ind]
                z = z_points[ind]
                """ Upper left quadrant"""
                mu_d[i2, j2, :] = np.sum(mu_dict[:, 
                    ref_fov[y, x, z]]*v_d[ind], axis=1)
                """ upper right quadrant"""
                mu_d[i2, grid_c[1]-1-j2, :] = np.sum(mu_dict[:,
                    ref_fov[y, x, grid_fov[2]-1-z]]*v_d[ind], axis=1)
                """ lower left quadrant"""
                mu_d[grid_c[0]-1-i2, j2,:] = np.sum(mu_dict[:, 
                    ref_fov[y, grid_fov[1] - 1 - x, z]] * v_d[ind], axis=1)
                """ lower right quadrant"""
                mu_d[grid_c[0]-1-i2, grid_c[1]-1-j2, :] = np.sum(mu_dict[:, 
                    ref_fov[y, grid_fov[1] - 1 - x, grid_fov[2] - 1 - z]] 
                    * v_d[ind], axis=1)
        # Loops over the part where the indencies can be reversed
        for i2 in range(delta_c, grid_qc[0]):
            for j2 in range(i2-delta_c, grid_qc[1]):
                ind = counter[i2]+j2-i2+delta_c
                x = x_points[ind]
                z = z_points[ind]
                """ Upper left quadrant"""
                mu_d[i2, j2, :] = np.sum(
                    mu_dict[:, ref_fov[y,  x,  z]] * v_d[ind], axis=1)
                """ upper right quadrant"""
                mu_d[i2, grid_c[1]-1-j2,:] = np.sum(mu_dict[:, 
                    ref_fov[y, x, grid_fov[2]-1-z]]*v_d[ind], axis=1)
                """ lower left quadrant"""
                mu_d[grid_c[0]-1-i2, j2,:] = np.sum(mu_dict[:, 
                    ref_fov[y, grid_fov[1] - 1 - x, z]] * v_d[ind], axis=1)
                """ lower right quadrant"""
                mu_d[grid_c[0]-1-i2, grid_c[1]-1-j2, :] = np.sum(mu_dict[:, 
                    ref_fov[y, grid_fov[1] - 1 - x, grid_fov[2] - 1 - z]] 
                    * v_d[ind], axis=1)
                """Reversed indencies """
                i3 = i2 - delta_c
                j3 = j2 + delta_c
                """ Upper left quadrant"""
                mu_d[j3, i3, :] = np.sum(mu_dict[:, 
                    ref_fov[y, z+rev_adjust, x-rev_adjust]] * v_d[ind], axis=1)
                """ upper right quadrant"""
                mu_d[j3, grid_c[1]-1-i3, :] = np.sum(mu_dict[:, 
                    ref_fov[y, z+rev_adjust, grid_fov[2] - 1 - x + rev_adjust]]
                    * v_d[ind], axis=1)
                """ lower left quadrant"""
                mu_d[grid_c[0]-1-j3, i3, :] = np.sum(mu_dict[:, 
                    ref_fov[y, grid_fov[1] - 1 - z-rev_adjust, x-rev_adjust]]
                    * v_d[ind], axis=1)
                """ lower right quadrant"""
                mu_d[grid_c[0]-1-j3, grid_c[1]-1-i3, :] = np.sum(mu_dict[:, 
                    ref_fov[y, grid_fov[1] - 1 - z -rev_adjust, 
                    grid_fov[2] - 1 - x + rev_adjust]]* v_d[ind], axis=1)
#       Assuming Poisson ditribution with Poisson varibel photon_count*mu_ray
        I_photon_temp = np.sum(np.random.poisson(
            ray_counts * np.exp(-mu_d)), axis=2)
        
        if saveImgs == True:
            # Saves an 8bit image of every subframe
            I8 = (normalize2D(I_photon_temp)*255.9).astype(np.uint)
            cv.imwrite(folder+filename+'/'+'subframes/' +
                        filename+f'{i+1}_{j+1}.png', I8)
        # Adds all the subframes together
        I_photon += I_photon_temp
        # dt = time() - t1
        # print(f'Created Subframe {j+1} out of {N_sub}, took {dt:02.2f} seconds\n')
        
    #The intensity at the CCD. Adds all the rays hitting the same CCD.
    #Save as a seperate file
    I_dummy = np.add.reduceat(I_photon, np.arange(0, I_photon.shape[0], 
            ray_ccd_ratio), axis=0)
    I_ccd[i,:,:] = np.add.reduceat(I_dummy, np.arange(0, I_photon.shape[1], 
            ray_ccd_ratio), axis=1)
    # Looks at the thresholdSets everything above threshold to threshold value
    I_ccd[i,I_ccd[i,:,:] > I_threshold] = I_threshold
    I_ccd[i,:,:] = (normalize2D(I_ccd[i,:,:], A_max = I_threshold, A_min = 0)
                *255.9).astype(np.uint8)
    if saveImgs == True:
        # Saves the frames as 8bit images
        cv.imwrite(folder+filename+'/'+'CCDreads/' +
                    filename+f'{i+1}.png', I_ccd[i, :, :])
    
    dt = time() - t0
    print(f'Created Frame {i+1} out of {N_frames}, took {dt:02.2f} seconds\n')
print('All Frames created')
plt.imshow(I_ccd[0,:,:])
#%% creates movie
from func.createMovie import createMovie
createMovie(I_ccd, N_frames, grid_ccd, folder, filename)

#%%
# Distance checker , the distancees gets off with a lot when m_m and n_m dosent 
# match in size
# real_distance = h_m/np.cos(alpha_x)
#print(real_distance)

plt.imshow(I_ccd[-1,:,:])

# Image with histogram equalization
I_equalizeHist = cv.equalizeHist(I_ccd[-1, :, :].astype('uint8'))
plt.imshow(I_equalizeHist)
cv.imwrite(folder+filename+'/'+'CCDreads/' +
           filename+'HistEqualization.png', I_equalizeHist)

# %%
