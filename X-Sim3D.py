"""
Created on Mon Jan 13 12:59:19 2020


@author: simonnyrup
X-ray simulator. Simulates the 2D image generated with x-rays 
emitted from a point source and penerating a material represented by an 3D array.

Generally the coordinate system is left-handed with center in x-ray point source. 
hat{y} points toward the camera(CCD object)
hat{x} (positive right) and hat{z}(inwards) are orthogonal and forms a plane with hat{y} as the normal vector

inputs:
    d1: The distance from point source to the top of material
    h_m: The "real" height of the density tensor representing the material
    d2: The distance from the buttom of the material to the camera (CCD)
    
    size_m: The "real" size of the density tensor representing the material:
            (height, length, width)
    grid_m: The amount of voxel in the y, x and z direction
            (m, n, p)
    
    size_c: the real size of image capturing device. Tuple with two inputs
            (length, width)
    grid_ccd: The resolution of the capturing device
            (n, p)
    ray_ccd_ratio: The amount of different angled rays hitting one ccd.   
    
    D: density matrix of size [grid_m[0] X grid_m[1] x grid_m[2]] - load a target object
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2 as cv
import os
from datetime import datetime
# import h5py

from func.calcPathAttenuation import calcPathAttenuation
from func.produceXray import produceXray
from func.findIO import findIO
from func.calcDiagonal import calcDiagonal
from func.generateMuDict import generateMuDict
from func.createD import drawBox, drawSphere, generateD
from func.normalize3D import normalize3D
from func.normalize2D import normalize2D
from func.fieldOfView import fieldOfView
from func.projectDtoFOV import projectDtoFOV
# from func.addNoise import addNoise
# import bohrium"
# Git hub test
#%% Set the constants 

#I/O
folder = "results/"
saveImgs = True
makeMovie = True

#Objects to be imported
folder_obj = 'objects/generated/'
# object_file = 'boxWithSmallSphere_carbon_iron200x200x200.npy'
# object_file = 'PotatoSingle.npy'
object_file = 'bunny.npy'
# object_file = 'horse.npy'
object_file2 = "Sphere_aluminium_carbon8x8x8.npy"
# object_file = "boxWithSphere_carbon_air200x200x200.npy"

filename = object_file.split('.')[0]
# If BOOL multiply by a number corresponding to the material it should be made of / Define before?
D = np.load(folder_obj+object_file)*1
D2 = np.load(folder_obj+object_file2)
D = np.rot90(D, 1, (0,1))

# D[140:148, 100:108, 100:108] = D2
#Finds the bounding box for D in x and z(The space in D with material)
def boundingbox(D):
    D_box = np.nonzero(np.sum(D,axis=0))
    D_ybox = np.nonzero(np.sum(D, axis=1))
    """ Below example could make it more readable? """
    # Creates a region of interest as a slice tuple - easy to pass
    # RegOfInt = (slice(500, 1100), slice(750, 1250), slice(None))
    D = D[D_ybox[0][0]:D_ybox[0][-1]+1, D_box[0][0]:D_box[0][-1]+1,np.min(D_box[1]):np.max(D_box[1]+1)]
    return D
D = boundingbox(D)

d1 = 2
d2 = 1
#Information about the material size _m is _material
size_v_m = (0.001, 0.001, 0.001) # volume of voxel in material

#Information about the grid definition [m x n x p] # The should all match in size? or 
grid_m = np.shape(D)
size_m = np.multiply(grid_m, size_v_m)

#Information about the size of the camera, _c is _camera
size_ccd_pixel = (.002, .002)

#number of ccds in the camera as [n x p]
#If non symmetric, first input should be largest
grid_ccd = (258, 258)
size_c = np.multiply(size_ccd_pixel, grid_ccd)

#The "resolution" of each CCD. Makes more rays hit the same CCD
ray_ccd_ratio = 1

# The finer grid for the camera calculations. 
grid_c = np.multiply(grid_ccd, ray_ccd_ratio)

#Number of Photons in a frame
photons_frame = 500

"""
Information of FOV size and grid. Length and with needs to be bigger than
material. For 1 to 1 transposability the amount of voxel per length needs 
to be the same as in the material.
"""
size_fov = (size_m[0], (size_c[0]*d1)/(d1+d2), (size_c[1]*d1)/(d1+d2))
#Increases the size of FOV by a maximum of 1. to make pretty numbers and 
# guarantee that the voxel volumes of D and fov match
size_fov = size_fov + (size_v_m-np.mod(size_fov,size_v_m))

""" 
Define FOV grid so each voxel is same size as in D
might be a rounding error
"""
grid_fov = np.round(np.divide(size_fov, size_v_m)).astype(int)
size_v_fov = np.divide(size_fov, grid_fov)

move = (0, 0)

ref_fov = projectDtoFOV(grid_fov, grid_m, D, move)

""" Defines relevant constants for making a movie"""
""" Needs to rewrite moment to fit to newest method. Look at image creation"""
# Movement in a single subframe in x and z dire
mat_movement = (0.00, 0.0)
# The starting position in x and z
mat_start = (0.0, 0)
# Rotation in a single subframe in y, x and z direction in radians
mat_rotate = (0, 0, 0)
mat_rotstart = (0, 0, 0)
# The total of Image frames
N_frames = 1
# The amount of iterations made to create one frame.
# Generating N_sub frames for every frame. Can be seen as integration time
N_sub = 10

if makeMovie == False:
    mat_movement = (-0.00, -0.00)
    mat_start = (0, 0)
    N_frames = 1
    N_sub = 1

#%% Creates Directories for all types of files and saves constants
if saveImgs == True:

    dirName = filename
    try:
        # Create target Directory
        os.mkdir(folder+filename)
        print("Directory ", dirName,  " Created ")
    except FileExistsError:
        print("Directory ", dirName,  " already exists")

    dirName = 'subframes'
    try:
        # Create target Directory
        os.mkdir(folder+filename+'/'+dirName)
        print("Directory ", dirName,  " Created ")
    except FileExistsError:
        print("Directory ", dirName,  " already exists")
    # Create directory
    dirName = 'CCDreads'
    try:
        # Create target Directory
        os.mkdir(folder+filename+'/'+dirName)
        print("Directory ", dirName,  " Created ")
    except FileExistsError:
        print("Directory ", dirName,  " already exists")

    const_names = ['d1', 'd2', 'size_m', 'grid_m',
                    'size_c', 'grid_ccd', 'ray_ccd_ratio', 'mat_movement', 'mat_start', 
                    'N_frames', 'N_sub', 'photons_frame']
    const_vals = [d1, d2, size_m, grid_m, size_c, grid_ccd, ray_ccd_ratio, mat_movement, 
                 mat_start, N_frames, N_sub, photons_frame]
    import csv

    with open(folder+filename+'/''details.csv', 'w', ) as myfile:
        wr = csv.writer(myfile)
        # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        test = zip(const_names, const_vals)
        for row in test:
           wr.writerow([row])


#%% Ray spectrum and intensities #The spectrum for the energies
from xraySpec.dataA import arr80

spec = np.asarray(arr80)
spec = spec[9:-1]

#For a given spectrum find the energies and the count of photons with each energy
ray_energy, ray_counts = produceXray(spec, N_points = photons_frame, Emin = 10, Emax = 80, Estep = 1)

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
#%% Generates Field of View list. An list containing all indencies a ray goes through in the field of view
# Only generates a Field of view list if one with those specifics dosen't ecist
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

#%% Creates the movie
I_ccd = np.zeros([N_frames, grid_ccd[0], grid_ccd[1]])
mat_move = mat_start
mu_d = np.zeros([grid_c[0],grid_c[1], len(ray_energy)])
#The number of iterations in the loop
ind_len = int(grid_qc[0]*grid_qc[1]+(grid_qc[1]-grid_qc[1]**2)/2)
# transDtoFOVcenter = np.subtract(grid_fov, grid_m)/2
y = np.arange(grid_fov[0])
# Parameter to adjust for the unequal parts of FOV, in the reversed indencies
""" Maybe find a better expresions as it seems a little random factor to add """
rev_adjust = (grid_fov[1] - grid_fov[2])//2
# D_broad = np.broadcast_to(fov_ref,(fov_list.shape[1],D.shape[0],D.shape[1],D.shape[2]))
x_points, z_points = fov_list[0, :, :], fov_list[1, :, :]
v_d_broad = np.broadcast_to(v_d, (grid_fov[0], ind_len))

# The necessary points to evaluate
# Counter is made to evaluate the loop properly.
# Only calculate the upper triangle in the FOV. 1/8 of the whole camera area
delta_c = grid_qc[0] - grid_qc[1]
count_start = np.arange(0, grid_qc[1]*(delta_c+1), grid_qc[1])
count_end = np.cumsum(np.arange(grid_qc[1], 1, -1))+grid_qc[1]*delta_c
counter = np.insert(count_end, 0, count_start)
move_step = (1,0)
move = (int(-N_frames*N_sub/2),0)

def calcAttenuation(mu_dict, ref_fov, v_d, y, x, z): 
    return np.sum(mu_dict[:, ref_fov[y, x, z]]*v_d, axis=1)

for i in range(N_frames):
    t0 = time()
    I_photon = np.zeros(tuple(grid_c))
    for j in range(N_sub):
        t1 = time()
        #Finds photons from a new distribution. Needs to sample everytime?
        ray_energy, ray_counts = produceXray(
                                spec, N_points=photons_frame, Emin=10, Emax=80, Estep=1)
        # Calculates all entry and exit points of the rays in the density matrix
        ref_fov = projectDtoFOV(grid_fov, grid_m, D, move)
        move = np.add(move, move_step)
        # D_fov_conversion = np.round(np.divide(mat_move,size_v_m[1::])-transDtoFOVcenter[1::]).astype(int)
        #Loops over the part where the indencies cannot be reversed 
        for i2 in range(delta_c):
            for j2 in range(grid_qc[1]):
                ind = counter[i2]+j2
                x = x_points[ind]
                z = z_points[ind]
                """ Upper left quadrant"""
                mu_d[i2, j2, :] = np.sum(mu_dict[:, ref_fov[y, x, z]]*v_d[ind], axis=1)
                """ upper right quadrant"""
                mu_d[i2, grid_c[1]-1-j2, :] = np.sum(mu_dict[:, ref_fov[y, x, grid_fov[2]-1-z]]*v_d[ind], axis=1)
                """ lower left quadrant"""
                mu_d[grid_c[0]-1-i2, j2,:] = np.sum(mu_dict[:, ref_fov[y, grid_fov[1] - 1 - x, z]] * v_d[ind], axis=1)
                """ lower right quadrant"""
                mu_d[grid_c[0]-1-i2, grid_c[1]-1-j2, :] = np.sum(mu_dict[:, ref_fov[y, grid_fov[1] - 1 - x, grid_fov[2] - 1 - z]]* v_d[ind], axis=1)
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
                mu_d[i2, grid_c[1]-1-j2,:] = np.sum(mu_dict[:, ref_fov[y, x, grid_fov[2]-1-z]]*v_d[ind], axis=1)
                """ lower left quadrant"""
                mu_d[grid_c[0]-1-i2, j2,
                    :] = np.sum(mu_dict[:, ref_fov[y, grid_fov[1] - 1 - x, z]] * v_d[ind], axis=1)
                """ lower right quadrant"""
                mu_d[grid_c[0]-1-i2, grid_c[1]-1-j2, :] = np.sum(
                    mu_dict[:, ref_fov[y, grid_fov[1] - 1 - x, grid_fov[2] - 1 - z]] * v_d[ind], axis=1)
                """Reversed indencies """
                i3 = i2 - delta_c
                j3 = j2 + delta_c
                """ Upper left quadrant"""
                mu_d[j3, i3, :] = np.sum(
                             mu_dict[:, ref_fov[y, z+rev_adjust, x-rev_adjust]] * v_d[ind], axis=1)
                """ upper right quadrant"""
                mu_d[j3, grid_c[1]-1-i3, :] = np.sum(mu_dict[:, ref_fov[y, z+rev_adjust, grid_fov[2] - 1 - x + rev_adjust]]
                            * v_d[ind], axis=1)
                """ lower left quadrant"""
                mu_d[grid_c[0]-1-j3, i3, :] = np.sum(mu_dict[:, ref_fov[y, grid_fov[1] - 1 - z-rev_adjust, x-rev_adjust]]
                                                * v_d[ind], axis=1)
                """ lower right quadrant"""
                mu_d[grid_c[0]-1-j3, grid_c[1]-1-i3, :] = np.sum(mu_dict[:, ref_fov[y, grid_fov[1] - 1 - z -rev_adjust, grid_fov[2] - 1 - x + rev_adjust]]
                            * v_d[ind], axis=1)
#       Assuming Poisson ditribution with Poisson varibel photon_count*mu_ray
        I_photon_temp = np.sum(np.random.poisson(
            ray_counts * np.exp(-mu_d)), axis=2)
        
        # #The intensity of every photon beam without stochasticprosec
        # I_photon_temp = np.sum(ray_counts * np.exp(-mu_d), axis=2)
        # I_photon_temp = np.sum(ray_counts * np.exp(-mu_d.reshape([grid_c[0],grid_c[1],len(ray_energy)])), axis=2)
        if saveImgs == True:
            # plt.imsave(folder+filename+'/'+'subframes/' +
            #            filename+f'{i+1}_{j+1}.ppm', I_photon_temp)
            # I = I_photon_temp
            # I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
            I8 = (normalize2D(I_photon_temp)*255.9).astype(np.uint)
            cv.imwrite(folder+filename+'/'+'subframes/' +
                        filename+f'{i+1}_{j+1}.png', I8)

        I_photon += I_photon_temp
        # dt = time() - t1
        # print(f'Created Subframe {j+1} out of {N_sub}, took {dt:02.2f} seconds\n')
        
        #The intensity at the CCD. Adds all the rays hitting the same CCD.
        #Save as a seperate file

    I_dummy = np.add.reduceat(I_photon, np.arange(0, I_photon.shape[0], ray_ccd_ratio), axis=0)
    I_ccd[i,:,:] = np.add.reduceat(I_dummy, np.arange(0, I_photon.shape[1], ray_ccd_ratio), axis=1)

    I_ccd[i,I_ccd[i,:,:] > I_threshold] = I_threshold
    I_ccd[i,:,:] = (normalize2D(I_ccd[i,:,:], A_max = I_threshold, A_min = 0)*255.9).astype(np.uint8)
    if saveImgs == True:
        # plt.imsave(folder+filename+'/'+'CCDreads/' +
                #    filename+f'{i+1}.png', I_ccd[i, :, :], vmin = 0, vmax = 255)
        cv.imwrite(folder+filename+'/'+'CCDreads/' +
                    filename+f'{i+1}.png', I_ccd[i, :, :])
    
    dt = time() - t0
    print(f'Created Frame {i+1} out of {N_frames}, took {dt:02.2f} seconds\n')
print('All Frames created')
plt.imshow(I_ccd[0,:,:])
#%% creates movie
img_array = []
for i in range(N_frames):
    #Converts a frame to BGR scale
    frame = np.tile(I_ccd[i, :, :], (3, 1, 1)).T.astype('uint8')
    img_array.append(frame)

fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')

fps = 5
out = cv.VideoWriter(folder+filename+'/'+filename+'.avi', fourcc, fps, tuple(grid_ccd))

for frame in img_array:
    out.write(frame)
    plt.imshow(frame)
out.release()

#%%
# Distance checker , the distancees gets off with a lot when m_m and n_m dosent match in size
# real_distance = h_m/np.cos(alpha_x)
#print(real_distance)

plt.imshow(I_ccd[-1,:,:])
# plt.imshow(img_array[0])
# cv.imshow('image', frame)
# cv.waitKey(0)



# %%


# %%
