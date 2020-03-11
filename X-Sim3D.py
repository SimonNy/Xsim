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
    h_m: The "real" height of the density matrix representing the material
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

from func.calcPathAttenuation import calcPathAttenuation
from func.produceXray import produceXray
from func.findIO import findIO
from func.calcDiagonal import calcDiagonal
from func.generateMuDict import generateMuDict
from func.createD import drawBox, drawSphere, generateD
from func.normalize3D import normalize3D
from func.normalize2D import normalize2D
from func.fieldOfView import fieldOfView
# import bohrium

#%% Set the constants 

#I/O
folder = "results/"
# date_time = datetime.now().strftime("%Y-%d-%m--%H-%M-%S")
# date_time = ''
# filename = 'Testwup' + date_time
# filename = "WupWup"
saveImgs = True

makeMovie = True

#Objects to be imported
folder_obj = 'objects/generated/'
# object_file = 'boxWithSmallSphere_carbon_iron200x200x200.npy'
object_file = 'PotatoSingle.npy'

filename = object_file.split('.')[0]
D = np.load(folder_obj+object_file)*5
# D = np.rot90(D, 3, (0,2))

#Finds the bounding box for D in x and z(The space in D with material)
D_box = np.nonzero(np.sum(D,axis=0))
D_ybox = np.nonzero(np.sum(D, axis=1))
D = D[D_ybox[0][0]:D_ybox[0][-1], D_box[0][0]:D_box[0][-1],np.min(D_box[1]):np.max(D_box[1])]


# object_file2 = 'Sphere_carbon_water15x15x15.npy'
# filename2 = object_file2.split('.')[0]
# D2 = np.load(folder_obj+object_file2) 


# D[D.shape[0]//4:D.shape[0]//4+D2.shape[0], D.shape[1] //
#     4:D.shape[1]//4+D2.shape[1], D.shape[2]//4:D.shape[2]//4+D2.shape[2]] = D2

# D[D==1] = 4
# D = np.rot90(D, 1, (1, 2))
print(D.shape)
# object_file = 'horse_skeleton.npy'
# filename = object_file.split('.')[0]
# D2= np.load(folder_obj+object_file)*5
# D2 = np.rot90(D2, 1, (0, 2))*5
# D[D2 == 1] = 1
# D -= D2
# D_air_mask = D!=0

d1 = 2
d2 = 1
#Information about the material size _m is _material
size_m = (0.2, 0.2, 0.2)
#Information about the grid definition [m x n x p] # The should all match in size? or 
grid_m = np.shape(D)

"""
Information of FOV size and grid. Length and with needs to be bigger than
material. For 1 to 1 transposability the amount of voxel per length needs 
to be the same as in the material.
"""
# size_fov = (size_m[0], size_m[1]*2, size_m[2]*2)
size_fov = (size_m[0], size_m[1]*2, size_m[2]*2)

fov_grid = np.max(grid_m)
# grid_fov = (grid_m[0], int(grid_m[1]/size_m[1]*size_fov[1]),
            # int(grid_m[2]/size_m[2]*size_fov[2]))
# grid_fov = (256, int(256/size_m[1]*size_fov[1]),
            # int(256/size_m[2]*size_fov[2]))

grid_fov = (fov_grid, fov_grid*2, fov_grid*2)

#Information about the size of the camera, _c is _camera
size_c = (0.5, 0.5)

#number of ccds in the camera as [n x p]
grid_ccd = (256, 256)

#The "resolution" of each CCD. Makes more rays hit the same CCD
ray_ccd_ratio = 1

# The finer grid for the camera calculations. 
grid_c = np.multiply(grid_ccd, ray_ccd_ratio)

""" Defines relevant constants for making a movie"""
# Movement in a single subframe in x and z dire
mat_movement = (0.01, 0.0)
# The starting position in x and z
mat_start = (-0.2, 0)

# Rotation in a single subframe in y, x and z direction in radians
mat_rotate = (0, 0, 0)
mat_rotstart = (0,0,0)


# The total of Image frames
N_frames = 2
# The amount of iterations made to create one frame.
# Generating N_sub frames for every frame. Can be seen as integration time
N_sub = 5

if makeMovie == False:
    mat_movement = (0, 0)
    mat_start = (0, 0)
    N_frames = 1
    N_sub = 1

#Number of Photons in a frame
photons_frame = 10000

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

#The distance to each center point of the CCD(centered in the middle)
xc = (np.arange(grid_c[0]) - grid_c[0]/2 + 0.5) * size_c[0] / (grid_c[0]) 
zc = (np.arange(grid_c[1]) - grid_c[1]/2 + 0.5) * size_c[1] / (grid_c[1])   
# The angles between the center and the midpoint of a given input of the CCD 
alpha = (np.arctan(xc/h), np.arctan(zc/h))

# the physical size of each voxel in D
size_v = np.divide(size_m, grid_m)

v_d = calcDiagonal(size_v, alpha)
#%% Generates Field of View list. An list containing all indencies a ray goes through in the field of view
# Only generates a Field of view list if one with those specifics dosen't ecist
fov_name = f'{grid_fov[0]}x{grid_fov[1]}x{grid_fov[2]}_{grid_c[0]}x{grid_c[1]}_{size_fov[0]}x{size_fov[1]}x{size_fov[2]}_{size_c[0]}x{size_c[1]}_{d1}_{d2}'
if os.path.isfile('fov/'+fov_name+'.npy'):
    print (f"FOV File exist with name: {fov_name}")
    fov_list = np.load('fov/'+fov_name+'.npy')
else:
    t0 = time()
    print ("FOV File dose not exist.\n Creates FOV list,")
    fov_list = fieldOfView(grid_fov, grid_c, size_v, d1, size_fov, alpha)
    np.save('fov/'+fov_name,fov_list)
    dt = time() - t0
    print(f'took {dt:02.2f} seconds\n')

#%% Creates the movie
I_ccd = np.zeros([N_frames, grid_ccd[0], grid_ccd[1]])
mat_move = mat_start
mu_d = np.zeros([grid_c[0], grid_c[1], len(ray_energy)])

transDtoFOVcenter = np.subtract(grid_fov, grid_m)/2

for i in range(N_frames):
    t0 = time()
    I_photon = np.zeros(tuple(grid_c))
    for j in range(N_sub):
        t1 = time()
        #Finds photons from a new distribution
        ray_energy, ray_counts = produceXray(
                                spec, N_points=photons_frame, Emin=10, Emax=80, Estep=1)
        # Calculates all entry and exit points of the rays in the density matrix
        mat_move = np.add(mat_move, mat_movement)
        D_fov_conversion = np.round(np.divide(mat_move,size_v[1::])-transDtoFOVcenter[1::]).astype(int)

        for i2 in range(grid_c[0]):
            for j2 in range(grid_c[1]):
                y_points, x_points, z_points = tuple(fov_list[i2*grid_c[1]+j2])
                points_mask = y_points >= 0
                points_mask[y_points > grid_m[0]-1] = False
                # D_fov_conversion = np.add((x_points, z_points), D_fov_conversion)
                #Translates the FOV points to the coordinates of D
                x_points = D_fov_conversion[0] + x_points
                z_points = D_fov_conversion[1] + z_points
                #Makes sure to only calculate points inside of D
                x_points_mask = x_points >= 0
                x_points_mask[x_points > grid_m[1]-1] = False
                z_points_mask = z_points >= 0
                z_points_mask[z_points > grid_m[2]-1] = False

                points_mask[x_points_mask*1+z_points_mask*1 != 2] = False

                mu_d[i2, j2,:] = np.sum(mu_dict[:, D[y_points[points_mask], x_points[points_mask], z_points[points_mask] ]]
                             * v_d[i2,j2], axis =1)

        #The intensity of every photon beam
        I_photon_temp = np.sum(ray_counts * np.exp(-mu_d), axis=2)
        if saveImgs == True:
            plt.imsave(folder+filename+'/'+'subframes/' +
                       filename+f'{i+1}_{j+1}.png', I_photon_temp)

        I_photon += I_photon_temp
        # dt = time() - t1
        # print(f'Created Subframe {j+1} out of {N_sub}, took {dt:02.2f} seconds\n')
        
        #The intensity at the CCD. Adds all the rays hitting the same CCD.
        #Save as a seperate file

    I_dummy = np.add.reduceat(I_photon, np.arange(0, I_photon.shape[0], ray_ccd_ratio), axis=0)
    I_ccd[i,:,:] = np.add.reduceat(I_dummy, np.arange(0, I_photon.shape[1], ray_ccd_ratio), axis=1)

    I_ccd[i,I_ccd[i,:,:] > I_threshold] = I_threshold
    I_ccd[i,:,:] = (normalize2D(I_ccd[i,:,:], A_max = I_threshold, A_min = 0)*255).astype('uint8')
    if saveImgs == True:
        plt.imsave(folder+filename+'/'+'CCDreads/' +
                   filename+f'{i+1}.png', I_ccd[i, :, :], vmin = 0, vmax = 255)

    
    dt = time() - t0
    print(f'Created Frame {i+1} out of {N_frames}, took {dt:02.2f} seconds\n')
print('All Frames created')
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
