

#%%
""" Load and define object 
    Two types of objects exists:

    - User-defined objects created with the gernerateD.py file
    These types carry values reffering to the desired materials in the 
    materials folder

    - 3D graphic meshes converted to a .npy file with binvox
    In general they are homogeneous and needs to change the values to match the
    desired materials
"""

# Imports stuff
import numpy as np
import matplotlib.pyplot as plt

# To get a list of avaliable materials run
from func.materialList import materialList
materialList();
# imports the boundingbox function to remove wasted space from the object
from func.boundingbox import boundingbox

# Path to the folder carrying objects
folder_obj = 'objects/generated/'

""" Which of the following examples to run """
choose_object = "example1"
# choose_object = "example2"
# choose_object = "example3"

# object_file = 'bunny.npy'
if choose_object == "example1":
    """ Loads a bunny and defines it as water """
    object_file = 'bunny.npy'
    filename = object_file.split('.')[0]

    # The bunny object is loaded as a boolean and can be multiplied by the desired
    # material value
    mat_val1 = 5
    D = np.load(folder_obj+object_file) * mat_val1
    # Removes as much space as possible
    D = boundingbox(D)
    D = np.rot90(D, 1, (0, 2))

elif choose_object == "example2":
    """ Loads a sphere with a small sphere inside of it """
    object_file = "sphereWithSmallSphere_carbon_air100x100x100.npy"
    filename = object_file.split('.')[0]

    # the object has a predefined material value
    D = np.load(folder_obj+object_file)
    # Removes as much space as possible
    D = boundingbox(D)

elif choose_object == "example3": 
    """ Example of loading a potato object as carbon and putting a hole in it """
    object_file = 'potatoSingle.npy'
    filename = object_file.split('.')[0]

    object_file2 = "sphere_air_carbon30x30x30.npy"
    filename2 = object_file2.split('.')[0]
    filename = filename + "With" + filename2
    # The potato object is loaded as a boolean and can be multiplied by the 
    # desired material value
    mat_val1 = 1
    D = np.load(folder_obj+object_file) * mat_val1
    # Removes as much space as possible
    D = boundingbox(D)
    # Rotates object to desired position
    D = np.rot90(D, 1, (0, 2))

    D2 = np.load(folder_obj+object_file2)
    D2[D2 == 0] = mat_val1

    # places D2 in the center of D
    grid_D = np.shape(D)
    grid_D2 = np.shape(D2)
    D[grid_D[0]//2-grid_D2[0]//2:grid_D[0]//2+grid_D2[0]//2,
    grid_D[1]//2-grid_D2[1]//2:grid_D[1]//2+grid_D2[1]//2,
    grid_D[2]//2-grid_D2[2]//2:grid_D[2]//2+grid_D2[2]//2
    ] = D2


print(f"D has a grid size of: {np.shape(D)}")
print("Naive bottom view of object")
plt.imshow(np.sum(D, axis=0), cmap="gray")

# %%
