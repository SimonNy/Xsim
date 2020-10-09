

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


def loadObject(filename):

    # object_file = 'bunny.npy'
    if filename == "bunny":
        """ Loads a bunny and defines it as carbon """
        object_file = 'bunny.npy'
        filename = object_file.split('.')[0]

        # The bunny object is loaded as a boolean and can be multiplied by the desired
        # material value
        mat_val1 = 1
        D = np.load(folder_obj+object_file) * mat_val1
        # Removes as much space as possible
        D = boundingbox(D)
        D = np.rot90(D, 1, (0, 2))
        D = np.rot90(D, 2, (1, 2))

    elif filename == "sphereWithSphere":
        """ Loads a sphere with a small sphere inside of it """
        object_file = "sphereWithSmallSphere_carbon_iron200x200x200.npy"
        filename = object_file.split('.')[0]

        # the object has a predefined material value
        D = np.load(folder_obj+object_file)
        # Removes as much space as possible
        D = boundingbox(D)

    elif filename == "simplePotato": 
        """ Example of loading a potato object containing potato starch and 
        putting a hole in it """
        object_file = 'potatoNew255.npy'
        filename = object_file.split('.')[0]
        # The potato object is loaded as a boolean and can be multiplied by the 
        # desired material value
        mat_val1 = 10

        D = np.load(folder_obj+object_file) * mat_val1
        # Removes as much space as possible
        D = boundingbox(D)
        # Rotates object to desired position
        D = np.rot90(D, 1, (0, 2))

    elif filename == "potatoWithHeart":
        """ Example of loading a potato object as potatoStarch and putting a hollow heart 
            in it + a nail and a paper clip

        """
        object_file = 'potatoSingle300.npy'
        filename = object_file.split('.')[0]

        object_file2 = "heart70.npy"
        filename2 = object_file2.split('.')[0]

        object_file3 = "nail100.npy"
        filename3 = object_file3.split('.')[0]
        object_file4 = "paperClip100.npy"

        filename = filename + "With" + filename2
        # The potato object is loaded as a boolean and can be multiplied by the
        # desired material value
        mat_val1 = 10
        mat_val2 = 4
        mat_val3 = 3
        mat_val4 = 2
        D = np.load(folder_obj+object_file) * mat_val1
        # Removes as much space as possible
        D = boundingbox(D)
        # Rotates object to desired position
        D = np.rot90(D, 2, (1, 2))
        """ puts the heart in it """
        D2 = np.load(folder_obj+object_file2) * mat_val2
        # outer box is same material as the outer material
        D2[D2 == 0] = mat_val1
        D2 = np.rot90(D2, 1, (0, 1))

        # places D2 in the center of D
        grid_D = np.shape(D)
        grid_D2 = np.shape(D2)
        D[grid_D[0]//2-grid_D2[0]//2:grid_D[0]//2+grid_D2[0]//2,
        grid_D[1]//2-grid_D2[1]//2:grid_D[1]//2+grid_D2[1]//2,
        grid_D[2]//2-grid_D2[2]//2:grid_D[2]//2+grid_D2[2]//2
        ] = D2

        """ Puts the nail in it """
        D3 = np.load(folder_obj+object_file3) * mat_val3
        # outer box is same material as the outer material
        D3[D3 == 0] = mat_val1
        grid_D3 = np.shape(D3)
        D[grid_D[0]//4:grid_D[0]//4+grid_D3[0],
        grid_D[1]//4:grid_D[1]//4+grid_D3[1],
        grid_D[2]//4:grid_D[2]//4+grid_D3[2]
        ] = D3


        D4 = np.load(folder_obj+object_file4) * mat_val4
        # outer box is same material as the outer material
        D4[D4 == 0] = mat_val1
        D4 = np.rot90(D4, 1, (0, 1))
        grid_D4 = np.shape(D4)
        D[grid_D[0]//2:grid_D[0]//2+grid_D4[0],
        grid_D[1]*3//4:grid_D[1]*3//4+grid_D4[1],
        grid_D[2]//4:grid_D[2]//4+grid_D4[2]
        ] = D4

        print(f"Shape of D2 is: {grid_D2}")
        print(f"Shape of D3 is: {grid_D3}")
        print(f"Shape of D4 is: {grid_D4}")
        # plt.imshow(np.sum(D2, axis=0), cmap="gray")
        # plt.show()
        # plt.imshow(np.sum(D3, axis=0), cmap="gray")
        # plt.show()
        # plt.imshow(np.sum(D4, axis=0), cmap="gray")
        # plt.show()

    elif filename == "boardWithMarks":
        """ Loads a soft_tissue box with a bone cylinder insie it """
        object_file = "boardWithMarks_carbon_iron100x200x300.npy"
        filename = object_file.split('.')[0]

        # the object has a predefined material value
        D = np.load(folder_obj+object_file)
        # Removes as much space as possible
        D = boundingbox(D)
        D = np.rot90(D, 1, (1, 2))

    print(f"D has a grid size of: {np.shape(D)}")
    print("Naive bottom view of object")
    plt.imshow(np.sum(D, axis=0), cmap="gray")
    # plt.show()
    return D
    # %%
