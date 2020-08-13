"""
    Generates a specified object with reference to a material list
    grid_D: size of the grid defining D
    mat1, mat2: type of material to use in D
    kind: Type of object generated

    The generated obeject are placed in the objects/generated/ folder
    The filename is printed to the terminal
"""
#%%
from func.createD import drawBox, drawSphere, generateD
from func.generateMuDict import generateMuDict

import numpy as np
import matplotlib.pyplot as plt
from time import time

plot3D = False

# The size of the grid for the object
grid_D = (30, 30, 30)
# The materials the object should consist of
mat1 = 'air'
mat2 = 'carbon'
#The kind of D to be created
"""kind: boxWithBox, boxWithSphere, oneMat, boxWithSmallSphere, sphere, boardWithMarks"""
# kind = "boardWithMarks"
# kind = "oneMat"
kind = "sphere"
# kind = "boxWithSmallSphere"
# kind = "twoBoxesWithSmallSphere"
# kind = "sphereWithSmallSphere"


folder = 'materials/'
mat_names = []
with open(folder+'mat_density.txt', 'r') as file:
    for line in file:
        row = line.split('\t')
        mat_names.append(row[0])

mat_names.insert(0, 'none')

t0 = time()
print(f'Creates Density Matrix D as {kind},')
D = generateD(grid_D, kind, mat_names, mat1, mat2).astype(int)
if 'mat3' in locals():
    D = generateD(grid_D, kind, mat_names, mat1, mat2).astype(int)
dt = time() - t0
print(f'took {dt:02.2f} seconds\n')


folder_out = 'objects/generated/'
if 'mat3' in locals():
    filename = kind+'_'+mat1+'_'+mat2+'_'+mat3+f'{grid_D[0]}x{grid_D[1]}x{grid_D[2]}'
else:
    filename = kind+'_'+mat1+'_'+mat2+f'{grid_D[0]}x{grid_D[1]}x{grid_D[2]}'

np.save(folder_out+filename, D)
print(f'Name of generated file '+filename+'.npy')

print("Naive view of object")
plt.imshow(np.sum(D, axis=0))
# %%
if plot3D == True:
    # Playing with constants for Latex margin figures
    pc = 400/2409  # the pc unit relative to inchens
    goldenRatio = 1.618  # ratio between width and height
    marginWidth = 11.5  # width of latex margin document
    resize = 1/0.25 # 1/scale_factor_in_latex


    fig = plt.figure(figsize=(marginWidth*pc*resize,
                            marginWidth*pc*resize))
    ax = fig.gca(projection='3d')
    ax.voxels(D, facecolor='#d62728', edgecolor='k')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_axis_off()

    # The fix
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig("generatedFigure.pdf", dpi=600)

    plt.show
