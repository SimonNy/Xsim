"""Generates a specified object with reference to a material list"""
#%%
from func.createD import drawBox, drawSphere, generateD
from func.generateMuDict import generateMuDict

import numpy as np
from time import time

#Grid definition in D
grid_m = (15, 15, 15)
mat1 = 'carbon'
mat2 = 'water'

#The kind of D to be created
"""kind: boxWithBox, boxWithSphere, oneMat, boxWithSmallSphere, Sphere"""
kind = "Sphere"
# kind = "oneMat"
# kind = "boxWithSmallSphere"

folder = 'materials/'
mat_names = []
with open(folder+'mat_density.txt', 'r') as file:
    for line in file:
        row = line.split('\t')
        mat_names.append(row[0])

mat_names.insert(0, 'none')

t0 = time()
print(f'Creates Density Matrix D as {kind},')
D = generateD(grid_m, kind, mat1, mat2, mat_names).astype(int)
dt = time() - t0
print(f'took {dt:02.2f} seconds\n')


folder_out = 'objects/generated/'
filename = kind+'_'+mat1+'_'+mat2+f'{grid_m[0]}x{grid_m[1]}x{grid_m[2]}'
np.save(folder_out+filename, D)
print(f'Name of generated file '+filename+'.npy')



# %%
