""" 
Prints a list of all the materials defined in the system with the corresponding 
reference values

Pulls all the materials form the mat_density.txt 
"""
#%%
import numpy as np

def materialList():
    folder = 'materials/'
    mat_names = []
    mat_rho = []
    i = 1
    with open(folder+'mat_density.txt', 'r') as file:
        for line in file:
            row = line.split('\t')
            row[1] = float(row[1].replace('\n', ''))
            mat_names.append(f"{i}: {row[0]}")
            mat_rho.append(row[1])
            i += 1

    mat_names.insert(0, '0: none')
    print(mat_names)
    return mat_names


# %%
