""" 
takes the material list and all the different materials and generates a dictionary for given energies 

returns a dictionary refering a index to a range of mu values and a reference list between indencies and materials

For now it multiplies with a density taken as a standard value at 25C0 at 1 at.
"""
#%%
import numpy as np
from func.muRescale import muRescale

def generateMuDict(energies):

    folder = 'materials/'

# makes a list of all materials and their densities(Keep in mind that the densities is just an estimate)
    # with open(folder+'matlist.txt', 'r') as file:
    #     mat_names = file.read().replace('\n', ',')
    # mat_names = mat_names.split(',')


    mat_names = []
    mat_rho = []
    with open(folder+'mat_density.txt', 'r') as file:
        for line in file:
            row = line.split('\t')
            row[1] = float(row[1].replace('\n', ''))
            mat_names.append(row[0])
            mat_rho.append(row[1])

    mat_rho = np.asarray(mat_rho)

#Creates a dictionary refering different materials to their attenuation coefficients for given energies  
    mu_dict = np.zeros([len(energies), len(mat_names)+1])

    for i in range(len(mat_names)):
        mu = np.loadtxt(folder+mat_names[i]+'.txt')
        mu_dict[:,i+1] = muRescale(mu, energies)[:,1]*mat_rho[i]
    #converts to meters
    mu_dict = mu_dict * 10**(-2)

    mat_names.insert(0, 'none')

    return mu_dict, mat_names

# %%
