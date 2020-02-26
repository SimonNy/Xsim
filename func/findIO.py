"""
Created on Thu Jan 23 10:51:52 2020

@author: simonnyrup

A function that finds the coordinate for all the input and outputs of an given
array of ray angles and positions.
"""
#%%
import numpy as np

def findIO(m_m, p_y, p_x, p_z, d1, h_m, l_m, b_m, alpha_x, alpha_z = 0):
    
    # Finds the distance from the starting point of the ray to the entry point in the density matrix
    x_entry = (d1 - h_m) * np.tan(alpha_x)
    z_entry = (d1 - h_m) * np.tan(alpha_z)
    
    # Finds the distance from the starting point of the ray to the exit point in the density matrix
    x_exit = d1 * np.tan(alpha_x)
    z_exit = d1 * np.tan(alpha_z)
    
    # Changes the reference system to be in the top left corner of the density matrix
    x_entry_Dref = x_entry + l_m/2
    z_entry_Dref = z_entry + b_m/2
    x_exit_Dref = x_exit + l_m/2
    z_exit_Dref = z_exit + b_m/2
    
    #Finds the closest index relating to the entry point
    x_entry_index = np.round(x_entry_Dref / p_x).astype(int)
    z_entry_index = np.round(z_entry_Dref / p_z).astype(int)
    
    x_exit_index = np.round(x_exit_Dref / p_x).astype(int)
    z_exit_index = np.round(z_exit_Dref / p_z).astype(int)
    
    #makes an array of all relevant points, y starting point is 0 and end points is m_m -1
    ray_points  = (x_entry_index,  z_entry_index, x_exit_index, z_exit_index)
    return ray_points
#%%