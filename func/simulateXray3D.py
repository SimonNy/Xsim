#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:59:19 2020

@author: simonnyrup

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

X-ray simulator. Simulates the image generated on 1D image generated with x-rays 
emitted from a point source and penerating a material, represented by an 2D array.

System is build up as a left handed coordinate system. Center point is at the point source,
with x going right, y going down, and z going into the script

inputs:
    d1: The distance from point source to the top of material
    h_m: The "real" height of the density matrix representing the material
    d2: The distance from the buttom of the material to the camera (CCD)
    
    l_m: The "real" length of the density matrix representing the material
    b_m: The "real" broadness of the density matrix representing the material
    
    n_m: The grid size of the height of the density matrix
    m_m: The grid size of the length of the density matrix
    p_m: The grid size of the broadness of the density matrix
    
    l_c: The length of the camera
    m_c: The grid size of the length of the camera
    p_c: The grid size of the broadness of the camera
    
    D: density matrix of size n_m X m_m
"""
import numpy as np


def simulateXray3D(d1, d2, h_m, l_m, b_m, l_c, b_c, m_c, p_c, D, moving_N = 0, moving_step_x = 0, moving_step_z = 0):
    n_m, m_m, p_m = D.shape

    # The total height of the system
    h = d1 + h_m + d2 
    
    #The distance to each center point of the CCD(centered in the middle)
    xc = (np.arange(m_c) - m_c/2 + 0.5) * l_c / (m_c) 
    zc = (np.arange(p_c) - p_c/2 + 0.5) * b_c / (p_c) 
    
    # The angles between the center and the midpoint of a given input of the CCD 
    alpha_x = np.arctan(xc/h)
    alpha_z = np.arctan(zc/h) 
    
    # The distance from the point source to the midpoint in a given layer of the density matrix
    y = (np.arange(n_m) + 0.5) * h_m / n_m + d1
    y_x = np.repeat(y, m_c).reshape([n_m, m_c])
    y_z = np.repeat(y, p_c).reshape([n_m, p_c])
    
    if moving_N == 0:
        # A matrix representing the distance to the center every ray hits for a given layer in the density matrix
        x = np.tan(alpha_x) * y_x
        z = np.tan(alpha_z) * y_z
        
        #The horizontal subindex relating the ray points to the density matrix in x and y
        R_x = (x + l_m/2) * m_m/l_m - 0.5
        R_z = (z + b_m/2) * p_m/b_m - 0.5
        
        
        #removes all values not within the densitymatrix.
        R_mask_x = R_x >= 0
        R_mask_x[R_x >= m_m -1] = False
        
        R_mask_z = R_z >= 0
        R_mask_z[R_z >= p_m -1] = False # '''Should this be strictly larger?'''
        
        #Makes sure that Rx and Rz is of the same dimensions
        R_mask_x = np.repeat(R_mask_x, p_c).reshape([n_m, m_c, p_c])
        R_mask_z = np.repeat(R_mask_z, m_c, axis = 0).reshape([n_m, m_c, p_c])
        R_x = np.repeat(R_x, p_c).reshape([n_m, m_c, p_c])
        R_z = np.repeat(R_z, m_c, axis = 0).reshape([n_m, m_c, p_c])
        
        #Makes a mask for all indicies within the dimensions of D
        R_mask = (R_mask_x*1 + R_mask_z*1)//2 == True
        
        R_x = R_x[R_mask]
        R_z = R_z[R_mask]
        
        #The vertical index relating the ray points to the density matrix
        R_y = np.repeat(np.arange(n_m), m_c * p_c).reshape([n_m, m_c, p_c])
        R_y = R_y[R_mask]
        
        #The material value at ever point
        R = np.zeros([n_m, m_c, p_c])
        R[R_mask] = D[R_y,(R_x//1).astype('int'),(R_z//1).astype('int')]*(2-R_x%1-R_z%1) \
                    + D[R_y,(R_x//1).astype('int')+1,(R_z//1).astype('int')+1]*(R_x%1+R_z%1)
        
        
        img = n_m - np.sum(R, axis=0)
    else: 
        img = np.zeros([moving_N, m_c, p_c])
        for i in range(moving_N):
             # A matrix representing the distance to the center every ray hits for a given layer in the density matrix
            x = np.tan(alpha_x) * y_x + moving_step_x*i - moving_step_x*moving_N/2
            z = np.tan(alpha_z) * y_z + moving_step_z*i - moving_step_z*moving_N/2
            
            #The horizontal subindex relating the ray points to the density matrix in x and y
            R_x = (x + l_m/2) * m_m/l_m - 0.5
            R_z = (z + b_m/2) * p_m/b_m - 0.5
            
            
            #removes all values not within the densitymatrix.
            R_mask_x = R_x >= 0
            R_mask_x[R_x >= m_m -1] = False
            
            R_mask_z = R_z >= 0
            R_mask_z[R_z >= p_m -1] = False # '''Should this be strictly larger?'''
            
            #Makes sure that Rx and Rz is of the same dimensions
            R_mask_x = np.repeat(R_mask_x, p_c).reshape([n_m, m_c, p_c])
            R_mask_z = np.repeat(R_mask_z, m_c, axis = 0).reshape([n_m, m_c, p_c])
            R_x = np.repeat(R_x, p_c).reshape([n_m, m_c, p_c])
            R_z = np.repeat(R_z, m_c, axis = 0).reshape([n_m, m_c, p_c])
            
            #Makes a mask for all indicies within the dimensions of D
            R_mask = (R_mask_x*1 + R_mask_z*1)//2 == True
            
            R_x = R_x[R_mask]
            R_z = R_z[R_mask]
            
            #The vertical index relating the ray points to the density matrix
            R_y = np.repeat(np.arange(n_m), m_c * p_c).reshape([n_m, m_c, p_c])
            R_y = R_y[R_mask]
            
            #The material value at ever point
            R = np.zeros([n_m, m_c, p_c])
            R[R_mask] = D[R_y,(R_x//1).astype('int'),(R_z//1).astype('int')]*(2-R_x%1-R_z%1) \
                        + D[R_y,(R_x//1).astype('int')+1,(R_z//1).astype('int')+1]*(R_x%1+R_z%1)
            

            img[i,:,:] = n_m - np.sum(R, axis=0)

        
    return img
#    plt.imshow(np.repeat(img, 100).reshape([len(img),100]))
    #plt.plot(img)