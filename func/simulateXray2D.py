#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:59:19 2020

@author: simonnyrup


X-ray simulator. Simulates the image generated on 1D image generated with x-rays 
emitted from a point source and penerating a material, represented by an 2D array.

inputs:
    d1: The distance from point source to the top of material
    h_m: The "real" height of the density matrix representing the material
    d2: The distance from the buttom of the material to the camera (CCD)
    
    l_m: The "real" length of the density matrix representing the material
    
    l_c: The length of the camera
    m_c: The number of inputs in the camera vector
    
    D: density matrix of size n_m X m_m
"""
import numpy as np


def simulateXray2D(d1, d2, h_m, l_m, l_c, m_c, D, moving_N = 0, moving_step = 0.01):
    n_m, m_m = D.shape
    # The total height of the system
    h = d1 + h_m + d2 
    
    #The distance to each center point of the CCD(centered in the middle)
    xc = (np.arange(m_c) - m_c/2 + 0.5) * l_c / (m_c) 
    # The angles between the center and the midpoint of a given input of the CCD 
    alpha = np.arctan(xc/h) 
    
    # The distance from the point source to the midpoint in a given layer of the density matrix
    y = (np.arange(n_m) + 0.5) * h_m / n_m + d1
    y = np.repeat(y, m_c).reshape([n_m, m_c])
    

    
    if moving_N == 0:
        # A matrix representing the distance to the center every ray hits for a given layer in the density matrix
        x = np.tan(alpha) * y
        
        #The horizontal subindex relating the ray points to the density matrix
        R_x  = (x + l_m/2) * m_m/l_m - 0.5
        #removes all values not within the densitymatrix.
        R_mask = R_x >= 0
        R_mask[R_x > m_m -1] = False
        R_x = R_x[R_mask]
        
        #The vertical index relating the ray points to the density matrix
        R_y = np.repeat(np.arange(n_m), m_c).reshape([n_m, m_c])
        R_y = R_y[R_mask]
        
        #The material value at ever point
        R = np.zeros(R_mask.shape)
        R[R_mask] = D[R_y,(R_x//1).astype('int')]*(1-R_x%1) + D[R_y,(R_x//1).astype('int')+1]*(R_x%1)
        
        img = n_m - np.sum(R, axis=0)
    else: 
        img = np.zeros([m_c, moving_N])
        for i in range(moving_N):
            """Moves the object, by moving the x coordinates relating the point source to the sensor"""
            # A matrix representing the distance to the center every ray hits for a given layer in the density matrix
            x = np.tan(alpha) * y + moving_step*i - moving_step*moving_N/2
            #The horizontal subindex relating the ray points to the density matrix

            R_x  = (x + l_m/2) * m_m/l_m - 0.5 

            #removes all values not within the densitymatrix.
            R_mask = R_x >= 0           
            R_mask[R_x >= m_m -1] = False 
            """ should it be strictly larger?!"""
            R_x = R_x[R_mask]
            #The vertical index relating the ray points to the density matrix
            R_y = np.repeat(np.arange(n_m), m_c).reshape([n_m, m_c])
            R_y = R_y[R_mask]
            
            #The material value at ever point
            R = np.zeros(R_mask.shape)
            R[R_mask] = D[R_y,(R_x//1).astype('int')]*(1-R_x%1) + D[R_y,(R_x//1).astype('int')+1]*(R_x%1)

            img[:,i] = n_m - np.sum(R, axis=0)
            
        
    return img
#    plt.imshow(np.repeat(img, 100).reshape([len(img),100]))
    #plt.plot(img)