#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:22:47 2020

@author: simonnyrup

A function that pulls x-ray energies from a given probability distribution
"""
#%%
import numpy as np
import matplotlib.pyplot as plt


def produceXray(spec, N_points, Emin, Emax, Estep):
    """Draws x-rays energies for a given distribution. 
    Spec: is the energy spectrum
    N_points: is the number of points generated
    Emin: The minimum energy wanted
    Emax: The maximum energy wanted
    Estep: The Energy step 

    Returns: Array of all energies, Count of photons with given energy
    """
    r = np.random
#    r.seed(42) # If seed is needed
    
    # number of random points generated
#    N_points = 1000 
    
    #normalize the array 
    spec = spec / np.sum(spec)
    
#    Emin, Emax = 10, 80 
#    Estep = 5
    ymax = np.max(spec)
    
    N_bins = len(np.arange(Emin, Emax, Estep))+1
    
    N_try = 0
    xhit = np.zeros(N_points)
    yhit = np.zeros(N_points)
    
    for i in range(N_points):
        
        while True:
            
            # Count the number of tries, to get efficiency/integral
            N_try += 1                    
            
            # Range that f(x) is defined/wanted in
            x = r.uniform(Emin, Emax) 
            x = int(x-Emax)
            
            # Upper bound for function values (a better bound exists!)
            y = r.uniform(0, ymax)       
            
            if (y < spec[x]):
                break
                
        xhit[i] = x
        yhit[i] = y
    # Scales to the right energies       
    xhit = xhit + Emax
#    fig4, ax4 = plt.subplots(figsize=(10, 6))
#    ax4.plot(np.arange(len(spec))+Emax,spec)
#    ax4.scatter(xhit, yhit, s=1, label='Scatter plot of data')
    
#    fig, ax = plt.subplots(figsize=(10, 6))
    counts, bin_edges = np.histogram(xhit, bins=N_bins, range=(Emin-Estep/2, Emax+Estep/2) )
#    counts, bin_edges, _  = ax.hist(xhit, bins=N_bins, range=(Emin-Estep/2, Emax+Estep/2), histtype='step', label='histogram' )
#    ax.set(xlabel="x (f(x) distributed)", ylabel=f"Frequency/{2/N_bins}", xlim=(Emin, Emax));
    
    #The energies given from the distributions
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    return bin_centers, counts
#    s_counts = np.sqrt(counts)


# %%
