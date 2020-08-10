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

    """Plotting part """
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(np.arange(len(spec))+Emax,spec)
    # ax.scatter(xhit, yhit, s=1, label='Scatter plot of data')
    
    # Playing with constants for Latex margin figures
    pc = 400/2409 # the pc unit relative to inchens
    goldenRatio = 1.618 # ratio between width and height
    marginWidth = 11.5 # width of latex margin document
    resize = 10 # scale=0.1 in latex

    fig, ax = plt.subplots(figsize=(marginWidth*pc*resize, marginWidth*pc*resize/goldenRatio))
    counts, bin_edges, _ = ax.hist(xhit, bins=N_bins, range=(
        Emin-Estep/2, Emax+Estep/2), histtype='step', label='Photon counts', linewidth=3)
    ax.plot(np.arange(len(spec))+Emin, spec/np.max(spec)
            * np.max(counts), linewidth=3)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    # ax.yaxis.label.set_size(18)
    # ax.xaxis.label.set_size(18)
    # add legend
    # ax.legend(loc='best', fontsize=16)
    # fig.tight_layout()

    fig.savefig("hitNMissSpectrum.pdf", dpi=600)
    """ End of Plotting part """
    
    
    counts, bin_edges = np.histogram(xhit, bins=N_bins, range=(Emin-Estep/2, Emax+Estep/2) )

    #The energies given from the distributions
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    return bin_centers, counts
#    s_counts = np.sqrt(counts)


# %%
