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
plt.style.use("bmh")

def produceXray(spec, N_points, Emin, Emax, Estep, makePlot = False):
    """Draws x-rays energies for a given distribution. 
    Spec: is the energy spectrum
    N_points: is the number of points generated
    Emin: The minimum energy sampled
    Emax: The maximum energy sampled
    Estep: The Energy step 

    Returns: Array of all energies, Count of photons with given energy
    """
    r = np.random
    #normalize 
    spec = spec / np.sum(spec)
    # upper limit for the sampling
    ymax = np.max(spec)
    # defines the bins for the historgram 
    # N_bins = len(np.arange(Emin, Emax, Estep))+1
    N_bins = int((Emax-Emin)/Estep)+1
    # Empty arrays to be filled with values
    xhit = np.zeros(N_points)
    yhit = np.zeros(N_points)
    
    for i in range(N_points):    
        while True:            
            # Sample an x value from the spectrum
            x = r.uniform(Emin, Emax) 
            x = int(x-Emax)
            
            # sample an y value between zero and the maximum specvalue
            y = r.uniform(0, ymax)       
            
            # if the values lies withing the spectrum save
            if (y < spec[x]):
                break
                
        xhit[i] = x
        yhit[i] = y
    # Scales to the right energies       
    xhit = xhit + Emax
    if makePlot == True:
        """Plotting part """
        # Playing with constants for Latex margin figures
        pc = 400/2409 # the pc unit relative to inchens
        goldenRatio = 1.618 # ratio between width and height
        marginWidth = 11.5 # width of latex margin document
        resize = 1/0.25 # scale=0.1 in latex

        fig, ax = plt.subplots(figsize=(marginWidth*pc*resize, 
                marginWidth*pc*resize/goldenRatio))
        counts, bin_edges, _ = ax.hist(xhit, bins=N_bins, range=(
                Emin-Estep/2, Emax+Estep/2), histtype='step', 
                label='Photon counts', linewidth=3)

        ax.plot(np.arange(len(spec))+Emin, spec/np.max(spec)
                * np.max(counts), linewidth=3)
        ax.set(xlabel='Photon energy [keV]', ylabel='Photon count')
        # ax.get_yaxis().set_visible(False)
        # ax.get_xaxis().set_visible(False)
        ax.tick_params(axis='x', labelsize=40)
        ax.tick_params(axis='y', labelsize=40)
        ax.yaxis.label.set_size(38)
        ax.xaxis.label.set_size(38)
        # add legend
        # ax.legend(loc='best', fontsize=16)
        # fig.tight_layout()
        fig.tight_layout()
        fig.savefig("hitNMissSpectrum.png", dpi=600)
        """ End of Plotting part """
    # counts the number of photons at each energy level
    counts, bin_edges = np.histogram(xhit, bins=N_bins, 
                    range=(Emin-Estep/2, Emax+Estep/2) )

    #The energies given from the distributions
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    return bin_centers, counts



# %%
