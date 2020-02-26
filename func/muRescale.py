#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:57:43 2020

@author: simonnyrup

Energy and X-Ray Mass Attenuation Coefficients match. 
Takes a table of a mass attenuation coefficients from a table and 
matches them to given energy values in a new table. 
This is done by linear interpolation.
"""

import numpy as np

def muRescale(mu, energies):
    """ mu is array containg energies in the first column and mu values in the second"""
    #Converts from MeV to KeV
    mu[:,0] = mu[:,0] *10e+3
    
    mu_new = np.zeros([len(energies),2])
    mu_new[:,0] = energies
    mu_new[:,1] = np.interp(energies, mu[:,0], mu[:,1])
    return mu_new

