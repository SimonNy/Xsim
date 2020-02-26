#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:58:42 2020

@author: simonnyrup
"""
#%%
import numpy as np

def normalize2D(A, A_max = True, A_min = True):
    """Normalizes so  every array sums to 1 on the first axis of dim [N x M x K]"""
    # Rescales so every value between 0 and 1
    if A_max == True:
        A_max = np.max(A)
    if A_min == True:
        A_min = np.min(A)

    A = (A - A_min)/(A_max - A_min)
    # Makes every []
#    A_sum = np.sum(np.sum(A, axis=1), axis=1)
#    A = A/A_sum[:, None, None]
    return A

# %%
