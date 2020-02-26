#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:58:42 2020

@author: simonnyrup
"""
import numpy as np

def normalize3D(A, Amin = False):
    """Normalizes so  every array sums to 1 on the first axis of dim [N x M x K]"""
    # Rescales so every value between 0 and 1
    A_max = np.max(np.max(A, axis=1), axis=1)
    if Amin == False:
        A_min = np.min(np.min(A, axis=1), axis=1)
    A_min[A_min == A_max] = 0
    A = (A - A_min[:, None, None])/(A_max[:, None, None] - A_min[:, None, None])
    # Makes every []
#    A_sum = np.sum(np.sum(A, axis=1), axis=1)
#    A = A/A_sum[:, None, None]
    return A