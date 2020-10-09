# Python3 code for generating points on a 3-D line  
# using Bresenham's Algorithm
""" Taken from https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/""" 
""" Modified to only take z as the driving axis which is the direction from
    source to camera
    The biggest difference usually defines the driving axis, so the rays 
    should not travel longer than the height of the FOV in the any direction
"""
import numpy as np
def Bresenham3D(x1, y1, x2, y2, fov_height): 
    dx = abs(x2 - x1) 
    dy = abs(y2 - y1) 
    dz = fov_height
    z1 = 0
    ListOfPoints = np.zeros((2, dz))
    if (x2 > x1): 
        xs = 1
    else: 
        xs = -1
    if (y2 > y1): 
        ys = 1
    else: 
        ys = -1

    # Driving axis is Z-axis" 
    p1 = 2 * dy - dz 
    p2 = 2 * dx - dz 
    # while (z1 != z2):
    for i in range(dz): 
        z1 += 1
        if (p1 >= 0): 
            y1 += ys 
            p1 -= 2 * dz 
        if (p2 >= 0): 
            x1 += xs 
            p2 -= 2 * dz 
        p1 += 2 * dy 
        p2 += 2 * dx 
        ListOfPoints[:, i] = (x1, y1) 
    return ListOfPoints
  
  
# def main(): 
#    (x1, y1) = (-1, 1) 
#    (x2, y2) = (5, 3) 
#    fov_height = 10
#    ListOfPoints = Bresenham3D(x1, y1, x2, y2, fov_height) 
#    print(ListOfPoints) 
 
# main() 