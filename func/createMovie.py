import numpy as np
import cv2 as cv

def createMovie(I_ccd, Nframes, grid_ccd, folder, filename):
    """ Creates a movie including all the frames in I_ccd """

    img_array = []

    for i in range(Nframes):
        #Converts a frame to BGR scale
        frame = np.tile(I_ccd[i, :, :], (3, 1, 1)).T.astype('uint8')
        img_array.append(frame)

    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')

    fps = 5
    out = cv.VideoWriter(folder+filename+'/'+filename+'.avi', fourcc, fps, tuple(grid_ccd))

    for frame in img_array:
        out.write(frame)
    out.release()