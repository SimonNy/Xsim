
from xraySpec.dataA import arr80
from runXsim import runXsim
import numpy as np
from loadObject import loadObject

# loads the object defineed in the loadObject file

filename = "bunny"
# filename = "sphereWithSphere"
# filename = "simplePotato"
# filename = "potatoWithHeart"
# filename = "boardWithMarks"
D = loadObject(filename)


""" User-defined settings for the system 
    All distance and sizes are in units of cm
"""
# conversion from cm2m all distances written in m instead
cm2m = 100
# distance from point souce to the bottom of the material
d1 = 1*cm2m
# distance from the bottom of the material to the CCD array
d2 = 1*cm2m
#Information about the material 
voxelSizeD = 0.5e-4*cm2m  # volume of voxel in material
#Information about the size of the camera

pixelArea = 7e-5*cm2m
#number of ccds in the camera as [n x p]
#If asymmetric, first input should be largest
grid_ccd = (400, 400)
#The "resolution" of each CCD. Makes more rays hit the same CCD
Nsubpixels = 2

#Number of Photons in a single pixel direction
photonsRay = 4500
# Energies in the sampled spectrum
Emin = 10
Emax = 80
Estep = 1

# The total of Image frames
Nframes = 1
# The amount of iterations made to create one frame.
# Generating Nsubframes frames for every frame. Can be seen as integration time
Nsubframes = 1
photonsRay = int(photonsRay/Nsubframes)
# Should i do this instead ? scales with the amount of rays in pixel
# Maybe influences the distribution to much?
# photonsRay = int(photonsRay/(Nsubframes*Nsubpixels**2))

#The Thresholdvalue(Amount of photons maximally counted) for a Single CCD
saturation = 4000*Nsubpixels**2
# saturation = photonsRay*Nsubpixels**2*Nsubframes
# saturation = 5000*Nsubpixels**2*Nsubframes

#The spectrum for the energies
spectrum = np.asarray(arr80)
spectrum = spectrum[9:-1]


# Movement in a single subframe in x and z dire
moveStep = (0, 0)
# The start position of the object, 0 is centered
# moveStart = (0, 0)
# Makes the total movement symmetrical across the center
moveStart = np.multiply(int(-Nframes*Nsubframes/2), moveStep)

# Saves the FOV for multiple use
keepFOV = False
# Saves the output images in the dedicated folders
saveImgs = True

size_ccd_pixel = (pixelArea, pixelArea)
size_v_D = (voxelSizeD, voxelSizeD, voxelSizeD)

systemParameters = {
    "d1": d1,
    "d2": d2,
    "size_v_D": (voxelSizeD, voxelSizeD, voxelSizeD),
    "size_ccd_pixel": (pixelArea, pixelArea),
    "grid_ccd": grid_ccd,
    "Nsubpixels": Nsubpixels,
    "photonsRay": photonsRay,
    # "spectrum": spectrum,
    "Emin": Emin,
    "Emax": Emax,
    "Estep": Estep,
    "saturation": saturation,
    "Nframes": Nframes,
    "Nsubframes": Nsubframes,
    "moveStep": moveStep,
    "moveStart": moveStart,
    "imgName": "",
    "keepFOV": keepFOV,
    "saveImgs": saveImgs,
    }

# Takes a single image with no translations and movement, for reference
simpleSystem = systemParameters.copy()
simpleSystem["Nframes"] = 1
simpleSystem["Nsubframes"] = 1
simpleSystem["moveStep"] = (0,0)
simpleSystem["moveStart"] = (0,0)
simpleSystem["imgName"] = "ref"

if Nframes != 1 or Nsubframes != 1:
    runXsim(filename, D, spectrum, **simpleSystem)
runXsim(filename, D, spectrum, **systemParameters)
