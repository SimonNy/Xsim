""" User-defined settings for the system 
    All distance and sizes are in units of cm
"""
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

# conversion from cm2m all distances written in m instead
cm2m = 100

""" Iterates over the same object, with different system settings """
d1List = [1, 1, 1]
d2List = [1, 1, 1]
voxelSizeList = [5e-5, 5e-5, 5e-5]
pixelAreaList = [7e-5, 7e-5, 7e-5]
gridCCDListX = [400, 400, 400]
gridCCDListY = [400, 400, 400]
NsubpixelsList = [2, 2, 2]
NframesList = [1, 1, 1]
NsubframesList = [1, 10, 10]
photonsRayList = [3000, 3000, 3000]
moveStepListX = [0, 1, 0]
moveStepListY = [0, 0, 1]
# Images the same object in different defined systems
for i in range(len(d1List)):

    # distance from point souce to the bottom of the material
    d1 = d1List[i]*cm2m
    # distance from the bottom of the material to the CCD array
    d2 = d2List[i]*cm2m
    #Information about the material 
    voxelSizeD = voxelSizeList[i]*cm2m  # volume of voxel in material
    #Information about the size of the camera

    pixelArea = pixelAreaList[i]*cm2m
    #number of ccds in the camera as [n x p]
    #If asymmetric, first input should be largest
    grid_ccd = (gridCCDListX[i], gridCCDListY[i])
    #The "resolution" of each CCD. Makes more rays hit the same CCD
    Nsubpixels = NsubpixelsList[i]

    #Number of Photons in a single pixel direction
    photonsRayRef = photonsRayList[i]
    # Energies in the sampled spectrum
    Emin = 10
    Emax = 80
    Estep = 1

    # The total of Image frames
    Nframes = NframesList[i]
    # The amount of iterations made to create one frame.
    # Generating Nsubframes frames for every frame. Can be seen as integration time
    Nsubframes = NsubframesList[i]
    photonsRay = int(photonsRayRef/Nsubframes)
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
    moveStep = (moveStepListX[i], moveStepListY[i])
    # The start position of the object, 0 is centered
    # moveStart = (0, 0)
    # Makes the total movement symmetrical across the center
    moveStart = np.multiply(int(-Nframes*Nsubframes/2), moveStep)

    # Saves the FOV for multiple use
    keepFOV = True
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
    # Uses the ref here as incresed number of subframes reduces the amount
    simpleSystem["photonsRay"] = photonsRayRef

    if Nframes != 1 or Nsubframes != 1:
        runXsim(filename + f"{i}", D, spectrum, **simpleSystem)
    runXsim(filename + f"{i}", D, spectrum, **systemParameters)


