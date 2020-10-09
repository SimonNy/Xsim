import numpy as np


def addNoise(img, sigma, type = "poisson")
    img_noise = img + np.random.normal(size=img.shape, scale=sigma)
