from skimage.feature import graycomatrix, graycoprops
import numpy as np

def extract_glcm_features(image):

    glcm = graycomatrix(
        image,
        distances=[1,2],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    features = []

    properties = [
        'contrast',
        'energy',
        'homogeneity',
        'correlation',
        'ASM',
        'dissimilarity'
    ]

    for prop in properties:
        values = graycoprops(glcm, prop)
        features.extend(values.flatten())

    return features