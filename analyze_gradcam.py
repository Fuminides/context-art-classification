'''
Analyzes the Grad-CAM results for a given image and model.

author: @fuminides (Javier Fumanal Idocin)
'''

import numpy as np
import scipy.stats.powerlaw as powerlaw


def relevant_area(image_numpy):
    '''
    Returns the relevant area of the image.
    '''
    # Get the image size
    height, width = image_numpy.shape


    return np.mean(image_numpy > np.min(image_numpy, axis=-1), axis=-1)


def atention_focus(image_numpy):
    '''
    '''
    vector_data = image_numpy.flatten()

    return np.max(np.diff(sorted(vector_data)))
    

def relevant_parts(image_numpy):
    '''
    '''
    # Partition the  image in 9 parts
    height, width = image_numpy.shape

    # Get the mean of each part
    mean_parts = np.zeros((3,3))
    std_parts = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            mean_parts[i,j] = np.mean(image_numpy[i*height//3:(i+1)*height//3, j*width//3:(j+1)*width//3])
            std_parts[i,j] = np.std(image_numpy[i*height//3:(i+1)*height//3, j*width//3:(j+1)*width//3])
    
    min_mean_idx = mean_parts.flatten().argmin()
    min_mean = mean_parts.flatten()[min_mean_idx]
    min_std = std_parts.flatten()[min_mean_idx]

    
    relevant_areas = 0
    for i in range(3):
        for j in range(3):
            if mean_parts[i,j] > min_mean + min_std:
                relevant_areas += 1

    return relevant_areas

