'''
Analyzes the Grad-CAM results for a given image and model.

author: @fuminides (Javier Fumanal Idocin)
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image

def relevant_area(image_numpy):
    '''
    Returns the relevant area of the image.
    '''
    # Get the image size
    height, width = image_numpy.shape
    avg_value = np.mean(image_numpy, axis=None)
    std_value = np.std(image_numpy, axis=None)

    return np.mean(image_numpy > np.maximum(avg_value, 0), axis=None)


def atention_focus(image_numpy):
    '''
    '''
    # vector_data = image_numpy.flatten()
    dx = ndimage.sobel(image_numpy, axis=0)
    dy = ndimage.sobel(image_numpy, axis=1)
    magnitude = np.hypot(dx, dy)

    return np.max(magnitude)
    

def relevant_parts(image_numpy):
    '''
    '''
    avg_value = np.mean(image_numpy, axis=None)

    # Partition the  image in 9 parts
    height, width = image_numpy.shape
    subdivisions = 6
    # Get the mean of each part
    mean_parts = np.zeros((subdivisions,subdivisions))

    for i in range(subdivisions):
        for j in range(subdivisions):
            mean_parts[i,j] = np.mean(image_numpy[i*height//subdivisions:(i+1)*height//subdivisions, j*width//subdivisions:(j+1)*width//subdivisions])
    
    min_mean_idx = mean_parts.flatten().argmin()
    min_mean = mean_parts.flatten()[min_mean_idx]

    relevant_areas = np.zeros((subdivisions,subdivisions))

    for i in range(subdivisions):
        for j in range(subdivisions):
            if mean_parts[i,j] > avg_value:
                relevant_areas[i, j] = 1
    
    _, n_components = ndimage.label(relevant_areas)
    return n_components

def sobel_filter(np_image):
    # Apply Sobel filter
    edges_x = ndimage.sobel(np_image, axis=0)
    edges_y = ndimage.sobel(np_image, axis=1)
    # Compute gradient magnitude
    gradient_magnitude = np.hypot(edges_x, edges_y)
    return gradient_magnitude

if __name__ == '__main__':
    # Load the Grad-CAM results
    import os
    os.chdir('/home/javierfumanal/Documents/GitHub/')
    images_path = './GradCams/'
    im_list = os.listdir(images_path)

    # Filter non csv files
    im_list = [im for im in im_list if im[-3:] == 'csv']

    number_of_images = len(im_list)
    results = pd.DataFrame(np.zeros((number_of_images, 4)), columns=['Image', 'Relevant_area', 'Atention_focus', 'Relevant_parts'])
    for ix, image in enumerate(im_list):
        image_name = image
        image_numpy = pd.read_csv(images_path + image_name).values
        results.iloc[ix, 0] = image_name
        results.iloc[ix, 1] = relevant_area(image_numpy)
        results.iloc[ix, 2] = atention_focus(image_numpy)
        results.iloc[ix, 3] = relevant_parts(image_numpy)
    
    results.to_csv('./context-art-classification/Infus_experiments_rules_data/GradCam_feats.csv')


