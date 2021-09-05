# credit: Juan Carlos Niebles and Ranjay Krishna

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io


def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    >> load('./Astana.jpg')
    True
    >> load('./KhanShatyr.jpg')
    True
    """
    out = None
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    # Use skimage io.imread
    out = io.imread(fname=image_path)
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    out = 0.5 * np.square(image/255)
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width)
    """
    out = None
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    out = 0.2989 * R + 0.5870 * G + 0.1140 * B
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return out


def rgb_decomposition(image, channel):
    """ Return image with the rgb channel specified
    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: string specifying the channel
    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    rgb = {'R': [1, 0, 0], 'G': [0, 1, 0], 'B': [0, 0, 1]}
    out = image[:, :, rgb[channel]]
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return out


def mix_images(image1, image2, channel1=False, channel2=False):
    """ Return image which is the left of image1 and right of image 2 including only
    the specified channels for each image
    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: string specifying channel used for image1
        channel2: string specifying channel used for image2
    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    if channel1:
        image1 = rgb_decomposition(image1, channel1)
    if channel2:
        image2 = rgb_decomposition(image2, channel2)
    out = np.zeros_like(image1)
    center = image1.shape[1] // 2
    out[:, :center, :] = image1[:, :center, :]
    out[:, center:, :] = image2[:, center:, :]
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return out
