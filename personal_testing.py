from train import get_files, kNN
from PIL import Image

import numpy as np

def flatten_image(image: list[list[int]]) -> list:
    """
    Given a two dimensional, this function flattens the list.
    """
    return [pixel for sub in image for pixel in sub]

if __name__ == '__main__':
    """
    This is a simple script designed for personal testing. The code here opens the image turns it into a numpy array which is then
    converted back into python-native flattened list which is then used for predicting what digits the image could correspond to.
    """

    image = flatten_image(np.asarray(Image.open('./tests/personal.png'))[:, :, 0].tolist())
    training_images, training_labels, testing_images, testing_labels = get_files(600)

    prediction = kNN(training_images, training_labels, image)
    print(prediction)