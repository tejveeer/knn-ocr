from PIL import Image
import numpy as np

from train import kNN, get_files

def flatten_image(image) -> list:
    return [pixel for sub in image for pixel in sub]

image = flatten_image(np.asarray(Image.open('./personal.png'))[:, :, 0].tolist())
training_images, training_labels, testing_images, testing_labels = get_files(600)

prediction = kNN(training_images, training_labels, image)
print(prediction)