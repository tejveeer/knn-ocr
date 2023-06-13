# kNN OCR Summative
The purpose of this summative project was to implement k-Nearest Neighbors to images to determine the digit an image represents. The algorithm was trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) as other datasets had images of varying sizes which would require unnecessary image manipulation.

## Installation
Use `pip install -r requirements.txt` to install the necessary packages for running the files.

## Files
| Filename | Description |
| -------- | ----------- |
| `train.py` | This file includes the implementation of the kNN algorithm. |
| `test.py` | This file includes the unit tests for the kNN algorithm. You can change the `get_files` function to increase the training data and change the distance metric for the `gather_data` function as well.|
| `personal_testing.py` | You can use paint to modify `personal.png` to draw any digit between 0 - 9 and run this file to see what digit the algorithm predicts it to be |
| `analysis.py` | This file makes an image amount vs. confidence graph to compare which of the distance metrics provides better confidence |
