
from train import *
from test import get_files, gather_data

import matplotlib.pyplot as plt

class Analysis:
    def __init__(self, initial_data: list[Images | Labels]):
        self.amount = len(initial_data[0])
        self.training_images, self.training_labels, self.testing_images, self.testing_labels = initial_data
    
    def get_average_confidence(self, amount, distance_metric) -> int:
        """
        Given amount of images to collect the data from and the distance metric, this function returns the average
        confidence of the algorithm.
        """
        data = gather_data(
            self.training_images[:amount],
            self.training_labels[:amount], 
            self.testing_images[:amount], 
            self.testing_labels[:amount],
            dist_metric=distance_metric
        )
        
        return sum(confidence for (*_, confidence) in data) / amount
    
    def show_graph(self) -> None:
        """
        This function shows a graph of the confidence over an increase in images for the Euclidean and Manhattan metric.
        """

        # euclidean metric
        x1, y1 = list(zip(*(
            (amount, self.get_average_confidence(amount, euclidean_metric)) 
            for amount in range(5, self.amount, 5)
        )))

        # manhattan metric
        x2, y2 = list(zip(*(
            (amount, self.get_average_confidence(amount, manhattan_metric)) 
            for amount in range(5, self.amount, 5)
        )))

        plt.plot(x1, y1, label='euclidean')
        plt.plot(x2, y2, label='manhattan')

        plt.xlabel('Image Amount')
        plt.ylabel('Average Confidence')

        plt.legend()
        plt.show()

analyzer = Analysis(get_files(200))
analyzer.show_graph()