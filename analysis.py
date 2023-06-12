from train import *
from test import get_files, gather_data

import matplotlib.pyplot as plt

class Analysis:
    def __init__(self, initial_data: list[Images | Labels]):
        self.amount = len(initial_data[0])
        self.training_images, self.training_labels, self.testing_images, self.testing_labels = initial_data
    
    # returns average confidence of testing data over a certain amount of images
    def get_point(self, amount, distance_metric) -> int:
        data = gather_data(
            self.training_images[:amount],
            self.training_labels[:amount], 
            self.testing_images[:amount], 
            self.testing_labels[:amount],
            dist_metric=distance_metric
        )
        
        return sum(confidence for (*_, confidence) in data) / amount
    
    def show_graph(self) -> None:
        print(self.amount)

        # euclidean metric
        x1, y1 = list(zip(*(
            (amount, self.get_point(amount, euclidean_metric)) 
            for amount in range(5, self.amount, 5)
        )))

        # manhattan metric
        x2, y2 = list(zip(*(
            (amount, self.get_point(amount, manhattan_metric)) 
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