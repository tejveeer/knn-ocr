from train import *

Predictions = list[Label]
Actual = Label
Confidence = float
Data = list[tuple[Predictions, Actual, Confidence]]

def gather_data(training_images: Images, 
                training_labels: Labels, 
                testing_images: Images, 
                testing_labels: Labels,
                **kwargs) -> Data:
    """
    Given the testing and training data, this function gathers the results, actual digit in the image as well as the
    algorithm's confidence in the prediction.
    """
    return [
        ((result := kNN(training_images, training_labels, testing_image, **kwargs)),
         label,
         100 * result.count(label) / len(result)) # confidence
        
        for (label, testing_image) in zip(testing_labels, testing_images)
    ]

def show_statistics(data: Data) -> None:
    """
    Given the data gathered from the gathering function, this function presents the statistics.
    """

    average_confidence = sum(confidence for (*_, confidence) in data) / len(data)
    correct_predictions = 100 * sum(1 for (*_, confidence) in data if confidence > 0) / len(data)
    wrong_predictions = 100 * sum(1 for (*_, confidence) in data if confidence == 0) / len(data)

    print(f"Average Confidence: {round(average_confidence)}%\nCorrect Predictions: {round(correct_predictions)}%\nWrong Predictions: {round(wrong_predictions)}%\n")

    print("Actual Number -> Predictions [P1, P2, P3] | Confidence")
    for (predictions, actual_number, confidence) in data:
        print(f"{actual_number} -> {predictions} | {round(confidence)}%")

if __name__ == '__main__':
    training_images, training_labels, testing_images, testing_labels = get_files(10_000)

    data = gather_data(training_images, training_labels, testing_images, testing_labels)
    show_statistics(data)