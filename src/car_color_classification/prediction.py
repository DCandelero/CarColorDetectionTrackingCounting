from car_color_classification import knn_classifier
from car_color_classification import feature_extraction

def get_car_color_predictions(cars):
    predictions = [] # (car_id, predicted_label)

    for (car_id, car_img) in cars:
        feature_extraction.color_histogram_of_test_image(car_img)
        prediction = knn_classifier.knn_classifier('./car_color_classification/Data/training.data', './car_color_classification/Data/test.data')

        predictions.append((car_id, car_img, prediction))

    return predictions