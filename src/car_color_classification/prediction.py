from .knn_classifier import knn_classifier
from .feature_extraction import color_histogram_of_test_image

def get_car_color_predictions(cars):
    predictions = [] # (car_id, predicted_label)

    for (car_id, car_img) in cars:
        color_histogram_of_test_image(car_img)
        prediction = knn_classifier('src/car_color_classification/Data/training.data', 'src/car_color_classification/Data/test.data')

        predictions.append((car_id, car_img, prediction))

    return predictions