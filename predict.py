import numpy as np
from models.random_forest_model import RandomForestModel


def predict_sample(sample):

    model = RandomForestModel()

    model.load_model("rf_model.pkl")

    prediction = model.predict(sample)

    return prediction


if __name__ == "__main__":

    sample = np.random.rand(1, 100)

    result = predict_sample(sample)

    print("Prediction:", result)
