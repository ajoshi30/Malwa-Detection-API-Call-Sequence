import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.random_forest_model import RandomForestModel


def load_dataset():

    data = np.load("dataset.npz")

    X = data["X"]
    y = data["y"]

    return X, y


def train_random_forest():

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestModel()

    model.train(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)

    print("Random Forest Accuracy:", acc)

    model.save_model()


if __name__ == "__main__":

    train_random_forest()
