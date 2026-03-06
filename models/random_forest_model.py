from sklearn.ensemble import RandomForestClassifier
import joblib


class RandomForestModel:

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, path="rf_model.pkl"):
        joblib.dump(self.model, path)

    def load_model(self, path="rf_model.pkl"):
        self.model = joblib.load(path)
