import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class PriceForecaster:
    def __init__(self, data):
        self.data = data
        self.model_regressor = None
        self.model_classifier = None
        
    def prepare_data(self, target_column, features):
        """Prepare data for training"""
        X = self.data[features]
        y = self.data[target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_regression_model(self, X_train, y_train):
        """Train regression model for price forecasting"""
        self.model_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_regressor.fit(X_train, y_train)
        return self.model_regressor
    
    def train_classification_model(self, X_train, y_train):
        """Train classification model for price direction prediction"""
        self.model_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_classifier.fit(X_train, y_train)
        return self.model_classifier
    
    def predict_prices(self, X_test):
        """Make price predictions"""
        if self.model_regressor:
            return self.model_regressor.predict(X_test)
        else:
            raise ValueError("Model not trained yet!")
    
    def predict_direction(self, X_test):
        """Predict price direction (up/down)"""
        if self.model_classifier:
            return self.model_classifier.predict(X_test)
        else:
            raise ValueError("Classifier not trained yet!")

# Example usage
if __name__ == "__main__":
    # Load your data
    # data = pd.read_csv('your_price_data.csv')
    
    # Initialize forecaster
    # forecaster = PriceForecaster(data)
    
    # Prepare data
    # features = ['feature1', 'feature2', 'feature3']
    # target_regression = 'price'
    # target_classification = 'direction'
    
    # Split data
    # X_train, X_test, y_train_reg, y_test_reg = forecaster.prepare_data(target_regression, features)
    # _, _, y_train_clf, y_test_clf = forecaster.prepare_data(target_classification, features)
    
    # Train models
    # forecaster.train_regression_model(X_train, y_train_reg)
    # forecaster.train_classification_model(X_train, y_train_clf)
    
    # Make predictions
    # price_predictions = forecaster.predict_prices(X_test)
    # direction_predictions = forecaster.predict_direction(X_test)
    
    print("Price forecasting and prediction script ready!")
