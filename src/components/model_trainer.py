import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Training data and testing data split completed")
            logging.info("Initializing model training")
            models = {
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'Linear Regression': LinearRegression(),
                'DecisionTree Regressor': DecisionTreeRegressor(),
                'RandomForest Regressor': RandomForestRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'GradientBoosting Regressor': GradientBoostingRegressor(),
                'CatBoost Regressor': CatBoostRegressor(verbose=0),
                'XGB Regressor': XGBRegressor(eval_metric='rmse')
            }

            params = {
                'DecisionTree Regressor': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                'RandomForest Regressor': {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                'GradientBoosting Regressor': {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [0.01, 0.1, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
                },
                'Linear Regression': {},
                'K-Neighbors Regressor': {
                    'n_neighbors': [3, 5, 7, 9, 11]
                },
                'XGB Regressor': {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [0.01, 0.1, 0.05, 0.001]
                },
                'CatBoost Regressor': {
                    'iterations': [30, 50, 100],
                    'learning_rate': [0.01, 0.1, 0.05],
                    'depth': [6, 8, 10]
                },
                'AdaBoost Regressor': {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [0.01, 0.1, 0.5]
                }
            }
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test, models=models, params=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            
            logging.info("Saving the best model")
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            logging.info("Model training completed successfully")
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square
        except Exception as e:
            raise CustomException(e, sys)