import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.

    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj (object): The object to be saved.

    Raises:
    - CustomException: If there is an error during saving the object.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate multiple regression models and return their performance metrics.

    Parameters:
    - X_train (np.ndarray): Training feature set.
    - y_train (np.ndarray): Training target variable.
    - X_test (np.ndarray): Testing feature set.
    - y_test (np.ndarray): Testing target variable.
    - models (dict): Dictionary of model names and their instances.

    Returns:
    - dict: A dictionary containing model names and their R-squared scores.
    """
    try:
        model_report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param_model = params[list(models.keys())[i]]

            grid_search = GridSearchCV(model, param_model, cv=3, verbose=0)
            grid_search.fit(X_train, y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            model_report[list(models.keys())[i]] = test_model_score
        return model_report

    except Exception as e:
        raise CustomException(e, sys) from e
    
def load_object(file_path):
    """
    Load an object from a file using pickle.

    Parameters:
    - file_path (str): The path from which the object will be loaded.

    Returns:
    - object: The loaded object.

    Raises:
    - CustomException: If there is an error during loading the object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e