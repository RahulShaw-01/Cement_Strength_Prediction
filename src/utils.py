import os
import sys
import pickle
import numpy as np 
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score,mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluation_model(model_name, mse, r2):
    logging.info(f"Evaluation Metrics for {model_name}:")
    logging.info(f"Mean Squared Error (MSE): {mse:.4f}")
    logging.info(f"R2 Score: {r2:.4f}")
    print(f"Model: {model_name} | MSE: {mse:.4f} | R2 Score: {r2:.4f}")


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info('Exception occured in load_objcer function utils')
        raise CustomException(e, sys)