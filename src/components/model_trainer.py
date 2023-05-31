import os 
import sys
import pickle
import numpy as np
import pandas as pd
from src.exception import CustomException 
from src.logger import logging 
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor 
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from src.utils import save_object, evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    train_model_file_path= os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting Indepenedent an dependent variable from train and test dataset.")
            X_train, y_train, X_test, y_test= (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )

            #Train the multiple models
            models= {
                "Linear Regression": LinearRegression(),
                'Ridge': Ridge(),
                "Lasso": Lasso(),
                "KNieghborsRegressor": KNeighborsRegressor(),
                "DecisionTree Regressor": DecisionTreeRegressor(),
                "RandomForest Regressor": RandomForestRegressor(),
                "XGB Regressor": XGBRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor()
            }

            
            model_report:dict= evaluate_model(X_train, y_train, X_test, y_test,models)
            print(model_report)
            print("\n=\n"*5)
            logging.info(f"Model Report: {model_report}")

            #To get best model score from dictionary 
            best_model_score= max(sorted(model_report.values()))

            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model= models[best_model_name]
            
            print(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")
            print("\n=\n"*5)
            logging.info(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")

            save_object(
                file_path= self.model_trainer_config.train_model_file_path,
                obj= best_model
            )



        except Exception as e:
            logging.info("Exception occured at Model Training")
            raise CustomException(e, sys)


