import os
import sys
import numpy as np
import pandas as pd
import pickle
from src.logger import logging 
from src.exception import CustomException
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#This function will save pickle file
def save_object(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report= {}
        for i in range(len(models)):
            model= list(models.values())[i]
            #Train model
            model.fit(X_train, y_train)

            #Predict the test data
            y_test_pred= model.predict(X_test)

            #Get R2 scores for train and test data
            # train_model_score= r2_score(y_train, y_train_pred)
            test_model_score= r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report 
    
    except Exception as e:
        logging.info("Exception ocurred during model training.")
        raise CustomException(e, sys)


