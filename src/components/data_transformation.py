import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join("artifacts", 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation initiated.")
            #Define which column should be onehot encoded and which should be scaled.
            categorical_cols= ['sex', 'smoker', 'region']
            numerical_cols= ['age', 'bmi', 'children']

            
            logging.info("Pipeline Initiated.")

            #Numerical Pipeline
            num_pipeline= Pipeline(
                steps= [
                    ('scaler', StandardScaler())
                ]
                )

            #Categorical Pipeline
            cat_pipeline= Pipeline(
                steps= [
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )


            preprocessor= ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)]
            )

            return preprocessor

            logging.info("Pipeline Completed.")

        except Exception as e:
            logging.info("Error in Data Transformation.")
            raise CustomExeption(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            #Reading Train and Test Data
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("Read train and test data completed.")
            logging.info(f"Train DataFrame Head: \n{train_df.head().to_string()}")

            logging.info("Obtaining Preprocessing Object.")

            preprocessing_obj= self.get_data_transformation_object() 

            #Target Column
            target_column_name= 'expenses'

            input_feature_train_df= train_df.drop(columns= target_column_name, axis= 1)
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df= test_df.drop(columns= target_column_name, axis= 1)
            target_feature_test_df= test_df[target_column_name]

            #Transforming using preprocessing obj
            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )

            logging.info('Preprocessor Pickle file saved.')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,

            )
        except Exception as e:
            logging.info("Exception occurred in Initiated data transformation.")
            raise CustomException(e, sys)


