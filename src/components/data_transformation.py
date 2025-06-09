import sys
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['writing score', 'reading score']
            categorical_features = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]

            numerical_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            categorical_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', OneHotEncoder(sparse_output=False)),  # or sparse=False for older sklearn
                    ('scaler', StandardScaler())
                ]
            )

            logging.info('Numerical and categorical transformers created successfully')

            logging.info('Creating preprocessor object using ColumnTransformer')
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', numerical_transformer, numerical_features),
                    ('cat_pipeline', categorical_transformer, categorical_features)
                ]
            )

            logging.info('Preprocessor object created successfully')
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test dataframes successfully')

            preprocessing_obj = self.get_data_transformer_object()
            target_column = 'math score'
            numerical_features = ['writing score', 'reading score']

            input_features_train = train_df.drop(columns=[target_column, 'Total Score', 'Average Score'], axis=1)
            target_feature_train = train_df[target_column]

            input_features_test = test_df.drop(columns=[target_column, 'Total Score', 'Average Score'], axis=1)
            target_feature_test = test_df[target_column]

            logging.info('Input features and target features separated successfully')
            logging.info('Applying preprocessing object on training and testing dataframes')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test)
            ]
            logging.info('Preprocessing completed successfully')

            # Save the preprocessor object
            save_object (
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys) from e