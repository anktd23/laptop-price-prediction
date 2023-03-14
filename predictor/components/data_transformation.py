from predictor.entity import artifact_entity,config_entity
from predictor.exception import LapException
from predictor.logger import logging
from typing import Optional
import os,sys 
from sklearn.pipeline import Pipeline
import pandas as pd
from predictor import utils
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,StandardScaler
from predictor.config import TARGET_COLUMN,num_col,cat_col
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


class DataTransformation:


    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise LapException(e, sys)


    @classmethod
    def get_data_transformer_object_input(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0,add_indicator=True)
            robust_scaler =  RobustScaler(with_centering=False)
            ohe_enc = make_column_transformer((OneHotEncoder(handle_unknown='ignore'),cat_col), remainder='passthrough')
            pipeline_input = make_pipeline(ohe_enc,simple_imputer,robust_scaler)
            return pipeline_input
        except Exception as e:
            raise LapException(e, sys)

    @classmethod
    def get_data_transformer_object_target(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0,add_indicator=True)
            robust_scaler =  RobustScaler(with_centering=False)
            pipeline_target = Pipeline(steps=[
                    ('Imputer',simple_imputer),
                    ('robust_scaler',robust_scaler)
                ])
            return pipeline_target
        except Exception as e:
            raise LapException(e, sys)


    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            #reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            #selecting input feature for train and test dataframe
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)

            #selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            #fit train input features
            transformation_pipeline = DataTransformation.get_data_transformer_object_input()
            transformation_pipeline.fit(input_feature_train_df)

            #transforming input features
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            transformation_pipeline_target = DataTransformation.get_data_transformer_object_target()
            transformation_pipeline_target.fit(target_feature_train_df)

            target_feature_train_arr = transformation_pipeline_target.transform(target_feature_train_df)
            target_feature_test_arr =transformation_pipeline_target.transform(target_feature_test_df)

            #target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr ]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]


            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)


            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
             obj=transformation_pipeline)

            utils.save_object(file_path=self.data_transformation_config.transformed_target_path,
             obj=transformation_pipeline_target)



            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                transformed_target_path = self.data_transformation_config.transformed_target_path

            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise LapException(e, sys)

