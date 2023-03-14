from predictor.logger import logging
from predictor.exception import LapException
from predictor.utils import get_collection_as_dataframe
import sys,os
from predictor.entity import config_entity
from predictor.components.data_ingestion import DataIngestion
from predictor.components.data_validation import DataValidation
from predictor.components.data_transformation import DataTransformation



print(__name__)
if __name__=="__main__":
     try:
          training_pipeline_config = config_entity.TrainingPipelineConfig()
          data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
          print(data_ingestion_config.to_dict())
          data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
          data_ingestion_artifact= data_ingestion.initiate_data_ingestion()
          
          data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
          data_validation = DataValidation(data_validation_config=data_validation_config,
                         data_ingestion_artifact=data_ingestion_artifact)

          data_validation_artifact = data_validation.initiate_data_validation()

        
          data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
          data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                         data_ingestion_artifact=data_ingestion_artifact)

          data_transformation_artifact = data_transformation.initiate_data_transformation()
     except Exception as e:
          print(e)