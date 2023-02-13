from predictor.logger import logging
from predictor.exception import SensorException
from predictor import get_collection_as_dataframe
import sys,os
from predictor.entity import config_entity
from predictor.components.data_ingestion import DataIngestion



if __name__=="__main__":
     try:
          training_pipeline_config = config_entity.TrainingPipelineConfig()
          data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
          print(data_ingestion_config.to_dict())
          data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
          print(data_ingestion.initiate_data_ingestion())
     except Exception as e:
          print(e)