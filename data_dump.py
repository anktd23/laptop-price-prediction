import pandas as pd
import pymongo
import os
import json
from predictor.config import mongo_client

DATA_FILE_PATH = "F:\Project\laptop-price-prediction\cleaned_data.csv"
DATABASE_NAME = "laptop_data"
COLLECTION_NAME = "laptop"

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns:{df.shape}")
    #convert dataframe to JSON so that we can dump this records in MongoDB.
    df.reset_index(drop=True,inplace=True)
    json_record = list((json.loads(df.T.to_json()).values()))
    print(json_record[0])

    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
