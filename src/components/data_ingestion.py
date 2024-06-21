import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv') # Create a file and save to the folder 'artifacts' , with the name 'train.csv' for the file
    test_data_path: str=os.path.join('artifacts', 'test.csv') # Same as above
    raw_data_path: str=os.path.join('artifacts', 'data.csv') # Same as above

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv('notebook\Dataset\StudentsPerformance.csv')
           
            logging.info('Read the dataset into a dataframe') #This gives information on what is going on at this point, and also helps to know what line is giving an issue in the case an exception occurs.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path)exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)

            logging.info('Train-Test Split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2,random_state=93)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)

            logging.info('Ingestion of data is completed...')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
            )
        except Exception as e:
            raise CustomException(e,sys)
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion
