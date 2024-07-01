import os
import sys
import pandas as pd 

from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


class CustomData:
    def __init__(self,
                  gender: str,
                  race_ethnicity: str,
                  parental_level_of_education: str,
                  lunch: str,
                  test_preparation_course: str,
                  reading_score: int,
                  writing_score: int
                  ):
        self.gender = self.check_missing(gender, "Unknown")
        self.race_ethnicity = self.check_missing(race_ethnicity, "Unknown")
        self.parental_level_of_education = self.check_missing(parental_level_of_education, "Unknown")
        self.lunch = self.check_missing(lunch, "Unknown")
        self.test_preparation_course = self.check_missing(test_preparation_course, "Unknown")
        self.reading_score = self.check_missing(reading_score, 0)
        self.writing_score = self.check_missing(writing_score, 0)
        
    def check_missing(self, value, default):
        if value is None or value == "":
            return default
        return value

    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],  
                "parental level of education": [self.parental_level_of_education],  
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],  
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(str(e))  # Only pass the error message
        logging.info('Input data is converted to pandas dataframe')