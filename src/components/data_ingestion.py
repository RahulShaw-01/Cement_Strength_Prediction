import os
import sys
import logging
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


## initialize the data ingestion configuration
@dataclass
class DataIngestionconfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')


## create a data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            df = pd.read_csv(os.path.join('notebooks', 'concrete_data.csv'))
            logging.info('Dataset read as pandas Dataframe')

            initial_shape = df.shape
            df = df.drop_duplicates(keep='first', ignore_index=True)
            logging.info(f"Removed duplicates: {initial_shape[0] - df.shape[0]} rows dropped")

            # Handle outliers using IQR method (added part)
            def handle_outliers_iqr(df, columns):
                for col in columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    upper_bound_cast = df[col].dtype.type(upper_bound)
                    lower_bound_cast = df[col].dtype.type(lower_bound)
                    df.loc[df[col] < lower_bound, col] = lower_bound_cast
                    df.loc[df[col] > upper_bound, col] = upper_bound_cast
                return df

            numerical_cols = ['cement', 'blast_furnace_slag', 'fly_ash', 'water',
                              'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age']

            # Strip any whitespace from column names to avoid key errors
            df.columns = df.columns.str.strip()

            # Check columns exist
            missing_cols = [col for col in numerical_cols if col not in df.columns]
            if missing_cols:
                raise Exception(f"Missing columns in dataset for outlier handling: {missing_cols}")

            # Outlier handling call
            df = handle_outliers_iqr(df, numerical_cols)

            # Continue with rest of your code unchanged
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Train test split")
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Error occurred in Data Ingestion config')
            raise CustomException(e, sys)
