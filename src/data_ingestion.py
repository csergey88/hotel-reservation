import os
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml
from config.paths_config import *

load_dotenv()

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_ratio = self.config["train_ratio"]

        self.aws_access_key_id = os.getenv('MINIO_ACCESS_KEY')
        self.aws_secret_access_key = os.getenv('MINIO_SECRET_KEY')
        
        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(f"Created directory: {RAW_DIR}")
        logger.info(f"Date Ingeston started with bucket: {self.bucket_name} and file: {self.file_name}")
    
    def download_csv_from_s3(self):
        logger.info("Downloading CSV file from S3...")
        try:
            client = boto3.client(
                's3',
                region_name='us-east-1',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                endpoint_url='http://localhost:9000',
                use_ssl=False,
                verify=False
            )
            print(f"Client created: {client}")
            logger.info(f"Attempting to download {self.file_name} from bucket {self.bucket_name}")
            client.download_file(self.bucket_name, self.file_name, RAW_FILE_PATH)
            logger.info(f"Downloaded file {self.file_name} from bucket {self.bucket_name} to {RAW_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error downloading file from S3: {str(e)}")
            raise CustomException(f"Error downloading file from S3: {str(e)}")

    def split_data(self):
        logger.info("Splitting data into train and test sets...")
        # Add your data splitting logic here
        try:
            data = pd.read_csv(RAW_FILE_PATH)
            train_data, test_data = train_test_split(data, test_size=1-self.train_ratio, random_state=42)
            
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)
            
            logger.info(f"Split data into train ({TRAIN_FILE_PATH}) and test ({TEST_FILE_PATH}) sets")
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise CustomException(f"Error splitting data: {str(e)}")
    
    def run(self):
        try:
            logger.info("Starting data ingestion process...")
            self.download_csv_from_s3()
            self.split_data()
            logger.info("Data ingestion completed successfully")
        except Exception as e:
            logger.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(f"Error in data ingestion: {str(e)}")
        finally:
            logger.info("Data ingestion process finished")
    
if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)
    data_ingestion = DataIngestion(config)
    data_ingestion.run()