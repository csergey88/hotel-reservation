import os
import pandas as pd
import yaml

from src.logger import logging
from src.custom_exception import CustomException


logger = logging.getLogger(__name__)

def read_yaml(path_to_yaml: str) -> dict:
    """Read yaml file and return dictionary"""
    try:
        if not os.path.exists(path_to_yaml):
            raise CustomException(f"YAML file not found: {path_to_yaml}")
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Successfully loaded YAML file: {path_to_yaml}")
            return content
    except Exception as e:
        logger.error(f"Error reading yaml file {path_to_yaml}: {str(e)}")
        raise CustomException(f"Error reading yaml file {path_to_yaml}: {str(e)}")
        

def load_data(path):
    try:
        logger.info(f"Loading data from {path}")
        if not os.path.exists(path):
            raise CustomException(f"Data file not found: {path}")
        data = pd.read_csv(path)
        logger.info(f"Data loaded successfully from {path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {path}: {str(e)}")
        raise CustomException(f"Error loading data from {path}: {str(e)}") 
    
    