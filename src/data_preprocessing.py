import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            logger.info(f"Created directory for processed data at {self.processed_dir}")
    
    def preprocess_data(self, df):
        try:
            logger.info("Starting data preprocessing...")
            
            logger.info("Dropping the columns")
            df.drop(columns=["Unamed: 0", "Booking_ID"], inplace=True, errors='ignore')

            logger.info("Dropping duplicates")
            df.drop_duplicates(inplace=True)


            cat_cols = self.config['data_preprocessing']['categorical_columns']
            num_cols = self.config['data_preprocessing']['numerical_columns']

            logger.info("Label encoding")

            label_encoders = LabelEncoder()
            mappings = {}

            for col in cat_cols:
                df[col] = label_encoders.fit_transform(df[col])
                mappings[col] = {label:code for label, code in zip(label_encoders.classes_, label_encoders.transform(label_encoders.classes_))}
            logger.info(f"Label encoding mappings: {mappings}")

            logger.info("Label Mappings are: ")
            for col, mapping in mappings.items():
                logger.info(f"{col}: {mapping}")

            logger.info("Doing Skewness Handling")
            skewed_threshold = self.config['data_preprocessing'].get('skewed_threshold', 5)
            skewness = df[num_cols].apply(lambda x: x.skew())

            for column in skewness[skewness > skewed_threshold].index:
                df[column] = np.log1p(df[column]) 
                logger.info(f"Applied log transformation to {column} due to high skewness: {skewness[column]}")

            return df
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise CustomException(f"Error in data preprocessing: {str(e)}")
    
    def balance_data(self, df):
        try:
            logger.info("Handling inbalanced data")
            
            X = df.drop(columns="booking_status")
            y = df["booking_status"]

            smote = SMOTE(random_state=42)

            X_resampled, y_resampled = smote.fit_resample(X, y)

            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df["booking_status"] = y_resampled

            logger.info("Successfully balanced the dataset using SMOTE")

            return balanced_df
        except Exception as e:
            logger.error(f"Error in balancing data: {str(e)}")
            raise CustomException(f"Error in balancing data: {str(e)}")
    
    def select_features(self, df):
        try:
            logger.info("Starting feature selection using RandomForestClassifier")
            
            X = df.drop(columns="booking_status")
            y = df["booking_status"]

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            feature_importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importances
            })
            top_features_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            
            num_features_to_select = self.config['data_preprocessing'].get('no_of_features', 10)
            top_features = top_features_importance_df['Feature'].head(num_features_to_select).values

            top_features_df = df[top_features.tolist() + ['booking_status']]

            logger.info(f"Selected top {num_features_to_select} features based on importance")
            
            return top_features_df
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            raise CustomException(f"Error in feature selection: {str(e)}")

    def save_data(self, df, file_path):
        try:
            logger.info(f"Saving processed data to {file_path}") 
            df.to_csv(file_path, index=False)
            logger.info(f"Saved processed data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {str(e)}")
            raise CustomException(f"Error saving data to {file_path}: {str(e)}")
        
    def process(self):
        try:
            logger.info("Starting data preprocessing pipeline...")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            processed_train_df = self.preprocess_data(train_df)
            processed_test_df = self.preprocess_data(test_df)

            balanced_train_df = self.balance_data(processed_train_df)

            final_train_df = self.select_features(balanced_train_df)
            final_test_df = processed_test_df[final_train_df.columns]

            self.save_data(final_train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(final_test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data preprocessing completed successfully.")
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            raise CustomException(f"Error in data processing pipeline: {str(e)}")
        

if __name__ == "__main__":
    data_preprocessor = DataPreprocessor(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH
    )
    data_preprocessor.process()

    logger.info("Data preprocessing completed successfully.")