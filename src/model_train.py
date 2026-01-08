import os
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.logger import logging
from src.custom_exception import CustomException
from config.model_params import LIGHTGBM_PARAMS, RANDON_SEARCH_PARAMS
from config.paths_config import PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH
from utils.common_functions import load_data, read_yaml
from scipy.stats import uniform, randint

import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_data_path = train_path
        self.test_data_path = test_path
        self.model_output_path = model_output_path

        self.param_distributions = LIGHTGBM_PARAMS
        self.search_params = RANDON_SEARCH_PARAMS   
   
    def load_and_split_data(self):
        try:
            train_df = load_data(self.train_data_path)
            test_df = load_data(self.test_data_path)

            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df["booking_status"]
            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df["booking_status"]

            logger.info("Data loaded and split successfully")

            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error in loading and splitting data: {str(e)}")
            raise CustomException(f"Error in loading and splitting data: {str(e)}")
        
    
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Starting model training...")
            lgbm_model = lgb.LGBMClassifier(random_state=self.search_params['random_state'])

            logger.info("Starting hyperparameter tuning using RandomizedSearchCV...")


            rand_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.param_distributions,
                n_iter=self.search_params['n_iter'],
                cv=self.search_params['cv'],
                verbose=self.search_params['verbose'],
                n_jobs=self.search_params['n_jobs'],
                random_state=self.search_params['random_state'],
                scoring=self.search_params['scoring']
            )

            logger.info("Model training started...")

            rand_search.fit(X_train, y_train)

            logger.info("Model training completed successfully")

            best_parameters = rand_search.best_params_
            best_lgbm_model = rand_search.best_estimator_

            logger.info(f"Best parameters found: {best_parameters}")
            
            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise CustomException(f"Error in model training: {str(e)}")
        
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model...")
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            logger.info(f"Model evaluation completed successfully")
            logger.info(f"Accuracy: {accuracy}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}")

            return {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            }
        
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise CustomException(f"Error in model evaluation: {str(e)}")
        
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved successfully at {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error in saving model: {str(e)}")
            raise CustomException(f"Error in saving model: {str(e)}")
        

    def run(self):
        try:
            mlflow.set_tracking_uri("http://localhost:8082")
            with mlflow.start_run():
                logging.info("Logging the training and dataset to MLflow")
                mlflow.log_artifact(self.train_data_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_data_path, artifact_path="datasets")
                mlflow.set_tag("model", "lgbm")
                logger.info("Running ModelTrainer...")
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(model, X_test, y_test)
                self.save_model(model)

                mlflow.log_artifact(self.model_output_path, artifact_path="model")
                mlflow.log_metrics(metrics)
                mlflow.log_params(model.get_params())
                return metrics
        except Exception as e:
            logger.error(f"Error in running ModelTrainer: {str(e)}")
            raise CustomException(f"Error in running ModelTrainer: {str(e)}")  
        
if __name__ == "__main__":
    trainer = ModelTrainer(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    trainer.run()
    
