import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split training & Test Data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(), 
                "SVC Classifier": SVC(),
                "GaussianNB": GaussianNB()
            }
            param={
                    "Decision Tree": {
                # "class_weight":["balanced"],
                "criterion":["gini", "entropy", "log_loss"],
                # "splitter":['best','random'],
                #"max_depth":[3,4,5,6,10],
                # "min_samples_split":[2,3,4,5],
                # "min_samples_leaf":[1,2,3],
                # "max_features":["auto","sqrt","log2"]
            },
                    "Random Forest Classifier":{
                # "class_weight":["balanced"],
                "n_estimators":[150,200],
                'max_depth': [10, 8, 5,20],
                # 'min_samples_split': [2, 5, 10],
            },
                    "Logistic Regression":{
                # "penalty":["l1", "l2", "elasticnet", None],
                "class_weight":["balanced"],
                # 'C': [0.001, 0.01, 0.1, 1, 10, 100],
                # "solver":["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
            },
                    "SVC Classifier": {
                        # 'C': [0.1, 1, 10, 100, 1000], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                # 'kernel': ['rbf']
                },
        
                    "GaussianNB":{"var_smoothing": [1e-9,0.1, 0.001, 0.5,0.05,0.01,1e-8,1e-7,1e-6,1e-10,1e-11]}
            }
            
            
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models,param)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]


            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model found on both training and testing dataset")

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            raise CustomException(e,sys)

