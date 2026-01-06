import os
import sys

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
) 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score,accuracy_score,classification_report







@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and test input data")

            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                # "Decision Tree": DecisionTreeClassifier(random_state = 42),
                # "Gradient Boosting": GradientBoostingClassifier(),
                # "Logistic Regression": LogisticRegression(),
                # "XGBClassifier":XGBClassifier(),
                # "AdaBoost classifier": AdaBoostClassifier(),
            }
            params ={
                "Decision Tree":{
                    'criterion':['gini', 'entropy', 'log_loss'],
                    'splitter':['best','random'],
                    'max_depth':[3]

                },
                "Random Forest" :{
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                    "bootstrap": [True]

                }
            }

            model_report:dict = evaluate_models(X_train = X_train,Y_train = Y_train,X_test = X_test,Y_test = Y_test,models = models,params= params)

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6 :
                raise CustomException("No best model found")
            
            logging.info(f"best model found on both training and testing dataset")

            save_object(
                file_path = self.model_trainer.trained_model_file_path,
                obj = best_model
            )

            logging.info(f"saved as .pkl of model.pkl")

            best_model.fit(X_train,Y_train)

            predicted = best_model.predict(X_test)

            # p_score = precision_score(Y_test,predicted)

            print(classification_report(
                Y_test, predicted, target_names=["Human", "Bot"]
            ))

            # return p_score
        
        except Exception as e:
            raise CustomException(e,sys)
        









