import os
import sys
import pickle

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import precision_score,accuracy_score,f1_score
from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV
import numpy as np




def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb")  as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_models(X_train,Y_train,X_test,Y_test,models,params):
    try:
        report ={}
        best_estimator ={}

        # for i in range(len(list(models))):
        #     model = list(models.values())[i]

           


        #     model.fit(X_train,Y_train)


        #     # Y_train_pred = model.predict(X_train)
        #     Y_test_pred = model.predict(X_test)

        #     # train_model_score = accuracy_score(Y_train,Y_train_pred)
        #     test_model_score = precision_score(Y_test,Y_test_pred)

        #     report[list(models.keys())[i]] = test_model_score

        cv = StratifiedKFold(n_splits=5,shuffle = True,random_state=42)
        for model_name, model in models.items():
            
            param_grid = params[model_name]

            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring="f1",
                n_jobs=-1
            )

            grid.fit(X_train, Y_train)

            best_score = grid.best_score_
            best_model = grid.best_estimator_

            report[model_name] = best_score
            best_estimator[model_name] = best_model

        return report,best_estimator
        
    except Exception as e:
        raise CustomException(e,sys)
    


