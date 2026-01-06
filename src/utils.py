import os
import sys
import pickle

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import precision_score,accuracy_score,f1_score
from sklearn.model_selection import cross_val_score,StratifiedKFold
import numpy as np




def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,Y_train,X_test,Y_test,models):
    try:
        report ={}

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
            
            scores = np.mean(cross_val_score(
                model,
                X_train,
                Y_train,
                cv = cv,
                scoring ='f1'
            ))
            report[model_name] = scores;

        return report
        
    except Exception as e:
        raise CustomException(e,sys)
    


