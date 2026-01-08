import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,feature):
        # print("TYPE:", type(feature))
        # print(feature)

        print("predict func called")

        


        try:

            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
           

            model = load_object(file_path = model_path)
            print("MODEL LOADED")
            preprocessor = load_object(file_path = preprocessor_path)
            print("PREPROCESSOR LOADED")

            print("INPUT COLUMNS:", feature.columns.tolist())
            print("INPUT DTYPES:\n", feature.dtypes)

            print("EXPECTED COLUMNS:", preprocessor.feature_names_in_)
          

            

            # transfromed_data = preprocessor.transform(feature)
            try:
                transfromed_data = preprocessor.transform(feature)
                print("TRANSFORM SUCCESS")
                print("TRANSFORM SHAPE:", transfromed_data.shape)
            except Exception as e:
                print("TRANSFORM FAILED:", e)
                raise

            try:
                preds = model.predict(transfromed_data)
                print("PREDICTION:", preds)
            except Exception as e:
                print("prediction FAILED:", e)
                raise




            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:

    def __init__(self,
        mousePointsCount,
        avgSpeed,speedStdDev,
        accelStdDev,
        jitterCount,
        angleStd,
        diffScore,
        hoverTime,
        hesitation,
        clickLatency,
        clickOffset,
        isChecked):

        self.mousePointsCount = mousePointsCount
        self.avgSpeed = avgSpeed
        self.speedStdDev = speedStdDev
        self.accelStdDev = accelStdDev
        self.jitterCount = jitterCount
        self.angleStd = angleStd
        self.diffScore = diffScore
        self.hoverTime = hoverTime
        self.hesitation = hesitation
        self.clickLatency= clickLatency
        self.clickOffset = clickOffset
        self.isChecked = isChecked

    def get_data_as_df(self):
        try:
            custom_input = {
                "mousePointsCount":[self.mousePointsCount],
                "avgSpeed":[self.avgSpeed],
                "speedStdDev":[self.speedStdDev],
                "accelStdDev":[self.accelStdDev],
                "jitterCount":[self.jitterCount],
                "angleStd":[self.angleStd],
                "diffScore":[self.diffScore],
                "hoverTime":[self.hoverTime],
                "hesitation":[self.hesitation],
                "clickLatency":[self.clickLatency],
                "clickOffset":[self.clickOffset],
                "isChecked":[self.isChecked],


            }

            df = pd.DataFrame(custom_input)

            return df
        
        except Exception as e:
            raise CustomException(e,sys)
        # except Exception as e:
        #     print("ACTUAL ERROR:", e)
        #     import traceback
        #     traceback.print_exc()
        #     raise




    

    