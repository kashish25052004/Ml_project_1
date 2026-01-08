from flask import Flask,request,render_template,jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
import sys

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/home", methods=["GET"])
def home():
    return render_template("home.html")

@app.route('/test_data',methods = ['POST'])
def predict_datapoints():
    try:
        data = request.get_json()

        custom_data = CustomData(
            mousePointsCount=data["mousePointsCount"],
            avgSpeed=data["avgSpeed"],
            speedStdDev=data["speedStdDev"],
            accelStdDev=data["accelStdDev"],
            jitterCount=data["jitterCount"],
            angleStd=data["angleStd"],
            diffScore=data["diffScore"],
            hoverTime=data["hoverTime"],
            hesitation=data["hesitation"],
            clickLatency=data["clickLatency"],
            clickOffset=data["clickOffset"],
            isChecked = int(data["isChecked"])

        )

        input_df = custom_data.get_data_as_df()
        # print(input_df)


        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500






if __name__ == "__main__":
    app.run()


