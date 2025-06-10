from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

# create a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race'),
            parental_level_of_education=request.form.get('parental_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_prep'),
            reading_score=int(request.form.get('reading_score')),
            writing_score=int(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)
        return render_template('index.html', prediction=prediction[0])
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
