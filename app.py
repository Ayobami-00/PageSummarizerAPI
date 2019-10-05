from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np


app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    # Give message to user
    return {"info": "This is a model used to predict if a person had survived the shipwreck of the Titanic ship in 1912 based on the person's Age, Sex and the Port of embarkation. Use the format {Age: 85, Sex: male, Embarked: S} and POST to get prediction."}




@app.route('/', methods=['POST'])
def predict():
        try:
            input = request.json
            print(input)
            query = pd.get_dummies(pd.DataFrame(input,['0']))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(model.predict(query))
            if prediction == [0]:
                result = 'Did not survive'
            else:
                result = 'Survived'

            return jsonify({'result': result})

        except:

            return jsonify({'trace': traceback.format_exc()})
   
if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) 
    except:
        port = 12395 
    #loading up the models
    model = joblib.load('models/model.pkl') 
    print ('Model loaded')
    #loading up the model columns
    model_columns = joblib.load("models/model_columns.pkl")
    print ('Model columns loaded')

    app.run(port=port, debug=True)
