"""
    This module is an API for a Page Summarizer model.
"""
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import pandas as pd



APP = Flask(__name__)

@APP.route('/', methods=['GET'])
def home():
    """
    This function called up when a GET request is done to the API
    """
    # Give message to user
    return {"info": "This is a model used to predict if a person had survived the shipwreck of the Titanic ship in 1912 based on the person's Age, Sex and the Port of embarkation. Use the format {Age: 85, Sex: male, Embarked: S} and POST to get prediction."}




@APP.route('/', methods=['POST'])
def predict():
    """
    This function called up when a POST request is done to the API
    """
    input_json = request.json
    print(input_json)
    query = pd.get_dummies(pd.DataFrame(input_json, ['0']))
    query = query.reindex(columns=MODEL_COLUMNS, fill_value=0)
    prediction = list(MODEL.predict(query))
    if prediction == [0]:
        result = 'Did not survive'
    else:
        result = 'Survived'

    return jsonify({'result': result})
if __name__ == '__main__':
    try:
        PORT = int(sys.argv[1])
    except:
        PORT = 12395
    #loading up the models
    MODEL = joblib.load('models/model.pkl')
    print('Model loaded')
    #loading up the model columns
    MODEL_COLUMNS = joblib.load("models/model_columns.pkl")
    print('Model columns loaded')

    APP.run(port=PORT, debug=True)
