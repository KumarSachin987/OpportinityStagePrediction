import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
app = Flask(__name__)
regmodel = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    result = None
    data = [float(x) for x in request.form.values()]
    #test_data = request.form.get("firstname")
    print('Receiving data',data)
    #final_input = scaler.transform(np.array(data).reshape(1,-1))
    output = regmodel.predict(np.array(data).reshape(1,-1))[0]
    if(output == 0):
        result = 'Closed Lost'
    else:
        result = 'Closed Won'    
    return render_template('home.html',prediction_text = 'The Final Stage of opportunity will be {}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)    