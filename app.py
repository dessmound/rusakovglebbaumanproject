import flask
from flask import render_template
import pickle
from sklearn import preprocessing
from joblib import load

app = flask.Flask(__name__, template_folder = 'templates', static_folder='css')

@app.route('/', methods = ['POST', 'GET'])

@app.route('/index', methods = ['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('index.html')

    if flask.request.method == 'POST':
        with open('DecisionTreeRegressor.pkl', 'rb+') as f:
            loaded_model1 = pickle.load(f)

            IW = float(flask.request.form['IW'])
            IF = float(flask.request.form['IF'])
            VW = float(flask.request.form['VW'])
            FP = float(flask.request.form['FP'])
            input = [IW, IF, VW, FP]
            scaler = load('minmaxscaler.joblib')
            input_scaled = scaler.transform([input])
            
            y_pred = loaded_model1.predict(input_scaled)

            return render_template('index.html', result1 = y_pred[0][0].round(2), result2 = y_pred[0][1].round(2))


if __name__ == '__main__':
    app.run()