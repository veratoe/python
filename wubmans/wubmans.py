import sys
sys.path.append('../rnn')

import flask
import json
from flask import request

import rnn

app = flask.Flask(__name__)

results_json = '/home/diede/Documents/python/rnn/resultaten.json'
status_json = '/home/diede/Documents/python/rnn/status.json'

@app.route("/")
def root():
    return app.send_static_file('index.html')

@app.route("/status")
def status():
    return app.response_class(
        response = json.dumps(open(status_json).read()),
        status = 200,
        mimetype = 'application/json'
    )

@app.route("/results")
def results():
    return app.response_class(
        response = json.dumps(open(results_json).read()),
        status = 200,
        mimetype = 'application/json'
    )


@app.route("/seed", methods = ['POST'])
def seed():
    print('we got a request')
    print(request.get_json())
    a = rnn.predict(request.get_json())

    return app.response_class(
        response = json.dumps(a),
        status = 200,
        mimetype = 'application/json'
    )

app.run(debug = False, host = '192.168.1.14', port = 8000)
