import sys

import flask
import json
import base64

from flask import request
from PIL import Image
from io import BytesIO

#import rnn

app = flask.Flask(__name__)


@app.route("/")
def root():
    return app.send_static_file('index.html')

#@app.route("/status")
#def status():
#    return app.response_class(
#        response = json.dumps(open(status_json).read()),
#        status = 200,
#        mimetype = 'application/json'
#    )

#@app.route("/results")
#def results():
#    return app.response_class(
#        response = json.dumps(open(results_json).read()),
#        status = 200,
#        mimetype = 'application/json'
#    )


@app.route("/upload", methods = ['POST'])
def upload():
    print('we got a request')
    data = request.form['url']

    image = Image.open(BytesIO(base64.b64decode(data.replace("data:image/png;base64,", ""))))
    image = image.resize((28, 28))
    image.save("sample.png")

    import mnist

    prediction = mnist.predict()

    return app.response_class(
        response = json.dumps(prediction.tolist()[0]),
        status = 200,
        mimetype = "application/json"
    )

app.run(debug = False, host = '192.168.1.14', port = 8000)
