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
    return app.send_static_file('index_s2.html')

@app.route("/weights")
def status():
    return app.response_class(
        response = json.dumps(open("weights.json").read()),
        status = 200,
        mimetype = 'application/json'
    )

app.run(debug = False, host = '192.168.1.14', port = 8000)
