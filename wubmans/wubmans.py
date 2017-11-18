import flask
import json
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

app.run(debug = True, host = '0.0.0.0', port = 8000)
