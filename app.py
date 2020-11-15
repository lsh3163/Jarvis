from flask import Flask, render_template, request, url_for
from inference import *
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == "POST":
        pass
    elif request.method == "GET":
        char_text = request.args.get("char_text")
        print(char_text)
        str_predict = predict(char_text)
        print(str_predict)
    return render_template("index.html", pred=str_predict)