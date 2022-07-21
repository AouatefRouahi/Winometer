from flask import Flask, render_template, request, jsonify, abort
import json
import joblib

import warnings
warnings.filterwarnings("ignore")

from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# load model
model = joblib.load("models/model.joblib")

# useful functions for errors handling
def validateJSON_TypeError(jsonData):
    if not (type(jsonData) is dict):
        abort(500, description="Incompatible Input Type Error!! Please check the required input format!!")
        return False
    return True


def validateJSON_KeyError(jsonData):
    if len(jsonData.keys()) < 1:
        abort(500, description="Incompatible Key Error!! No key is supplied!!")
        return False
    elif len(jsonData.keys()) > 1:
        abort(500, description="Incompatible Key Error!! Only one key is accepted!!")
        return False
    elif "input" not in jsonData.keys():
        abort(500, description="Incompatible Key Error nomination!! The key must be equal to 'input'!!")
        return False
    return True


def floats(l):
    res = True
    i = 0
    if not (type(l) is list):
        return False
    while (i < len(l)) and res:
        res = type(l[i]) is float
        i += 1
    return res


def validateJSON_ValueError(jsonData):
    x = list(jsonData.values())[0]
    if type(x) is list:
        are_lists = [type(l) is list for l in x]

        are_complete = [len(l) == 11 for l in x]

        are_floats = [floats(l) for l in x]

        result = ((False in are_lists) or (False in are_floats) or (False in are_complete))

        if result:
            abort(500, description="Incompatible Request Value Error!! The recieved parameters must be lists of 11 float parameters!!")
        return not (result)

    abort(500, description="Incompatible Request Value Error!! The recieved parameters must be lists of 11 float parameters!!")
    return False


# errors handlers
@app.errorhandler(500)
def internalServerError(e):
    return jsonify(error=str(e)), 500


# endpoints
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # get data
    input_ = request.get_json()

    if validateJSON_TypeError(input_):
        if validateJSON_KeyError(input_):
            if validateJSON_ValueError(input_):
                # output
                output = {"Prediction": []}

                # get prediction
                prediction = model.predict(input_["input"])
                output["Prediction"] = prediction.tolist()
                print("/predict endpoint:::: ", str(output))

                return jsonify(str(output)), 200


@app.route("/gui", methods=["POST"])
def gui():
    # access the data from form
    fixed_acidity = float(request.form["fixed-acidity"])
    volatile_acidity = float(request.form["volatile-acidity"])
    citric_acid = float(request.form["citric-acid"])
    residual_sugar = float(request.form["residual-sugar"])
    chlorides = float(request.form["chlorides"])
    free_sulfur_dioxide = float(request.form["free-sulfur-dioxide"])
    total_sulfur_dioxide = float(request.form["total-sulfur-dioxide"])
    density = float(request.form["density"])
    ph = float(request.form["pH"])
    sulphates = float(request.form["sulphates"])
    alcohol = float(request.form["alcohol"])

    # get prediction
    input_cols = [
        [
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            ph,
            sulphates,
            alcohol,
        ]
    ]

    prediction = model.predict(input_cols)
    output = prediction[0]
    print("/gui endpoint:::: ", str(output))
    
    msg = "import requests \n" +\
    "url = \"https://wineometer-ar.herokuapp.com/predict\"\n" +\
    "my_input = { \n" +\
    "\"input\": {}\n".format(input_cols) +  "}\n" +\
    "res = requests.post(url, json=my_input)\n"+\
    "print(res.json()) \n\n"
    

    print(msg)
    return (
        render_template(
            "index.html",
            prediction_text="The predicted wine quality score is {}/10".format(output),
            input_ = "{}".format(input_cols)
        ),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
