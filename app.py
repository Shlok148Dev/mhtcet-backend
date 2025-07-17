from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)
data = pd.read_csv("mhtcet_colleges.csv")

@app.route("/query", methods=["POST"])
def query():
    req = request.get_json()
    percentile = float(req["percentile"])
    category = req["category"]
    branch = req["branch"]

    result = data[
        (data["Percentile"] <= percentile) &
        (data["Category"].str.lower() == category.lower()) &
        (data["Branch"].str.lower() == branch.lower())
    ]

    return jsonify(result.to_dict(orient="records"))

app.run(host="0.0.0.0", port=10000)
