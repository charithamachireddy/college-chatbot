from flask import Flask, render_template, request, jsonify
import pickle
import random

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
data = pickle.load(open("data.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json["message"]

    vec = vectorizer.transform([msg])
    tag = model.predict(vec)[0]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return jsonify({
                "response": random.choice(intent["responses"])
            })

    return jsonify({"response": "Please ask about college details 😊"})

if __name__ == "__main__":
    app.run(debug=True)