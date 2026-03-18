from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_engine import ask_question

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("message")

    response = ask_question(query)

    return jsonify({
        "response": response
    })

if __name__ == "__main__":
    app.run(debug=True)