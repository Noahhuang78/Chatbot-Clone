from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from faq_module import query_faq, load_faq, embed_faq
from add_FAQ import scrap_new
import os

app = Flask(__name__, static_folder="static")
CORS(app)

@app.route('/chatResponse', methods = ["POST"])
def chatResponse():
    data = request.get_json()
    print("Received data:", data)   
    history = data.get("contents", [])
    print("Parsed history:", history)
    try:
        i = len(history) - 1
        user_question = history[i]["text"]
        print("User QUESTIOn", user_question)
        bot_message = query_faq(user_question, history)
        return jsonify({"response": bot_message}) 
    
    except(KeyError, TypeError):
         return jsonify({"response": "error"}) 

@app.route("/updateFAQs")          #scrape for new faqs, load and embed updated delta_faq.jsonl file
def update():
    scrap_new()
    load_faq()
    embed_faq()
    return "Scrap Success!"

@app.route("/")
def serve():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # fallback to 8080 locally
    app.run(host="0.0.0.0", port=port, threaded=True)


