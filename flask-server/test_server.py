from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from faq_module import query_faq, load_faq, embed_faq
from add_FAQ import scrap_new
from celery_worker import scrape_faqs, process_faqs, celery
from celery import chain
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

@app.route("/update")          #scrape for new faqs, load and embed updated delta_faq.jsonl file
def update():
    try:
        job = chain(                #promise chain: scrape_faqs, then(process_faqs(result_from_scrape_faqs)).
            scrape_faqs.s(),
            process_faqs.s()
        ).apply_async()
        return jsonify({"status": "queued", "task_id": job.id})  #job id refers to the chain id

    except Exception as e:
        return jsonify({"status": "error", "task_id": str(e)})

@app.route("/status/<task_id>")
def status(task_id):
    from celery.result import AsyncResult
    job = AsyncResult(task_id, app=celery)
    return jsonify({"status": job.state, "result": job.result if job.ready() else None})

@app.route("/redis-test")
def redis_test():
    try:
        from celery_worker import celery
        i = celery.control.inspect()
        active = i.active()
        return jsonify({"redis_connection": "OK", "active_tasks": active})
    except Exception as e:
        return jsonify({"redis_connection": "FAILED", "error": str(e)})

@app.route("/")
def serve():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # fallback to 8080 locally
    app.run(host="0.0.0.0", port=port, threaded=True)

