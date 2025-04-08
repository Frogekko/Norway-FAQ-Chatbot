from flask import Flask, request, jsonify, render_template, Response
import queue
from threading import Thread
import torch
import torch.nn as nn
from chat import #(placeholder), bot_name
from chatbot_train import

def create_app():
    app = Flask(__name__)
    
    #Components important for chatbot, the variables are set to None, as they have no values yet.
    encode = decode = vocab = search = None
    loq_queue = queue

    def load_model():
        nonlocal encode, decode, vocab, search
        if not all([encode, decode, vocab, search]):
            encode, decode, vocab, search = """path to the bot"""
        
    @app.route("/")
    def index():
        return render_template("website.html") #må lagre senere, kan være jeg stjeler meisam sin template
    
    #this is so that the request goes through the URL
    @app.route("/api/chat", methods = ["POST"])
    def chat():
        load_model()
        data = request.json
        u_input = data.get("message","")

        max = data.get("max length", 100)

        if not u_input:
            return jsonify({"Error": "No Message Read"}), 400

        response = #Add the model here(u_input, encode, decode, vocab, search, max_length = max)
        return jsonify({"Response": response})
    
    @app.route("/api/train-more", methods=["POST"])
    def train_more():
        data = request.json
        iter = data.get("iterations", 1000)
        batch_size = data.get("Batch_size", 64)

        def background_train(log_queue, iterations, batch_size):
            #placeholder_train_model()
        
        thread = Thread(target=background_train, args=(loq_queue, iter, batch_size))
        thread.start()
        
        return jsonify({"status":"Training started","iterations":iter})
    
    @app.route("/api/train-stream")
    def train_stream():
        def generate():
            while True:
                message = loq_queue.get()
                yield f"data: {message}\n\n"
                if "[Training] Done" in message:
                    break
            return Response(generate(), mimetype="text/event_stream")
    return app
