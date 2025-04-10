from flask import Flask, request, jsonify, render_template, Response
from threading import Thread
import queue
import os
from chat_group1 import generations, bot_name
from chatbot_train_group1 import training


def create_app():
    app = Flask(__name__, template_folder = "/home/engebret/testPedroChat/Api")
    
    loq_queue = queue.Queue()
    
    @app.route("/")
    def index():
        return render_template("website.html") #må lagre senere, kan være jeg stjeler meisam sin template
    
    #this is so that the request goes through the URL
    @app.route("/api/chat", methods = ["POST"])
    def chat():
        data = request.json
        u_input = data.get("message","")

        if not u_input:
            return jsonify({"Error": "No Message Read"}), 400

        response = generations(u_input)
        return jsonify({"bot_name": bot_name, "Response": response})
    
    @app.route("/api/train-more", methods=["POST"])
    def train_more():
        data = request.json
        iter = data.get("iterations", 1000)
        batch_size = data.get("Batch_Size", 64)

        def background_train(queue, iter, batch_size):
            training(iter, batch_size, message=False)
        
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
