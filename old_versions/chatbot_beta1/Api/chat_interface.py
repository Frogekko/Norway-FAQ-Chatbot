from flask import Flask, request, jsonify, render_template, Response
from threading import Thread
import queue
import os
from .chat import generations, bot_name
from .chatbot_train import training

log_queue = queue.Queue()

def create_app():
    # Use a relative path for templates
    app = Flask(__name__, template_folder='static')
    
    @app.route("/")
    def index():
        return render_template("website.html", bot_name=bot_name) 
    
    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.json
        u_input = data.get("message", "")

        if not u_input:
            return jsonify({"Error": "No Message Read"}), 400

        response = generations(u_input)
        # Return exactly what generations() returns
        return jsonify(response)
    
    @app.route("/api/train-more", methods=["POST"])
    def train_more():
        data = request.json
        iter = data.get("iterations", 1000)
        batch_size = data.get("Batch_Size", 64)

        print("Training started with iterations:", iter, "and batch size:", batch_size)
        
        def background_train(queue, iter, batch_size):
            # Use a relative path for the intents file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            intents_file = os.path.join(current_dir, "intents.json")
            training(batch_size, iter, intents_file=intents_file, message=queue)
        
        thread = Thread(target=background_train, args=(log_queue, iter, batch_size))
        thread.start()
        
        return jsonify({"status": "Training started", "iterations": iter})

    @app.route("/test-queue")
    def test_queue():
        log_queue.put("Test message to frontend!")
        return "Message sent to queue!"
    
    @app.route("/api/train-stream")
    def train_stream():
        def generate():
            while True:
                try:
                    message = log_queue.get(timeout=30)  # Add timeout to prevent hanging
                    print(f"Sending message to client: {message}")
                    yield f"data: {message}\n\n"
                    if "[Training] Done" in message:
                        break
                except queue.Empty:
                    # Send a keepalive message if no updates for 30 seconds
                    yield f"data: Training still in progress...\n\n"
                    
        return Response(generate(), mimetype="text/event-stream")
    
    return app