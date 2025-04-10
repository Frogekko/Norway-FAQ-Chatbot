from flask import Flask, request, jsonify, render_template, Response
from threading import Thread
import queue
from chat import generations, bot_name
from chatbot_train import training

loq_queue = queue.Queue()

def create_app():
    app = Flask(__name__, template_folder='/home/engebret/group1_chatbot/Api/static')

    
    @app.route("/")
    def index():
        return render_template("website.html", bot_name = bot_name) 
    
    #this is so that the request goes through the URL
    @app.route("/api/chat", methods = ["POST"])
    def chat():
        data = request.json
        u_input = data.get("message","")

        if not u_input:
            return jsonify({"Error": "No Message Read"}), 400

        response = generations(u_input)
        return jsonify({"bot_name": bot_name, "response": response})
    
    @app.route("/api/train-more", methods=["POST"])
    def train_more():
        data = request.json
        iter = data.get("iterations", 1000)
        batch_size = data.get("Batch_Size", 64)

        print("Training started with iterations:", iter, "and batch size:", batch_size)
        
        def background_train(queue, iter, batch_size):
            training(batch_size, iter, message=queue)
        
        thread = Thread(target=background_train, args=(loq_queue, iter, batch_size))
        thread.start()
        
        return jsonify({"status":"Training started","iterations":iter})

    @app.route("/test-queue")
    def test_queue():
        loq_queue.put("Test message to frontend!")
        return "Message sent to queue!"
    
    @app.route("/api/train-stream")
    def train_stream():
        def generate():
            while True:
                try:
                    message = loq_queue.get(timeout=5)
                except:
                    print("Nothing in Queue")
                    continue
                print(f"Sending message to client: {message}")
                yield f"data: {message}\n\n"
                if "[Training] Done" in message:
                    break
        return Response(generate(), mimetype="text/event_stream")
    return app