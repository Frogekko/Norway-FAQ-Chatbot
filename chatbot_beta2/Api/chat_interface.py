from flask import Flask, request, jsonify, render_template, Response
import os
from .chat import generations, bot_name

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
    
    return app