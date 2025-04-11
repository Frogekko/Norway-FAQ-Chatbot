from flask import Flask, request, jsonify, render_template, Response
import os
from chat import generations, bot_name

# This ensures that it can read the file from any PC
basedir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(basedir, "static")

def create_app():
    # Initializes flask with template directory
    app = Flask(__name__, template_folder=template_dir)

    @app.route("/") # This code reads the website HTML
    def index():
        return render_template("website.html", bot_name = bot_name) # Puts the bot name in the html frontend
    
    # This is so that the request goes through the URL
    @app.route("/api/chat", methods = ["POST"]) # Initializes the button
    def chat():
        data = request.json
        u_input = data.get("message","") # Translates the input so that it can be used as 

        if not u_input:
            return jsonify({"Error": "No Message Read"}), 400 # Error message so that 

        response = generations(u_input) # Uses the function to create and awnser to the input of the user
        return jsonify({"bot_name": bot_name, "response": response}) # Returns the response
    return app