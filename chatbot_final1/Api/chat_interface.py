from flask import Flask, request, jsonify, render_template, Response
import os
from chat import generations, bot_name

#This ensures that it can read the file from any PC
basedir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(basedir, "static")

def create_app():
    #Initializes flask with template directory
    app = Flask(__name__, template_folder=template_dir)

    @app.route("/") #this code reads the website HTML
    def index():
        return render_template("website.html", bot_name = bot_name) #puts the bot name in the html frontend
    
    #this is so that the request goes through the URL
    @app.route("/api/chat", methods = ["POST"]) #initializes the button
    def chat():
        data = request.json
        u_input = data.get("message","") #translates the input so that it can be used as 

        if not u_input:
            return jsonify({"Error": "No Message Read"}), 400 #error message so that 

        response = generations(u_input) #uses the function to create and awnser to the input of the user
        return jsonify({"bot_name": bot_name, "response": response}) #returns the response
    return app