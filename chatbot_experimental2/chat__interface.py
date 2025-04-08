from flask import Flask, request, jsonify, render_template, Response
import torch
import torch.nn as nn
#this import is placeholder until i can get a proper response
#from (name of chatbot python file) import (name of response model), (name of bot)

def create_app():
    app = Flask(__name__)

    #placeholder for modelnames from the imports of other files
    #chat_model = (name of response model)()
