import os
import sys

#This code ensures that this file can read the Api folder from any pc.
basedir = os.path.abspath(os.path.dirname(__file__))
api_path = os.path.join(basedir, "Api")
sys.path.append(api_path)

#Imports the create app function from the interface code
from Api.chat_interface import create_app

#which leads to the code initializing
Mimir_app = create_app()

#and then giving the app a host IP
if __name__ == "__main__":
    Mimir_app.run(host="0.0.0.0", port = 5000, debug=True)