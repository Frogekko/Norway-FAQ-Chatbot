import os
import sys

basedir = os.path.abspath(os.path.dirname(__file__))
api_path = os.path.join(basedir, "Api")
sys.path.append(api_path)

from Api.chat_interface import create_app

Mimir_app = create_app()

if __name__ == "__main__":
    Mimir_app.run(host="0.0.0.0", port = 5000, debug=True)