import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the current directory to path (not needed if you're already in the right directory)
sys.path.append(current_dir)

from Api.chat_interface import create_app

Mimir_app = create_app()

if __name__ == "__main__":
    Mimir_app.run(host="0.0.0.0", port=5000, debug=True)