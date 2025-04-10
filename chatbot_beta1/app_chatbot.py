import sys
sys.path.append('/home/engebret/group1_chatbot/Api')

from Api.chat_interface import create_app

Mimir_app = create_app()

if __name__ == "__main__":
    Mimir_app.run(host="0.0.0.0", port = 5000, debug=True)