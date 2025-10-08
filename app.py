import base64
from flask import Flask, request, jsonify, render_template
from chatbot.chat import get_bot_response, bot_name  # Updated import
from chatbot.nltk_utils import text_to_speech as tts  # Updated import

# Initializes Flask
# Flask automatically looks for HTML files in a folder named "templates"
app = Flask(__name__)

@app.route("/")
def index():
    # Renders the website.html file from the 'templates' folder
    return render_template("website.html", bot_name=bot_name)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    u_input = data.get("message", "")

    if not u_input:
        return jsonify({"Error": "No Message Read"}), 400

    response = get_bot_response(u_input)
    response_text = response.get("Response")
    
    # Generate audio and encode it
    audio = tts(response_text)
    audio_base64 = base64.b64encode(audio).decode('utf-8')
    response['audio_base64'] = audio_base64
    
    return jsonify({"bot_name": bot_name, "response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)