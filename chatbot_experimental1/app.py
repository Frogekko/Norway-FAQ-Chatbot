# Inside app.py

# Add near the top imports
import requests
import random
import json
# ... other imports like Flask, torch, model, utils ...

# --- Load Model Components (as before) ---
# device = ...
# intents = ...
# FILE = ...
# data = ...
# model = ...
# all_words = ...
# tags = ...
# bot_name = ...
# confidence_threshold = ...
# -----------------------------------------

def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("website.html")

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.json
        u_input = data.get("message", "")

        if not u_input:
            return jsonify({"Error": "No Message Read"}), 400

        try:
            # Preprocess user input (same as before)
            tokens = tokenize(u_input)
            X = bag_of_words(tokens, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)

            # Get prediction (same as before)
            output = model(X)
            _, predicted = torch.max(output, dim=1)
            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]

            response_text = ""
            if prob.item() > confidence_threshold:
                # --- Check for the specific exchange_rate tag ---
                if tag == "exchange_rate":
                    try:
                        # Define target currencies
                        targets = ["EUR", "USD", "GBP"] # Example targets
                        rates_str_parts = []

                        # Fetch rates relative to NOK (How many Target Currencies 1 NOK buys)
                        api_url = f"https://api.frankfurter.app/latest?from=NOK&to={','.join(targets)}"
                        api_response = requests.get(api_url, timeout=5) # Add timeout
                        api_response.raise_for_status() # Raise error for bad status codes
                        rate_data = api_response.json()
                        fetched_rates = rate_data.get('rates', {})
                        date = rate_data.get('date', 'latest')

                        if fetched_rates:
                            for target_currency in targets:
                                if target_currency in fetched_rates:
                                    # Calculate 1 Target Currency = X NOK
                                    rate_nok_per_target = 1 / fetched_rates[target_currency]
                                    rates_str_parts.append(f"1 {target_currency} ≈ {rate_nok_per_target:.2f} NOK")

                            if rates_str_parts:
                                response_text = f"As of {date}: " + " | ".join(rates_str_parts) + ". (Rates via Frankfurter.app)"
                            else:
                                response_text = "Sorry, I couldn't retrieve the specific rates from the API."
                        else:
                             response_text = "Sorry, failed to get rates data from the API."

                    except requests.exceptions.RequestException as e:
                        print(f"API Error fetching exchange rate: {e}")
                        response_text = "Sorry, I couldn't fetch the live exchange rate due to a connection issue."
                    except Exception as e:
                        print(f"Error processing exchange rate data: {e}")
                        response_text = "Sorry, an error occurred while processing the exchange rate."
                else:
                    # --- Original Logic for other intents ---
                    for intent_data in intents['intents']:
                        if tag == intent_data["tag"]:
                            response_text = random.choice(intent_data['responses'])
                            break
                    if not response_text:
                        response_text = "I recognized the topic, but couldn't find a specific response."

            else:
                # --- Original Fallback Logic ---
                fallback_tag = 'fallback_suggestions' # Ensure this tag exists
                found_fallback = False
                for intent_data in intents['intents']:
                    if fallback_tag == intent_data["tag"]:
                        fallback_responses = intent_data['responses']
                        response_text = random.choice(fallback_responses)
                        found_fallback = True
                        break
                if not found_fallback:
                    response_text = "Sorry, I don't understand."
            # --- End Logic ---

            return jsonify({"Response": response_text})

        except Exception as e:
            print(f"Error processing chat API request: {e}")
            return jsonify({"Error": "Internal server error processing message"}), 500

    # --- Add other routes (/api/train-more, /api/train-stream) as before ---
    # ...

    return app

# --- Code to run the app (as before) ---
# if __name__ == "__main__":
#  app = create_app()
#  app.run(debug=True)