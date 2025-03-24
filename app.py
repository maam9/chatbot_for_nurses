
import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_cors import CORS
import json
import os
app = Flask(__name__)
CORS(app)
app.secret_key = "1234"  # For session management
RASA_URL = "http://localhost:5005/webhooks/rest/webhook"
# Load worker data (example)
with open("json_data/worker_informations.json", "r") as file:
    workers = json.load(file)
@app.route('/')
def index():
    # If user not logged in, go to login
    if "worker_id" not in session:
        return redirect(url_for('login'))
    # Else go to chat
    return redirect(url_for('chat'))
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    data = request.get_json()
    worker_id = data.get("worker_id")
    password = data.get("password")
    # Find worker
    worker = next((w for w in workers if w["worker_id"] == worker_id), None)

    if worker and worker["password"] == password:
        # Store in session
        session["worker_id"] = worker_id
        session["first_name"] = worker["first_name"]
        session["last_name"] = worker["last_name"]
        return jsonify({
            "success": True,
            "first_name": worker["first_name"],
            "last_name": worker["last_name"]
        })
    else:
        return jsonify({"success": False, "message": "Invalid credentials"})
@app.route('/chat')
def chat():
    # Must be logged in
    if "worker_id" not in session:
        return redirect(url_for('login'))

    # Pass user’s full name to the template
    user_full_name = f"{session['first_name']} {session['last_name']}"
    return render_template('index.html', user_full_name=user_full_name)


@app.route('/logout')
def logout():
    # Clear session
    session.clear()
    return redirect(url_for('login'))


@app.route('/send_message', methods=['POST'])
def send_message():
    # Check login
    if "worker_id" not in session:
        return jsonify({"error": "Unauthorized. Please log in."}), 401

    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    payload = {
        "sender": session["worker_id"],
        "message": user_message
    }

    try:
        rasa_response = requests.post(RASA_URL, json=payload)
        rasa_response.raise_for_status()
        responses = rasa_response.json()

        if not responses:
            return jsonify({"bot_message": "No response from the bot."})

        # Combine Rasa messages into one text
        bot_messages = [resp.get('text', '') for resp in responses if 'text' in resp]
        bot_response_text = "\n".join(bot_messages)

        return jsonify({"bot_message": bot_response_text})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


@app.route('/check_login')
def check_login():
    return jsonify({"logged_in": "worker_id" in session})


if __name__ == '__main__':
    app.run(debug=True, port=3000)
