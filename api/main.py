import os
import tempfile
from flask import Flask, request, jsonify
import google.generativeai as genai

# Configure the Google Generative AI SDK
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)

# Store chat histories in memory (for demonstration purposes)
chat_histories = {}

def upload_to_gemini(file_path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(file_path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

@app.route('/api/process', methods=['POST'])
def process_image_and_prompt():
    if 'image' not in request.files or 'prompt' not in request.form:
        return jsonify({"error": "Image and prompt are required."}), 400

    user_id = request.form.get('user_id', 'default_user')  # Default user ID if not provided
    image = request.files['image']
    prompt = request.form['prompt']

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        image_path = temp_file.name
        image.save(image_path)

        # Upload the image to Gemini
        file_uri = upload_to_gemini(image_path, mime_type=image.mimetype)

        # Retrieve or start a new chat session history
        history = chat_histories.get(user_id, [])

        # Add the new prompt to the history
        history.append({
            "role": "user",
            "parts": [
                file_uri,
                prompt,
            ],
        })

        # Start a chat session with the accumulated history
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(prompt)

        # Add the bot's response to the history
        history.append({
            "role": "bot",
            "parts": [response.text],
        })

        # Update the chat history for this user
        chat_histories[user_id] = history

    # Clean up temporary file
    os.remove(image_path)

    return jsonify({"response": response.text})

@app.route('/api/query', methods=['GET'])
def query_prompt():
    user_id = request.args.get('user_id', 'default_user')  # Default user ID if not provided
    prompt = request.args.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    # Retrieve or start a new chat session history
    history = chat_histories.get(user_id, [])

    # Add the new prompt to the history
    history.append({
        "role": "user",
        "parts": [prompt],
    })

    # Start a chat session with the accumulated history
    chat_session = model.start_chat(history=history)
    response = chat_session.send_message(prompt)

    # Add the bot's response to the history
    history.append({
        "role": "bot",
        "parts": [response.text],
    })

    # Update the chat history for this user
    chat_histories[user_id] = history

    return jsonify({"response": response.text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
