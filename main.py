from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # <-- import CORS
import os, io, json, numpy as np
from core import GeneticCore
from learning import Updater
from gtts import gTTS
import soundfile as sf
import tempfile
import openai

# ===== CONFIG =====
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
CORS(app)  # <-- enables CORS for all routes

core = GeneticCore(voice=None)  # voice handled separately
updater = Updater(core=core, voice=None)

# ===== ROUTE: STREAM AI =====
@app.route("/stream", methods=["POST"])
def stream_ai():
    """
    Receives raw WAV audio in POST body
    Returns JSON:
      { "text": "...", "audio_bytes": base64-encoded or streamed audio }
    """
    if "file" not in request.files:
        return jsonify({"error":"No audio file sent"}), 400
    audio_file = request.files["file"]

    # Save temporarily
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_file.save(tmp_wav.name)

    # ---- Whisper STT ----
    try:
        with open(tmp_wav.name, "rb") as f:
            result = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        user_text = result['text']
    except Exception as e:
        user_text = ""
        print("Whisper error:", e)

    os.remove(tmp_wav.name)

    # ---- AI processing ----
    core.process_input(user_text)
    response_text = core.memory.get("last_speech", "I have nothing to say.")

    # ---- TTS generation ----
    tts = gTTS(response_text)
    tmp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_mp3.name)

    # Return audio as file
    return send_file(tmp_mp3.name, mimetype="audio/mpeg", as_attachment=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)
