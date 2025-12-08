from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os, tempfile
from core import GeneticCore
from learning import Updater
from gtts import gTTS
import openai

# ---------------- CONFIG ----------------
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY: raise RuntimeError("OPENAI_API_KEY not set")
openai.api_key=OPENAI_API_KEY

WAKE_WORD=os.environ.get("WAKE_WORD","genetic").lower()
PORT=int(os.environ.get("PORT",5000))

app=Flask(__name__)
CORS(app)

core=GeneticCore(voice=None)
updater=Updater(core)

@app.route("/stream",methods=["POST"])
def stream_ai():
    if "file" not in request.files:
        return jsonify({"error":"No audio file sent"}),400
    audio_file=request.files["file"]
    tmp_wav=tempfile.NamedTemporaryFile(delete=False,suffix=".wav")
    audio_file.save(tmp_wav.name)

    # 1) Whisper STT
    try:
        with open(tmp_wav.name,"rb") as f:
            result=openai.audio.transcriptions.create(model="whisper-1",file=f)
        transcript=result.get("text","").strip()
    except Exception as e:
        print("Whisper error:",e)
        transcript=""
    finally:
        os.remove(tmp_wav.name)

    # 2) Wake word check
    if WAKE_WORD and WAKE_WORD.lower() in transcript.lower():
        cleaned=transcript.lower().replace(WAKE_WORD,"").strip()
        user_text=cleaned if cleaned else "[wake word detected]"
        core.process_input(user_text)
        response_text=core.memory.get("last_speech","I heard you, but need more info.")
        tmp_mp3=tempfile.NamedTemporaryFile(delete=False,suffix=".mp3")
        try:
            gTTS(response_text).save(tmp_mp3.name)
            return send_file(tmp_mp3.name,mimetype="audio/mpeg",as_attachment=False)
        except Exception as e:
            print("TTS error:",e)
            return jsonify({"error":"TTS failed"}),500
    return ("",204)

if __name__=="__main__":
    print(f"Starting server on port {PORT}, wake word='{WAKE_WORD}'")
    app.run(host="0.0.0.0",port=PORT)
