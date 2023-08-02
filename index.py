# header
from fastapi import FastAPI, File, UploadFile, Response
import requests
import os
import speech_recognition as sr
import gpt_2_simple as gpt2
from googletrans import Translator
from configs import config

# deklarasi
app = FastAPI()
translator = Translator()
r = sr.Recognizer()

# function
model_name = "124M"
model_folder = os.path.join("models", model_name)
if not os.path.exists(model_folder):
    gpt2.download_gpt2(model_name=model_name)

def chat(pesan: str):
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, model_name="124M")
    
    text_response = gpt2.generate(sess, model_name="124M", prefix=pesan, return_as_list=True)[0]
    return text_response

def request_audio(text):
    url = 'https://deprecatedapis.tts.quest/v2/voicevox/audio/'
    params = {
        'key': config.api_key_voicevox,
        'speaker': '0',
        'pitch': '0',
        'intonationScale': '1',
        'speed': '1',
        'text': text
    }
    response = requests.get(url, params=params)
    return response.content

# endpoint
@app.post("/talk/{language_used}")
async def talk(language_used: str, audio_talk: UploadFile = File(...)):
    with sr.AudioFile(audio_talk.file) as audio_file:
        audio_data = r.record(audio_file)
        transcript = r.recognize_google(audio_data, language=language_used)
    jawaban = chat(pesan=transcript)
    translation = translator.translate(jawaban, dest='ja')
    audio_response = request_audio(text=translation.text)
    return Response(content=audio_response, media_type="audio/wav")