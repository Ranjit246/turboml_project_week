import torch
from faster_whisper import WhisperModel
import requests
from ttsmms import TTS

# Create the WhisperModel instance
model_size = "small"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(model_size, device=device, compute_type="int8")

print("Model Loaded.")
print("Checkpoint 01 passed.")

# Transcribe the audio file
audio_file = "test.wav"
segments, info = model.transcribe(audio_file, beam_size=5)

# Combine all segments into one text
transcribed_text = " ".join(segment.text for segment in segments)
print("Transcribed Text:", transcribed_text)

print("Checkpoint 02 passed.")

# LLM endpoint - created by giving a PDF file input
url = "https://020b-34-71-71-183.ngrok-free.app/api/query"   ## Get the Colab running and pass the ngrok endpoint:
data = {"query": transcribed_text}                             # https://colab.research.google.com/drive/1BwkGHJ-ETdxaBq974k10BoqRuHeuCEzU?usp=sharing

# Query the LLM endpoint
response = requests.post(url, json=data)
response_json = response.json()

# Get the answer from the response
answer = response_json.get("answer")
print("LLM Answer:", answer)

print("Checkpoint 03 passed.")

# Text-to-Speech synthesis
dir_path = "eng"
tts = TTS(dir_path)
tts.synthesis(answer, wav_path="output.wav")
print("Synthesis complete. Audio saved at output.wav")
print("Final Checkpoint passed.")