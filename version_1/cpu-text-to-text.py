import torch
from faster_whisper import WhisperModel
import requests
from ttsmms import TTS

user_text = "What is Antibody?"

# LLM endpoint - created by giving a PDF file input
url = "https://f9a0-35-237-28-246.ngrok-free.app/api/query"   ## Get the Colab running and pass the nginix endpoint:
data = {"query": user_text}                             # https://colab.research.google.com/drive/1VUagz3UfC4jfXiqPj8aYlmsnh76Di3UX?usp=sharing

# Query the LLM endpoint
response = requests.post(url, json=data)
response_json = response.json()

# Get the answer from the response
answer = response_json.get("answer")
print("LLM Answer:", answer)