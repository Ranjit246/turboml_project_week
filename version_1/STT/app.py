import torch
from faster_whisper import WhisperModel

# Create the WhisperModel instance
model_size = "small"
model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")

audio_file = "trim_3sec.wav"
segments, info = model.transcribe(audio_file, beam_size=5)

for segment in segments:
    print("%s" % (segment.text))