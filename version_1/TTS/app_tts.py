from ttsmms import TTS
dir_path = "eng"
tts=TTS(dir_path) 
text = "I am Ranjit Patro. Hello How are you?"
tts.synthesis(text, wav_path="data/example.wav")