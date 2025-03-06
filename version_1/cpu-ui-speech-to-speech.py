import tkinter as tk
from tkinter import filedialog
import threading
import torch
from faster_whisper import WhisperModel
import requests
from ttsmms import TTS
import sounddevice as sd
import numpy as np
import wavio
from gtts import gTTS
import wave
import librosa  # Import librosa for audio loading
import soundfile as sf  # For file validation and some audio handling

# Set up Whisper Model
model_size = "small"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(model_size, device=device, compute_type="int8")

# Set up TTS
dir_path = "eng"
tts = TTS(dir_path)

# Global variables
audio_file = "test.wav"
transcribed_text = ""
output_answer = ""
recording = False  # To track if recording is active

# Global variables for audio playback
playback_thread = None
playback_data = None
playback_samplerate = None
is_paused = False
is_playing = False
playback_position = 0

# Function to verify WAV file integrity
def verify_wav_file(file_path):
    try:
        with wave.open(file_path, 'r') as wav_file:
            wav_file.getnchannels()  # Check if the file has valid WAV structure
        return True
    except wave.Error as e:
        print(f"Invalid WAV file: {e}")
        return False

# Function to record audio from the microphone
def record_audio():
    global audio_file, recording
    duration = 5  # seconds
    fs = 16000  # Sample rate
    recording = True

    print("Recording audio...")
    audio_data = np.empty((0, 1), dtype=np.int16)

    def callback(indata, frames, time, status):
        if not recording:
            raise sd.CallbackAbort  # Stop callback when recording is finished
        nonlocal audio_data
        audio_data = np.concatenate((audio_data, indata), axis=0)

    try:
        with sd.InputStream(samplerate=fs, channels=1, callback=callback, dtype=np.int16):
            sd.sleep(int(duration * 1000))  # Wait for the duration or until stopped
    except sd.CallbackAbort:
        print("Recording stopped.")
    except Exception as e:
        print(f"Error during recording: {e}")
        return

    if audio_data.size > 0:
        # Save the audio file
        wavio.write(audio_file, audio_data, fs, sampwidth=2)
        print("Audio recording complete. Saved to test.wav")
        status_label.config(text="Audio recording complete.")
        transcribe_audio()
    else:
        status_label.config(text="No audio data recorded.")
        print("No audio data recorded.")

# Function to stop recording
def stop_recording():
    global recording
    if recording:
        recording = False
        status_label.config(text="Recording stopped.")

# Function to transcribe the audio
def transcribe_audio():
    global transcribed_text
    status_label.config(text="Transcribing audio...")
    
    if verify_wav_file(audio_file):
        segments, info = model.transcribe(audio_file, beam_size=5, language="en")
        transcribed_text = " ".join(segment.text for segment in segments)
        input_text_box.delete("1.0", tk.END)  # Correct usage of the delete method
        input_text_box.insert("1.0", transcribed_text)  # Correct usage of the insert method
        status_label.config(text="Transcription complete.")
        query_llm(transcribed_text)
    else:
        status_label.config(text="Invalid audio file. Unable to transcribe.")

# Function to load and play audio using librosa and sounddevice
def play_audio(audio_path):
    global playback_data, playback_samplerate, is_paused, is_playing, playback_thread
    try:
        # Load the audio using librosa
        playback_data, playback_samplerate = librosa.load(audio_path, sr=None)
        
        # Check if playback is already happening and stop it
        if is_playing:
            stop_audio()

        # Start playback in a new thread to avoid blocking UI
        playback_thread = threading.Thread(target=audio_playback_thread)
        playback_thread.start()
        is_paused = False
        is_playing = True
        status_label.config(text="Playing audio...")

    except Exception as e:
        status_label.config(text=f"Error playing audio: {e}")

# Function to handle playback in a separate thread
def audio_playback_thread():
    global playback_data, playback_samplerate, is_paused, is_playing
    try:
        sd.play(playback_data, samplerate=playback_samplerate)
        sd.wait()  # Wait until playback is finished
        is_playing = False
        status_label.config(text="Audio playback completed.")
    except Exception as e:
        print(f"Error during playback: {e}")
        status_label.config(text=f"Playback error: {e}")

# Function to pause the audio
def pause_audio():
    global is_paused
    if is_playing and not is_paused:
        is_paused = True
        sd.stop()
        status_label.config(text="Audio paused.")

# Function to resume the audio
def resume_audio():
    global is_paused
    if is_paused:
        is_paused = False
        play_audio(audio_file)  # Restart playback since librosa doesn't support resume natively

# Function to stop the audio
def stop_audio():
    global is_playing, is_paused
    if is_playing:
        sd.stop()
        is_playing = False
        is_paused = False
        status_label.config(text="Audio stopped.")

# Function to synthesize text to speech
def synthesize_audio(transcribed_text):
    status_label.config(text="Synthesizing speech...")
    tts = gTTS(transcribed_text)
    tts.save('test_out.wav')
    status_label.config(text="Synthesis complete. Ready to play audio.")

# Function to query the LLM and get a response
def query_llm(query):
    global output_answer
    status_label.config(text="Querying LLM...")
    url = "https://9c78-34-138-37-168.ngrok-free.app/api/query"
    data = {"query": query}
    
    try:
        response = requests.post(url, json=data)
        response_json = response.json()
        # output_answer = response_json.get("answer")
        output_answer = response_json
        output_text_box.delete("1.0", tk.END)  # Correct usage of the delete method
        output_text_box.insert("1.0", output_answer)  # Correct usage of the insert method
        status_label.config(text="LLM query complete. Answer displayed.")
        synthesize_audio(output_answer)
    except Exception as e:
        print(f"Error querying LLM: {e}")
        status_label.config(text=f"Error querying LLM: {e}")

# Function to start the process in a new thread
def start_process():
    threading.Thread(target=record_audio).start()

# Tkinter UI Setup
root = tk.Tk()
root.title("Question Recorder")

# Labels
tk.Label(root, text="Input Transcription:").pack(pady=5)
input_text_box = tk.Text(root, height=5, width=50)
input_text_box.pack(pady=5)

tk.Label(root, text="LLM Output Answer:").pack(pady=5)
output_text_box = tk.Text(root, height=5, width=50)
output_text_box.pack(pady=5)

status_label = tk.Label(root, text="Click 'Start' to record your question.", fg="red")
status_label.pack(pady=10)

# Start button to record audio
start_button = tk.Button(root, text="Start", command=start_process, bg="green", fg="blue", width=20, height=2)
start_button.pack(pady=10)

# Stop button to stop recording
stop_button = tk.Button(root, text="Stop", command=stop_recording, bg="red", fg="black", width=20, height=2)
stop_button.pack(pady=10)

# Play, Pause, Resume, Stop buttons for audio control
play_button = tk.Button(root, text="Play Audio", command=lambda: play_audio('test_out.wav'), bg="blue", fg="red", width=20, height=2)
play_button.pack(pady=5)

pause_button = tk.Button(root, text="Pause Audio", command=pause_audio, bg="orange", fg="red", width=20, height=2)
pause_button.pack(pady=5)

resume_button = tk.Button(root, text="Resume Audio", command=resume_audio, bg="yellow", fg="black", width=20, height=2)
resume_button.pack(pady=5)

stop_audio_button = tk.Button(root, text="Stop Audio", command=stop_audio, bg="red", fg="black", width=20, height=2)
stop_audio_button.pack(pady=5)

# Run the app
root.geometry("500x700")
root.mainloop()