import streamlit as st
import time
import speech_recognition as sr
from gtts import gTTS  # Google Text-to-Speech
import os
from vectors import EmbeddingsManager
from chatbot import ChatbotManager

# Initialize session_state variables
if 'temp_pdf_path' not in st.session_state:
    st.session_state['temp_pdf_path'] = None

if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Set page configuration
st.set_page_config(
    page_title="AI Assistant",
    layout="wide"
)

# Header section styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 40px;
    }
    .column-header {
        font-size: 20px;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #888;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="main-header">Smart Document Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload, Analyze, and Interact with Your Documents Seamlessly</div>', unsafe_allow_html=True)

# Columns layout
col1, col2 = st.columns([1, 2], gap="large")

# Speech Recognition (STT) - Microphone Input
recognizer = sr.Recognizer()

def transcribe_audio():
    with sr.Microphone() as source:
        st.info("🎤 Listening... Please speak for up to 5 seconds.")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
        try:
            # Listen with a timeout of 5 seconds (listening stops after 5 seconds of silence)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            st.success(f"🗣 You said: {text}")
            return text
        except sr.UnknownValueError:
            st.warning("⚠️ Could not understand audio.")
        except sr.RequestError:
            st.error("⚠️ Speech Recognition service is unavailable.")
        except sr.WaitTimeoutError:
            st.warning("⚠️ Listening timed out, no speech detected.")
        return ""

# Column 1: File Upload and Embeddings Creation
with col1:
    st.markdown('<div class="column-header">📂 Upload Document</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.success("📄 File Uploaded Successfully!", icon="📁")
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**File Size:** {uploaded_file.size / 1000:.2f} KB")
        
        temp_pdf_path = "temp.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state['temp_pdf_path'] = temp_pdf_path

    st.markdown("---")
    
# Column 2: Chatbot Interface and TTS with STT in Chat Input
with col2:
    st.markdown('<div class="column-header">💬 Chat Interface</div>', unsafe_allow_html=True)

    create_embeddings = st.button("🧠 Analyse Document")
    
    if create_embeddings:
        if st.session_state['temp_pdf_path'] is None:
            st.warning("⚠️ Please upload a PDF first.")
        else:
            try:
                embeddings_manager = EmbeddingsManager(
                    model_name="BAAI/bge-small-en",
                    device="cpu",
                    encode_kwargs={"normalize_embeddings": True},
                    qdrant_url="http://localhost:6333",
                    collection_name="vector_db"
                )
                with st.spinner("🔄 Analysing Document..."):
                    result = embeddings_manager.create_embeddings(st.session_state['temp_pdf_path'])
                    time.sleep(1)
                st.success("Document Analysed successfully!", icon="✅")

                if st.session_state['chatbot_manager'] is None:
                    st.session_state['chatbot_manager'] = ChatbotManager(
                        model_name="BAAI/bge-small-en",
                        device="cpu",
                        encode_kwargs={"normalize_embeddings": True},
                        llm_model="llama3.2:1b",
                        llm_temperature=0.7,
                        qdrant_url="http://localhost:6333",
                        collection_name="vector_db"
                    )
            except Exception as e:
                # Log the full error message for debugging
                st.error(f"⚠️ An error occurred: {str(e)}")
                st.write(f"**Error Details:** {str(e)}")  # This will display the error message

    # Chat Interface
    if st.session_state['chatbot_manager'] is None:
        st.info("🤖 Please upload a document and analyse to start chatting.")
    else:
        st.markdown("**Interact with Your Document Below:**")
        
        for msg in st.session_state['messages']:
            role = "user" if msg['role'] == "user" else "assistant"
            st.chat_message(role).markdown(msg['content'])

        # Speech-to-Text button inside chat input
        st.markdown("### 🗣 Speak or Type Your Question:")
        st.write("Press the microphone button to speak or type your question below.")

        # Microphone button to get input from speech
        if st.button("🎤 Use Microphone for Input"):
            user_input = transcribe_audio()
            if user_input:
                st.session_state['messages'].append({"role": "user", "content": user_input})

                # Pass the recognized speech directly to the chatbot
                with st.spinner("🤖 Thinking..."):
                    try:
                        response = st.session_state['chatbot_manager'].get_response(user_input)
                        time.sleep(1)
                    except Exception as e:
                        # Log the full error message for debugging
                        st.error(f"⚠️ An error occurred: {str(e)}")
                        st.write(f"**Error Details:** {str(e)}")  # This will display the error message

                # Display chatbot response
                st.chat_message("assistant").markdown(response)
                st.session_state['messages'].append({"role": "assistant", "content": response})

                # Text-to-Speech (TTS) synthesis
                tts = gTTS(text=response, lang='en')
                tts.save("response.mp3")

                # Play TTS audio
                st.audio("response.mp3", format="audio/mp3")

        # Regular text input as well
        user_input = st.chat_input("Type your question here...")
        
        if user_input:
            st.chat_message("user").markdown(user_input)
            st.session_state['messages'].append({"role": "user", "content": user_input})

            with st.spinner("🤖 Thinking..."):
                try:
                    response = st.session_state['chatbot_manager'].get_response(user_input)
                    time.sleep(1)
                except Exception as e:
                    # Log the full error message for debugging
                    st.error(f"⚠️ An error occurred: {str(e)}")
                    st.write(f"**Error Details:** {str(e)}")  # This will display the error message

            # Display chatbot response
            st.chat_message("assistant").markdown(response)
            st.session_state['messages'].append({"role": "assistant", "content": response})

            # Text-to-Speech (TTS) synthesis
            tts = gTTS(text=response, lang='en')
            tts.save("response.mp3")

            # Play TTS audio
            st.audio("response.mp3", format="audio/mp3")

# Footer
st.markdown('<div class="footer">© 2024. All rights reserved.</div>', unsafe_allow_html=True)