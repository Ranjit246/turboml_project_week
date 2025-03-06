# Setup Guide

## 1. Install Python 3.11 and Create a Virtual Environment
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev -y
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Install Required Dependencies
```bash
sudo apt update
sudo apt install -y poppler-utils mesa-utils libgl1 tesseract-ocr docker.io
```

## 3. Fix Docker Permission Issues
```bash
ls -l /var/run/docker.sock
sudo chown root:docker /var/run/docker.sock
sudo chmod 660 /var/run/docker.sock
sudo usermod -aG docker $USER
```
Restart your terminal after executing the above commands.

---

## 4. Install and Run Qdrant
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -v .:/qdrant/storage qdrant/qdrant
```
Qdrant runs at: [localhost:6333/dashboard](http://localhost:6333/dashboard/)

---

## 5. Install and Run Ollama
```bash
sudo apt update
sudo apt install -y curl
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b
ollama run llama3.2:1b  # Exit after a test run
```

---

## 6. Running the Setup
Keep the following services running in two terminal tabs:

### Terminal 01: Start Qdrant
```bash
docker run -p 6333:6333 -v .:/qdrant/storage qdrant/qdrant
```

### Terminal 02: Start Streamlit UI
```bash
streamlit run app.py
```

---

## 7. Ports Used
- **8501**: Streamlit Server (UI)
- **6333**: Qdrant
- **11434**: Ollama