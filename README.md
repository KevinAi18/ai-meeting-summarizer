# AI-Powered Meeting Summarizer

Engineers waste hours writing, formatting, and correcting meeting notes. Let's fix that.

This project is a high-performance local pipeline that converts raw audio recordings of meetings into clean, accurate transcripts and generates actionable, concise summaries. By running everything locally, you keep your data secure without sacrificing speed or quality.

---

## 🛠️ The Pipeline

The architecture is designed to be lightweight, modular, and fast.

```
[ Raw Audio ] ──► [ FFmpeg Preprocessing ] ──► [ whisper.cpp Transcription ] 
                                                               │
                                                               ▼
[ Gradio UI Web Interface ] ◄── [ Ollama Local Summarization ] ◄┘
```

1. **Audio Preprocessing**: Converts incoming audio (MP3, WAV, M4A, etc.) using `FFmpeg` to 16kHz mono WAV format to match Whisper's model requirements.
2. **Local Transcription**: Runs `whisper.cpp` to perform high-performance speech-to-text.
3. **Local Summarization**: Calls a local `Ollama` instance to analyze the transcript and summarize key concepts, action items, and decisions.
4. **Gradio Interface**: Provides an interactive local web interface for uploading files, configuring options, viewing results, and downloading the full transcript.

---

## 💻 Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Core Language** | Python | Orchestrates the pipeline and UI |
| **Transcription Engine** | [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) | Ultra-fast local C/C++ Whisper inference |
| **LLM Orchestration** | [Ollama](https://ollama.com/) | Powers the local summarization LLM |
| **Web Interface** | [Gradio](https://gradio.app/) | Interactive GUI for model options and downloads |
| **Audio Processing** | [FFmpeg](https://ffmpeg.org/) | Preprocesses audio files for transcription |

---

## ✨ Features

- **Local Speech-to-Text**: High-accuracy local transcription powered by `whisper.cpp`.
- **Intelligent Summarization**: Automatic extracting of meeting notes, key action points, and action items via Ollama LLMs.
- **Multi-Model Support**: Select between Whisper models (`base`, `small`, `medium`, `large-V3`) and any LLM model installed in your local Ollama server (e.g., `llama3.2`, `mistral`, `gemma2`).
- **Translation**: Built-in translation of non-English audio files directly to English transcripts.
- **Export Capabilities**: Instantly download the raw transcript as a text file.

---

## 🚀 Setup & Installation

### Prerequisites

1. Install **FFmpeg** on your system.
2. Download and run **Ollama** from [ollama.com](https://ollama.com/).
3. Pull your preferred summarization model via Ollama:
   ```bash
   ollama pull llama3.2
   ```

### Quick Start

Run the automated setup and startup script:

```bash
# Clone the repository
git clone https://github.com/KevinAi18/ai-meeting-summarizer.git
cd ai-meeting-summarizer

# Make the setup script executable and run it
chmod +x run_meeting_summarizer.sh
./run_meeting_summarizer.sh
```

The script automatically sets up a Python virtual environment, installs dependencies, clones and builds `whisper.cpp`, downloads the default `small` Whisper model, and launches the Gradio server.

Access the application in your browser at `http://127.0.0.1:7860`.

---

## 🎙️ Deep Dive: Interview Q&A

### **Why use `whisper.cpp` instead of the official Python `whisper` library?**

For production-grade local tools, Python's official Whisper library has several drawbacks:
1. **System Overhead & Bloat**: The Python library pulls in heavy PyTorch and CUDA dependencies, easily exceeding 4GB of disk space and requiring substantial memory.
2. **Speed & Efficiency**: `whisper.cpp` is written in pure C/C++ with no heavy runtime dependencies. It supports hardware acceleration (Metal on macOS, CoreML, OpenMP/AVX on CPU, and CUDA/OpenCL on GPU) making it significantly faster, especially on standard consumer laptops or CPU-only systems.
3. **Deployment**: Running a standalone C++ binary is cleaner and more reliable to package in automation scripts than orchestrating complex PyTorch runtime environments.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
 
## Output Includes 
- Concise meeting summary 
- List of action items with assigned owners where mentioned 
- Key decisions made during the meeting 
- Full transcript with speaker labels 
 
## Why This Project 
Built to reduce time spent writing meeting notes manually and to make it easy to search past discussions. 
