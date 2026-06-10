import subprocess
import os
import gradio as gr
import requests
import json

OLLAMA_SERVER_URL = "http://localhost:11434"
WHISPER_MODEL_DIR = "./whisper.cpp/models"

def get_available_models() -> list[str]:
    """Retrieves a list of all available models from the Ollama server."""
    try:
        response = requests.get(f"{OLLAMA_SERVER_URL}/api/tags")
        if response.status_code == 200:
            models = response.json()["models"]
            return [model["model"] for model in models]
        return []
    except Exception:
        return []

def get_available_whisper_models() -> list[str]:
    """Retrieves available Whisper models based on downloaded .bin files."""
    valid_models = ["base", "small", "medium", "large", "large-V3"]
    if not os.path.exists(WHISPER_MODEL_DIR):
        return ["small"]
    try:
        model_files = [f for f in os.listdir(WHISPER_MODEL_DIR) if f.endswith(".bin")]
        whisper_models = [
            os.path.splitext(f)[0].replace("ggml-", "")
            for f in model_files
            if any(valid_model in f for valid_model in valid_models) and "test" not in f
        ]
        whisper_models = list(set(whisper_models))
        return whisper_models if whisper_models else ["small"]
    except Exception:
        return ["small"]

def preprocess_audio_file(audio_file_path: str) -> str:
    """Converts the input audio file to a WAV format with 16kHz sample rate and mono channel."""
    output_wav_file = f"{os.path.splitext(audio_file_path)[0]}_converted.wav"
    cmd = f'ffmpeg -y -i "{audio_file_path}" -ar 16000 -ac 1 "{output_wav_file}"'
    subprocess.run(cmd, shell=True, check=True)
    return output_wav_file

def summarize_with_model(llm_model_name: str, context: str, text: str) -> str:
    """Uses a specified model on the Ollama server to generate a summary."""
    prompt = f"""You are given a transcript from a meeting, along with some optional context.
    
    Context: {context if context else 'No additional context provided.'}
    
    The transcript is as follows:
    
    {text}
    
    Please summarize the transcript."""

    headers = {"Content-Type": "application/json"}
    data = {"model": llm_model_name, "prompt": prompt}

    response = requests.post(
        f"{OLLAMA_SERVER_URL}/api/generate", json=data, headers=headers, stream=True
    )

    if response.status_code == 200:
        full_response = ""
        try:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    json_line = json.loads(decoded_line)
                    full_response += json_line.get("response", "")
                    if json_line.get("done", False):
                        break
            return full_response
        except json.JSONDecodeError:
            return f"Failed to parse response. Raw response: {response.text}"
    else:
        raise Exception(f"Failed to summarize with model {llm_model_name}: {response.text}")

def translate_and_summarize(
    audio_file_path: str, context: str, whisper_model_name: str, llm_model_name: str, translate: bool
) -> tuple[str, str]:
    """Processes audio and generates a summary and transcript file. Supports optional translation."""
    output_file = "output.txt"
    audio_file_wav = preprocess_audio_file(audio_file_path)

    # Use the -tr flag for translation to English if checkbox is ticked
    tr_flag = "-tr " if translate else ""
    whisper_command = f'./whisper.cpp/main -m ./whisper.cpp/models/ggml-{whisper_model_name}.bin {tr_flag}-f "{audio_file_wav}" > {output_file}'
    subprocess.run(whisper_command, shell=True, check=True)

    with open(output_file, "r") as f:
        transcript = f.read()

    # Save transcript for download
    transcript_file = "transcript.txt"
    with open(transcript_file, "w", encoding="utf-8") as tf:
        tf.write(transcript)

    summary = summarize_with_model(llm_model_name, context, transcript)

    os.remove(audio_file_wav)
    os.remove(output_file)
    return summary, transcript_file

def gradio_app(audio, context: str, whisper_model_name: str, llm_model_name: str, translate: bool) -> tuple[str, str]:
    return translate_and_summarize(audio, context, whisper_model_name, llm_model_name, translate)

if __name__ == "__main__":
    ollama_models = get_available_models()
    whisper_models = get_available_whisper_models()
    
    iface = gr.Interface(
        fn=gradio_app,
        inputs=[
            gr.Audio(type="filepath", label="Upload an audio file"),
            gr.Textbox(
                label="Context (optional)",
                placeholder="Provide any additional context for the summary",
            ),
            gr.Dropdown(
                choices=whisper_models,
                label="Select a Whisper model for audio-to-text conversion",
                value=whisper_models[0] if whisper_models else None,
            ),
            gr.Dropdown(
                choices=ollama_models,
                label="Select a model for summarization",
                value=ollama_models[0] if ollama_models else None,
            ),
            gr.Checkbox(
                label="Translate non-English audio to English",
                value=False,
            ),
        ],
        outputs=[
            gr.Textbox(label="Summary", show_copy_button=True),
            gr.File(label="Download Transcript"),
        ],
        analytics_enabled=False,
        title="Meeting Summarizer",
        description="Upload an audio file of a meeting and get a summary of the key concepts discussed.",
    )

    iface.launch(debug=True)
