 
## 2026-06-15 
### Meeting Summarization Techniques 
- Extractive summarization picks key sentences from transcript 
- Abstractive summarization generates new text using LLM 
- Whisper used for audio to text transcription 
- Speaker diarization helps assign dialogue to correct speaker 
 
## 2026-06-17 
### Action Item Extraction from Meetings 
- Action items extracted using structured output from LLM 
- Prompt engineering key to getting consistent JSON output 
- Named Entity Recognition helps identify owners of action items 
- Deadline detection uses regex combined with LLM date parsing 
 
## 2026-06-19 
### Whisper Integration for Meeting Transcription 
- OpenAI Whisper converts meeting audio to text transcript 
- Supports multiple languages and accents out of the box 
- Large-v3 model gives best accuracy for meeting transcription 
- Faster-Whisper library reduces transcription time by 4x 
 
## 2026-06-22 
### Sentiment Analysis on Meeting Transcripts 
- Sentiment analysis detects tone of each speaker in meeting 
- Positive sentiment linked to agreement and decision making 
- Negative sentiment flags conflict or confusion in discussion 
- VADER and FinBERT both tested for meeting sentiment scoring 
 
## 2026-06-24 
### Multi Speaker Diarization Notes 
- Diarization assigns spoken segments to individual speakers 
- Pyannote audio library used for speaker diarization pipeline 
- Speaker embeddings clustered to identify unique participants 
- Diarization output merged with Whisper transcript for full log 
