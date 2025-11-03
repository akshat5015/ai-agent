# Audio-to-Summary AI Agent

An intelligent AI agent that automates the conversion of lengthy class recordings, meeting audios, or study session transcripts into concise summaries and actionable task lists.

## 🎯 Project Overview

This AI agent addresses the manual task of processing long-form audio content by:
- Converting audio recordings to text using advanced speech recognition
- Generating concise, informative summaries using fine-tuned models
- Extracting actionable tasks and key insights
- Providing reasoning and planning capabilities for content organization

## 🧠 Core Features

### Mandatory Features
- **Audio Processing Pipeline**: Converts audio files to high-quality transcripts
- **Fine-Tuned Summarization Model**: Custom LoRA-tuned model for domain-specific summarization
- **Task Extraction Engine**: Intelligent identification of actionable items
- **Reasoning System**: AI agent that can plan and execute processing workflows
- **Evaluation Metrics**: Quality assessment for summaries and task extraction

### Bonus Features
- **Multi-Agent Collaboration**: Specialized agents for different content types
- **RAG Integration**: Enhanced context understanding through retrieval
- **Web Interface**: User-friendly Streamlit application
- **Custom Tools**: Extensible pipeline for various audio formats

## 🏗️ Architecture

```
src/
├── agents/          # AI agent implementations
├── models/          # Fine-tuned model components
├── processors/      # Audio and text processing
├── extractors/      # Task and insight extraction
├── evaluators/      # Quality assessment metrics
└── ui/              # User interface components
```

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Run the AI Agent**:
   ```bash
   python src/main.py --input audio_file.wav --output summary.json
   ```

3. **Launch Web Interface**:
   ```bash
   streamlit run src/ui/app.py
   ```

## 📊 Evaluation Metrics

- **Summary Quality**: ROUGE scores, BLEU scores, semantic similarity
- **Task Extraction Accuracy**: Precision, recall, F1-score for identified tasks
- **Processing Efficiency**: Time-to-completion, audio-to-text accuracy
- **User Satisfaction**: Subjective quality ratings

## 🔧 Fine-Tuning Process

The agent uses LoRA (Low-Rank Adaptation) fine-tuning on a base language model to specialize in:
- Academic content summarization
- Meeting minutes generation
- Task identification and prioritization
- Context-aware content structuring

## 📁 Data Sources

- Academic lecture recordings
- Meeting audio files
- Study session transcripts
- Podcast episodes (educational content)

## 🤝 Contributing

This project is part of an AI Agent Prototype assignment focusing on practical automation of manual tasks using fine-tuned models and intelligent reasoning systems.
