# AI Agent Prototype: Audio & Video Summarizer

[cite_start]This is a submission for the AI Agent Prototype internship assignment[cite: 2, 31].

- **Name:** [Your Name]
- **University:** [Your University]
- **Department:** [Your Department]

---

## 1. Project Overview

[cite_start]This project is an advanced AI agent that automates the manual task of processing and understanding long-form media[cite: 4]. It accepts both **audio and video files**, transcribes the content, extracts on-screen text, generates a detailed summary, and identifies all actionable tasks.

[cite_start]This project successfully implements all mandatory core features and all optional bonus features from the assignment[cite: 3, 14].

---

## 2. Core Features & Feature Weights

This agent is a multi-component system where each part adds significant value. Here is a breakdown of the features and their "weight" or importance to the project's success.

### 🌟 1. Fine-Tuned Summarizer (Heavy Weight)

[cite_start]This is the core intelligence of the agent, directly fulfilling a mandatory requirement[cite: 6].

* [cite_start]**Model:** Uses a `facebook/bart-large-cnn` model fine-tuned with **LoRA** (Low-Rank Adaptation)[cite: 8].
* **Key Feature (Map-Reduce):** A 10-minute video creates a transcript far too long for a standard model's token limit. This summarizer solves this by implementing a **recursive Map-Reduce strategy**.
    1.  **Map:** The long transcript is automatically split into overlapping chunks.
    2.  **Summarize (Map):** Each chunk is summarized individually using a special `"chunk_summary"` prompt.
    3.  **Combine:** All the "chunk summaries" are combined into one document.
    4.  **Reduce:** This *new document* is recursively passed back into the summarizer with a `"final_summary"` prompt to create the final, coherent, and detailed summary.
* [cite_start]**Why Fine-Tune?** [cite: 12] The base model does not understand the custom prompts (`"chunk_summary"`, `"final_summary"`) needed for this advanced logic. Fine-tuning for **Task Specialization** and **Adapted Style** was mandatory to make the model a reliable component in this summarization chain.

### 🌟 2. Task & Deadline Extractor (Heavy Weight)

This component moves the agent from a simple summary tool to a true productivity assistant.

* **Hybrid System:** It uses a sophisticated hybrid approach in `task_extractor.py` to ensure high accuracy.
* **Methods Used:**
    1.  **NLP (spaCy):** Parses sentences to find action verbs and their subjects/objects.
    2.  **Regex (Pattern Matching):** Uses a library of regular expressions to find common task-oriented keywords (e.g., "we need to," "the assignment is").
    3.  **Semantic Analysis:** Identifies imperative sentences or future-tense statements as potential tasks.
* **Deadline Extractor:** This is a sub-feature of the task extractor. It uses advanced regex in `_parse_deadline` to find and parse deadlines, successfully converting relative terms like **"tomorrow"**, **"next week"**, or **"EOD" (End of Day)** into standard `datetime` objects.
* **Categorization:** Each task is automatically categorized (e.g., 'academic', 'communication') and given an 'effort' score ('high', 'medium', 'low').

### 🌟 3. Multi-Tool/Agent Collaboration (Medium-Heavy Weight)

[cite_start]This project is built as a **Planner + Multi-Tool Executor** system, fulfilling a bonus requirement[cite: 15].

* **Planner:** `AudioSummaryAgent` acts as the "brain." It analyzes the file (`_analyze_and_plan`) and decides *which* tools to use.
* [cite_start]**Executors (Tools):** [cite: 16]
    * **`VideoProcessor`:** Handles video files, using `moviepy` to extract audio and `easyocr` to perform **OCR** on frames, capturing on-screen text.
    * **`AudioProcessor`:** Transcribes all audio using `whisper`.
    * **`FineTunedSummarizer`:** The "summarization" tool.
    * **`TaskExtractor`:** The "task" tool.

### 🌟 4. Insight & Evaluation Generation (Medium Weight)

[cite_start]This fulfills the mandatory requirement to implement evaluation metrics[cite: 13].

* **Quality Score:** The `quality_assessor.py` module runs a full evaluation **on every run**. It uses **ROUGE** scores (for summary content), readability metrics, and custom heuristics to calculate a final "Quality Score".
* **Insights Tab:** The agent provides meta-analysis on its own performance. It uses the quality score to provide plain-English recommendations, such as warning the user if the summary quality is low or if no tasks were found in a "meeting" type file.

### 🌟 5. Web Interface (Bonus)

* [cite_start]A full-featured Streamlit application (`src/ui/app.py`) provides an easy-to-use interface for uploading files and viewing the multi-tabbed results (Transcript, Summary, Tasks, Quality, Insights)[cite: 17].

---

## 3. AI Agent Architecture Document

[cite_start]This section covers the "AI agent architecture document" deliverable[cite: 24].

### Interaction Flow
1.  A user uploads a media file (e.g., `.mp4` or `.mp3`) to the Streamlit UI.
2.  The `AudioSummaryAgent` (Planner) receives the file path.
3.  **Plan:** The agent calls `_analyze_and_plan` to determine content type (e.g., 'academic', 'meeting') and file format (video/audio).
4.  **Execute (Process):**
    * If **Video**, the `VideoProcessor` is called to extract audio *and* run OCR on video frames.
    * The audio is passed to the `AudioProcessor` for transcription.
    * The agent combines the audio transcript and OCR text into a single document.
5.  **Execute (Summarize):** The combined transcript is sent to the `FineTunedSummarizer` (Tool), which automatically triggers its internal Map-Reduce logic for long text.
6.  **Execute (Extract):** The transcript and summary are sent to `TaskExtractor` (Tool) to find all tasks.
7.  **Execute (Evaluate):** The transcript, summary, and tasks are sent to `QualityAssessor` (Tool) to generate a final quality score and insights.
8.  **Respond:** The agent combines all these results and sends the final, structured output to the Streamlit UI for display.

---

## 4. Data Science Report

[cite_start]This section covers the "Data science report" deliverable[cite: 25].

### [cite_start]Fine-Tuning Setup [cite: 26]

* **Base Model:** `facebook/bart-large-cnn`
* **Method:** Parameter-Efficient Fine-Tuning (PEFT) using **LoRA** (Low-Rank Adaptation), as implemented in `train_model.py`.
* **Training Data:** A small, custom dataset of `(input, target)` pairs is included in `train_model.py`. This data trains the model to respond to specific instructional prompts.
* [cite_start]**Reason for Fine-Tuning:** The primary reason was **task specialization**[cite: 12]. The base model does not know how to handle the custom prompts required for our Map-Reduce strategy (e.g., `"Summarize this section..."` and `"Combine these summaries..."`). Fine-tuning teaches the model to follow these instructions, allowing it to act as an intelligent part of the agent's summarization chain.

### [cite_start]Evaluation Methodology [cite: 28]

The `quality_assessor.py` module implements a custom evaluation framework:
* **Transcript Quality:** Measured by the Whisper model's confidence, text readability (Flesch ease), and coherence.
* **Summary Quality:** Measured quantitatively using **ROUGE-1, ROUGE-2, and ROUGE-L** scores against the source transcript.
* **Task Quality:** Measured using a custom model that scores each task on **Relevance, Clarity, and Actionability**.
* **Overall Score:** A final weighted average of the Transcript, Summary, and Task scores, which is displayed in the UI.

---

## 5. How to Run

1.  **Clone Repository:**
    ```bash
    git clone [https://github.com/akshat5015/ai-agent.git](https://github.com/akshat5015/ai-agent.git)
    cd ai-agent
    ```

2.  **Setup Environment:**
    (Create a new conda/venv environment for Python 3.10+)
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

3.  **(Optional) Train the Model:**
    The project already includes a fine-tuned LoRA model. To re-train (e.g., if you change the base model), run:
    ```bash
    python -m scripts.train_model --training-data sample
    ```

4.  **Run the AI Agent:**
    ```bash
    # (Recommended on Windows to prevent a common torch/opencv warning)
    set KMP_DUPLICATE_LIB_OK=TRUE
    
    # Launch the Streamlit UI
    streamlit run src/ui/app.py
    ```

## 6. Demo Screenshots

*Include your demo screenshots here, like the one you sent.*

![Processing Results](image_eb4302.png)