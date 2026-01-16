---
title: AI Text Toolkit Pro
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.29.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🚀 AI Text Toolkit Pro

![AI Text Toolkit Pro Banner](https://raw.githubusercontent.com/shreyasTalwar/text_summariser/main/assets/banner.png)

> **Empowering text analysis with State-of-the-Art AI. Summarize, Analyze, and Transcribe with ease.**

AI Text Toolkit Pro is a comprehensive web application designed to handle complex text processing tasks using advanced deep learning models. Whether you're a researcher, student, or professional, this toolkit provides the tools you need to extract insights from text and video content instantly.

---

## ✨ Key Features

### 📝 Text Summarization
- **Smart Summaries**: Condense long articles, research papers, or reports into concise summaries.
- **Adjustable Length**: Control the summary size (50-500 words) to fit your needs.
- **Chunking Engine**: Automatically handles large documents by splitting them into manageable segments for processing.

### 🎬 YouTube Summarizer
- **Instant Transcripts**: Fetches transcripts directly from YouTube URLs.
- **Video Summaries**: Summarizes the video content, saving you hours of watch time.
- **Full Transparency**: View the raw transcript alongside the AI-generated summary.

### 📊 Sentiment Analysis
- **Deep Insight**: Understand the emotional tone (Positive/Negative) of any text.
- **Single Analysis**: Check individual reviews or comments.
- **Bulk Processing**: Upload **Excel (.xlsx)** files for mass sentiment analysis with confidence scores.
- **Visual Analytics**: Automatically generates distribution charts for bulk data.

---

## 🛠️ Tech Stack

- **Core**: Python 3.8+
- **Deep Learning**: Hugging Face Transformers (`BART`, `DistilBERT`)
- **Backend**: PyTorch
- **UI Framework**: Gradio (Soft Theme with Custom CSS)
- **Data Handling**: Pandas & OpenPyxl
- **Visualization**: Matplotlib

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU (Optional for acceleration)

### Installation

1. **Clone the Repo**
   ```bash
   git clone https://github.com/shreyasTalwar/text_summariser.git
   cd text_summariser
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the App**
   ```bash
   python app.py
   ```

Navigate to `http://localhost:7860` in your browser.

---

## 🔧 Optimization

The application automatically detects your hardware configurations:
- **🚀 GPU Accelerated**: Uses CUDA for lightning-fast inference if available.
- **💻 CPU Mode**: Optimized for standard hardware performance.

---

## 📋 Sample Excel Format for Bulk Analysis

To use the bulk analysis feature, ensure your Excel file has a column named **"Reviews"**.

| Reviews |
| :--- |
| This tool is absolutely incredible! |
| I found the summary a bit too short for my liking. |
| Speed is impressive even on CPU. |

---

## 📄 License

Distributed under the **Apache 2.0 License**. See `LICENSE` for more information.

---

## 👋 Contact

**Shreyas V Talwar**  
GitHub: [@shreyasTalwar](https://github.com/shreyasTalwar)

---

*Built with ❤️ using Hugging Face & Gradio*
