---
title: TextSummarizer
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.29.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Text Summarizer

A web application that automatically summarizes long text using state-of-the-art deep learning models. Built with Hugging Face Transformers and Gradio for an intuitive user interface.

## Features

- **Automatic Text Summarization** - Condenses long text into concise summaries
- **Fast & Efficient** - Uses DistilBART model optimized for performance
- **User-Friendly Interface** - Simple web-based UI powered by Gradio
- **GPU Optimized** - Uses bfloat16 precision for reduced memory usage

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd TextSummarizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Locally

```bash
python app.py
```

The application will start a local web server. Open your browser and navigate to the URL displayed (typically `http://localhost:7860`).

### How to Use

1. Enter or paste the text you want to summarize in the input box
2. Click the "Submit" button
3. The summarized text will appear in the output box

## Model Details

- **Model**: `sshleifer/distilbart-cnn-12-6`
- **Type**: DistilBART (distilled BART)
- **Task**: Abstractive Text Summarization
- **Input**: Plain text (recommended: 50-1024 tokens)
- **Output**: Concise summary capturing key information

## Requirements

- `transformers` - Hugging Face NLP library
- `torch` - PyTorch deep learning framework
- `gradio` - Web UI framework

## Limitations

- Maximum recommended input length: ~1024 tokens (approximately 4000 characters)
- Works best with English text
- Requires internet connection for first model download

## License

Apache 2.0

## References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Gradio Documentation](https://www.gradio.app/)
- [DistilBART Model Card](https://huggingface.co/sshleifer/distilbart-cnn-12-6)
