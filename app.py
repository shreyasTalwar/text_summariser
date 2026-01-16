import re
import torch
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from functools import lru_cache
import logging

# ---------------------------------------------------------
# LOGGING & ERROR HANDLING
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# DEVICE SETUP (AUTO GPU / CPU)
# ---------------------------------------------------------
device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device_name = "🚀 GPU Accelerated" if torch.cuda.is_available() else "💻 CPU Mode"

# ---------------------------------------------------------
# LOAD MODELS WITH CACHING
# ---------------------------------------------------------
@lru_cache(maxsize=1)
def get_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=device,
        torch_dtype=dtype
    )

@lru_cache(maxsize=1)
def get_sentiment_analyzer():
    return pipeline(
        "text-classification",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )

summarizer = get_summarizer()
sentiment_analyzer = get_sentiment_analyzer()

# ---------------------------------------------------------
# TEXT SUMMARIZER FUNCTIONS
# ---------------------------------------------------------
def chunk_text(text, max_chars=1000):
    """Split text into manageable chunks"""
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_chars:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_text(input_text, summary_size):
    """Summarize text with advanced error handling"""
    if not input_text or not input_text.strip():
        return "⚠️ Please enter text to summarize."

    text_clean = input_text.strip()
    if len(text_clean) < 50:
        return "⚠️ Text too short. Minimum 50 characters required."

    try:
        with torch.no_grad():
            if len(text_clean) <= 1024:
                result = summarizer(
                    text_clean,
                    max_length=summary_size,
                    min_length=max(30, int(summary_size * 0.3)),
                    do_sample=False
                )
                return result[0]["summary_text"]

            # For longer texts, use chunking strategy
            chunks = chunk_text(text_clean, 1000)
            partial_summaries = []

            for chunk in chunks:
                if len(chunk.split()) < 10:
                    continue
                output = summarizer(
                    chunk,
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )
                partial_summaries.append(output[0]["summary_text"])

            if not partial_summaries:
                return "⚠️ Text could not be summarized. Please try different text."

            combined = " ".join(partial_summaries)
            final_summary = summarizer(
                combined,
                max_length=summary_size,
                min_length=max(30, int(summary_size * 0.4)),
                do_sample=False
            )
            return final_summary[0]["summary_text"]
    
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return f"❌ Error: {str(e)}"

# ---------------------------------------------------------
# SENTIMENT ANALYZER FUNCTIONS
# ---------------------------------------------------------
def analyze_single_sentiment(review):
    """Analyze sentiment of a single review"""
    try:
        sentiment = sentiment_analyzer(review)[0]
        return sentiment['label']
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        return "UNKNOWN"

def create_sentiment_chart(df):
    """Create professional sentiment visualization"""
    try:
        sentiment_counts = df['Sentiment'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#4CAF50' if label == 'POSITIVE' else '#F44336' for label in sentiment_counts.index]
        
        wedges, texts, autotexts = ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        ax.set_title('Sentiment Distribution Analysis', fontsize=16, fontweight='bold', pad=20)
        
        # Add count labels
        for i, (label, count) in enumerate(zip(sentiment_counts.index, sentiment_counts.values)):
            texts[i].set_text(f'{label}\n(n={count})')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Chart creation error: {str(e)}")
        return None

def analyze_reviews_from_file(file_object):
    """Analyze reviews from Excel file"""
    if file_object is None:
        return None, None
    
    try:
        df = pd.read_excel(file_object)
        
        if 'Reviews' not in df.columns:
            error_msg = "❌ Excel file must contain a 'Reviews' column."
            return pd.DataFrame({'Error': [error_msg]}), None
        
        # Remove empty reviews
        df = df[df['Reviews'].notna()].copy()
        
        if df.empty:
            return pd.DataFrame({'Error': ['No valid reviews found']}), None
        
        # Analyze sentiments
        df['Sentiment'] = df['Reviews'].apply(lambda x: analyze_single_sentiment(str(x)))
        df['Confidence'] = df['Reviews'].apply(
            lambda x: f"{sentiment_analyzer(str(x))[0]['score']:.1%}"
        )
        
        chart = create_sentiment_chart(df)
        return df, chart
    
    except Exception as e:
        logger.error(f"File analysis error: {str(e)}")
        return pd.DataFrame({'Error': [str(e)]}), None

def analyze_single_text(text):
    """Analyze sentiment of single text"""
    if not text or not text.strip():
        return "⚠️ Please enter text to analyze."
    
    try:
        result = sentiment_analyzer(text)[0]
        label = result['label']
        score = result['score']
        
        emoji = "😊" if label == "POSITIVE" else "😞"
        return f"{emoji} **{label}** (Confidence: {score:.1%})"
    except Exception as e:
        logger.error(f"Single text analysis error: {str(e)}")
        return f"❌ Error analyzing text: {str(e)}"

# ---------------------------------------------------------
# YOUTUBE TRANSCRIPT FUNCTIONS (FIXED)
# ---------------------------------------------------------
def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com\/v\/)([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_id):
    """Fetch YouTube transcript with proper error handling"""
    try:
        # List available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get English transcript
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # If English not available, get first available
            transcript = transcript_list.find_transcript(transcript_list.get_available_languages())
        
        # Fetch transcript data
        transcript_data = transcript.fetch()
        
        # Convert to plain text
        text_transcript = " ".join([entry['text'] for entry in transcript_data])
        return text_transcript, None
    
    except Exception as e:
        error_msg = f"❌ Error fetching transcript: {str(e)}"
        logger.error(error_msg)
        return "", error_msg

def summarize_youtube_video(video_url, summary_size):
    """Summarize YouTube video from transcript"""
    if not video_url or not video_url.strip():
        return "⚠️ Please enter a YouTube URL.", ""
    
    video_id = extract_video_id(video_url)
    if not video_id:
        return "❌ Invalid YouTube URL format. Please check the URL.", ""

    # Fetch transcript
    text_transcript, error = get_youtube_transcript(video_id)
    
    if error:
        return error, ""
    
    if len(text_transcript) < 50:
        return "⚠️ Transcript too short to summarize.", text_transcript
    
    try:
        summary = summarize_text(text_transcript, summary_size)
        return summary, text_transcript
    except Exception as e:
        return f"❌ Error summarizing: {str(e)}", text_transcript

# ---------------------------------------------------------
# GRADIO UI - PROFESSIONAL DESIGN
# ---------------------------------------------------------
css = """
.container { max-width: 1200px; margin: auto; }
.header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
          color: white; padding: 30px; border-radius: 10px; text-align: center; }
.tab-button { font-weight: bold; padding: 10px 20px; }
.result-box { background: #f8f9fa; padding: 20px; border-radius: 8px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, title="🚀 AI Text Toolkit") as demo:
    
    # Header
    with gr.Row(elem_classes="header"):
        gr.Markdown(
            f"""
            # 🚀 AI Text Toolkit Pro
            **Summarize • Analyze Sentiment • YouTube Transcripts**
            
            {device_name}
            """
        )
    
    with gr.Tabs():
        # ============ TAB 1: TEXT SUMMARIZER ============
        with gr.Tab("📝 Text Summarizer", elem_classes="tab-button"):
            gr.Markdown("### ✨ Condense long text into concise summaries instantly")
            
            with gr.Row():
                with gr.Column(scale=2):
                    sum_input = gr.Textbox(
                        label="📄 Enter Text",
                        lines=10,
                        placeholder="Paste articles, research papers, reports, or long content (50+ characters)...",
                        elem_classes="result-box"
                    )
                    with gr.Row():
                        sum_slider = gr.Slider(
                            minimum=50,
                            maximum=500,
                            value=150,
                            step=10,
                            label="📏 Summary Length (words)"
                        )
                        sum_btn = gr.Button("✨ Summarize", variant="primary", scale=1)
                
                with gr.Column(scale=1):
                    sum_output = gr.Textbox(
                        label="📌 Summary",
                        lines=10,
                        interactive=False,
                        elem_classes="result-box"
                    )
            
            sum_btn.click(fn=summarize_text, inputs=[sum_input, sum_slider], outputs=sum_output)
        
        # ============ TAB 2: YOUTUBE SUMMARIZER ============
        with gr.Tab("🎬 YouTube Summarizer", elem_classes="tab-button"):
            gr.Markdown("### 🎥 Summarize any YouTube video from its transcript")
            
            with gr.Row():
                with gr.Column(scale=2):
                    yt_url = gr.Textbox(
                        label="🔗 YouTube URL",
                        lines=2,
                        placeholder="Paste YouTube video URL (e.g., https://www.youtube.com/watch?v=...)",
                        elem_classes="result-box"
                    )
                    with gr.Row():
                        yt_slider = gr.Slider(
                            minimum=50,
                            maximum=500,
                            value=150,
                            step=10,
                            label="📏 Summary Length (words)"
                        )
                        yt_btn = gr.Button("🎬 Summarize Video", variant="primary", scale=1)
                
                with gr.Column(scale=1):
                    yt_summary = gr.Textbox(
                        label="📌 Video Summary",
                        lines=8,
                        interactive=False,
                        elem_classes="result-box"
                    )
            
            with gr.Accordion("📜 Full Transcript", open=False):
                yt_transcript = gr.Textbox(
                    label="Raw Transcript",
                    lines=12,
                    interactive=False,
                    elem_classes="result-box"
                )
            
            yt_btn.click(fn=summarize_youtube_video, inputs=[yt_url, yt_slider], outputs=[yt_summary, yt_transcript])
        
        # ============ TAB 3: SENTIMENT ANALYZER ============
        with gr.Tab("📊 Sentiment Analyzer", elem_classes="tab-button"):
            gr.Markdown("### 💭 Analyze sentiment from text or Excel files")
            
            with gr.Tabs():
                # Sub-tab: Single Text Analysis
                with gr.Tab("Single Text"):
                    with gr.Row():
                        with gr.Column():
                            sent_text = gr.Textbox(
                                label="💬 Enter Review/Text",
                                lines=5,
                                placeholder="Type a review, comment, or text to analyze sentiment...",
                                elem_classes="result-box"
                            )
                            sent_text_btn = gr.Button("🔍 Analyze Sentiment", variant="primary")
                        
                        with gr.Column():
                            sent_text_output = gr.Markdown(
                                label="Result",
                                elem_classes="result-box"
                            )
                    
                    sent_text_btn.click(fn=analyze_single_text, inputs=sent_text, outputs=sent_text_output)
                
                # Sub-tab: Bulk File Analysis
                with gr.Tab("Bulk Analysis (Excel)"):
                    gr.Markdown("Upload an Excel file with a **'Reviews'** column to analyze multiple reviews at once.")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            file_input = gr.File(
                                file_types=[".xlsx", ".xls"],
                                label="📁 Upload Excel File"
                            )
                            file_btn = gr.Button("📊 Analyze File", variant="primary")
                        
                        with gr.Column(scale=1):
                            gr.Markdown(
                                """
                                **Sample Format:**
                                | Reviews |
                                |---------|
                                | Great product! | 
                                | Not satisfied |
                                """
                            )
                    
                    file_output = gr.Dataframe(label="📋 Results (with Sentiment & Confidence)")
                    chart_output = gr.Plot(label="📈 Sentiment Distribution")
                    
                    file_btn.click(fn=analyze_reviews_from_file, inputs=file_input, outputs=[file_output, chart_output])
    
    # Footer
    gr.Markdown(
        """
        ---
        **🔧 Features:**
        - Advanced text summarization with smart chunking
        - Multi-language sentiment analysis
        - Batch processing for large datasets
        - GPU-accelerated when available
        
        *Built with Hugging Face Transformers | Gradio | PyTorch*
        """
    )

# ---------------------------------------------------------
# LAUNCH
# ---------------------------------------------------------
if __name__ == "__main__":
    demo.launch(share=True, show_error=True)