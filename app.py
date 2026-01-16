import re
import torch
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

# ---------------------------------------------------------
# DEVICE SETUP (AUTO GPU / CPU)
# ---------------------------------------------------------
device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
# Text Summarizer Model
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=device,
    torch_dtype=dtype
)

# Sentiment Analyzer Model
sentiment_analyzer = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

# ---------------------------------------------------------
# TEXT SUMMARIZER FUNCTIONS
# ---------------------------------------------------------
def chunk_text(text, max_chars=1000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks

def summarize_text(input_text, summary_size):
    if not input_text or not input_text.strip():
        return "⚠️ Please enter text to summarize."

    if len(input_text) < 50:
        return "⚠️ Text too short. Minimum 50 characters required."

    with torch.no_grad():
        if len(input_text) <= 1200:
            result = summarizer(
                input_text,
                max_length=summary_size,
                min_length=int(summary_size * 0.4),
                do_sample=False
            )
            return result[0]["summary_text"]

        chunks = chunk_text(input_text)
        partial_summaries = []

        for chunk in chunks:
            output = summarizer(
                chunk,
                max_length=200,
                min_length=80,
                do_sample=False
            )
            partial_summaries.append(output[0]["summary_text"])

        combined = " ".join(partial_summaries)
        final_summary = summarizer(
            combined,
            max_length=summary_size,
            min_length=int(summary_size * 0.5),
            do_sample=False
        )
        return final_summary[0]["summary_text"]

# ---------------------------------------------------------
# SENTIMENT ANALYZER FUNCTIONS
# ---------------------------------------------------------
def analyze_single_sentiment(review):
    sentiment = sentiment_analyzer(review)
    return sentiment[0]['label']

def create_sentiment_chart(df):
    sentiment_counts = df['Sentiment'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#4CAF50' if label == 'POSITIVE' else '#F44336' for label in sentiment_counts.index]
    sentiment_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    plt.tight_layout()
    return fig

def analyze_reviews_from_file(file_object):
    if file_object is None:
        return None, None
    
    try:
        df = pd.read_excel(file_object)
        
        if 'Reviews' not in df.columns:
            return pd.DataFrame({'Error': ["Excel file must contain a 'Reviews' column."]}), None
        
        df['Sentiment'] = df['Reviews'].apply(analyze_single_sentiment)
        chart = create_sentiment_chart(df)
        return df, chart
    
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]}), None

def analyze_single_text(text):
    if not text or not text.strip():
        return "⚠️ Please enter text to analyze."
    
    result = sentiment_analyzer(text)
    label = result[0]['label']
    score = result[0]['score']
    
    emoji = "😊" if label == "POSITIVE" else "😞"
    return f"{emoji} **{label}** (Confidence: {score:.1%})"

# ---------------------------------------------------------
# YOUTUBE TRANSCRIPT SUMMARIZER FUNCTIONS
# ---------------------------------------------------------
def extract_video_id(url):
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

def summarize_youtube_video(video_url, summary_size):
    if not video_url or not video_url.strip():
        return "⚠️ Please enter a YouTube URL.", ""
    
    video_id = extract_video_id(video_url)
    if not video_id:
        return "❌ Invalid YouTube URL. Could not extract video ID.", ""

    try:
        # New API: use fetch() method
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['en']).fetch()
        
        # Format transcript to plain text
        text_transcript = " ".join([entry['text'] for entry in transcript])
        
        if len(text_transcript) < 50:
            return "⚠️ Transcript too short to summarize.", text_transcript
        
        summary = summarize_text(text_transcript, summary_size)
        return summary, text_transcript
    
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

# ---------------------------------------------------------
# UNIFIED UI WITH TABS
# ---------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚀 AI Text Toolkit
        **Summarize • Analyze Sentiment • YouTube Transcripts**
        """
    )
    
    with gr.Tabs():
        # ============ TAB 1: TEXT SUMMARIZER ============
        with gr.Tab("📝 Text Summarizer"):
            gr.Markdown("### Condense long text into concise summaries")
            
            with gr.Row():
                with gr.Column(scale=2):
                    sum_input = gr.Textbox(
                        label="📄 Enter Text",
                        lines=8,
                        placeholder="Paste articles, research papers, or long content (50+ characters)..."
                    )
                    sum_slider = gr.Slider(
                        minimum=80,
                        maximum=300,
                        value=150,
                        step=10,
                        label="📏 Summary Length (words)"
                    )
                    sum_btn = gr.Button("✨ Summarize", variant="primary")
                
                with gr.Column(scale=1):
                    sum_output = gr.Textbox(label="📌 Summary", lines=8)
            
            sum_btn.click(fn=summarize_text, inputs=[sum_input, sum_slider], outputs=sum_output)
        
        # ============ TAB 2: YOUTUBE SUMMARIZER ============
        with gr.Tab("🎬 YouTube Summarizer"):
            gr.Markdown("### Summarize any YouTube video from its transcript")
            
            with gr.Row():
                with gr.Column(scale=2):
                    yt_url = gr.Textbox(
                        label="🔗 YouTube URL",
                        lines=1,
                        placeholder="Paste YouTube video URL (e.g., https://youtube.com/watch?v=...)"
                    )
                    yt_slider = gr.Slider(
                        minimum=80,
                        maximum=300,
                        value=150,
                        step=10,
                        label="📏 Summary Length (words)"
                    )
                    yt_btn = gr.Button("🎬 Summarize Video", variant="primary")
                
                with gr.Column(scale=1):
                    yt_summary = gr.Textbox(label="📌 Video Summary", lines=6)
            
            with gr.Accordion("📜 Full Transcript", open=False):
                yt_transcript = gr.Textbox(label="Raw Transcript", lines=10)
            
            yt_btn.click(fn=summarize_youtube_video, inputs=[yt_url, yt_slider], outputs=[yt_summary, yt_transcript])
        
        # ============ TAB 3: SENTIMENT ANALYZER ============
        with gr.Tab("📊 Sentiment Analyzer"):
            gr.Markdown("### Analyze sentiment from text or Excel files")
            
            with gr.Tabs():
                # Sub-tab: Single Text Analysis
                with gr.Tab("Single Text"):
                    with gr.Row():
                        with gr.Column():
                            sent_text = gr.Textbox(
                                label="💬 Enter Review/Text",
                                lines=4,
                                placeholder="Type a review or comment to analyze..."
                            )
                            sent_text_btn = gr.Button("🔍 Analyze Sentiment", variant="primary")
                        
                        with gr.Column():
                            sent_text_output = gr.Markdown(label="Result")
                    
                    sent_text_btn.click(fn=analyze_single_text, inputs=sent_text, outputs=sent_text_output)
                
                # Sub-tab: Bulk File Analysis
                with gr.Tab("Bulk Analysis (Excel)"):
                    gr.Markdown("Upload an Excel file with a **'Reviews'** column")
                    
                    with gr.Row():
                        with gr.Column():
                            file_input = gr.File(
                                file_types=[".xlsx"],
                                label="📁 Upload Excel File"
                            )
                            file_btn = gr.Button("📊 Analyze File", variant="primary")
                        
                        with gr.Column():
                            file_output = gr.Dataframe(label="📋 Results")
                    
                    chart_output = gr.Plot(label="📈 Sentiment Distribution")
                    
                    file_btn.click(fn=analyze_reviews_from_file, inputs=file_input, outputs=[file_output, chart_output])
    
    gr.Markdown("---\n*Powered by Hugging Face Transformers • GPU-accelerated when available*")

demo.launch(share=True)
