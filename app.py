import torch
import gradio as gr
from transformers import pipeline

# ---------------------------------------------------------
# DEVICE SETUP (AUTO GPU / CPU)
# ---------------------------------------------------------
device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=device,
    torch_dtype=dtype
)

# ---------------------------------------------------------
# CHUNKING LOGIC FOR LONG TEXT
# ---------------------------------------------------------
def chunk_text(text, max_chars=1000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks

# ---------------------------------------------------------
# MAIN SUMMARY FUNCTION
# ---------------------------------------------------------
def summarize_text(input_text, summary_size):
    if not input_text or not input_text.strip():
        return "⚠️ Please enter text to summarize."

    if len(input_text) < 50:
        return "⚠️ Text too short. Minimum 50 characters required."

    with torch.no_grad():
        # If text is short, summarize directly
        if len(input_text) <= 1200:
            result = summarizer(
                input_text,
                max_length=summary_size,
                min_length=int(summary_size * 0.4),
                do_sample=False
            )
            return result[0]["summary_text"]

        # For long text → chunk + summarize
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

        # Second pass summary (meta-summary)
        combined = " ".join(partial_summaries)

        final_summary = summarizer(
            combined,
            max_length=summary_size,
            min_length=int(summary_size * 0.5),
            do_sample=False
        )

        return final_summary[0]["summary_text"]

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧠 GenAILearniverse – Advanced Text Summarizer
        **Fast • Accurate • Handles Long Documents**
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="📄 Enter Text",
                lines=10,
                placeholder="Paste articles, research papers, or long content (50+ characters)..."
            )

            summary_size = gr.Slider(
                minimum=80,
                maximum=300,
                value=150,
                step=10,
                label="📝 Summary Length (words)"
            )

            summarize_btn = gr.Button("✨ Summarize", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="📌 Summary Output",
                lines=10
            )

    summarize_btn.click(
        fn=summarize_text,
        inputs=[input_text, summary_size],
        outputs=output_text
    )

demo.launch()
