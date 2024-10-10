import torch
import gradio as gr
from transformers import pipeline
from huggingface_hub import model_info
import numpy as np
from api import *

MODEL_NAME = "razhan/whisper-small-ckb"

device = 0 if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(task="transcribe")


def transcribe(audio):
    sr, y = audio
    
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    text = pipe({"sampling_rate": sr, "raw": y})["text"]
    api_resp = send_to_claude(text)
    return api_resp


with gr.Blocks(css="""
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');
    
    .gradio-container {
        background-color: #f0f4f8;
        font-family: 'Poppins', sans-serif;
        direction: rtl;
        text-align: center;
        padding: 40px;
    }

    #header {
        font-size: 48px;
        font-weight: 700;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 50px;
        padding: 10px;
        text-align: center;
    }

    .gr-textbox, .gr-audio {
        border-radius: 20px;
        box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.1);
        border: 1px solid #d1e0eb;
        padding: 10px;
        transition: all 0.3s ease;
        font-size: 14px;
    }
    
    .gr-textbox:hover, .gr-audio:hover {
        box-shadow: 0px 6px 20px rgba(0, 0, 0, 0.2);
    }

    .gr-button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: white;
        border-radius: 30px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 500;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .gr-button:hover {
        transform: translateY(-3px);
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.1);
    }

    .gr-row {
        margin-top: 20px;
    }
    
    .gr-audio, .gr-textbox {
        width: 70%;
        margin: 0 auto;
    }
    
    @media (max-width: 768px) {
        .gr-audio, .gr-textbox {
            width: 90%;
        }
    }

""") as demo:
    
    # Large, stylish header with gradient text
    with gr.Row():
        gr.Markdown('<h1 id="header">شیفا</h1>')
    
    # Textbox to display transcrib  ed and AI-generated text
    with gr.Row():
        text_output = gr.Textbox(label="بەخێربێی بۆ شیفا چۆن بتوانین یارمەتیت بدەم", lines=4, elem_classes="gr-textbox")
    
    # Microphone and file upload inputs with smaller, sleek boxes
    with gr.Row():
        microphone_input = gr.Audio(label="تۆمارکردن", elem_classes="gr-audio")
    
    # Gradient submit button with hover effect
    with gr.Row():
        submit_button = gr.Button(value="ناردن", variant="primary", elem_classes="gr-button")
    
    # Click event to process transcription and display result
    submit_button.click(
        transcribe, 
        inputs=[microphone_input], 
        outputs=text_output
    )
demo.launch(share=True)