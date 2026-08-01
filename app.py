
import gradio as gr
from predict import predict

demo=gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="My First AI Image Classifier",
    description="Bird vs Cat vs Dog (ONNX Runtime)"
)

import os

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT",7860))
)
