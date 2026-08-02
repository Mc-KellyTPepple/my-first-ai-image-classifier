import os
import tempfile

import gradio as gr
from gtts import gTTS

from predict import predict


def predict_with_voice(image):
    # Get prediction probabilities
    result = predict(image)

    # Highest confidence class
    best_class = max(result, key=result.get)
    confidence = result[best_class] * 100

    text = (
        f"The image is predicted to be a "
        f"{best_class} "
        f"with {confidence:.1f} percent confidence."
    )

    # Create temporary mp3
    audio_path = tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False
    ).name

    gTTS(text=text, lang="en").save(audio_path)

    return result, audio_path


demo = gr.Interface(
    fn=predict_with_voice,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Prediction"),
        gr.Audio(label="Voice Result", autoplay=True)
    ],
    title="My First AI Image Classifier",
    description="""
Upload a Bird, Cat or Dog image.

✓ ONNX Runtime
✓ Voice feedback
✓ Render Deployment
"""
)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
