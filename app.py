
import os

import gradio as gr

from inference import predict_image


# ============================================================
# CLASSIFICATION FUNCTION
# ============================================================

def classify_image(image):

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if image is None:

        return (

            "Please upload an image.",

            {}

        )


    # --------------------------------------------------------
    # RUN PREDICTION
    # --------------------------------------------------------

    try:

        result = predict_image(

            image

        )

    except Exception as error:

        return (

            f"Prediction error: {str(error)}",

            {}

        )


    # --------------------------------------------------------
    # READ RESULTS
    # --------------------------------------------------------

    predicted_class = (

        result[
            "class"
        ]

    )


    confidence = (

        result[
            "confidence"
        ]

    )


    probabilities = (

        result[
            "probabilities"
        ]

    )


    # --------------------------------------------------------
    # CREATE DISPLAY TEXT
    # --------------------------------------------------------

    prediction_text = (

        f"Prediction: "
        f"{predicted_class.title()}\n\n"

        f"Confidence: "
        f"{confidence:.2%}"

    )


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return (

        prediction_text,

        probabilities

    )


# ============================================================
# CREATE GRADIO INTERFACE
# ============================================================

demo = gr.Interface(

    fn=classify_image,

    inputs=gr.Image(

        type="pil",

        label="Upload an image"

    ),

    outputs=[

        gr.Textbox(

            label="Prediction"

        ),

        gr.Label(

            label="Class Probabilities"

        )

    ],

    title=(

        "🐱🐶🐦 "
        "My First AI Image Classifier"

    ),

    description=(

        "Upload an image of a cat, dog, "
        "or bird. The AI model will "
        "classify the image."

    )

)


# ============================================================
# LAUNCH GRADIO ON RENDER
# ============================================================

if __name__ == "__main__":

    port = int(

        os.environ.get(

            "PORT",

            7860

        )

    )


    demo.launch(

        server_name="0.0.0.0",

        server_port=port,

        share=False

    )
