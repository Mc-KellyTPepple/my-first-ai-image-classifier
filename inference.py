
from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image

from torchvision import models
from torchvision import transforms


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (

    BASE_DIR
    /
    "model"
    /
    "my_first_ai_image_classifier.pth"

)


# ============================================================
# RENDER DEPLOYMENT DEVICE
# ============================================================

# Render Web Services normally run on CPU.
# We therefore explicitly use CPU for stable deployment.

DEVICE = torch.device(
    "cpu"
)


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(

        f"Model file not found: {MODEL_PATH}\n"
        f"Expected location: "
        f"{BASE_DIR / 'model'}"

    )


# ============================================================
# LOAD MODEL CHECKPOINT
# ============================================================

checkpoint = torch.load(

    MODEL_PATH,

    map_location=DEVICE

)


# ============================================================
# READ MODEL INFORMATION
# ============================================================

CLASS_NAMES = checkpoint[
    "class_names"
]


NUM_CLASSES = checkpoint[
    "num_classes"
]


IMG_SIZE = checkpoint[
    "img_size"
]


IMAGENET_MEAN = checkpoint[
    "imagenet_mean"
]


IMAGENET_STD = checkpoint[
    "imagenet_std"
]


# ============================================================
# CREATE MOBILEV3-SMALL ARCHITECTURE
# ============================================================

model = models.mobilenet_v3_small(

    weights=None

)


# ============================================================
# RECREATE FINAL CLASSIFIER
# ============================================================

in_features = (

    model
    .classifier[3]
    .in_features

)


model.classifier[3] = nn.Linear(

    in_features,

    NUM_CLASSES

)


# ============================================================
# LOAD TRAINED MODEL WEIGHTS
# ============================================================

model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]

)


# ============================================================
# MOVE MODEL TO CPU
# ============================================================

model = model.to(

    DEVICE

)


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()


# ============================================================
# IMAGE PREPROCESSING
# MUST MATCH VALIDATION TRANSFORM
# ============================================================

inference_transform = transforms.Compose([

    transforms.Resize(

        (
            IMG_SIZE,
            IMG_SIZE
        )

    ),

    transforms.ToTensor(),

    transforms.Normalize(

        IMAGENET_MEAN,

        IMAGENET_STD

    )

])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if image is None:

        raise ValueError(
            "No image was provided."
        )


    # --------------------------------------------------------
    # ACCEPT PIL IMAGE OR FILE PATH
    # --------------------------------------------------------

    if isinstance(

        image,

        (str, Path)

    ):

        image = Image.open(

            image

        )


    elif not isinstance(

        image,

        Image.Image

    ):

        raise TypeError(

            "Input must be a PIL image "
            "or a valid image file path."

        )


    # --------------------------------------------------------
    # CONVERT IMAGE TO RGB
    # --------------------------------------------------------

    image = image.convert(

        "RGB"

    )


    # --------------------------------------------------------
    # APPLY MODEL PREPROCESSING
    # --------------------------------------------------------

    input_tensor = (

        inference_transform(

            image

        )

        .unsqueeze(

            0

        )

        .to(

            DEVICE

        )

    )


    # --------------------------------------------------------
    # RUN MODEL INFERENCE
    # --------------------------------------------------------

    with torch.inference_mode():

        outputs = model(

            input_tensor

        )


        probabilities = (

            torch.softmax(

                outputs,

                dim=1

            )[0]

        )


        confidence, predicted_index = (

            probabilities.max(

                dim=0

            )

        )


    # --------------------------------------------------------
    # GET PREDICTED CLASS
    # --------------------------------------------------------

    predicted_class = (

        CLASS_NAMES[

            predicted_index.item()

        ]

    )


    # --------------------------------------------------------
    # CREATE PROBABILITY DICTIONARY
    # --------------------------------------------------------

    probability_results = {

        CLASS_NAMES[index]:

            float(

                probabilities[index].item()

            )

        for index

        in range(

            len(CLASS_NAMES)

        )

    }


    # --------------------------------------------------------
    # RETURN PREDICTION RESULTS
    # --------------------------------------------------------

    return {

        "class":
            predicted_class,

        "confidence":
            float(

                confidence.item()

            ),

        "probabilities":
            probability_results

    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    import sys


    if len(sys.argv) < 2:

        print(

            "Usage: "
            "python inference.py "
            "<image_path>"

        )

        raise SystemExit(1)


    image_path = sys.argv[1]


    result = predict_image(

        image_path

    )


    print()

    print(
        "Predicted class:",
        result["class"]
    )


    print(

        "Confidence:",

        f"{result['confidence']:.2%}"

    )


    print()

    print(
        "Probabilities:"
    )


    for (

        class_name,

        probability

    ) in (

        result[
            "probabilities"
        ]

        .items()

    ):

        print(

            f"  {class_name}: "
            f"{probability:.2%}"

        )
