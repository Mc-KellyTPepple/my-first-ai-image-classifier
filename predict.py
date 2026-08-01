
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT=Path(__file__).parent

with open(ROOT/"labels.json") as f:
    info=json.load(f)

CLASSES=info["classes"]

MEAN=np.array(info["mean"],dtype=np.float32)
STD=np.array(info["std"],dtype=np.float32)

IMG_SIZE=info["img_size"]

session=ort.InferenceSession(
    str(ROOT/"model"/"my_first_ai_image_classifier.onnx"),
    providers=["CPUExecutionProvider"]
)

def preprocess(image):

    image=image.convert("RGB")

    image=image.resize((IMG_SIZE,IMG_SIZE))

    x=np.array(image).astype(np.float32)/255.0

    x=(x-MEAN)/STD

    x=np.transpose(x,(2,0,1))

    x=np.expand_dims(x,0)

    return x

def softmax(x):

    e=np.exp(x-x.max())

    return e/e.sum()

def predict(image):

    x=preprocess(image)

    out=session.run(None,{"input":x})[0][0]

    prob=softmax(out)

    return {
        CLASSES[i]:float(prob[i])
        for i in range(len(CLASSES))
    }
