# My First AI Image Classifier

A beginner-friendly transfer-learning image classification project that uses MobileNetV3-Small to classify images into three categories:

- Cat
- Dog
- Bird

## Model

- Architecture: MobileNetV3-Small
- Input image size: 224 x 224
- Number of classes: 3
- Classes: bird, cat, dog
- Training framework: PyTorch
- Pretrained weights: ImageNet

## Training Results

- Best validation accuracy: 92.33%
- Best validation loss: 0.1774
- Best epoch: 2

## Project Structure

```text
my-first-ai-image-classifier/
│
├── model/
│   └── my_first_ai_image_classifier.pth
│
├── app.py
│
├── inference.py
│
├── requirements.txt
│
├── project_info.json
│
└── README.md
```

## How to Run Locally

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Gradio application:

```bash
python app.py
```

## Command-Line Inference

You can test the model directly with:

```bash
python inference.py path/to/your/image.jpg
```

## Deployment

This project uses Gradio for the web interface.

- `app.py` - Gradio web application
- `inference.py` - Model loading and prediction logic
- `model/my_first_ai_image_classifier.pth` - Trained model
- `requirements.txt` - Python dependencies
- `project_info.json` - Project metadata

## Technologies Used

- Python
- PyTorch
- Torchvision
- MobileNetV3-Small
- Gradio
- Pillow

## License

This project is intended for educational and demonstration purposes.
