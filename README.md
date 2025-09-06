# FVBNet – Food Vision Benchmark Network

FVBNet is a food image classification model based on EfficientNetV2B0, trained on the Food-101 dataset. It surpasses the original benchmark reported in the Food-101 paper, achieving **79.55% accuracy**. Ideal for computer vision applications in gastronomy, nutrition analysis, and recommendation systems.

---

## 📑 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference Example](#inference-example)
- [Model Download](#model-download)
- [Dependencies](#dependencies)
- [Contributors](#contributors)
- [License](#license)

---

## 🧠 Introduction

This project presents a food classification model for `224x224` RGB images, based on **EfficientNetV2B0** and trained using the **Food-101 dataset from Kaggle**. It achieves a **top-1 accuracy of 79.55%**, improving upon the benchmark presented in the original Food-101 paper (77.4%).

---

## ✨ Features

- ✅ Pretrained on Food-101
- 🧠 Based on EfficientNetV2B0
- 📈 Outperforms original Food-101 benchmark
- 🖼️ Input: RGB images of size 224x224
- 🔍 Output: Predicted class among 101 food categories

---

## 💾 Installation

Clone this repository:

```bash
git clone https://github.com/jcm-developer/FVBNet.git
cd your-repo
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Usage

Use the included script to run predictions:

```bash
python predict.py --image path/to/image.jpg
```

Make sure the image is a JPG or PNG with food content.

---

## 🧪 Inference Example (Manual in Python)

```python
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Download model from Hugging Face
model_path = hf_hub_download(repo_id="jcm-developer/FVBNet", filename="FVBNet.keras")
model = load_model(model_path)

# Preprocess image
def preprocess_image(path):
    img = Image.open(path).resize((224, 224)).convert("RGB")
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Predict
input_image = preprocess_image("path/to/image.jpg")
prediction = model.predict(input_image)
predicted_class = prediction.argmax()
print(f"Predicted class index: {predicted_class}")
```

---

## 🏋️ Training

- **Dataset**: [Food-101 on Kaggle](https://www.kaggle.com/dansbecker/food-101)
- **Architecture**: EfficientNetV2B0
- **Image Size**: 224x224
- **Optimization**: Modern training strategies and hyperparameter tuning
- **Metric**: Top-1 Accuracy

---

## 📊 Evaluation

| Model                       | Top-1 Accuracy |
|----------------------------|----------------|
| Random Forest (Food-101)   | 77.4%           |
| FVBNet – EfficientNetV2B0  | **79.55%**      |

---

## 📥 Model Download

The trained model is hosted on Hugging Face:

🔗 [FVBNet on Hugging Face](https://huggingface.co/jcm-developer/FVBNet)

To download and load the model:

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="jcm-developer/FVBNet", filename="FVBNet.keras")
```

---

## 📦 Dependencies

The project uses the following libraries:

- Python 3.8+
- TensorFlow
- Pillow
- NumPy
- huggingface_hub

You can install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 👤 Contributors

Developed by **Jaume Cortés**

---

## 📄 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it, with proper attribution.

See the [LICENSE](LICENSE) file for full details.

---

## 📬 Contact

For questions or suggestions:
- GitHub: [jcm-developer](https://github.com/jcm-developer)
