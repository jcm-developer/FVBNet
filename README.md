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
- [Dependencies](#dependencies)
- [Model Download](#model-download)
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
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

Install required packages:

```bash
pip install tensorflow pillow numpy huggingface_hub
```

---

## ⚙️ Usage

Load the model and run prediction on an image:

```python
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Download model from Hugging Face
model_path = hf_hub_download(repo_id="jcm-developer/FVBNet", filename="deep_food_enhanced_model.keras")
model = load_model(model_path)

# Preprocess image
def preprocess_image(path):
    img = Image.open(path).resize((224, 224)).convert("RGB")
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Predict
img_path = 'path/to/image.jpg'
input_image = preprocess_image(img_path)
prediction = model.predict(input_image)

predicted_class = prediction.argmax()
print(f"Predicted class: {predicted_class}")
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

## 🧪 Inference Example

```bash
python predict.py --image path/to/image.jpg
```

> Ensure the model is either downloaded or accessible through the Hugging Face integration.

---

## 📦 Dependencies

- Python 3.8+
- TensorFlow (latest)
- Pillow
- NumPy
- huggingface_hub

---

## 📥 Model Download

The trained model is publicly available on Hugging Face:

🔗 [FVBNet on Hugging Face](https://huggingface.co/jcm-developer/FVBNet)

You can also programmatically download it using:

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="jcm-developer/FVBNet", filename="deep_food_enhanced_model.keras")
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
