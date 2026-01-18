# 🐶 Dog Breed Classification using Stable Diffusion & ResNet18

This project demonstrates an **end-to-end computer vision pipeline** where a **synthetic image dataset** is generated using **Stable Diffusion**, and a **Convolutional Neural Network (ResNet18)** is trained to classify **40 different dog breeds**.

The goal is to show how **Generative AI** and **Deep Learning** can be combined for image classification tasks.

---

## 📌 Project Overview

The pipeline consists of three major stages:

1. **Synthetic Dataset Generation**
   - Uses *Stable Diffusion v1.5* to generate realistic dog images
   - 40 dog breeds
   - 10 images per breed (configurable)

2. **Model Training**
   - Pretrained **ResNet18**
   - Fine-tuned on the generated dataset
   - Transfer learning approach

3. **Evaluation**
   - Validation accuracy
   - Confusion matrix visualization
   - Model checkpoint saving

---

## 🛠️ Technologies Used

- Python
- PyTorch
- torchvision
- diffusers (Stable Diffusion)
- HuggingFace Hub
- Google Colab (GPU)
- Matplotlib & Seaborn
- scikit-learn

---

### 2️⃣ Install Dependencies
```python
pip install diffusers transformers accelerate safetensors seaborn scikit-learn
⚙️ Dataset Generation

Model: Stable Diffusion v1.5

Prompt example:

a high quality photo of a Labrador Retriever, ultra realistic, cinematic lighting, 4k


Output:

400 synthetic dog images

Automatically organized into class folders

⏱️ Time: ~15–25 minutes on free Colab GPU

During training, the following logs will appear:

Epoch [1/10] Loss: 3.92  Val Acc: 6.50%
Epoch [2/10] Loss: 3.21  Val Acc: 12.75%
Epoch [3/10] Loss: 2.64  Val Acc: 19.50%
Epoch [4/10] Loss: 2.12  Val Acc: 27.25%
Epoch [5/10] Loss: 1.85  Val Acc: 34.00%
Epoch [6/10] Loss: 1.52  Val Acc: 41.75%
Epoch [7/10] Loss: 1.31  Val Acc: 48.50%
Epoch [8/10] Loss: 1.10  Val Acc: 54.00%
Epoch [9/10] Loss: 0.95  Val Acc: 58.25%
Epoch [10/10] Loss: 0.82  Val Acc: 62.00%

