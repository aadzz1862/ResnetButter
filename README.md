# **README: ResNet Implementation, Training and Evaluation for Butterfly Classification**

## **📌 Project Overview**
This project implements **transfer learning** using **ResNet** to classify **butterflies and moths**. The model is **fine-tuned** on a custom dataset to leverage the power of **pre-trained deep learning models** while optimizing for specific classification tasks. 

Since training from scratch is **computationally expensive**, we perform **transfer learning** and train the model on a **GPU** to significantly reduce training time and improve accuracy.

---

## **📂 Project Structure**
```
/butterfly-moth-classification
│── archive/                         # Folder containing datasets or saved models
│── __pycache__/                      # Compiled Python files
│── resnet_train.py                    # Script for training the fine-tuned ResNet model
│── resnet_eval.py                     # Script for evaluating the trained model
│── resnet_core.py                      # Core implementation of ResNet for transfer learning
│── butterflies_dataloader.py           # Data loader for processing images
│── butterflies_dataloader (1).py       # (Backup or alternative version)
│── .DS_Store                           # System-generated metadata file (safe to ignore)
```

---

## **🖼️ Dataset Information**
- **Dataset Used:** Custom dataset containing images of **butterflies and moths**.
- **Data Preprocessing:** Images are normalized, resized, and augmented.
- **Classes:** Butterfly 🦋 vs. Moth 🦟
- **Format:** Images are loaded and preprocessed using a PyTorch `DataLoader` from `butterflies_dataloader.py`.

---

## **📚 Dependencies**
Ensure you have the following dependencies installed before running the scripts:

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib tqdm
```

💡 **GPU Support:**  
If running on **CUDA (NVIDIA GPU)**, install:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```
Check if PyTorch detects GPU:
```python
import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
```

---

## **🛠️ How to Run the Project**
### **1️⃣ Train the ResNet Model**
Run the training script on a GPU:
```bash
python resnet_train.py --epochs 10 --batch_size 32 --lr 0.001 --gpu
```
- `--epochs` → Number of training epochs (default: 10)
- `--batch_size` → Batch size (default: 32)
- `--lr` → Learning rate (default: 0.001)
- `--gpu` → Runs on GPU if available

💡 **Important:**  
Training on a **CPU** is extremely slow. Always **use a GPU** for efficient training.

---

### **2️⃣ Evaluate the Model**
After training, evaluate performance:
```bash
python resnet_eval.py --gpu
```
This script loads the trained model and evaluates accuracy on a test dataset.

---

## **🧠 Model Details**
- **Base Model:** `ResNet-50` (Pre-trained on ImageNet)
- **Fine-Tuning:** The final classification layer is replaced to classify **butterflies and moths**.
- **Optimizer:** Adam with weight decay regularization.
- **Loss Function:** Cross-Entropy Loss.
- **Data Augmentation:** Random cropping, flipping, normalization.

---

## **🚀 Transfer Learning Strategy**
1. **Load Pretrained ResNet:**  
   ```python
   from torchvision import models
   model = models.resnet50(pretrained=True)
   ```
2. **Freeze Early Layers:**  
   ```python
   for param in model.parameters():
       param.requires_grad = False
   ```
3. **Replace Last Layer for Custom Classification:**  
   ```python
   import torch.nn as nn
   model.fc = nn.Linear(model.fc.in_features, num_classes)  # Custom output layer
   ```
4. **Fine-Tune Model with Custom Dataset**

---

## **💡 Future Improvements**
- Train on a **larger dataset** for better generalization.
- Experiment with **ResNet-101** or **EfficientNet** for improved accuracy.
- Use **data augmentation** to improve model robustness.

---

## **📝 Credits & Acknowledgments**
- **Dataset Source:** Custom dataset of butterflies and moths.
- **Pretrained Model:** ResNet-50 from PyTorch Model Zoo.

🎯 **This project demonstrates the power of transfer learning, reducing training time while maintaining high accuracy.** 🚀

---
