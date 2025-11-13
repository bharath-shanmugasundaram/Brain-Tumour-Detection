# 🧠 Brain Tumor Classification using Custom CNN (PyTorch)

This project implements a **binary brain tumor detection model** using a custom-built Convolutional Neural Network (CNN).  
The goal is to classify MRI images into **Tumor** vs **No Tumor** categories.

---

## 📌 Project Overview

This project demonstrates the complete pipeline of:

- Loading MRI images from a structured dataset  
- Preprocessing: resizing, grayscale conversion, normalization  
- Training a custom CNN from scratch  
- Testing and evaluating the model on unseen MRI scans  
- Deploying the model for real-time predictions  

The approach is minimal yet effective for learning CNN design, medical imaging preprocessing, and PyTorch workflow.

---

## 📂 Dataset Structure

The dataset consists of two folders:

Each folder contains MRI images in `.jpg`/`.png` format.

Images are manually separated into:

- **90 training images** per class  
- Remaining images used as **test images**

---

## 🔄 Preprocessing

Each image undergoes:

- Conversion to RGB  
- Normalization  
- Grayscale conversion (`1 channel`)
- Resizing to **64×64**  
- Conversion to PyTorch tensor  

A custom function applies these transforms to all samples.

---

## 🧱 Model Architecture — Custom CNN

A compact CNN architecture designed specifically for medical image patterns:

### **Convolutional Layers**
- Conv2d(1 → 3)
- Conv2d(3 → 256)
- Max-pooling
- Double Conv2d(256 → 256)
- Final max-pooling

### **Fully Connected Layers**
- Dense: 4096 → 4096  
- Dropout (0.5)  
- Dense: 4096 → 4096  
- Dropout (0.5)  
- Dense: 4096 → 2 (Tumor vs No Tumor)

### **Activation**
- ReLU used throughout

This architecture captures edges, textures, and medical-region patterns inside MRI images.

---

## 🏋️ Training Details

- **Epochs:** 5  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Batch Size:** 64  
- **Device:** MPS (Apple Silicon GPU) or CPU  

A full training loop logs average epoch loss.

---

## 📈 Testing & Accuracy

The test images (from both `yes` and `no` folders) are preprocessed with the same pipeline.

The script computes:

- Total samples  
- Total correct predictions  
- Final test accuracy percentage  

This provides a clear measure of how well the model generalizes to unseen MRI scans.

---

## 💾 Model Saving

The trained model is saved as:


This can later be loaded for:

- Medical imaging demos  
- Streamlit/Gradio apps  
- Deployment in diagnosis assistance systems  

---

## 🛠️ Technologies Used

- **PyTorch**
- **OpenCV**
- **NumPy**
- **Scikit-Learn**
- **Torchvision**
- **Matplotlib** (optional for visualizations)

---

## 🚀 Future Improvements

- Add **data augmentation** to avoid overfitting  
- Train deeper architectures (ResNet/EfficientNet)  
- Plot confusion matrix for detailed evaluation  
- Use larger and more diverse MRI datasets  
- Add Grad-CAM heatmaps to visualize tumor regions  

---

## 🙌 Acknowledgements

- Public MRI brain tumor dataset contributors  
- PyTorch developers  
- OpenCV and NumPy open-source ecosystem  

---

If you'd like:

✅ A downloadable `.md` file  
✅ A better-structured GitHub-ready README  
✅ A version with images/visual diagrams  

Just tell me!  
