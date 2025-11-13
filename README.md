# Cervical Cancer Detection with Deep Learning 🔬

![Prediction Demo](<img width="1919" height="1022" alt="Screenshot 2025-11-12 162719" src="https://github.com/user-attachments/assets/97a9f3c1-5cfc-4504-a4aa-5324fafd452b" />)  


This project is an **AI-based Computer Vision** application designed to detect and classify cervical cancer cells from microscopic (Pap smear) images. It is built with **Python** and **TensorFlow**, using *Transfer Learning* techniques.

## 🎯 Objective
To build a model capable of classifying cervical cells into three clinical categories:
1. **NORMAL**: Healthy cervical cells.  
2. **LSIL** (*Low-Grade Squamous Intraepithelial Lesion*): Mild precancerous lesions.  
3. **HSIL** (*High-Grade Squamous Intraepithelial Lesion*): Severe precancerous lesions with high risk.

## 🚀 Features & Technologies
* **Deep Learning Engine:** TensorFlow & Keras  
* **Model Architecture:** MobileNetV2 (Pre-trained on ImageNet)  
* **Optimization Techniques:**  
  * **Data Augmentation:** Enhances dataset variability (rotation, zoom, flip)  
  * **Early Stopping:** Prevents overfitting by halting training at optimal accuracy  
* **Accuracy:** Achieves around **85% validation accuracy** and **>99% confidence** on clear cases  

## 📂 Dataset Structure
The dataset used is adapted from the **SipakMed Dataset**.  
Folder structure for training:
```text
data/
├── train/
│   ├── NORMAL/
│   ├── LSIL/
│   └── HSIL/
└── validation/
    ├── NORMAL/
    ├── LSIL/
    └── HSIL/
```

## 🛠️ How to Run Locally
1. **Clone this repository:**
   ```bash
   git clone https://github.com/NayPyon/CervicalCancerDetection.git
   cd Cervical-Cancer-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data (Optional)**  
   If you want to retrain the model (`train.py`), download the SipakMed dataset from Kaggle and arrange it according to the folder structure above.  
   Skip this step if you only want to test the pre-trained model.

4. **Run Prediction**
   Use the pre-trained model **(cervical_cancer_model_v2.keras)**  
   1. Prepare a cell image (**.bmp**, **.jpg**, or **.png**)  
   2. Run the command:
      ```bash
      python predict.py
      ```
   3. Enter the image filename when prompted  

## 🧠 Main Files
1. **train.py:** Script for preprocessing, training, and saving the model  
2. **predict.py:** Script for predicting new images  
3. **cervical_cancer_model_v2.keras:** The trained model file (the “brain” of the app)  
