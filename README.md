# 💡 Sentiment Image Detector

A deep learning-based application that analyzes facial expressions from an image and predicts the corresponding sentiment or emotion such as *Happy*, *Sad*, *Angry*, *Surprised*, *Neutral*, etc.  
It supports both image upload and real-time camera input.

---

## 🚀 Features

- 📷 Detects emotions from static images or webcam feed  
- 🧠 Trained deep learning model (.h5) for accurate sentiment classification  
- 🧹 Image preprocessing (grayscale, resize) for better prediction  
- 🖼️ Supports formats: `.jpg`, `.png`, etc.  
- 🐍 Built with Python, TensorFlow, Keras, OpenCV, Pillow

---

## 📁 Project Structure

sentiment-image-detector/
│
├── main.py # Main application file
├── model.h5 # Trained emotion classification model
├── requirements.txt # Required Python libraries
├── README.md # Project documentation
└── assets/ # (Optional) Sample images or outputs

yaml
Copy
Edit

---

## ✅ How It Works

1. User chooses to upload an image or capture one via webcam  
2. The image is preprocessed (grayscale, resized to 48x48)  
3. The trained model (`model.h5`) predicts the dominant emotion  
4. The result is displayed in the console or GUI

---

## 🛠 Technologies Used

- Python 3.10  
- TensorFlow 2.17.1  
- Keras  
- OpenCV  
- Pillow  
- NumPy

---

## 📷 Supported Emotions

- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral

---

## 📌 Note

Do **not** push large files (e.g. `.h5`, `venv`) directly to GitHub. Use [Git LFS](https://git-lfs.github.com/) or keep them in `.gitignore`.

---

## 👨‍💻 Author

*Sashi kumar Reddy - CSE(AI&ML)**  
*Sri Venkateswara College of Engineering and Technology, Chittoor*  
📧 reddysashi4242@gmail.com  
