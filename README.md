# breast-cancer-image-classification-using-deep-learning-technique

# 🔬 Breast Cancer Diagnostic Platform

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-94%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

A deep learning-powered diagnostic platform for breast cancer detection using histopathological image analysis. Achieves **94% classification accuracy**, aiding early and reliable diagnosis.

---

## 📌 Features

- Binary classification: Malignant vs. Benign tumor detection
- Trained on histopathological image datasets using CNNs
- Built with TensorFlow and Keras deep learning frameworks
- Flask-based REST API for real-time inference
- Clean web interface for image upload and result display

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow, Keras |
| Backend API | Flask |
| Frontend | HTML, CSS, JavaScript |
| Data Processing | NumPy, OpenCV, Pandas |
| Visualization | Matplotlib, Seaborn |

---

## 📁 Project Structure

```
breast-cancer-diagnostic-platform/
│
├── model/
│   ├── train.py          # Model training script
│   └── model.h5          # Saved trained model
│
├── static/               # CSS, JS, uploaded images
├── templates/            # HTML templates
│   └── index.html
│
├── app.py                # Flask application entry point
├── predict.py            # Inference logic
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 94% |
| Framework | TensorFlow / Keras |
| Task | Binary Classification |

---

## 👤 Author

**E. Sainath Reddy**  
GitHub: [@sainath1308](https://github.com/sainath1308)  
LinkedIn: [e-sainath-reddy](https://linkedin.com/in/e-sainath-reddy-a64667287)

---

## 📄 License

This project is licensed under the MIT License.

