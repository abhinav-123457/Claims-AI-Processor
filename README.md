# 🚗 Car Damage AI – Automated Insurance Claim Processing

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)]()
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange?logo=yolo)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?logo=opencv)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

### 🧠 **98%+ Accurate** Car Damage Detection using **YOLOv8** fine-tuned on the **CARDD Dataset**

---

## 🌍 Overview

**Car Damage AI** is an end-to-end solution for **automated insurance claim analysis**.
It detects, validates, and estimates repair costs for damaged vehicles — ensuring speed, accuracy, and fraud prevention.

---

## ⚙️ Features

✅ **Upload car damage photo** through a simple web interface
✅ **AI Image Checker** – detects blur, tampering, duplicates, and AI-generated fakes
✅ **Real-time YOLOv8 Damage Detection** – bounding boxes with confidence scores
✅ **Cost Estimation Engine** – converts detected damages to estimated ₹ costs
✅ **PDF Report Generator** – export annotated image + cost breakdown
✅ **Fraud Prevention** – prevents fake or reused photos from claims

---

## 🧩 Tech Stack

| Category             | Tools / Libraries                                                                  |
| -------------------- | ---------------------------------------------------------------------------------- |
| **Object Detection** | YOLOv8 (Ultralytics)                                                               |
| **Image Processing** | OpenCV, Pillow                                                                     |
| **AI Detection**     | Hugging Face Transformers (`umm-maybe/AI-image-detector`)                          |
| **Backend**          | Flask                                                                              |
| **Frontend**         | HTML, CSS, JS                                                                      |
| **Dataset**          | [CARDD Dataset](https://universe.roboflow.com/capstone-car-damage-detection/cardd) |

---

## 📁 Project Structure

```
car-damage-ai/
│
├── model/
│   └── best.pt              # Trained YOLOv8 weights
│
├── app.py                   # Flask backend app
├── cardd.yaml               # Dataset configuration
├── index.html               # Frontend UI
├── README.md                # Documentation
└── requirements.txt         # Python dependencies
```

---

## 🚀 How to Run

```bash
# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/car-damage-ai.git
cd car-damage-ai

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the Flask app
python app.py
```

Then open your browser at **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)** 🌐

---

## 📸 Example Output
![](output1.png)
![](output.png)



---

## 🔍 Core AI Modules

| Step                                | Function                                               |
| ----------------------------------- | ------------------------------------------------------ |
| **1. Data Preprocessing**           | OpenCV noise reduction, resizing, contrast enhancement |
| **2. Damage Detection**             | YOLOv8 custom fine-tuned on 6 classes                  |
| **3. AI-Generated Image Detection** | Hugging Face `AI-image-detector`                       |
| **4. Quality Validation**           | Laplacian variance (blur) + MD5 hashing (duplicate)    |
| **5. Cost Estimation**              | Rule-based damage-to-₹ mapping                         |
| **6. Report Generation**            | Auto PDF via FPDF/jsPDF                                |

---

## 🧠 Future Enhancements

* License plate OCR integration
* Real-time deployment (Vercel / Render / Hugging Face Spaces)
* Cloud database for claim tracking
* Multi-language support

---

## 🪪 License

Licensed under the **MIT License**.
Free for educational and research use.
