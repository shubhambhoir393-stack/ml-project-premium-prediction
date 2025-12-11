# Health Insurance Cost Prediction

***A Machine Learning Project***
This project predicts **health insurance costs** based on a user’s demographic, medical, and lifestyle information.
It uses trained ML models and is deployed using **Streamlit Cloud**.

---
## 🚀 Live Demo

Try the live app here:

👉 **[https://ml-project-premium-prediction-ct4k6qpvc8ofcxy5wrpmcc.streamlit.app/](https://ml-project-premium-prediction-ct4k6qpvc8ofcxy5wrpmcc.streamlit.app/)**

---

## 🧠 Overview

This project builds an ML model to estimate health insurance premiums using the following features:

* Age
* Number of dependants
* Income (Lakhs)
* Insurance Plan
* Genetic Risk
* Gender
* BMI Category
* Employment Status
* Region
* Smoking Status
* Marital Status
* Medical History

Two separate models are used depending on age:

* **Young Model** (Age ≤ 25)
* **Regular Model** (Age > 25)

---

## 🛠 Tech Stack

| Category        | Tools                        |
| --------------- | ---------------------------- |
| Programming     | Python                       |
| ML              | Pandas, Scikit-Learn, Joblib |
| Deployment      | Streamlit                    |
| Version Control | Git & GitHub                 |

---

## 📁 Project Structure

```
ml-project-premium-prediction/
│── artifacts/                     # Trained ML models & scalers  
│── main.py                        # Streamlit UI  
│── prediction_helper.py           # Preprocessing + prediction logic  
│── requirements.txt               # Dependencies for deployment  
│── LICENSE                        # Project license  
│── README.md                      # Project documentation  
```

---

## 🧩 How It Works

### 1. User Inputs

The Streamlit UI collects user data like age, BMI category, region, etc.

### 2. Preprocessing
* Encodes categorical variables
* Normalizes medical risk
* Applies the correct scaler (young or adult)

### 3. Model Selection
The app chooses a young or regular model based on age.

### 4. Prediction
The model returns the **estimated insurance cost**.

---

## 🖥 Streamlit App UI
<img width="975" height="493" alt="image" src="https://github.com/user-attachments/assets/f3842cd3-affe-4307-98d4-82e92b2a7733" />

```
---

## ▶ Run Locally

Clone the repo:

```bash
git clone https://github.com/shubhambhoir393-stack/ml-project-premium-prediction.git
cd ml-project-premium-prediction
```
Install dependencies:
```bash
pip install -r requirements.txt
```

Run Streamlit:
```bash
streamlit run main.py
```

## 📦 Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Connect repository in Streamlit Cloud
3. Set `main.py` as the entry point
4. Deploy — Streamlit auto-installs dependencies from `requirements.txt`
---

## 📜 License
Distributed under the **Apache-2.0 License**.

---
**Shubham Bhoir**
GitHub: [https://github.com/shubhambhoir393-stack](https://github.com/shubhambhoir393-stack)

