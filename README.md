# Health Insurance Cost Prediction
*A Machine Learning Project from the Codebasics ML Course*

This project predicts **health insurance costs** based on user attributes such as age, BMI category, income, medical history, and lifestyle behaviors.
The model is deployed using **Streamlit Cloud**, offering a clean and interactive UI.
---

## 🚀 Live Demo

Try the live application here:

👉 **[https://ml-project-premium-prediction-ct4k6qpvc8ofcxy5wrpmcc.streamlit.app/](https://ml-project-premium-prediction-ct4k6qpvc8ofcxy5wrpmcc.streamlit.app/)**

---

## 🧠 Project Overview

This project estimates health insurance premiums based on:

* Age
* Number of dependents
* Income (Lakhs)
* Insurance plan
* Genetic risk
* Smoking status
* BMI category
* Employment status
* Region
* Marital status
* Medical history

Two separate ML models are used depending on age:

* **Young model** (≤ 25 years)
* **Adult model** (> 25 years)

---

## 🧩 Tech Stack

| Component            | Technology                   |
| -------------------- | ---------------------------- |
| Programming          | Python                       |
| ML Libraries         | Pandas, Scikit-learn, Joblib |
| Model Deployment     | Streamlit                    |
| Code Hosting         | GitHub                       |
| Training Environment | Codebasics ML Course         |

---

## 📁 Project Structure
```
ml-project-premium-prediction/
│── artifacts/                     # Trained ML models & scalers  
│── main.py                        # Streamlit app UI  
│── prediction_helper.py           # Preprocessing & prediction logic  
│── requirements.txt               # App dependencies  
│── LICENSE  
│── README.md  
```

## 🔍 How the Prediction Works

### 1. Input Collection
User enters personal & health-related attributes.

### 2. Preprocessing
* Encodes categorical variables
* Computes normalized medical risk
* Applies correct scaler (young/adult)
* Aligns features with training schema

### 3. Model Selection
```
If Age ≤ 25 → model_young  
Else        → model_rest
```

### 4. Output
A single integer value representing predicted insurance cost.
---

## ▶ Running the App Locally

### Clone the repo:
```bash
git clone https://github.com/shubhambhoir393-stack/ml-project-premium-prediction.git
cd ml-project-premium-prediction
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the Streamlit app:
```bash
streamlit run main.py

---

## 📦 Deployment
The app is deployed on **Streamlit Cloud**:
1. Push project to GitHub
2. Connect repo in Streamlit Cloud
3. Choose `main.py` as entrypoint
4. Deploy & monitor logs

---

## 📜 License
This project is open-source under the **Apache-2.0 License**.

---
**Shubham Bhoir**
GitHub: [https://github.com/shubhambhoir393-stack](https://github.com/shubhambhoir393-stack)

Just send the screenshot or tell me to extract the image from your earlier upload!
