# 🩸 DiabetesIQ – AI Diabetes Risk Predictor

An **AI-powered diabetes risk screening web application** built using **Machine Learning and Streamlit**.
The system analyzes patient demographics and symptoms to estimate the **probability of diabetes in seconds**.

This project demonstrates how **clinical symptom data + machine learning** can be used to build a **medical screening tool** with an intuitive interface.

---

# 🚀 Features

* 🧠 **Machine Learning Prediction**

  * Decision Tree model trained on clinical symptom dataset
* ⚡ **Instant Risk Analysis**

  * Predict diabetes probability in real time
* 🧩 **16 Clinical Features**

  * Age, gender, metabolic symptoms, neurological symptoms, skin conditions
* 📊 **Risk Visualization**

  * Interactive risk meter showing probability percentage
* 🎨 **Modern Medical UI**

  * Dark glassmorphism theme with Streamlit + custom CSS
* 🔍 **Symptom Tracking**

  * Displays active symptoms selected by the user
* 📋 **Personalized Recommendations**

  * Provides health suggestions based on prediction
* ⚠️ **Medical Disclaimer Included**

---

# 🧠 Machine Learning Model

| Component | Description              |
| --------- | ------------------------ |
| Model     | Decision Tree Classifier |
| Features  | 16 clinical symptoms     |
| Scaling   | Standard Scaler          |
| Output    | Diabetes probability (%) |

### Input Features

* Age
* Gender
* Polyuria
* Polydipsia
* Polyphagia
* Sudden Weight Loss
* Weakness
* Obesity
* Visual Blurring
* Irritability
* Partial Paresis
* Muscle Stiffness
* Alopecia
* Itching
* Delayed Healing
* Genital Thrush

---

# 🖥️ Application Interface

The application contains **four main sections**:

### 1️⃣ Patient Demographics

* Age
* Gender

### 2️⃣ Primary Metabolic Symptoms

* Polyuria
* Polydipsia
* Polyphagia
* Weight loss
* Weakness
* Obesity

### 3️⃣ Secondary & Neurological Symptoms

* Visual blurring
* Irritability
* Partial paresis
* Muscle stiffness
* Alopecia
* Itching

### 4️⃣ Skin & Infection Indicators

* Delayed healing
* Genital thrush

After entering the data, the system **calculates the probability of diabetes and visualizes the risk level**.

---

# 🏗️ Tech Stack

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core programming language |
| Streamlit    | Web application framework |
| Scikit-learn | Machine learning model    |
| Pandas       | Data manipulation         |
| NumPy        | Numerical computations    |
| Joblib       | Model serialization       |

---

# 📂 Project Structure

```
DiabetesIQ/
│
├── app.py                # Streamlit application
├── DT.pkl                # Trained Decision Tree model
├── scaler.pkl            # Feature scaler
├── columns.pkl           # Expected feature order
│
├── dataset/
│   └── diabetes.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/DiabetesIQ.git
cd DiabetesIQ
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
streamlit run app.py
```

The application will open in your browser:

```
http://localhost:8501
```

---

# 📊 Prediction Workflow

1. User enters demographic and symptom data
2. Inputs are converted into numerical features
3. Features are aligned with trained model columns
4. Data is scaled using the saved scaler
5. Decision Tree model predicts:

   * Diabetes class
   * Probability score
6. Risk level is displayed visually in the interface

---

# ⚠️ Medical Disclaimer

This application is intended **only for educational and informational purposes**.

It **does not replace professional medical advice, diagnosis, or treatment**.
Always consult a qualified healthcare professional for medical decisions.

---

# 🎯 Future Improvements

* Add **more advanced ML models** (Random Forest, XGBoost)
* Integrate **SHAP explainability**
* Add **patient history tracking**
* Deploy the app using **Streamlit Cloud / Docker**
* Connect with **real medical datasets**

---

# 👨‍💻 Author

**Aditya Tupe**


Built with ❤️ using **Python, Machine Learning, and Streamlit**

---

# ⭐ If you like this project

Consider **starring the repository** and sharing it.
