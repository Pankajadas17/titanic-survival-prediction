# titanic-survival-prediction
End-to-end ML project predicting Titanic passenger survival with Streamlit deployment

# 🚢 Titanic Survival Prediction – End-to-End ML Project

## 📌 Project Overview
This project is an end-to-end Machine Learning application that predicts whether a passenger would survive the RMS Titanic disaster.  
It demonstrates a complete ML lifecycle — from data cleaning and exploratory analysis to model training, evaluation, and live deployment via a web interface.

The project is designed and structured following **industry-level best practices**, with clean code, version control, and deployment readiness.

---

## 🎯 Core Objective
- Predict passenger survival using demographic and socio-economic features
- Analyze survival patterns based on age, gender, class, and fare
- Apply feature engineering for improved predictive performance
- Deploy the trained model as a publicly accessible web application

---

## 📊 Dataset
- Source: Kaggle Titanic Dataset
- Files used:
  - `train.csv` – model training and evaluation
  - `test.csv` – unseen data for prediction demonstration

### Key Features
- Passenger Class (`Pclass`)
- Age
- Gender
- Fare
- Family relationships (`SibSp`, `Parch`)
- Embarkation port
- Engineered features such as `FamilySize`, `IsAlone`, and `Title`

---

## 🧹 Data Processing
- Handled missing values using statistical methods
- Removed non-informative columns
- Ensured fully numeric, model-ready dataset
- Preserved data integrity throughout preprocessing

---

## 🔍 Exploratory Data Analysis (EDA)
EDA was conducted to understand survival patterns and feature importance:
- Survival distribution analysis
- Gender-based and class-based survival trends
- Age and fare impact on survival
- Correlation analysis using heatmaps

Key insights were documented to guide feature engineering and modeling decisions.

---

## 🧠 Feature Engineering
The following engineered features were created:
- **FamilySize**: Combined family members aboard
- **IsAlone**: Binary indicator for solo travelers
- **Title extraction** from passenger names
- One-hot encoding of categorical variables

All features were converted to numeric format to ensure model compatibility.

---

## 🤖 Model Development
Multiple machine learning models were trained and evaluated:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve & AUC
- Cross-validation accuracy

📌 **Random Forest** was selected as the final model due to its superior performance and robustness.

---

## 🌐 Web Application (Streamlit)
A user-friendly web interface was built using Streamlit:
- Interactive input controls
- Real-time survival prediction
- Probability-based output
- Clean and intuitive UI

🔗 **Live App:** *(Add your Streamlit deployment link here)*

---

## 🛠️ Tech Stack
- **Programming:** Python
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Web Framework:** Streamlit
- **Version Control:** Git & GitHub

---

## 📁 Project Structure
titanic-survival-prediction/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── 01_data_cleaning.ipynb
│ ├── 02_eda.ipynb
│ ├── 03_feature_engineering.ipynb
│ └── 04_modeling.ipynb
│
├── app.py
├── model.pkl
├── requirements.txt
└── README.md

## 🚀 How to Run Locally
```bash
git clone https://github.com/<your-username>/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
streamlit run app.py