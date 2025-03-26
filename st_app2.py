import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display

st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🏥")

# Load trained model
@st.cache_data
def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()


# Custom Theme
custom_css = """
<style>
    body { background-color: #FFFFFF; color: #333333; }
    .stApp { background-color: #FFFFFF; }
    .css-18e3th9 { background-color: #F5F5F5; }  /* Sidebar */
    .st-bb { color: #4BC9FF !important; }  /* Primary Color */
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ---------------- Sidebar UI  ----------------
st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)
st.sidebar.image("logo.jpg", width=120)  # Display Logo Properly
st.sidebar.markdown("<h2 style='text-align: center;'>Diabetes Prediction</h2>", unsafe_allow_html=True)

# Sidebar Menu selectbox
menu = st.sidebar.selectbox("📌 Select a Page", ["🏥 Prediction", "📊 Study Report", "📈 Model Evaluation", "ℹ️ About"])

# ----------------------  PREDICTION ----------------------
if menu == "🏥 Prediction":
    st.title("🩺 Diabetes Prediction")
    prediction_type = st.radio("Choose Prediction Type:", ["Single Prediction", "Multiple Prediction"], horizontal=True)

    if prediction_type == "Single Prediction":
        st.subheader("📝 Enter Patient Details")
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.slider("🤰 Pregnancies", 0, 15, 1)
            glucose = st.number_input("🍬 Plasma Glucose", 50, 250, 120)
            blood_pressure = st.number_input("💓 Diastolic BP (mm Hg)", 30, 120, 70)
            skin_thickness = st.number_input("🩸 Triceps Thickness (mm)", 0, 99, 20)

        with col2:
            insulin = st.number_input("💉 Serum Insulin (mu U/ml)", 0, 900, 80)
            bmi = st.number_input("⚖️ BMI", 10.0, 50.0, 25.0)
            pedigree = st.number_input("👨‍👩‍👧 Diabetes Pedigree", 0.0, 2.5, 0.5)
            age = st.number_input("🎂 Age", 18, 90, 30)

        if st.button("🚀 Predict"):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]])
            prediction = model.predict(input_data)[0]
            if prediction == 1:
                st.warning(f"🔮 Prediction: **{'Diabetic'}**")
            else:
                st.success(f"🔮 Prediction: **{'Not Diabetic'}**")

    elif prediction_type == "Multiple Prediction":
        st.subheader("📂 Upload CSV File")
        uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            if st.button("🔍 Predict for Uploaded Data"):
                predictions = model.predict(df)
                df["Diabetic"] = predictions
                st.dataframe(df)
                st.download_button("📥 Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")

# ---------------------- STUDY REPORT ----------------------
if menu == "📊 Study Report":
    st.title("📊 Study Report")
    
    st.write(" Features/variables are of homogeneous types and there appears to be no problematic NaN values. The fact that there are 2 float type variables (BMI, DiabetesPedigree) as well as int type ones is not an issue for numerical analysis. According to our general statistics, no null values are to be noted in our dataset. However, there seems to be some outliers, class imbalances in our dataset to handle, and maybe the necessity to rescale and/or normalize our data.")

    df = pd.read_csv("TAIPEI_diabetes.csv").drop(columns=['PatientID'])
    
    #--- 
    df.hist(bins=60, figsize=(15, 10))
    st.pyplot(plt)
    #---

    #---
    st.subheader("📌 Diabetic vs Non-Diabetic Distribution")
    #
    diabetes_counts = df['Diabetic'].value_counts()

    # ---
    plt.figure(figsize=(8, 6))
    plt.pie(diabetes_counts, labels=['Non-Diabetic', 'Diabetic'], autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
    st.pyplot(plt)
    st.write(" As we can see, there is an imbalance in favor of the non-diabetic class. That's not extreme, but it's there. In reality, outside of our dataset, such an imbalance is not outrageous, since diabetes only affects about 11-12% of people worldwide (cf. study). However, even if the gap between classes isn't very large, our future model could be negatively influenced by class imbalance. The model could simply learn to predict the majority class (non-diabetics) with high accuracy, but perform poorly in predicting the minority class (diabetics). This could lead to a poor model, even if overall accuracy is high.    There are techniques to improve the model's ability to predict instances of the minority class (diabetics), which may be of greater interest here, since failing to detect a diabetic can be far more serious than incorrectly predicting a non-diabetic as a diabetic. Hence, we can try to use resampling techniques to see if it can be useful for the model performance metrics we'll choose.")
    #---


    #---
    st.subheader("📌 Correlation Matrix")
    # 
    correlation_matrix = df.corr()

    # ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    st.pyplot(plt)
    #---


    #---
    st.markdown("### 🔎 Insights from the Dataset (EDA)")
    eda_slides = [
        "🗂 **Dataset Overview:** 15,000 records with 8 features",
        "🛠 **Missing Values Handling:** No major missing data",
        "📊 **Feature Importance:** PlasmaGlucose and BMI are key indicators",
        "⚖️ **Diabetes Distribution:** 35% of the dataset has diabetes",
        "🧩 **Correlation Matrix:** High correlation between glucose and diabetes"
    ]
    
    for slide in eda_slides:
        st.write(slide)

    #---
# ---------------------- MODEL EVALUATION ----------------------
if menu == "📈 Model Evaluation":
    st.title("📈 Model Evaluation")
    st.write("### 🤖 Model Used: **RandomForest Classifier**")
    st.write("🏆 **Performance:** Random Forest performed best with an accuracy of **85%**.")

    eval_metrics = {
        "🎯 Accuracy": "85%",
        "📍 Precision": "83%",
        "🔄 Recall": "80%",
        "🧠 F1 Score": "81%"
    }
    
    for metric, value in eval_metrics.items():
        st.write(f"**{metric}:** {value}")

    st.subheader("📌 Feature Importance")
    feature_importance = model.feature_importances_
    features = ["Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure", "TricepsThickness",
                "SerumInsulin", "BMI", "DiabetesPedigree", "Age"]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=feature_importance, y=features, palette="viridis")
    st.pyplot(plt)

# ---------------------- ABOUT PAGE ----------------------
if menu == "ℹ️ About":
    st.title("ℹ️ About the Project")
    st.markdown("""
        ## 🏥 Diabetes Mellitus: A Chronic Metabolic Disorder  
        Diabetes mellitus is a condition characterized by the body's impaired ability to utilize blood sugar (glucose) effectively.  
        The **American Diabetes Association** classifies diabetes into two primary types:
        
        ### 🔹 Type 1 Diabetes (Insulin-Dependent)
        - Often manifests in **childhood**  
        - Caused by an **autoimmune response** that destroys insulin-producing beta cells  
        - The exact cause is **multifactorial**: genetic predisposition, environmental factors, and viral infections  
        
        ### 🔹 Type 2 Diabetes (Non-Insulin-Dependent)
        - More **prevalent** and typically diagnosed in **adulthood**  
        - Results from **insulin resistance** or **insufficient insulin secretion**  
        - **Risk Factors:** Family history, obesity, and physical inactivity  

        ### 🏡 Other Forms of Diabetes:
        - **Gestational Diabetes Mellitus (GDM)**: Temporary during pregnancy, increases risk of Type 2 diabetes later  
        - **Genetic Defects & Pancreatic Dysfunction**: Less common, caused by genetics or exposure to medications/chemicals  

        ---

        ## 👩‍👦 Maternal Inheritance of Diabetes:
        - 🤰 **Gestational diabetes** is unlikely to directly cause diabetes in the baby  
        - 👶 **Type 2 diabetes in the mother** increases the child's risk of Type 2 later in life  
        - 🧬 **Type 1 diabetes in the mother** slightly increases the risk of the child having Type 1 diabetes at birth  

        ---

        ## 📊 Machine Learning & Diabetes Prediction  
        - Diabetes is a **multi-factorial disease** 🏥  
        - Many **ML models** have been built to assist doctors in diagnosing diabetes  
        - The **PIMA Indian Diabetes dataset** is commonly used for research  
        - Our project is based on a **recent study**:  

        > **Chou et al., J.Pers.Med. 2023**:  
        > Study of **15,000 women (aged 20-80)** at the **Taipei Municipal Medical Center**  
        > Data collected from **2018–2020 & 2021–2022**  

        ---

        ## 📂 Dataset: TAIPEI_diabetes.csv
        **This dataset contains 15,000 records with 8 health features:**  
        - **🤰 Pregnancies:** Number of times pregnant  
        - **🍬 Plasma Glucose:** Glucose concentration after 2 hours in an oral glucose tolerance test  
        - **💓 Diastolic Blood Pressure:** Measured in mm Hg  
        - **🩸 Triceps Thickness:** Skin fold thickness (mm)  
        - **💉 Serum Insulin:** 2-Hour serum insulin (mu U/ml)  
        - **⚖️ BMI:** Body Mass Index (kg/m²)  
        - **👨‍👩‍👧 Diabetes Pedigree:** Probability of diabetes based on family history  
        - **🎂 Age:** Age in years  

        ---

    """)

    st.subheader("👨‍💻 Team Members")
    team = [
        {"name": "BRUNET Nathan", "role": "Data Scientist"},
        {"name": "IBITOWA Abraham", "role": "Data Scientist"},
        {"name": "HAOUA Anis Sofiane", "role": "Data Analyst"},
        {"name": "KAKY SUZY Joelly Magalie", "role": "Data Analyst"},
        {"name": "NIANG Falilou", "role": "Data Engineer"}
    ]

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, member in zip([col1, col2, col3, col4, col5], team):

        col.write(f"**{member['name']}**")
        col.write(member["role"])

    st.divider()
    st.write("#### 🎓 Sponsored by")
    st.image("dsti_logo.webp", width=80)  
