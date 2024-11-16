import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib

# Sidebar for option selection
option = st.sidebar.radio("Web App", ['Classification Task', 'Regression Task'])


# Function to load classification dataset
@st.cache_data
def load_classification_data(uploaded_file):
    names = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 
             'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 
             'smoking', 'time', 'DEATH_EVENT']
    dataframe = pd.read_csv(uploaded_file, names=names)
    return dataframe

def load_water_data(uploaded_file):
        names = ['aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium', 'chloramine',
                'chromium', 'copper', 'fluoride', 'bacteria', 'viruses', 'lead',
                'nitrates', 'nitrites', 'mercury', 'perchlorate', 'radium',
                'selenium', 'silver', 'uranium', 'is_safe']
        dataframe = pd.read_csv(uploaded_file, names=names)
        
        dataframe.replace('#NUM!', pd.NA, inplace=True)
        dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

        # Impute missing values for numeric data (mean strategy)
        numeric_imputer = SimpleImputer(strategy='mean')
        dataframe.iloc[:, :-1] = numeric_imputer.fit_transform(dataframe.iloc[:, :-1])

        # Encode target variable 'is_safe' if necessary (assuming it's categorical)
        if dataframe['is_safe'].dtype == 'object' or dataframe['is_safe'].isnull().any():
            label_encoder = LabelEncoder()
            dataframe['is_safe'] = label_encoder.fit_transform(dataframe['is_safe'].fillna(0))

        return dataframe

 
def get_water_quality_input():
    aluminium = st.number_input("Aluminium", min_value=0.000, value=0.000, key="aluminium1")
    ammonia = st.number_input("Ammonia", min_value=0.000, value=0.000, key="ammonia1")
    arsenic = st.number_input("Arsenic", min_value=0.000, value=0.000, key="arsenic1")
    barium = st.number_input("Barium", min_value=0.000, value=0.000, key="barium1")
    cadmium = st.number_input("Cadmium", min_value=0.000, value=0.000, key="cadmium1")
    chloramine = st.number_input("Chloramine", min_value=0.000, value=0.000, key="chloramine1")
    chromium = st.number_input("Chromium", min_value=0.000, value=0.000, key="chromium1")
    copper = st.number_input("Copper", min_value=0.000, value=0.000, key="copper1")
    fluoride = st.number_input("Fluoride", min_value=0.000, value=0.000, key="fluoride1")
    bacteria = st.number_input("Bacteria", min_value=0.000, value=0.000, key="bacteria1")
    viruses = st.number_input("Viruses", min_value=0.000, value=0.000, key="viruses1")
    lead = st.number_input("Lead", min_value=0.000, value=0.000, key="lead1")
    nitrates = st.number_input("Nitrates", min_value=0.000, value=0.000, key="nitrates1")
    nitrites = st.number_input("Nitrites", min_value=0.000, value=0.000, key="nitrites1")
    mercury = st.number_input("Mercury", min_value=0.000, value=0.000, key="mercury1")
    perchlorate = st.number_input("Perchlorate", min_value=0.000, value=0.000, key="perchlorate1")
    radium = st.number_input("Radium", min_value=0.000, value=0.000, key="radium1")
    selenium = st.number_input("Selenium", min_value=0.000, value=0.000, key="selenium1")
    silver = st.number_input("Silver", min_value=0.000, value=0.000, key="silver1")
    uranium = st.number_input("Uranium", min_value=0.000, value=0.000, key="uranium1")

    features = np.array([[aluminium, ammonia, arsenic, barium, cadmium, chloramine,
                        chromium, copper, fluoride, bacteria, viruses, lead,
                        nitrates, nitrites, mercury, perchlorate, radium,
                        selenium, silver, uranium]])
    return features


# Main function for the app
def main():
    if option == 'Classification Task':
        st.title("CLASSIFICATION TASK")
        
        # Upload model and predict
        st.subheader("Upload a Saved Model for Heart Failure Prediction")
        uploaded_model = st.file_uploader("Upload your model (joblib format)", type=["joblib"])

        if uploaded_model is not None:
            model = joblib.load(uploaded_model)
            st.subheader("Input Sample Data for Heart Disease Prediction")
            
            # Collect input data
            name = st.text_input("Enter your name:", "")
            age = st.number_input("Age", min_value=0, max_value=120, value=0)
            anaemia = st.number_input("Anaemia (1 = yes, 0 = no)", min_value=0, max_value=1, value=0)
            creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/l)", min_value=0, max_value=5000, value=0)
            diabetes = st.number_input("Diabetes (1 = yes, 0 = no)", min_value=0, max_value=1, value=0)
            ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=0)
            high_blood_pressure = st.number_input("High Blood Pressure (1 = yes, 0 = no)", min_value=0, max_value=1, value=0)
            platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0, max_value=1000000, value=0)
            serum_creatinine = st.number_input("Serum Creatinine (mg/dl)", min_value=0.0, max_value=10.0, value=0.0)
            serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=120, max_value=180, value=130)
            sex = st.number_input("Sex (1 = male, 0 = female)", min_value=0, max_value=1, value=0)
            smoking = st.number_input("Smoking (1 = yes, 0 = no)", min_value=0, max_value=1, value=0)
            time = st.number_input("Time (days)", min_value=0, max_value=1000, value=0)

            # Prepare data for prediction
            input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                                    high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex,
                                    smoking, time]])

            if st.button("Predict"):
                prediction = model.predict(input_data)
                st.subheader("Prediction Result")
                if prediction[0] == 0:
                    st.write("The predicted result is: **No Heart Failure**")
                else:
                    st.write("The predicted result is: **Heart Failure**")

    elif option == 'Regression Task':
        st.title("REGRESSION TASK")
        # Model upload for prediction
        st.subheader("Upload a Saved Model for Water Safety Prediction")
        uploaded_model = st.file_uploader("Upload your model (joblib format)", type=["joblib"])

        if uploaded_model is not None:
            model = joblib.load(uploaded_model)  # Load the uploaded model

            # Input fields for prediction
            st.subheader("Input Sample Data for Water Quality Prediction")
            input_data = get_water_quality_input()
            
            if st.button("Predict"):
                # Prediction
                prediction = model.predict(input_data)
                
                # Display prediction result
                st.subheader("Prediction Result and Interpretation")
                st.write(f"The predicted water safety is: **{'Safe' if prediction[0] == 1 else 'Not Safe'}**")
                interpretation = "The water is safe for human consumption." if prediction[0] == 1 else "The water is not safe for human consumption."
                st.write(interpretation)

if __name__ == "__main__":
    main()
