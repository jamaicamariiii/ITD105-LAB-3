import streamlit as st
import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score # sampling techniques
from sklearn.linear_model import LogisticRegression # ML algorithm
from sklearn.preprocessing import StandardScaler,LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score #performance metric
from sklearn.tree import DecisionTreeClassifier #hyperparameter tuning
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import altair as alt


import joblib

# Load CSS styling
st.markdown(
    """
    <style>
    .box {
        background-color: #f0f2f6;
        padding: 5px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Sidebar for option selection
option = st.sidebar.radio("Part I - Classification Task", ['Sampling Techniques', 'Performance Metric/s','Hyperparameter Tuning'])


# Classification Task
if option == 'Sampling Techniques':
    st.title("Heart Failure Prediction Model")

    # RESAMPLING TECHNIQUES
    st.header("SAMPLING TECHNIQUES")
    # Introduction and Instructions
    st.write("""      
    ### Hello, User! Here's how to use this tool:
    1. Start by uploading your CSV file.
    2. Click the button to train the model.
    3. Download the trained model to your device.
    4. Upload the saved model for predictions.
    5. Enter the required input data and press predict.
    6. The results will be shown below in the summary section.
""")

    # Function to load the dataset
    @st.cache_data
    def load_data(uploaded_file):
        # Define column names as strings
        names = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 
         'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 
         'smoking', 'time', 'DEATH_EVENT']
        dataframe = pd.read_csv(uploaded_file, names=names)
        return dataframe
    st.write("<br><br>", unsafe_allow_html=True)

    def main():
        #Split into Train and Test Sets
        st.title("Split into Train and Test Sets")

        # File uploader for dataset
        uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

        if uploaded_file is not None:
            # Read dataset
            data = pd.read_csv(uploaded_file)
            
            #st.write("### Dataset Preview:")
            #st.write(data.head())
            # Check dataset content
            st.write("Dataset:")
            st.write(data)

            # Split into input and output variables
            array = data.values
            X = array[:, 0:12]
            Y = array[:, 12]
            st.write("### Dataset Preview:")
            st.write(data.shape)  # This will show you how many rows and columns are in your dataset

            # Set the test size using a slider
            test_size = st.slider("Test size (as a percentage)", 10, 50, 20) / 100
            seed = 7

            # Split the dataset into test and train
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

            # Train the data on a Logistic Regression model
            model = LogisticRegression(max_iter=500)
            model.fit(X_train, Y_train)

            # Evaluate the accuracy
            result = model.score(X_test, Y_test)
            accuracy = result * 100
            st.write(f"Accuracy: {accuracy:.3f}%")

            # Interpretation based on accuracy ranges
            accuracy_interpretation = ""

            if accuracy >= 90:
                accuracy_interpretation = "Excellent performance! The model is highly accurate."
            elif 80 <= accuracy < 90:
                accuracy_interpretation = "Good performance. The model is performing well but can still be improved."
            elif 70 <= accuracy < 80:
                accuracy_interpretation = "Fair performance. The model might need some improvements."
            else:
                accuracy_interpretation = "Poor performance. Consider revisiting the model and features."

            # Display interpretation
            st.write(f"**Interpretation:** {accuracy_interpretation}")

            # Guide or Legend for easier understanding
            st.write("""
            ### Accuracy Guide:
            - **90% - 100%**: Excellent performance
            - **80% - 90%**: Good performance
            - **70% - 80%**: Fair performance
            - **Below 70%**: Poor performance, requires improvement
            """)
            
            # Train the model on the entire dataset and save it
            model.fit(X, Y)  # Train on the entire dataset
            model_filename = "logistic_regression_split_classification.joblib"  # Filename to save the model
            joblib.dump(model, model_filename)  # Save the model to disk
            st.success(f"Model saved as {model_filename}")

            # Option to download the model
            with open(model_filename, "rb") as f:
                st.download_button("Download Model", f, file_name=model_filename)

            # Model upload for prediction
            st.subheader("Upload a Saved Model for Prediction")
            uploaded_model = st.file_uploader("Upload your model (joblib format)", type=["joblib"])

            if uploaded_model is not None:
                model = joblib.load(uploaded_model)  # Load the uploaded model

                # Sample input data for prediction
                st.subheader("Input Sample Data for Heart Disease Prediction")
                name = st.text_input("Enter your name:", "")
                age = st.number_input("Age", min_value=0, max_value=120, value=0)  # Age range 0-120
                anaemia = st.number_input("Anaemia (1 = yes, 0 = no)", min_value=0, max_value=1, value=0)  # Binary: 0 or 1
                creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/l)", min_value=0, max_value=5000, value=0)
                diabetes = st.number_input("Diabetes (1 = yes, 0 = no)", min_value=0, max_value=1, value=0)  # Binary: 0 or 1
                ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=0)  # Ejection fraction range
                high_blood_pressure = st.number_input("High Blood Pressure (1 = yes, 0 = no)", min_value=0, max_value=1, value=0)  # Binary: 0 or 1
                platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0, max_value=1000000, value=0)
                serum_creatinine = st.number_input("Serum Creatinine (mg/dl)", min_value=0.0, max_value=10.0, value=0.0)
                serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=120, max_value=180, value=130)
                sex = st.number_input("Sex (1 = male, 0 = female)", min_value=0, max_value=1, value=0)  # Sex is binary (0 or 1)
                smoking = st.number_input("Smoking (1 = yes, 0 = no)", min_value=0, max_value=1, value=0)  # Binary: 0 or 1
                time = st.number_input("Time (days)", min_value=0, max_value=1000, value=0)  # Time range

               
                # Creating input data array for prediction (ensure correct number of features)         
                input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, 
                            high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, 
                            smoking, time]])
                
                if st.button("Predict"):

                     # Display the input data summary
                  
                    st.subheader("Input Data Summary")
                    input_summary = {
                        "Name": name if name else "Not Provided",
                        "Age": age,
                        "Anaemia": "Yes" if anaemia == 1 else "No",
                        "Creatinine Phosphokinase": creatinine_phosphokinase,
                        "Diabetes": "Yes" if diabetes == 1 else "No",
                        "Ejection Fraction": ejection_fraction,
                        "High Blood Pressure": "Yes" if high_blood_pressure == 1 else "No",
                        "Platelets": platelets,
                        "Serum Creatinine": serum_creatinine,
                        "Serum Sodium": serum_sodium,
                        "Sex": "Male" if sex == 1 else "Female",
                        "Smoking": "Yes" if smoking == 1 else "No",
                        "Time": time,
                    }
                    
                    st.write(input_summary)

                    #Prediction
                    prediction = model.predict(input_data)
                    st.subheader("Prediction Result")
                    
                    # Display prediction based on the model's output
                    if prediction[0] == 0:
                        st.write("The predicted result is: **No Heart Failure**")
                    else:
                        st.write("The predicted result is: **Heart Failure**")


                    # Interpretation of input data
                    st.write("### Interpretation of Input Data")
                    st.write("""
                        - **Age**: Older individuals are at higher risk for heart disease.
                             
                        - **Anaemia**: Low red blood cell count can stress the heart, increasing risk.
                             
                        - **Creatinine Phosphokinase (CPK)**: Higher CPK levels may indicate previous heart damage.
                             
                        - **Diabetes**: High blood sugar can damage blood vessels and increase heart disease risk.
                             
                        - **Ejection Fraction**: Low percentage of blood pumped out may indicate heart failure.
                             
                        - **High Blood Pressure**: A major risk factor for heart disease and strokes.
                             
                        - **Platelets**: Abnormal platelet counts can affect blood flow and heart health.
                             
                        - **Serum Creatinine**: High levels suggest kidney dysfunction, linked to heart disease.
                             
                        - **Serum Sodium**: Affects fluid balance and blood pressure, influencing heart health.
                             
                        - **Sex**: Men are at higher risk at younger ages; risk increases for women post-menopause.
                             
                        - **Smoking**: Increases blood pressure and heart disease risk.
                             
                        - **Time**: Longer observation may indicate progression of heart disease.
                    """)

    if __name__ == "__main__":
        main()      
####################################################################################################
####################################################################################################

# Performance Metric
elif option == 'Performance Metric/s':
    st.title("PERFORMANCE METRICS")
    # Introduction and Instruction
    st.write("""       
    ### Instructions:
    1. Upload your CSV dataset.
    2. Click the button to evaluate the model.
    """)
    st.write("<br><br>", unsafe_allow_html=True)

    @st.cache_data
    def load_data(uploaded_file):
        names = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 
                'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 
                'smoking', 'time', 'DEATH_EVENT']
        dataframe = pd.read_csv(uploaded_file, names=names)
        return dataframe

    def preprocess_data(dataframe):
        # Convert categorical features to numeric if necessary
        categorical_cols = dataframe.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            dataframe[col] = le.fit_transform(dataframe[col])
        return dataframe

    def classification_accuracy_kfold(dataframe):
        st.title("Classification Accuracy (K-Fold Cross Validation)")

        array = dataframe.values
        X = array[:, 0:12]  # Features
        Y = array[:, 12]    # Target

        num_folds = st.slider("Select number of folds for K-Fold Cross Validation:", 2, 20, 10)
        if len(dataframe) < num_folds:
            st.write("The dataset is too small for the number of folds selected.")
            return

        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=None)
        model = LogisticRegression(max_iter=210)
        scoring = 'accuracy'
        results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

        if results is None or len(results) == 0 or any(pd.isna(results)):
            st.write("Error in cross-validation results. Please check your dataset.")
        else:
            st.subheader("Cross-Validation Results")
            st.write(f"Mean Accuracy: {results.mean() * 100:.3f}%")
            st.write(f"Standard Deviation: {results.std() * 100:.3f}%")

            plt.figure(figsize=(10, 5))
            plt.boxplot(results)
            plt.title('K-Fold Cross-Validation Accuracy')
            plt.ylabel('Accuracy')
            plt.xticks([1], [f'{num_folds}-Fold'])
            st.pyplot(plt)

            st.write("### Interpretation of Results")
            st.write("- Mean accuracy represents the average percentage of correctly predicted instances over the total instances.")
            st.write("- Standard deviation indicates the variability of the model's performance across different folds.")
            st.write("- A high mean accuracy with a low standard deviation suggests a reliable model.")

    def classification_accuracy_train_test_split(dataframe):
        st.title("Classification Accuracy (Train-Test Split 75:25)")

        array = dataframe.values
        X = array[:, 0:12]
        Y = array[:, 12]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=None)

        model = LogisticRegression(max_iter=210)
        model.fit(X_train, Y_train)

        accuracy = model.score(X_test, Y_test)

        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy * 100:.3f}%")

        st.write("### Interpretation of Results")
        st.write("- High accuracy (e.g., >75%) means the model is performing well.")
        st.write("- If accuracy is low, consider feature engineering or trying different algorithms.")

    def classification_report_section(dataframe):
        st.title("Classification Report")

        array = dataframe.values
        X = array[:, 0:12]
        Y = array[:, 12]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

        model = LogisticRegression(max_iter=180)
        model.fit(X_train, Y_train)

        predicted = model.predict(X_test)
        report = classification_report(Y_test, predicted, output_dict=True)

        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.write(report_df)

        st.subheader("Interpretation of the Classification Report")
        st.write("""
            - **Precision**: Correct positive predictions (higher is better).
            - **Recall**: Correctly identified positives (higher is better).
            - **F1 Score**: Balance between precision and recall (higher is better).
            - **Support**: Number of actual class instances.
        """)

    def main():
        st.title("Performance Metrics")

        # File uploader
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader")
        if uploaded_file is not None:
            st.write("Loading the dataset...")
            dataframe = load_data(uploaded_file)

            # Check for missing values
            if dataframe.isnull().values.any():
                st.write("The dataset contains missing values. Please clean your data.")
                return

            # Preprocess data
            dataframe = preprocess_data(dataframe)

            # Display first few rows of the dataset
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

            # Choose the option
            option = st.selectbox("Choose the analysis to perform", 
                                ["K-Fold Cross Validation", "Train-Test Split", "Classification Report"])

            if option == "K-Fold Cross Validation":
                classification_accuracy_kfold(dataframe)
            elif option == "Train-Test Split":
                classification_accuracy_train_test_split(dataframe)
            elif option == "Classification Report":
                classification_report_section(dataframe)
        else:
            st.write("Please upload a CSV file to proceed.")

    if __name__ == "__main__":
        main()     
####################################################################################################
####################################################################################################

# Hyperparameter Tuning
elif option == 'Hyperparameter Tuning':
    st.title("HYPERPARAMETER TUNING")
    # Introduction and Instructions
    st.write("""       
        ### Instructions for Hyperparameter Tuning
        1. **Upload Dataset**  

        2. **Set Hyperparameters**  

        3. **Run Tuning**  

        4. **Result**  
    """)

    st.write("<br><br>", unsafe_allow_html=True)
    # Initialize accuracy dictionary
    accuracy_results = {}
    
   # Hyperparameter tuning function
    def main():
        # Load the dataset
        filename = r"C:\Users\Jamaica Marie Canedo\Downloads\lab3.2\heart.csv"
        dataframe = pd.read_csv(filename)
        array = dataframe.values
        X = array[:, 0:12]
        Y = array[:, 12]

        # Tabs 
        tabs = st.tabs(["CART", "Naive Bayes", "AdaBoost","K-NN","Logistic","MLP","Perceptron","Random Forest","SVM" ])

        with tabs[0]:
            # Hyperparameter input fields in the main content area
            st.header("CART (Classification and Regression Trees) - Decision Tree")
            st.subheader("Set Hyperparameters")

            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
            random_seed = st.slider("Random Seed", 1, 100, 50)
            max_depth = st.slider("Max Depth", 1, 20, 5)
            min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)

            # Split the dataset into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Initialize and train the Decision Tree Classifier with hyperparameters
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_seed
            )

            model.fit(X_train, Y_train)

            # Evaluate the accuracy
            accuracy = model.score(X_test, Y_test)
            accuracy_results["CART"] = accuracy
            # Display the accuracy in the app
            st.write(f"**Accuracy:** {accuracy * 100.0:.3f}%")

            # Interpretation and Guide for Results
            st.write("""

                - **Accuracy (%)**: A higher accuracy score means the model correctly classified more instances. 

                - **Max Depth**: This controls the depth of the tree. Adjusting max depth allows balancing complexity and generalization.

                - **Min Samples Split and Min Samples Leaf**: These parameters control how splits are made. Increase these values if the model is overfitting, or reduce if underfitting.
            """)

        with tabs[1]:
            # Hyperparameter input fields in the main content area
            st.header("Gaussian Naive Bayes")
            st.subheader("Set Hyperparameters")

            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test_size")
            random_seed = st.slider("Random Seed", 1, 100, 7, key="random_seed")
            var_smoothing_range = st.slider("Var Smoothing Range (Log Scale)", -15, -1, (-9, -5), key="var_smoothing_range")

            # Automatically run tuning and collect results based on parameter values
            results = []  # Store results of each run

            for log_smoothing in range(var_smoothing_range[0], var_smoothing_range[1] + 1):
                var_smoothing_value = 10 ** log_smoothing

                # Split the dataset
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

                # Initialize and train the model
                model = GaussianNB(var_smoothing=var_smoothing_value)
                model.fit(X_train, Y_train)

                # Evaluate the model
                Y_pred = model.predict(X_test)
                accuracy = accuracy_score(Y_test, Y_pred)
                accuracy_results["Naive Bayes"] = accuracy

            # Display the accuracy in the app
            st.write(f"**Accuracy:** {accuracy * 100.0:.3f}%")

            # Interpretation and Guide for Results
            st.write("""

                - **Accuracy (%)**: Higher accuracy means better performance.
                - **Test Size**: A typical value is 20% (0.2); larger test sizes offer more stability but less training data.
                - **Var Smoothing**: Controls the smoothness of class boundaries in the model.
            """)

        
        with tabs[2]:
            # Hyperparameter input fields in the main content area
            st.header("Gradient Boosting Machines (AdaBoost) ")
            st.subheader("Set Hyperparameters")

            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test_size2")
            random_seed = st.slider("Random Seed", 1, 100, 7, key="random_seed2")
            n_estimators = st.slider("Number of Estimators", 1, 100, 50)

            # Split the dataset into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Create an AdaBoost classifier
            model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_seed)

            # Train the model on the training data
            model.fit(X_train, Y_train)

            # Evaluate the accuracy
            accuracy = model.score(X_test, Y_test)
            accuracy_results["AdaBoost"] = accuracy

            # Display the accuracy in the app
            st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

            # Interpretation and Guide for Results
            st.write("""

                - **Accuracy (%)**: High accuracy means the model is classifying well.
                - **Number of Estimators**: Controls weak learners; start with 50 and adjust for performance and training speed.
                - **Test Size**: 20% is typical; it helps balance training and testing data.
                - **Random Seed**: Ensures reproducibility; test different seeds for stable results.
            """)


        with tabs[3]:
            # Hyperparameter input fields in the main content area
            st.header("K-Nearest Neighbors (K-NN) ")
            st.subheader("Set Hyperparameters")

            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="keytest3")
            random_seed = st.slider("Random Seed", 1, 100, 7, key="seed3")
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
            weights = st.selectbox("Weights", options=["uniform", "distance"])
            algorithm = st.selectbox("Algorithm", options=["auto", "ball_tree", "kd_tree", "brute"])
            
           # Split the dataset into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Create a K-Nearest Neighbors (K-NN) classifier
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

            # Train the model on the training data
            model.fit(X_train, Y_train)

            # Evaluate the accuracy
            accuracy = model.score(X_test, Y_test)
            accuracy_results["K-NN"] = accuracy
            # Display the accuracy in the app
            st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

            # Interpretation and Guide for Results
            st.write("""
                - **Accuracy (%)**: A higher accuracy score suggests the model is making correct predictions on most test instances. 

                - **Number of Neighbors**: The `n_neighbors` parameter represents the number of neighbors considered for majority voting in classification. Smaller values (e.g., `n_neighbors=1`) may capture more detail and work well with fewer patterns but can overfit with noisy data. Larger values generally smooth the decision boundary but may miss finer distinctions.

                - **Weights**: Setting `weights` to "uniform" treats all neighbors equally, while "distance" assigns more weight to closer neighbors. Using "distance" can help improve accuracy if closer neighbors are more similar to the test points.

                - **Algorithm**: This specifies the algorithm used to compute nearest neighbors. “auto” allows the model to select the best option based on the dataset size and structure.
            """)

        with tabs[4]:
            # Hyperparameter input fields in the main content area
            st.header("Logistic Regression")
            st.subheader("Set Hyperparameters")

            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test4")
            random_seed = st.slider("Random Seed", 1, 100, 7, key="seed4")
            max_iter = st.slider("Max Iterations", 100, 500, 200)
            solver = st.selectbox("Solver", options=["lbfgs", "liblinear", "sag", "saga", "newton-cg"])
            C = st.number_input("Inverse of Regularization Strength", min_value=0.01, max_value=10.0, value=1.0)
            
            # Split the dataset into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Create a Logistic Regression model
            model = LogisticRegression(max_iter=max_iter, solver=solver, C=C)

            # Train the model on the training data
            model.fit(X_train, Y_train)

            # Evaluate the accuracy
            accuracy = model.score(X_test, Y_test)
            accuracy_results["Logistic"] = accuracy

            # Display the accuracy in the app
            st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

            # Interpretation and Guide for Results
            st.write("""

                - **Accuracy (%)**: Higher accuracy indicates the model is making more correct predictions. 

                - **C Parameter**: Start with `C=1.0` and tune it based on observed accuracy changes.     

                - **Inverse of Regularization Strength (C)**: Controls the amount of regularization applied to the model. A smaller `C` value indicates stronger regularization, which can reduce overfitting but may lower accuracy. A larger `C` value reduces regularization, potentially increasing accuracy but risking overfitting.

                - **Solver**: Different solvers are optimized for various types of data and may impact training speed and performance. For example, “liblinear” is often suited for small datasets or binary classification, while “lbfgs” works well with larger datasets.

                - **Max Iterations**: The maximum number of iterations controls when the solver stops trying to improve. 

            """)  

        with tabs[5]:
            # Hyperparameter input fields in the main content area
            st.header("Multi-Layer Perceptron(MLP)")
            st.subheader("Set Hyperparameters")

            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test5")
            random_seed = st.slider("Random Seed", 1, 100, 7, key="seed5")
            hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 65,32)", "65,32")
            activation = st.selectbox("Activation Function", options=["identity", "logistic", "tanh", "relu"])
            max_iter = st.slider("Max Iterations", 100, 500, 200, key="max5")
            
            # Convert hidden_layer_sizes input to tuple
            hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))

            # Split the dataset into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Create an MLP-based model
            model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, 
                                solver='adam', max_iter=max_iter, random_state=random_seed)

            # Train the model
            model.fit(X_train, Y_train)

            # Evaluate the accuracy
            accuracy = model.score(X_test, Y_test)
            accuracy_results["MLP"] = accuracy

            # Display the accuracy in the app
            st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

            # Interpretation and Guide for Results
            st.write("""
                     
                - **Accuracy (%)**: Higher accuracy indicates better predictive performance. 

                - **Hidden Layer Sizes**: This parameter specifies the number of neurons in each hidden layer, significantly impacting the model's complexity and ability to learn intricate patterns. 

                - **Activation Function**: The activation function introduces non-linear properties to the network. For example:
                    - **relu** is generally a good starting choice and works well for most applications.
                    - **tanh** and **logistic** (sigmoid) may work better for smaller datasets or when a smoother decision boundary is desired.
                    - **identity** can be useful for specific linear data scenarios, but it's rarely chosen for complex problems.

                - **Max Iterations**: This controls the maximum number of passes over the data during training. If the model doesn’t converge (i.e., the loss stops decreasing), you may need to increase this value. 
               
            """)

        with tabs[6]:
            # Hyperparameter input fields in the main content area
            st.header("Perceptron")
            st.subheader("Set Hyperparameters")

            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test6")
            random_seed = st.slider("Random Seed", 1, 100, 7, key="seed6")
            max_iter = st.slider("Max Iterations", 100, 500, 200, key="max6")
            eta0 = st.number_input("Initial Learning Rate", min_value=0.001, max_value=10.0, value=1.0)
            tol = st.number_input("Tolerance for Stopping Criterion", min_value=0.0001, max_value=1.0, value=1e-3)
            
           # Split the dataset into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Create a Perceptron classifier
            model = Perceptron(max_iter=max_iter, random_state=random_seed, eta0=eta0, tol=tol)

            # Train the model
            model.fit(X_train, Y_train)

            # Evaluate the accuracy
            accuracy = model.score(X_test, Y_test)
            accuracy_results["Perceptron"] = accuracy

            # Display the accuracy in the app
            st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

            # Interpretation and Guide for Results
            st.write("""

                - **Max Iterations (max_iter)**: Controls the maximum number of iterations over the training data. 
                
                - **Initial Learning Rate (eta0)**: Determines the step size during each update. 
                
                - **Tolerance for Stopping Criterion (tol)**: Defines the minimum required change in the model's loss for each iteration. 
                     
            """)

        with tabs[7]:
            # Hyperparameter input fields in the main content area
            st.header("Random Forest")
            st.subheader("Set Hyperparameters")

            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test7")
            random_seed = st.slider("Random Seed", 1, 100, 7, key="seed7")
            n_estimators = st.slider("Number of Estimators (Trees)", 10, 200, 100)
            max_depth = st.slider("Max Depth of Trees", 1, 50, None)  # Allows None for no limit
            min_samples_split = st.slider("Min Samples to Split a Node", 2, 10, 2)
            min_samples_leaf = st.slider("Min Samples in Leaf Node", 1, 10, 1)
                    
           # Split the dataset into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Create a Random Forest classifier
            rfmodel = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_seed,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            )

            # Train the model
            rfmodel.fit(X_train, Y_train)

            # Evaluate the accuracy
            accuracy = rfmodel.score(X_test, Y_test)
            accuracy_results["Random Forest"] = accuracy
            # Display the accuracy in the app
            st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

            # Interpretation and Guide for Results
            st.write("""

                - **Accuracy (%)**: A higher accuracy means the model is correctly predicting the target variable for most instances in the test set.

                - **Number of Estimators (Trees)**: The number of trees in the forest can impact the model's performance. 

                - **Max Depth of Trees**: Controls the maximum depth of each tree in the forest. 

                - **Min Samples to Split a Node**: This parameter controls the minimum number of samples required to split an internal node. 

                - **Min Samples in Leaf Node**: Controls the minimum number of samples required to be in a leaf node. 

            """)

        with tabs[8]:
            # Hyperparameter input fields in the main content area
            st.header("Support Vector Machines (SVM)")
            st.subheader("Set Hyperparameters")

            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test8")
            random_seed = st.slider("Random Seed", 1, 100, 42, key="seed8")
            C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
            kernel = st.selectbox("Kernel Type", options=['linear', 'poly', 'rbf', 'sigmoid'])
                            
           # Split the dataset into training and testing sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Create an SVM classifier
            model = SVC(kernel=kernel, C=C, random_state=random_seed)

            # Train the model
            model.fit(X_train, Y_train)

            # Evaluate the accuracy
            result = model.score(X_test, Y_test)
            accuracy_results["SVM"] = result
            
            st.write(f"Accuracy: {result * 100.0:.3f}%")

            # Interpretation and Guide for Results
            st.write("""

                - **Accuracy (%)**: Higher accuracy indicates better generalization, i.e., the model performs well on unseen data. 

                - **Regularization Parameter (C)**: The value of `C` controls the trade-off between achieving a low error on the training data and minimizing the model complexity (overfitting). 

                - **Kernel Type**: The kernel defines the type of transformation applied to the data to allow for separation in a higher-dimensional space:
                    - **linear**: Best for linearly separable data.
                    - **poly**: Good for non-linear data with a polynomial decision boundary.
                    - **rbf (Radial Basis Function)**: Common choice for non-linear data; transforms data to infinite-dimensional space.
                    - **sigmoid**: Similar to a neural network; can be useful in some specific cases, but may require tuning.

                - **Test Size and Random Seed**: Experiment with different `test_size` and `random_seed` values to ensure that the model is stable and generalizes well across different splits of the data.

            """)
        st.write("<br><br>", unsafe_allow_html=True)
        # Accuracy Results Table and Visualization
        st.subheader("Model Comparison")
        accuracy_df = pd.DataFrame(list(accuracy_results.items()), columns=["Algorithm", "Accuracy"])
        highest = accuracy_df["Accuracy"].idxmax()
        lowest = accuracy_df["Accuracy"].idxmin()
        
        # Highlight highest and lowest
        def highlight_row(row):
            if row.name == highest:
                return ["background-color: green; color: white"] * len(row)
            elif row.name == lowest:
                return ["background-color: red; color: white"] * len(row)
            else:
                return [""] * len(row)

        st.dataframe(accuracy_df.style.apply(highlight_row, axis=1))
        st.write("<br><br>", unsafe_allow_html=True)
        # Visualization
        colors = ['red' if idx == lowest else 'green' if idx == highest else 'blue' for idx in range(len(accuracy_df))]

        #st.bar_chart(data=accuracy_df.set_index("Algorithm"), use_container_width=True)
        fig, ax = plt.subplots()
        ax.bar(accuracy_df["Algorithm"], accuracy_df["Accuracy"], color=colors)
        ax.set_title("Accuracy Comparison of ML Algorithms")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Algorithm")
        plt.xticks(rotation=45, ha="right")

        st.pyplot(fig)

        # Insights
        st.write("### Insights")
        st.write("""
        - The highlighted green algorithm had the highest accuracy.
        - The highlighted red algorithm had the lowest accuracy.
         """)


    # Call the main function to run the code
    if __name__ == "__main__":
        main()

