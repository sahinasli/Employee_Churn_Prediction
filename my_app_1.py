import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from skimpy import clean_columns

# Set the page configuration
st.set_page_config(page_title="Employee Churn Prediction App", layout="wide")

# Page 1: Project Information and Image
def page_info():
    # Display an image
    st.image("image3_churn.jpg", caption="Project Image", use_column_width=True)
    
    st.title("Project and Data Information")
    st.write("""
        **Project Description:**
        This project involves predicting employee churn using HR data. The dataset includes 14,999 samples and contains information about employees who have either stayed or left the company.

        **Dataset Attributes:**
        - `satisfaction_level`: Employee satisfaction level, ranging from 0 to 1.
        - `last_evaluation`: Performance evaluation score by the employer, ranging from 0 to 1.
        - `number_projects`: Number of projects assigned to the employee.
        - `average_monthly_hours`: Average number of hours worked per month by the employee.
        - `time_spent_company`: Number of years the employee has spent with the company, indicating experience.
        - `work_accident`: Whether the employee has had a work accident (1) or not (0).
        - `promotion_last_5years`: Whether the employee has had a promotion in the last 5 years (1) or not (0).
        - `departments`: Department where the employee works.
        - `salary`: Salary level of the employee (low, medium, high).
        - `left`: Whether the employee has left the company (1) or not (0).

        **Project Steps:**
        1. **Exploratory Data Analysis (EDA)**: Observing data structure, identifying outliers and missing values, and understanding features that affect the target variable using data visualization techniques.
        2. **Data Pre-Processing**: Scaling and encoding data to enhance the accuracy of classification algorithms.
        3. **Cluster Analysis**: Performing clustering based on characteristics identified during EDA to group similar data points.
        4. **Model Building**: Splitting data into training and test sets, then training classification models (Random Forest and XGBoost) and evaluating their performance.
        5. **Model Deployment**: Deploying the trained models using Streamlit to allow users to input data and get predictions.
    """)

# Page 2: Prediction
def page_prediction():
    st.title("Employee Churn Prediction")

    # Load the dataset and process it
    @st.cache_data
    def load_data():
        df = pd.read_csv('HR_Dataset.csv')
        df = clean_columns(df)  # Use skimpy to clean column names
        df.rename(columns={'average_montly_hours': 'average_monthly_hours'}, inplace=True)
        df.drop_duplicates(inplace=True, ignore_index=True)
        return df

    df = load_data()

    # Define feature columns
    numeric_features = ['satisfaction_level', 'last_evaluation', 'number_project', 
                        'average_monthly_hours', 'time_spend_company', 'work_accident', 
                        'promotion_last_5years']
    cat_onehot = ['departments']
    cat_ordinal = ['salary']

    # Define transformers
    numeric_transformer = StandardScaler()
    onehot_transformer = OneHotEncoder()
    ordinal_transformer = OrdinalEncoder(categories=[['low', 'medium', 'high']])

    # Combine preprocessing for numeric, one-hot categorical, and ordinal categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat_onehot', onehot_transformer, cat_onehot),
            ('cat_ordinal', ordinal_transformer, cat_ordinal)])

    # Define Random Forest and XGBoost pipelines
    operations_rf = [
        ("preprocessor", preprocessor),
        ("RF_model", RandomForestClassifier(class_weight="balanced", random_state=101)),
    ]
    pipe_model_rf = Pipeline(steps=operations_rf)

    operations_xgb = [
        ("preprocessor", preprocessor),
        ("XGB_model", XGBClassifier(eval_metric='logloss', random_state=101)),
    ]
    pipe_model_xgb = Pipeline(steps=operations_xgb)

    # Split data into train and test sets
    X = df.drop(columns=['left'])
    y = df['left']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the models
    pipe_model_rf.fit(X_train, y_train)
    pipe_model_xgb.fit(X_train, y_train)

    # Save the models
    joblib.dump(pipe_model_rf, 'rf_model_app.pkl')
    joblib.dump(pipe_model_xgb, 'xgb_model_app.pkl')

    # Load the models for prediction
    rf_model = joblib.load("rf_model_app.pkl")
    xgb_model = joblib.load("xgb_model_app.pkl")

    # User input for prediction
    st.sidebar.header("Enter Data for Prediction")
    
    def user_input():
        satisfaction_level = st.sidebar.slider('Satisfaction Level', 0.0, 1.0, 0.5)
        last_evaluation = st.sidebar.slider('Last Evaluation', 0.0, 1.0, 0.5)
        number_project = st.sidebar.slider('Number of Projects', 1, 7, 4)
        average_monthly_hours = st.sidebar.slider('Average Monthly Hours', 80, 320, 200)
        time_spend_company = st.sidebar.slider('Time Spent at Company', 1, 10, 3)
        work_accident = st.sidebar.selectbox('Work Accident', [0, 1])
        promotion_last_5years = st.sidebar.selectbox('Promotion in Last 5 Years', [0, 1])
        departments = st.sidebar.selectbox('Departments', df['departments'].unique())
        salary = st.sidebar.selectbox('Salary', ['low', 'medium', 'high'])

        data = {
            'satisfaction_level': satisfaction_level,
            'last_evaluation': last_evaluation,
            'number_project': number_project,
            'average_monthly_hours': average_monthly_hours,
            'time_spend_company': time_spend_company,
            'work_accident': work_accident,
            'promotion_last_5years': promotion_last_5years,
            'departments': departments,
            'salary': salary
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input()

    # Display user input
    st.write("**Input Data**")
    st.write(input_df)

    # Prediction buttons
    if st.button("Predict (Random Forest)"):
        rf_prediction = rf_model.predict(input_df)
        st.write("Random Forest Prediction: ", "**Churn**" if rf_prediction[0] == 1 else "**No Churn**")
        rf_pred_proba = rf_model.predict_proba(input_df)
        st.write(f"Probability of No Churn: **{rf_pred_proba[0][0]:.2f}**")
        st.write(f"Probability of Churn: **{rf_pred_proba[0][1]:.2f}**")
        rf_recall = recall_score(y_test, rf_model.predict(X_test))
        st.write(f"Random Forest Recall on test data: **{rf_recall:.2f}**")

    if st.button("Predict (XGBoost)"):
        xgb_prediction = xgb_model.predict(input_df)
        st.write("XGBoost Prediction: ", "**Churn**" if xgb_prediction[0] == 1 else "**No Churn**")
        xgb_pred_proba = xgb_model.predict_proba(input_df)
        st.write(f"Probability of No Churn: **{xgb_pred_proba[0][0]:.2f}**")
        st.write(f"Probability of Churn: **{xgb_pred_proba[0][1]:.2f}**")
        xgb_recall = recall_score(y_test, xgb_model.predict(X_test))
        st.write(f"XGBoost Recall on test data: **{xgb_recall:.2f}**")

# Page 3: Visualization
def page_visualization():
    st.title("Data Visualization and Model Evaluation")

    # Load the dataset
    @st.cache_data
    def load_data():
        df = pd.read_csv('HR_Dataset.csv')
        df = clean_columns(df)
        df.rename(columns={'average_montly_hours': 'average_monthly_hours'}, inplace=True)
        df.drop_duplicates(inplace=True, ignore_index=True)
        return df

    df = load_data()

    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe().T)
    
    # Pie chart of employees who left vs stayed using Plotly
    st.subheader("Employee Churn Distribution")
    sizes = df.left.value_counts()
    labels = ["Stay", "Churn"]

    # Create pie chart
    fig = px.pie(values=sizes, names=labels, title='Employee Churn Distribution')
    
    # Adjust the size of the pie chart
    fig.update_layout(width=800, height=800)
    
    # Display the pie chart with Streamlit
    st.plotly_chart(fig)

    # Bar chart of employees who left by department using Plotly with annotations
    st.subheader("Churn by Department")
    churn_by_department = df[df['left'] == 1]['departments'].value_counts().sort_values(ascending=False)
    fig = go.Figure(data=[go.Bar(
        x=churn_by_department.index,
        y=churn_by_department.values,
        text=churn_by_department.values,
        textposition='auto',
        marker_color='indianred'
    )])
    fig.update_layout(
        title="Number of Employees Who Left by Department",
        xaxis_title="Departments",
        yaxis_title="Number of Employees Who Left",
        width=800,
        height=600
    )
    st.plotly_chart(fig)

   
    # Heatmap for correlation matrix using Plotly
    st.subheader("Heatmap of Feature Correlations")

    # Compute the correlation matrix
    corr_matrix = df.corr(numeric_only=True)

    # Create a heatmap with column names
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,  # Column names as x-axis labels
        y=corr_matrix.columns,  # Column names as y-axis labels
        colorscale='Viridis',
        texttemplate="%{z:.2f}",  # Show numbers with 2 decimal places
        hovertemplate="Value: %{z:.2f}"  # Hover display with 2 decimal places
    ))

    # Update layout with title
    fig.update_layout(title='Heatmap with Formatted Numbers')

    # Display the heatmap in Streamlit
    st.plotly_chart(fig)

    # Model evaluation metrics
    st.subheader("Model Evaluation Metrics")
    rf_model = joblib.load('rf_model_app.pkl')
    xgb_model = joblib.load('xgb_model_app.pkl')

    # Split data into train and test sets
    X = df.drop(columns=['left'])
    y = df['left']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predictions for evaluation
    rf_y_pred = rf_model.predict(X_test)
    xgb_y_pred = xgb_model.predict(X_test)

    # Calculate evaluation metrics
    metrics = {
        'Model': ['Random Forest', 'XGBoost'],
        'Accuracy': [accuracy_score(y_test, rf_y_pred), accuracy_score(y_test, xgb_y_pred)],
        'Precision': [precision_score(y_test, rf_y_pred), precision_score(y_test, xgb_y_pred)],
        'Recall': [recall_score(y_test, rf_y_pred), recall_score(y_test, xgb_y_pred)],
        'F1 Score': [f1_score(y_test, rf_y_pred), f1_score(y_test, xgb_y_pred)]
    }

    metrics_df = pd.DataFrame(metrics)
    st.write(metrics_df)

    
# Streamlit app structure
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ["Project Information", "Employee Churn Prediction", "Data Visualization"])

    if page == "Project Information":
        page_info()
    elif page == "Employee Churn Prediction":
        page_prediction()
    elif page == "Data Visualization":
        page_visualization()

if __name__ == "__main__":
    main()
