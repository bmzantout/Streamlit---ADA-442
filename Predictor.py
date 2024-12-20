# import plotly.graph_objects as go
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle as pkl
# from streamlit_lottie import st_lottie
# import json

# st.set_page_config(
#     page_title="Term Deposit Success Predictor",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="📊",
# )

# st.title("Term Deposit Success Predictor")

# # sidebar
# st.sidebar.header("Term Deposit Success Predictor")
# st.sidebar.image("gadgets/data_analysis.jpeg")
# st.sidebar.subheader("Choose Your Favorite Predictor")

# # Filters
# model = st.sidebar.selectbox(
#     "predictors", ["Random Forest", "Logistic Regression", "KNN"]
# )

# # Load Cleaned Data
# df = pd.read_csv("datasets/cleaned_data_v2.csv")

# # !
# # # Load preprocessor
# # preprocessor = pkl.load(open('models/preprocessor.pkl', 'rb'))

# # Load models
# rf = pkl.load(open('models/best_rf.pkl', 'rb'))
# gb = pkl.load(open('models/best_logreg.pkl', 'rb'))
# xgb = pkl.load(open('models/best_knn.pkl', 'rb'))

# # model selection
# if model == "Random Forest":
#     model = rf
# elif model == "Logistic Regression":
#     model = gb
# elif model == "KNN":
#     model = xgb

# # Input Data
# # Create two columns
# col1, col2 = st.columns(2)

# # First column inputs
# with col1:
#     job_type = st.selectbox('Job Type', df['JobType'].unique())
#     marital_status = st.selectbox('Marital Status', df['MaritalStatus'].unique())
#     education_level = st.selectbox('Education Level', df['EducationLevel'].unique())
#     has_housing_loan = st.selectbox('Has Housing Loan', df['HasHousingLoan'].unique())
#     has_personal_loan = st.selectbox('Has Personal Loan', df['HasPersonalLoan'].unique())
#     contact_type = st.selectbox('Contact Communication Type', df['ContactCommunicationType'].unique())
#     last_contact_month = st.selectbox('Last Contact Month', df['LastContactMonth'].unique())
#     last_contact_day = st.selectbox('Last Contact Day', df['LastContactDayOfWeek'].unique())
#     campaign_outcome = st.selectbox('Previous Campaign Outcome', df['PreviousCampaignOutcome'].unique())

# # Second column inputs
# with col2:
#     age = st.number_input('Age', df.Age.min(), df.Age.max())
#     call_duration = st.number_input('Call Duration', df.CallDuration.min(), df.CallDuration.max())
#     campaign_contacts = st.number_input('Campaign Contacts', df.CampaignContacts.min(), df.CampaignContacts.max())
#     previous_contacts = st.number_input('Previous Campaign Contacts', df.PreviousCampaignContacts.min(), df.PreviousCampaignContacts.max())
#     emp_var_rate = st.number_input('Employment Variation Rate', df['EmploymentVariationRate'].min(), df['EmploymentVariationRate'].max())
#     consumer_price_idx = st.number_input('Consumer Price Index', df['ConsumerPriceIndex'].min(), df['ConsumerPriceIndex'].max())
#     consumer_conf_idx = st.number_input('Consumer Confidence Index', df['ConsumerConfidenceIndex'].min(), df['ConsumerConfidenceIndex'].max())
#     euribor_rate = st.number_input('Euribor 3M Rate', df['Euribor3MRate'].min(), df['Euribor3MRate'].max())
#     num_employees = st.number_input('Number of Employees', df['NumberOfEmployees'].min(), df['NumberOfEmployees'].max())

# # Preprocessing
# new_data = {
#     'JobType': job_type,
#     'MaritalStatus': marital_status,
#     'EducationLevel': education_level,
#     'HasHousingLoan': has_housing_loan,
#     'HasPersonalLoan': has_personal_loan,
#     'ContactCommunicationType': contact_type,
#     'LastContactMonth': last_contact_month,
#     'LastContactDayOfWeek': last_contact_day,
#     'PreviousCampaignOutcome': campaign_outcome,
#     'Age': age,
#     'CallDuration': call_duration,
#     'CampaignContacts': campaign_contacts,
#     'PreviousCampaignContacts': previous_contacts,
#     'EmploymentVariationRate': emp_var_rate,
#     'ConsumerPriceIndex': consumer_price_idx,
#     'ConsumerConfidenceIndex': consumer_conf_idx,
#     'Euribor3MRate': euribor_rate,
#     'NumberOfEmployees': num_employees
# }

# new_data = pd.DataFrame(new_data, index=[0])

# # !
# # # Preprocessing
# # new_data = preprocessor.transform(new_data)



# # Prediction
# prediction = model.predict(new_data)

# if prediction == 0:
#     prediction = 'NO'
# else:
#     prediction = 'YES'

# # Output
# if st.button('Predict'):
#     st.markdown('# Deposit Prediction')
#     st.markdown(prediction)

# f1_scores = [60.0, 60.0, 59.0]
# recall_scores = [95.0, 94.0, 94.0]
# precision_scores = [44.0, 44.0, 43.0]

# fig = go.Figure(data=[go.Table(
#     header=dict(
#         values=['<b>Model<b>', '<b>F1 Score<b>', '<b>Recall<b>', '<b>Precision<b>'],
#         line_color='darkslategray',
#         fill_color='whitesmoke',
#         align=['center', 'center'],
#         font=dict(color='black', size=18),
#         height=40
#     ),
#     cells=dict(
#         values=[
#             ['<b>Random Forest<b>', '<b>AdaBoost<b>', '<b>SVM<b>'],
#             [f1_scores[0], f1_scores[1], f1_scores[2]],
#             [recall_scores[0], recall_scores[1], recall_scores[2]],
#             [precision_scores[0], precision_scores[1], precision_scores[2]]
#         ]
#     )
# )])

# fig.update_layout(title='Model Results On Test Data')
# st.plotly_chart(fig, use_container_width=True)



# # /---------------------------------------------------------------------------------------------------------/
# # /---------------------------------------------------------------------------------------------------------/


# import plotly.graph_objects as go
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle as pkl
# from streamlit_lottie import st_lottie
# import json
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder

# st.set_page_config(
#     page_title="Term Deposit Success Predictor",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="📊",
# )

# st.title("Term Deposit Success Predictor")

# # sidebar
# st.sidebar.header("Term Deposit Success Predictor")
# st.sidebar.image("gadgets/data_analysis.jpeg")
# st.sidebar.subheader("Choose Your Favorite Predictor")

# # Filters
# model = st.sidebar.selectbox(
#     "predictors", ["Random Forest", "Logistic Regression", "KNN"]
# )

# # Load Cleaned Data
# df = pd.read_csv("datasets/cleaned_data_v2.csv")

# # Define categorical and numerical columns
# categorical_columns = ['JobType', 'MaritalStatus', 'EducationLevel', 'HasHousingLoan', 
#                       'HasPersonalLoan', 'ContactCommunicationType', 'LastContactMonth', 
#                       'LastContactDayOfWeek', 'PreviousCampaignOutcome']

# numerical_columns = ['Age', 'CallDuration', 'CampaignContacts', 'PreviousCampaignContacts',
#                     'EmploymentVariationRate', 'ConsumerPriceIndex', 'ConsumerConfidenceIndex',
#                     'Euribor3MRate', 'NumberOfEmployees']

# # Create preprocessing pipelines
# numeric_transformer = Pipeline(steps=[
#     ('scaler', StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
# ])

# # Combine transformers
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numerical_columns),
#         ('cat', categorical_transformer, categorical_columns)
#     ])

# # Fit the preprocessor
# preprocessor.fit(df)

# # Load models
# rf = pkl.load(open('models/best_rf.pkl', 'rb'))
# logreg = pkl.load(open('models/best_logreg.pkl', 'rb'))
# knn = pkl.load(open('models/best_knn.pkl', 'rb'))


# # model selection
# if model == "Random Forest":
#     model = rf
# if model == "Logistic Regression":
#     model = logreg
# elif model == "KNN":
#     model = knn

# # Input Data
# # Create two columns
# col1, col2 = st.columns(2)

# # First column inputs
# with col1:
#     job_type = st.selectbox('Job Type', df['JobType'].unique())
#     marital_status = st.selectbox('Marital Status', df['MaritalStatus'].unique())
#     education_level = st.selectbox('Education Level', df['EducationLevel'].unique())
#     has_housing_loan = st.selectbox('Has Housing Loan', df['HasHousingLoan'].unique())
#     has_personal_loan = st.selectbox('Has Personal Loan', df['HasPersonalLoan'].unique())
#     contact_type = st.selectbox('Contact Communication Type', df['ContactCommunicationType'].unique())
#     last_contact_month = st.selectbox('Last Contact Month', df['LastContactMonth'].unique())
#     last_contact_day = st.selectbox('Last Contact Day', df['LastContactDayOfWeek'].unique())
#     campaign_outcome = st.selectbox('Previous Campaign Outcome', df['PreviousCampaignOutcome'].unique())

# # Second column inputs
# with col2:
#     age = st.number_input('Age', df.Age.min(), df.Age.max())
#     call_duration = st.number_input('Call Duration', df.CallDuration.min(), df.CallDuration.max())
#     campaign_contacts = st.number_input('Campaign Contacts', df.CampaignContacts.min(), df.CampaignContacts.max())
#     previous_contacts = st.number_input('Previous Campaign Contacts', df.PreviousCampaignContacts.min(), df.PreviousCampaignContacts.max())
#     emp_var_rate = st.number_input('Employment Variation Rate', df['EmploymentVariationRate'].min(), df['EmploymentVariationRate'].max())
#     consumer_price_idx = st.number_input('Consumer Price Index', df['ConsumerPriceIndex'].min(), df['ConsumerPriceIndex'].max())
#     consumer_conf_idx = st.number_input('Consumer Confidence Index', df['ConsumerConfidenceIndex'].min(), df['ConsumerConfidenceIndex'].max())
#     euribor_rate = st.number_input('Euribor 3M Rate', df['Euribor3MRate'].min(), df['Euribor3MRate'].max())
#     num_employees = st.number_input('Number of Employees', df['NumberOfEmployees'].min(), df['NumberOfEmployees'].max())

# # Create input data dictionary
# new_data = {
#     'JobType': job_type,
#     'MaritalStatus': marital_status,
#     'EducationLevel': education_level,
#     'HasHousingLoan': has_housing_loan,
#     'HasPersonalLoan': has_personal_loan,
#     'ContactCommunicationType': contact_type,
#     'LastContactMonth': last_contact_month,
#     'LastContactDayOfWeek': last_contact_day,
#     'PreviousCampaignOutcome': campaign_outcome,
#     'Age': age,
#     'CallDuration': call_duration,
#     'CampaignContacts': campaign_contacts,
#     'PreviousCampaignContacts': previous_contacts,
#     'EmploymentVariationRate': emp_var_rate,
#     'ConsumerPriceIndex': consumer_price_idx,
#     'ConsumerConfidenceIndex': consumer_conf_idx,
#     'Euribor3MRate': euribor_rate,
#     'NumberOfEmployees': num_employees
# }

# # Convert to DataFrame and preprocess
# new_data = pd.DataFrame(new_data, index=[0])
# new_data_processed = preprocessor.transform(new_data)

# # Prediction
# X_train = df.drop(columns=['SubscribedToTermDeposit'])  # Assuming 'TermDepositSuccess' is your target column
# y_train = df['SubscribedToTermDeposit']
# model.fit(X_train, y_train)
# prediction = model.predict(new_data_processed)

# if prediction == 0:
#     prediction = 'NO'
# else:
#     prediction = 'YES'

# # Output
# if st.button('Predict'):
#     st.markdown('# Deposit Prediction')
#     st.markdown(prediction)

# f1_scores = [60.0, 60.0, 59.0]
# recall_scores = [95.0, 94.0, 94.0]
# precision_scores = [44.0, 44.0, 43.0]

# fig = go.Figure(data=[go.Table(
#     header=dict(
#         values=['<b>Model<b>', '<b>F1 Score<b>', '<b>Recall<b>', '<b>Precision<b>'],
#         line_color='darkslategray',
#         fill_color='whitesmoke',
#         align=['center', 'center'],
#         font=dict(color='black', size=18),
#         height=40
#     ),
#     cells=dict(
#         values=[
#             ['<b>Random Forest<b>', '<b>AdaBoost<b>', '<b>SVM<b>'],
#             [f1_scores[0], f1_scores[1], f1_scores[2]],
#             [recall_scores[0], recall_scores[1], recall_scores[2]],
#             [precision_scores[0], precision_scores[1], precision_scores[2]]
#         ]
#     )
# )])

# fig.update_layout(title='Model Results On Test Data')
# st.plotly_chart(fig, use_container_width=True)


# # /---------------------------------------------------------------------------------------------------------/
# # /---------------------------------------------------------------------------------------------------------/



# import plotly.graph_objects as go
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle as pkl
# from streamlit_lottie import st_lottie
# import json
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder

# st.set_page_config(
#     page_title="Term Deposit Success Predictor",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="📊",
# )

# st.title("Term Deposit Success Predictor")

# # sidebar
# st.sidebar.header("Term Deposit Success Predictor")
# st.sidebar.image("gadgets/data_analysis.jpeg")
# st.sidebar.subheader("Choose Your Favorite Predictor")
# # Load the pre-trained model
# model = pkl.load(open('models/bank_model.pkl.sav', 'rb'))

# # Function to preprocess the input data using LabelEncoder
# def preprocess(client_df):
#     # List of categorical columns
#     categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
#     # Initialize the LabelEncoder
#     le = LabelEncoder()
    
#     # Apply LabelEncoder to each categorical column
#     for col in categorical_columns:
#         client_df[col] = le.fit_transform(client_df[col])
    
#     return client_df

# # Function to ensure feature names match those used during training
# def ensure_feature_order(data):
#     model_features = [
#         'duration', 'previous', 'emp.var.rate', 'euribor3m', 'nr.employed',
#         'contacted_before', 'contact_cellular', 'contact_telephone',
#         'month_dec', 'month_mar', 'month_may', 'month_oct', 'month_sep',
#         'poutcome_nonexistent', 'poutcome_success'
#     ]
#     return data.reindex(columns=model_features, fill_value=0)

# # Streamlit app
# st.title("Bank Marketing Prediction App")

# st.header("Client Information")
# age = st.slider("Age", 18, 100)
# job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", 
#                            "retired", "self-employed", "services", "student", "technician", 
#                            "unemployed", "unknown"])
# marital = st.selectbox("Marital Status", ["divorced", "married", "single", "unknown"])
# education = st.selectbox("Education", ["basic.4y", "basic.6y", "basic.9y", "high.school", 
#                                        "illiterate", "professional.course", "university.degree", 
#                                        "unknown"])
# default = st.selectbox("Default", ["no", "yes", "unknown"])
# balance = st.number_input("Balance", min_value=0)
# housing = st.selectbox("Housing Loan", ["no", "yes", "unknown"])
# loan = st.selectbox("Personal Loan", ["no", "yes", "unknown"])
# contact = st.selectbox("Contact Communication Type", ["cellular", "telephone", "unknown"])
# day = st.slider("Day of Contact", 1, 31)
# month = st.selectbox("Month of Contact", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", 
#                                           "sep", "oct", "nov", "dec"])
# duration = st.number_input("Last Contact Duration", min_value=0)
# campaign = st.number_input("Number of Contacts", min_value=1)
# pdays = st.number_input("Days since Last Contact", min_value=-1)
# previous = st.number_input("Number of Previous Contacts", min_value=0)
# poutcome = st.selectbox("Previous Outcome", ["failure", "nonexistent", "success"])

# # Create a dictionary from the input
# client_data = {
#     'age': age,
#     'job': job,
#     'marital': marital,
#     'education': education,
#     'default': default,
#     'balance': balance,
#     'housing': housing,
#     'loan': loan,
#     'contact': contact,
#     'day': day,
#     'month': month,
#     'duration': duration,
#     'campaign': campaign,
#     'pdays': pdays,
#     'previous': previous,
#     'poutcome': poutcome
# }

# # Convert to DataFrame
# client_df = pd.DataFrame([client_data])

# # Preprocess the data
# processed_data = preprocess(client_df)

# # Ensure feature order matches model expectations
# processed_data = ensure_feature_order(processed_data)

# if st.button('Predict Subscription'):
#     # Make a prediction
#     prediction = model.predict(processed_data)

#     # Display the result
#     if prediction[0] == 1:
#         st.success("The client is likely to subscribe to a term deposit.")
#     else:
#         st.warning("The client is unlikely to subscribe to a term deposit.")


# /---------------------------------------------------------------------------------------------------------/
# /---------------------------------------------------------------------------------------------------------/

import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from streamlit_lottie import st_lottie
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, recall_score, precision_score


# st.set_page_config(
#     page_title="Term Deposit Success Predictor",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="📊",
# )

def predictor_page():

    st.title("Term Deposit Success Predictor")

    # sidebar
    st.sidebar.header("Term Deposit Success Predictor")
    st.sidebar.image("gadgets/data_analysis.jpeg")
    st.sidebar.subheader("Choose Your Favorite Predictor")

    # Filters
    model = st.sidebar.selectbox(
        "predictors", ["Random Forest", "Logistic Regression", "KNN"]
    )

    # Load Cleaned Data
    df = pd.read_csv("datasets/cleaned_data_v2.csv")

    # Define categorical and numerical columns
    categorical_columns = ['JobType', 'MaritalStatus', 'EducationLevel', 'HasHousingLoan', 
                        'HasPersonalLoan', 'ContactCommunicationType', 'LastContactMonth', 
                        'LastContactDayOfWeek', 'PreviousCampaignOutcome']

    numerical_columns = ['Age', 'CallDuration', 'CampaignContacts', 'PreviousCampaignContacts',
                        'EmploymentVariationRate', 'ConsumerPriceIndex', 'ConsumerConfidenceIndex',
                        'Euribor3MRate', 'NumberOfEmployees']

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # Fit the preprocessor
    preprocessor.fit(df)

    # Load models
    rf = pkl.load(open('models/best_rf2_model.pkl', 'rb'))
    logreg = pkl.load(open('models/best_logreg2_model.pkl', 'rb'))
    knn = pkl.load(open('models/best_knn2_model.pkl', 'rb'))

    model = logreg

    # # model selection
    if model == "Random Forest":
        model = rf
    elif model == "Logistic Regression":
        model = logreg
    elif model == "KNN":
        model = knn

    # Input Data
    # Create two columns
    col1, col2 = st.columns(2)

    # First column inputs
    with col1:
        job_type = st.selectbox('Job Type', df['JobType'].unique())
        marital_status = st.selectbox('Marital Status', df['MaritalStatus'].unique())
        education_level = st.selectbox('Education Level', df['EducationLevel'].unique())
        has_housing_loan = st.selectbox('Has Housing Loan', df['HasHousingLoan'].unique())
        has_personal_loan = st.selectbox('Has Personal Loan', df['HasPersonalLoan'].unique())
        contact_type = st.selectbox('Contact Communication Type', df['ContactCommunicationType'].unique())
        last_contact_month = st.selectbox('Last Contact Month', df['LastContactMonth'].unique())
        last_contact_day = st.selectbox('Last Contact Day', df['LastContactDayOfWeek'].unique())
        campaign_outcome = st.selectbox('Previous Campaign Outcome', df['PreviousCampaignOutcome'].unique())

    # Second column inputs
    with col2:
        age = st.number_input('Age', df.Age.min(), df.Age.max())
        call_duration = st.number_input('Call Duration', df.CallDuration.min(), df.CallDuration.max())
        campaign_contacts = st.number_input('Campaign Contacts', df.CampaignContacts.min(), df.CampaignContacts.max())
        previous_contacts = st.number_input('Previous Campaign Contacts', df.PreviousCampaignContacts.min(), df.PreviousCampaignContacts.max())
        emp_var_rate = st.number_input('Employment Variation Rate', df['EmploymentVariationRate'].min(), df['EmploymentVariationRate'].max())
        consumer_price_idx = st.number_input('Consumer Price Index', df['ConsumerPriceIndex'].min(), df['ConsumerPriceIndex'].max())
        consumer_conf_idx = st.number_input('Consumer Confidence Index', df['ConsumerConfidenceIndex'].min(), df['ConsumerConfidenceIndex'].max())
        euribor_rate = st.number_input('Euribor 3M Rate', df['Euribor3MRate'].min(), df['Euribor3MRate'].max())
        num_employees = st.number_input('Number of Employees', df['NumberOfEmployees'].min(), df['NumberOfEmployees'].max())

    # Create input data dictionary
    new_data = {
        'JobType': job_type,
        'MaritalStatus': marital_status,
        'EducationLevel': education_level,
        'HasHousingLoan': has_housing_loan,
        'HasPersonalLoan': has_personal_loan,
        'ContactCommunicationType': contact_type,
        'LastContactMonth': last_contact_month,
        'LastContactDayOfWeek': last_contact_day,
        'PreviousCampaignOutcome': campaign_outcome,
        'Age': age,
        'CallDuration': call_duration,
        'CampaignContacts': campaign_contacts,
        'PreviousCampaignContacts': previous_contacts,
        'EmploymentVariationRate': emp_var_rate,
        'ConsumerPriceIndex': consumer_price_idx,
        'ConsumerConfidenceIndex': consumer_conf_idx,
        'Euribor3MRate': euribor_rate,
        'NumberOfEmployees': num_employees
    }

    # Convert to DataFrame and preprocess
    new_data = pd.DataFrame(new_data, index=[0])
    new_data_processed = preprocessor.transform(new_data)

    # Prediction
    X_train = df.drop(columns=['SubscribedToTermDeposit'])  # Assuming 'SubscribedToTermDeposit' is your target column
    y_train = df['SubscribedToTermDeposit']
    X_train_processed = preprocessor.transform(X_train)
    model.fit(X_train_processed, y_train)
    prediction = model.predict(new_data_processed)

    if prediction == 0:
        prediction = 'NO'
    else:
        prediction = 'YES'

    # Output
    if st.button('Predict'):
        st.markdown('# Deposit Prediction')
        st.markdown(prediction)
