import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import sklearn
import joblib
import imblearn
model = joblib.load("Model.pkl")
inputs = joblib.load("inputs.pkl")

def prediction(Gender, Married, Dependents, Education, Self_Employed,ApplicantIncome, CoapplicantIncome, LoanAmount,Loan_Amount_Term, Credit_History, Property_Area):
    df = pd.DataFrame(columns=inputs)
    log_df = pd.DataFrame(columns = ["Log_Total_Income","Log_Loan_Per_Month"])
    df.at[0,'Gender'] =Gender
    df.at[0,'Married'] =Married
    df.at[0,'Dependents'] =Dependents
    df.at[0,'Education'] =Education
    df.at[0,'Self_Employed'] =Self_Employed
    df.at[0,'ApplicantIncome'] =ApplicantIncome
    df.at[0,'CoapplicantIncome'] =CoapplicantIncome
    df.at[0,'LoanAmount'] =LoanAmount
    df.at[0,'Loan_Amount_Term'] =Loan_Amount_Term
    df.at[0,'Credit_History'] =Credit_History
    df.at[0,'Property_Area'] =Property_Area

    log_df["Log_Total_Income"] = np.log(df.iloc[0]["ApplicantIncome"] + df.iloc[0]["CoapplicantIncome"])
    log_df["Log_Loan_Per_Month"] = np.log((df.iloc[0]["LoanAmount"] * 1000 ) / df.iloc[0]["Loan_Amount_Term"])
    
    df.drop(['ApplicantIncome' , 'CoapplicantIncome'  , 'Loan_Amount_Term' ] , axis = 1 , inplace = True)
    df = pd.concat([df, log_df], axis=1)
    res = model.predict(df)
    return res[0]

def main():
    st.title("Loan Approval Prediction")
    Gender = st.selectbox('Gender :', ['Male', 'Female'])
    Married = st.selectbox('marital status :', ['Yes', 'No'])
    Dependents = st.selectbox(' Number of Dependents :', ["0","1","2","3"])
    Education = st.selectbox('Education :', ['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox('Self Employed :', ['No', 'Yes'])
    ApplicantIncome = st.slider("Applicant Income :", min_value=150, max_value=82000, step = 10, value = 3800 )
    CoapplicantIncome = st.slider("Co-applicant Income :", min_value=0, max_value=45000, step = 10, value = 3800 )
    LoanAmount = st.slider("Loan Amount in thousands :", min_value=9, max_value=800, step = 1, value = 200 )
    Loan_Amount_Term = st.slider("Term of Loan in months :", min_value=10, max_value=500, step = 10, value = 200 )
    Credit_History = st.selectbox('Credit History Meets Guidelines :', ["0","1"])
    Property_Area = st.selectbox('Property Area :',  ['Urban', 'Rural', 'Semiurban'])
    
    if st.button("Predict"):
        result = prediction(Gender, Married, Dependents, Education, Self_Employed,ApplicantIncome, CoapplicantIncome, LoanAmount,Loan_Amount_Term, Credit_History, Property_Area)

        res_list = ['not be Approved',' be Approved']
        st.text(f"This loan will {res_list[result]}")

main()



