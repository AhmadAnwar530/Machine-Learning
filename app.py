from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from prettytable import PrettyTable
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained machine learning models
with open('./model/logistic_trained_model.pkl', 'rb') as f:
    logistic_trained_model = pickle.load(f)
with open('./model/logistic_k_model.pkl', 'rb') as f:
    logistic_k_model = pickle.load(f)
with open('./model/rfc_trained_model.pkl', 'rb') as f:
    rfc_trained_model = pickle.load(f)
with open('./model/rf_k_model.pkl', 'rb') as f:
    rf_k_model = pickle.load(f)

with open('./model/naive_bayes_trained_model.pkl', 'rb') as f:
    naive_bayes_trained_model = pickle.load(f)
with open('./model/nb_k_model.pkl', 'rb') as f:
    nb_k_model = pickle.load(f)

# Loading the Decision Tree model
with open('./model/dtc_trained_model.pkl', 'rb') as f:
    dtc_trained_model = pickle.load(f)
with open('./model/dt_k_model.pkl', 'rb') as f:
    dt_k_model = pickle.load(f)

with open('./model/svc_trained_model.pkl', 'rb') as f:
    svc_trained_model = pickle.load(f)
with open('./model/svm_k_model.pkl', 'rb') as f:
    svm_k_model = pickle.load(f)

# Loading the XGradient Boosting Classifier model
with open('./model/xgb_trained_model.pkl', 'rb') as f:
    xgb_trained_model = pickle.load(f)
with open('./model/xgb_k_model.pkl', 'rb') as f:
    xgb_k_model = pickle.load(f)

# Loading the KNN model
with open('./model/knn_trained_model.pkl', 'rb') as f:
    knn_trained_model = pickle.load(f)
with open('./model/knn_k_model.pkl', 'rb') as f:
    knn_k_model = pickle.load(f)

with open('./model/ada_trained_model.pkl', 'rb') as f:
    ada_trained_model = pickle.load(f)
with open('./model/ada_boost_k_model.pkl', 'rb') as f:
    ada_boost_k_model = pickle.load(f)

with open('./model/qda_trained_model.pkl', 'rb') as f:
    qda_trained_model = pickle.load(f)
with open('./model/qda_k_model.pkl', 'rb') as f:
    qda_k_model = pickle.load(f)

with open('./model/lda_trained_model.pkl', 'rb') as f:
    lda_trained_model = pickle.load(f)
with open('./model/lda_k_model.pkl', 'rb') as f:
    lda_k_model = pickle.load(f)


# Create LabelEncoder object for categorical feature encoding
label_encoder = LabelEncoder()


@app.route('/')
def home():
    return render_template('prediction.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Take input from the HTML form
    gender_input = request.form['gender']
    married_input = request.form['married']
    dependents_input = request.form['dependents']
    education_input = request.form['education']
    self_employed_input = request.form['self_employed']
    income_input = request.form['income']
    coapplicant_input = request.form['coapplicant']
    loan_amount_input = request.form['loan_amount']
    loan_amount_term_input = request.form['loan_amount_term']
    credit_input = request.form['credit']
    property_area_input = request.form['property_area']

    # Convert user input into feature vector
    user_input = pd.DataFrame({
        'Gender': [gender_input],
        'Married': [married_input],
        'Dependents': [dependents_input],
        'Education': [education_input],
        'Self_Employed': [self_employed_input],
        'ApplicantIncome': [income_input],
        'CoapplicantIncome': [coapplicant_input],
        'LoanAmount': [loan_amount_input],
        'Loan_Amount_Term': [loan_amount_term_input],
        'Credit_History': [credit_input],
        'Property_Area': [property_area_input]
    })

    # Encode categorical features
    user_input['Gender'] = label_encoder.fit_transform(user_input['Gender'])
    user_input['Married'] = label_encoder.fit_transform(user_input['Married'])
    user_input['Dependents'] = label_encoder.fit_transform(user_input['Dependents'])
    user_input['Education'] = label_encoder.fit_transform(user_input['Education'])
    user_input['Self_Employed'] = label_encoder.fit_transform(user_input['Self_Employed'])
    user_input['Property_Area'] = label_encoder.fit_transform(user_input['Property_Area'])

    predicted_loan_status = None
    # Make a prediction using the selected model
    
    split = request.form.get('split')
    model1 = request.form.get('model')
    

    if split == 'Train_Test' and model1 == 'Logistic_Regression':
        predicted_loan_status = logistic_trained_model.predict(user_input)
    elif split == 'KFold' and model1 == 'Logistic_Regression':
        predicted_loan_status = logistic_k_model.predict(user_input)
    elif split == 'Train_Test' and model1 == 'Random_Forest':
            predicted_loan_status = rfc_trained_model.predict(user_input)
    elif split == 'KFold' and model1 == 'Random_Forest':
            predicted_loan_status = rf_k_model.predict(user_input)
    elif split == 'Train_Test' and model1 == 'Naive_Bayes':
            predicted_loan_status = naive_bayes_trained_model.predict(user_input)
    elif split == 'KFold' and model1 == 'Naive_Bayes':
            predicted_loan_status = nb_k_model.predict(user_input)
    elif split == 'Train_Test' and model1 == 'Decision_Tree':
            predicted_loan_status = dtc_trained_model.predict(user_input)
    elif split == 'KFold' and model1 == 'Decision_Tree':
            predicted_loan_status = dt_k_model.predict(user_input)
    elif split == 'Train_Test' and model1 == 'Support_Vector':
            predicted_loan_status = svc_trained_model.predict(user_input)
    elif split == 'KFold' and model1 == 'Support_Vector':
            predicted_loan_status = svm_k_model.predict(user_input)
    elif split == 'Train_Test' and model1 == 'XGradientBoost':
            predicted_loan_status = xgb_trained_model.predict(user_input)
    elif split == 'KFold' and model1 == 'XGradientBoost':
            predicted_loan_status = xgb_k_model.predict(user_input)
    elif split == 'Train_Test' and model1 == 'KNN':
            predicted_loan_status = knn_trained_model.predict(user_input)
    elif split == 'KFold' and model1 == 'KNN':
            predicted_loan_status = knn_k_model.predict(user_input)
    elif split == 'Train_Test' and model1 == 'Ada_Boost':
            predicted_loan_status = ada_trained_model.predict(user_input)
    elif split == 'KFold' and model1 == 'Ada_Boost':
            predicted_loan_status = ada_boost_k_model.predict(user_input)
    elif split == 'Train_Test' and model1 == 'QDA':
            predicted_loan_status = qda_trained_model.predict(user_input)
    elif split == 'KFold' and model1 == 'QDA':
            predicted_loan_status = qda_k_model.predict(user_input)
    elif split == 'Train_Test' and model1 == 'Linear_Discriminant_Analysis':
            predicted_loan_status = lda_trained_model.predict(user_input)
    elif split == 'KFold' and model1 == 'Linear_Discriminant_Analysis':
            predicted_loan_status = lda_k_model.predict(user_input)
                
    if predicted_loan_status is not None:
        if predicted_loan_status == 1:
            prediction = "Approved"
        else:
            prediction = "NOT Approved"
    else:
        prediction = "No prediction available"
    return render_template('prediction.html', prediction=prediction, split=split, model=model1)


if __name__ == '__main__':
    app.run(debug=True)
