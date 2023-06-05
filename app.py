from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the trained machine learning model
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

# Load the vectorizer
with open('./vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


@app.route('/')
def home():
    return render_template('prediction.html')


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        # Get the form data
        Loan_Status = request.form.get('Loan_Status')
        split = request.form.get('split')
        model1 = request.form.get('model')
        if Loan_Status is not None:
            # Vectorize the text using the loaded vectorizer
            Loan_Status_vectorized = vectorizer.transform([Loan_Status.lower()])
            predicted_label = None

            # Make a prediction using the loaded model
            if split == 'Train_Test' and model1 == 'Logistic_Regression':
                predicted_label = logistic_trained_model.predict(Loan_Status_vectorized)
            elif split == 'KFold' and model1 == 'Logistic_Regression':
                predicted_label = logistic_k_model.predict(Loan_Status_vectorized)
            elif split == 'Train_Test' and model1 == 'Random_Forest':
                predicted_label = rfc_trained_model.predict(Loan_Status_vectorized)
            elif split == 'KFold' and model1 == 'Random_Forest':
                predicted_label = rf_k_model.predict(Loan_Status_vectorized)
            elif split == 'Train_Test' and model1 == 'Naive_Bayes':
                predicted_label = naive_bayes_trained_model.predict(Loan_Status_vectorized)
            elif split == 'KFold' and model1 == 'Naive_Bayes':
                predicted_label = nb_k_model.predict(Loan_Status_vectorized)
            elif split == 'Train_Test' and model1 == 'Decision_Tree':
                predicted_label = dtc_trained_model.predict(Loan_Status_vectorized)
            elif split == 'KFold' and model1 == 'Decision_Tree':
                predicted_label = dt_k_model.predict(Loan_Status_vectorized)
            elif split == 'Train_Test' and model1 == 'Support_Vector':
                predicted_label = svc_trained_model.predict(Loan_Status_vectorized)
            elif split == 'KFold' and model1 == 'Support_Vector':
                predicted_label = svm_k_model.predict(Loan_Status_vectorized)
            elif split == 'Train_Test' and model1 == 'XGradientBoost':
                predicted_label = xgb_trained_model.predict(Loan_Status_vectorized)
            elif split == 'KFold' and model1 == 'XGradientBoost':
                predicted_label = xgb_k_model.predict(Loan_Status_vectorized)
            elif split == 'Train_Test' and model1 == 'KNN':
                predicted_label = knn_trained_model.predict(Loan_Status_vectorized)
            elif split == 'KFold' and model1 == 'KNN':
                predicted_label = knn_k_model.predict(Loan_Status_vectorized)
            elif split == 'Train_Test' and model1 == 'Ada_Boost':
                predicted_label = ada_trained_model.predict(Loan_Status_vectorized)
            elif split == 'KFold' and model1 == 'Ada_Boost':
                predicted_label = ada_boost_k_model.predict(Loan_Status_vectorized)
            elif split == 'Train_Test' and model1 == 'QDA':
                predicted_label = qda_trained_model.predict(Loan_Status_vectorized)
            elif split == 'KFold' and model1 == 'QDA':
                predicted_label = qda_k_model.predict(Loan_Status_vectorized)
            elif split == 'Train_Test' and model1 == 'Linear_Discriminant_Analysis':
                predicted_label = lda_trained_model.predict(Loan_Status_vectorized)
            elif split == 'KFold' and model1 == 'Linear_Discriminant_Analysis':
                predicted_label = lda_k_model.predict(Loan_Status_vectorized)
            else:
                predicted_label = ['Invalid Input']

            print(predicted_label[0])
            return render_template('prediction.html', Loan_Status=Loan_Status, prediction=predicted_label[0], split=split, model=model1)
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
