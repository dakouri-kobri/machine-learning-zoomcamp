import pickle

# Load saved model from a file
with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# Prediction on a single case
cust = {'gender': 'male',
        'seniorcitizen': 0,
         'partner': 'no',
         'dependents': 'yes',
         'phoneservice': 'no',
         'multiplelines': 'no_phone_service',
         'internetservice': 'dsl',
         'onlinesecurity': 'no',
         'onlinebackup': 'yes',
         'deviceprotection': 'no',
         'techsupport': 'no',
         'streamingtv': 'no',
         'streamingmovies': 'no',
         'contract': 'month-to-month',
         'paperlessbilling': 'yes',
         'paymentmethod': 'electronic_check',
         'tenure': 6,
         'monthlycharges': 29.85,
         'totalcharges': 129.85}

# Predict probability of churning for this customer
churn = pipeline.predict_proba(cust)[0, 1]
print(f"Churning probability: {churn:.2f}")

if churn >= 0.5:
    print("Send a promo email")
else:
    print("Do nothing")