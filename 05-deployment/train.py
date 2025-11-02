#!/usr/bin/env python
# coding: utf-8

# This is a starter notebook for an updated module 5 of ML Zoomcamp
# 
# The code is based on the modules 3 and 4. We use the same dataset: [telco customer churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

# In[2]:


import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn.pipeline import make_pipeline


# In[3]:


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


# In[4]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# In[5]:


data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

df = pd.read_csv(data_url)

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[6]:


y_train = df.churn


# In[7]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# In[8]:


pipeline = make_pipeline(
    DictVectorizer(),
    LogisticRegression(solver='liblinear')
)    


# In[9]:


train_dict = df[categorical + numerical].to_dict(orient='records')
pipeline.fit(train_dict, y_train)


# In[10]:


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


# In[11]:


# Probability of churning for this customer
churn = pipeline.predict_proba(cust)[0,1]
churn


# In[12]:


# Save model to a file
with open('model.bin', 'wb') as f_out:
    pickle.dump(pipeline, f_out)


# In[14]:


# Load saved model from a file
with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


# In[15]:


pipeline


# In[16]:


# Using from download
cust_dict = {'gender': 'male',
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
         'paperlessbilling': 'no',
         'paymentmethod': 'electronic_check',
         'tenure': 8,
         'monthlycharges': 29.85,
         'totalcharges': 149.85}

# Prediction
churn = pipeline.predict_proba(cust_dict)[0, 1]
print(f"Churning probability: {churn:.2f}")

if churn >= 0.5:
    print("Send a promo email")
else:
    print("Do nothing")

