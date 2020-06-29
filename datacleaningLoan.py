# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:49:38 2020

@author: patel
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import re

# Load in the data
orig_data = pd.read_csv('loan.csv', header = 0,  encoding='utf8')

# Remove unwanted rows and create new data frame
loan_status = ['Fully Paid', 'Charged Off', 'Default']
loan_data = orig_data[orig_data['loan_status'].isin(loan_status)]
loan_data = loan_data[loan_data['application_type']=='INDIVIDUAL']

# Define and subset by the columns of interest for the model
col_keep = ['loan_amnt','term', 'int_rate', 'installment', 'grade', 'sub_grade',
            'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
            'loan_status', 'purpose','addr_state', 'dti', 'delinq_2yrs','earliest_cr_line', 
            'inq_last_6mths', 'mths_since_last_delinq','mths_since_last_record', 'open_acc', 
            'pub_rec', 'revol_bal','revol_util', 'total_acc', 'collections_12_mths_ex_med',
            'mths_since_last_major_derog', 'acc_now_delinq', 'tot_coll_amt', 
            'tot_cur_bal', 'open_acc_6m','open_il_6m', 'open_il_12m', 'open_il_24m', 
            'mths_since_rcnt_il','total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 
            'max_bal_bc','all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m']

loan_data = loan_data[col_keep]
# Count the number of values in each column
loan_data.count()
# Remove columns that have significant NULL values
loan_data = loan_data.dropna(axis = 1, thresh = loan_data.shape[0]*0.1)
# Remove rows with NULL in emp_length
loan_data = loan_data[loan_data['emp_length'].notna()]
# Fill all NA columns with 0
loan_data = loan_data.fillna(0)
# Only keep year in earliest_cr_line
loan_data['earliest_cr_line'] =  [int(re.sub(r'[^0-9]','', str(x))) for x in loan_data['earliest_cr_line']]
# Count the number of values in each column - should have no NULL values
loan_data.count()
# Plot a histogram for each numeric column in the data frame
for i in loan_data.select_dtypes(include = ['float']).columns:
    loan_data.hist(column= i) 

# Plot bar chart for categorical columns
for i in loan_data.select_dtypes(include = ['object']).columns:
    figure(figsize=(25,15))
    sns.catplot(x=i, kind="count", data=loan_data)

# Remove instances based on condition in home_ownership column
loan_data.drop(loan_data[(loan_data['home_ownership'] == 'OTHER') | (loan_data['home_ownership'] == 'NONE') | (loan_data['home_ownership'] == 'ANY')].index, inplace=True)

# Drop state column - not enough instances of each state
loan_data = loan_data.drop(['addr_state'], axis=1)

# Change 'Charged Off' to Default - binary classification
loan_data.loc[(loan_data['loan_status'] == 'Charged Off'),'loan_status']='Default'

# Export and save cleaned data frame to CSV file
loan_data.to_csv('loan_data_cleaned.csv', index = False)
