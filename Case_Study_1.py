#!/usr/bin/env python
# coding: utf-8

# In[640]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV,KFold, cross_val_score, train_test_split
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings('ignore')


# # Data Understanding

# In[506]:


my_data = pd.read_csv('loans_full_schema.csv')
my_data.info()


# In[507]:


pd.set_option("display.max_columns", None)


# In[508]:


my_data.describe()


# ### Drop columns because of data leakage

# In[509]:


my_data = my_data.drop(['issue_month','loan_status','initial_listing_status','disbursement_method','installment','balance','paid_total','paid_principal','paid_interest','paid_late_fees'],1)


# ### Replace categorical value with numerical value

# In[510]:


my_data.groupby(['homeownership']).size()
my_data['homeownership']=my_data['homeownership'].replace(['MORTGAGE','OWN','RENT'],[1,2,0])
my_data.groupby(['verified_income']).size()
my_data.groupby(['verification_income_joint']).size()
my_data['verified_income']=my_data['verified_income'].replace(['Not Verified','Source Verified','Verified'],[0,2,1])
my_data['verification_income_joint']=my_data['verification_income_joint'].replace(['Not Verified','Source Verified','Verified'],[0,2,1])
my_data.groupby(['sub_grade']).size()
l = list(range(1,36))
p =sorted(l,reverse = True)
q = list(range(1,14))
my_data['sub_grade']= my_data['sub_grade'].replace(['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5'],p)
my_data.groupby(['loan_purpose']).size()
my_data['loan_purpose']= my_data['loan_purpose'].replace(['car','credit_card','debt_consolidation','home_improvement','house','major_purchase','medical','moving','other','renewalbe_energy','small_business','vacation','renewable_energy'],q)


# In[ ]:





# ### Drop features dulicated or too hard to analyze 

# In[511]:


my_data = my_data.drop(['emp_title','state','grade'],1)


# ### Review features with large amount of duplicate values

# In[512]:


my_data['num_accounts_30d_past_due'].describe()


# In[513]:


my_data['num_accounts_120d_past_due'].describe()


# In[514]:


my_data['current_accounts_delinq'].describe()


# In[515]:


my_data = my_data.drop(['num_accounts_120d_past_due','num_accounts_30d_past_due','current_accounts_delinq'],1)


# ### Combine Joint Application and Individual Application

# In[516]:


index = my_data.index
condition = my_data['application_type']=='joint'
indice = index[condition]
joint_row = indice.tolist()
my_data.loc[joint_row,['annual_income']] = my_data[my_data['application_type']=='joint']['annual_income_joint']
my_data.loc[joint_row,['debt_to_income']] = my_data[my_data['application_type']=='joint']['debt_to_income_joint']
indx = my_data.columns.get_loc('verified_income')
for i in joint_row:
    single = my_data['verification_income_joint'][i]
    joint = my_data['verified_income'][i]
    if joint < single:
        my_data.iloc[i,indx] = my_data['verification_income_joint'][i]

my_data = my_data.drop(['annual_income_joint','verification_income_joint','debt_to_income_joint','application_type'],1)     


# ### Modify "year" with meaningful "time length"

# In[517]:


my_data['earliest_credit_line'] = my_data['earliest_credit_line'].apply(lambda x: 2021 -x )


# In[518]:


my_data = pd.DataFrame(my_data)


# ## Handle Missing Values

# In[519]:


## Missing Values
null_values = my_data.isnull().sum().to_frame().reset_index()
null_values.columns = ['Variables', 'Null_counts']
null_values['Null_pct'] = null_values['Null_counts'] / my_data.shape[0]
null_values.sort_values(by='Null_pct', ascending=False)


# ### Replace NA employment length with 0 

# In[520]:


my_data.loc[my_data['emp_length'].isnull(), 'emp_length'] = 0


# ### Look Closer into "months since", drop highly correlated, fill NA 

# In[521]:


my_data[['months_since_90d_late','months_since_last_delinq','months_since_last_credit_inquiry']].corr()


# In[522]:


my_data = my_data.drop('months_since_90d_late',1)


# In[523]:


my_data.loc[my_data['months_since_last_delinq'].isnull(), 'months_since_last_delinq'] = max(my_data['months_since_last_delinq'])
my_data.loc[my_data['months_since_last_credit_inquiry'].isnull(), 'months_since_last_credit_inquiry'] = max(my_data['months_since_last_credit_inquiry'])


# ## Review cleaned data, double check for highly correlated columns 

# In[524]:


cormatrix = my_data.corr()
plt.figure(figsize=(25,25))
sns.heatmap(cormatrix, annot=True, cmap='RdGy')


# In[525]:


my_data = my_data.drop(['sub_grade','open_credit_lines'],1)


# ### Data Scaling 

# In[526]:


y = my_data['interest_rate']
X = my_data.drop(['interest_rate'],1)


# In[527]:


new_data = X
cat_feature = new_data[['homeownership','loan_purpose', 'verified_income']]
num_feature = new_data.drop(['homeownership','loan_purpose', 'verified_income'],1)


# In[528]:


scaler = MinMaxScaler()
scaler.fit(num_feature)
num_feature = pd.DataFrame(scaler.transform(num_feature),columns=[num_feature.columns])


# In[529]:


my_processed_X = num_feature.join(cat_feature)
y = pd.DataFrame(y, columns = ['interest_rate'])


# ## Data Modeling 

# ### Lasso

# In[530]:


inner_cv = KFold(n_splits = 4, shuffle = True)
outer_cv = KFold(n_splits = 5, shuffle = True)

X_train, X_test, y_train, y_test = train_test_split(my_processed_X, y, test_size=0.3, random_state=11)


# In[533]:


lasso = Lasso(random_state=42) 
            
lasso_params = {'alpha':[0.1, 0.2, 0.3, 0.4, 0.5],
                'normalize': [True, False],
                'tol':[0.0001, 0.001, 0.01, 1]}

gs_lasso = GridSearchCV(lasso, param_grid = lasso_params, scoring = 'neg_root_mean_squared_error', cv=inner_cv) # cross-validation scores
gs_lasso = gs_lasso.fit(X_train, y_train)
print("Non-nested CV RMSE score: ", gs_lasso.best_score_)
print("Optimal Parameter: ", gs_lasso.best_params_)    
print("Optimal Estimator: ", gs_lasso.best_estimator_) 


# #### Generalization Performance Using Cross-validation

# In[536]:


lasso_best = Lasso(random_state=42, alpha=0.2, normalize=False, tol=0.001) 
best_lasso_scores = cross_val_score(lasso_best, X_train, y_train, scoring = 'neg_root_mean_squared_error', cv=outer_cv) # cross-validation scores
print("Lasso Best Model Generalized Performance: %0.2f (+/- %0.2f)" % (-best_lasso_scores.mean(), best_lasso_scores.std() * 2))


# ### Decision Tree Regressor 

# In[544]:


tree_params = {'max_depth':range(1,10), 
               'min_impurity_decrease': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,0.01]}

gs_dt = GridSearchCV(DecisionTreeRegressor(random_state = 42), tree_params, cv=inner_cv, scoring = 'neg_root_mean_squared_error',n_jobs = 5)
gs_dt = gs_dt.fit(X_train, y_train)
print("Non-nested CV RMSE score: ", gs_dt.best_score_)
print("Optimal Parameter: ", gs_dt.best_params_)    
print("Optimal Estimator: ", gs_dt.best_estimator_) 


# In[547]:


tree_best = DecisionTreeRegressor(random_state = 42, max_depth=5, min_impurity_decrease=0.001, criterion='mse') 
best_tree_scores = cross_val_score(tree_best, X_train, y_train, scoring = 'neg_root_mean_squared_error', cv=outer_cv) # cross-validation scores
print("Decision Tree Best Model Generalized Performance: %0.2f (+/- %0.2f)" % (-best_tree_scores.mean(), best_tree_scores.std() * 2))


# ### ElasticNet

# ### Test 

# In[574]:


lasso_yhat = lasso_best.fit(X_train, y_train).predict(X_test)
dt_yhat = tree_best.fit(X_train, y_train).predict(X_test)


# In[583]:


lasso_best.coef_


# In[652]:


coef = [2,4,15,27,28,29,30]
sig_feature = list()
for i in coef:
    sig_feature.append(X_train.columns[i])
sig_feature
    


# ### Result Visualization 

# In[592]:


x = list(range(1,len(lasso_yhat)+1))


# #### Lasso 

# In[627]:


plt.scatter(x, lasso_yhat,color = 'red',alpha = 0.5)
plt.title("Lasso predicted interest rate")
plt.ylim(3, 33)


# In[626]:


plt.scatter(x,y_test,color = 'green', alpha = 0.5)
plt.title("Real interest rate")
plt.ylim(3, 33)


# In[632]:


lass_rs = np.array(y_test).T-np.array(lasso_yhat)
plt.scatter(x, lass_rs, color = 'blue', alpha = 0.5)
plt.title("Lasso Residual")
plt.ylim(-10, 20)


# #### Decision Tree 

# In[595]:


dt_yhat = tree_best.fit(X_train, y_train).predict(X_test)
x = list(range(1,len(dt_yhat)+1))


# In[631]:


plt.scatter(x, dt_yhat,color = 'red',alpha = 0.5)
plt.title("Decision tree predicted interest rate")
plt.ylim(3, 33)


# In[630]:


plt.scatter(x,y_test,color = 'green', alpha = 0.5)
plt.title("Real interest rate")
plt.ylim(3, 33)


# In[633]:


dt_rs = np.array(y_test).T-np.array(dt_yhat)
plt.scatter(x, dt_rs, color = 'blue', alpha = 0.5)
plt.title("Decision Tree Residual")
plt.ylim(-10, 20)


# #### ElasticNet 

# In[ ]:


## More model should be tested on if more time allowed.  Such as ElastiNet mentioned about, and also randomforest for decision tree.
## Also, more detailed dataset cleaning should be done. I would look into more about company's background information while doing feature selecture.
## For example, I simply put max value for three '#month since' features which should acturally mean that client never pay late or get inquired. Better NA filled value should be considered.
## Regarding the geographical information "state", I simply drop it here. However, I do believe that the location matters. Train different models for different state respectvely may be applicable.

