#!/usr/bin/env python
# coding: utf-8

# In[2]:


## import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[11]:


import pandas as pd

file_path = 'C:\\Users\\GNQTB\\DocumentsP\\Python\\loan.csv'
loan_data = pd.read_csv(file_path)


# In[12]:


loan_data.head()


# In[13]:


loan_data.shape


# In[14]:


loan_data.isnull().sum()


# ### Many columns contain exclusively null values. Let's eliminate these columns to streamline the dataset.

# In[16]:


loan_data.dropna(axis = 1, how = 'all', inplace = True)
loan_data.head()


#    ##There are several columns which are single valued.
#    ## They cannot contribute to our analysis in any way. So removing them.

# In[20]:


loan_data.drop(['pymnt_plan', "initial_list_status",'collections_12_mths_ex_med','policy_code','acc_now_delinq', 'application_type', 'pub_rec_bankruptcies', 'tax_liens', 'delinq_amnt'], axis = 1, inplace = True)
loan_data.head()


# ####  To refine the dataset by eliminating irrelevant and redundant features for effective predictive modeling of loan default.
# 
# Feature Identification and Removal
# Post-approval features: Columns reflecting loan performance post-disbursement (e.g., payment history, recovery status) will be excluded as they are not available at the time of credit decisioning.
# Identifier and descriptive features: Columns serving as unique identifiers (e.g., 'id', 'member_id'), textual descriptions (e.g., 'title', 'desc', 'emp_title'), and geographic indicators (e.g., 'zip_code', 'addr_state') will be removed due to their lack of predictive power in the context of loan default.
# Redundant features: The column 'funded_amnt' will be dropped as it is superseded by 'funded_amnt_inv' which provides equivalent information.
# Identified Post-Approval Features
# The following columns are identified as post-approval metrics:
# 
# delinq_2yrs: Number of delinquencies in the past 2 years
# revol_bal: Revolving balance
# out_prncp: Outstanding principal
# total_pymnt: Total payment received
# total_rec_prncp: Total received principal
# total_rec_int: Total received interest
# total_rec_late_fee: Total received late fees
# recoveries: Recoveries
# collection_recovery_fee: Collection recovery fee
# last_pymnt_d: Last payment date
# last_pymnt_amnt: Last payment amount
# next_pymnt_d: Next payment date
# chargeoff_within_12_mths: Charged off within 12 months
# mths_since_last_delinq: Months since last delinquency
# mths_since_last_record: Months since last record
# 

# In[22]:


loan_data.drop(["id", "member_id", "url", "title", "emp_title", "zip_code", "last_credit_pull_d", "addr_state","desc","out_prncp_inv","total_pymnt_inv","funded_amnt", "delinq_2yrs", "revol_bal", "out_prncp", "total_pymnt", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d" , "chargeoff_within_12_mths", "mths_since_last_delinq", "mths_since_last_record"], axis = 1, inplace = True)


# loan_data.shape

# In[24]:


loan_data.columns


# ##To accurately predict loan default, the analysis requires a dataset of completed loan cycles. 
# ##Therefore, records with a 'Current' loan status will be excluded as they represent ongoing loans without definitive outcomes.
# ## This ensures that the modeling process is based solely on instances where the borrower's repayment behavior is fully observed.

# In[29]:


loan_data = loan_data[loan_data.loan_status != "Current"]
loan_data.loan_status.unique()


# In[ ]:


Checking for missing values


# In[94]:


(loan_data.isna().sum()/len(loan_data.index))*100


# In[31]:


loan_data.info()


# In[97]:


print("Mode : " + loan_data.emp_length.mode()[0])
loan_data.emp_length.value_counts()


# In[33]:


loan_data.emp_length.fillna(loan_data.emp_length.mode()[0], inplace = True)
loan_data.emp_length.isna().sum()


# In[34]:


loan_data.dropna(axis = 0, subset = ['revol_util'] , inplace = True)
loan_data.revol_util.isna().sum()

Standardizing the data
"revol_util" column although described as an object column, it has continous values.
So we need to standardize the data in this column
"int_rate" is one such column.
"emp_length" --> { (< 1 year) is assumed as 0 and 10+ years is assumed as 10 }
Although the datatype of "term" is arguable to be an integer, there are only two values in the whole column and it might as well be declared a categorical variable.
# In[35]:


loan_data.revol_util = pd.to_numeric(loan_data.revol_util.apply(lambda x : x.split('%')[0]))


# In[36]:


loan_data.int_rate = pd.to_numeric(loan_data.int_rate.apply(lambda x : x.split('%')[0]))


# In[37]:


loan_data.emp_length = pd.to_numeric(loan_data.emp_length.apply(lambda x: 0 if "<" in x else (x.split('+')[0] if "+" in x else x.split()[0])))


# In[38]:


loan_data.head()


# In[96]:


file_path = 'C:\\Users\\GNQTB\\DocumentsP\\Python\\loan.csv'
loan_data = pd.read_csv(file_path)

sns.boxplot(loan_data['annual_inc'])


# In[43]:


quantile_info = loan_data.annual_inc.quantile([0.5, 0.75,0.90, 0.95, 0.97,0.98, 0.99])
quantile_info


# In[44]:


per_95_annual_inc = loan_data['annual_inc'].quantile(0.95)
loan_data = loan_data[loan_data.annual_inc <= per_95_annual_inc]


# In[45]:


sns.boxplot(loan_data.annual_inc)


# In[46]:


sns.boxplot(loan_data.dti)


# In[47]:


sns.boxplot(loan_data.loan_amnt)


# In[48]:


loan_data.loan_amnt.quantile([0.75,0.90,0.95,0.97,0.975, 0.98, 0.99, 1.0])


# In[49]:


sns.boxplot(loan_data.funded_amnt_inv)


# In[50]:


loan_data.funded_amnt_inv.quantile([0.5,0.75,0.90,0.95,0.97,0.975, 0.98,0.985, 0.99, 1.0])


# In[51]:


sns.countplot(x = 'loan_status', data = loan_data)


# In[52]:


loan_data.sub_grade = pd.to_numeric(loan_data.sub_grade.apply(lambda x : x[-1]))
loan_data.sub_grade.head()


# In[100]:


fig, ax = plt.subplots(figsize=(12,7))
sns.set_palette('colorblind')
sns.countplot(x = 'grade', order = ['A', 'B', 'C', 'D', 'E', 'F', 'G'] , hue = 'sub_grade',data = loan_data[loan_data.loan_status == 'Charged Off'])


# In[54]:


sns.countplot(x = 'grade', data = loan_data[loan_data.loan_status == 'Charged Off'], order = ['A', 'B', 'C', 'D', 'E', 'F', 'G'])


# In[55]:


#checking unique values for home_ownership
loan_data['home_ownership'].unique()


# In[ ]:


There are only 3 records with 'NONE' value in the data. So replacing the value with 'OTHER'


# In[56]:


#replacing 'NONE' with 'OTHERS'
loan_data['home_ownership'].replace(to_replace = ['NONE'],value='OTHER',inplace = True)


# In[57]:


#checking unique values for home_ownership again
loan_data['home_ownership'].unique()


# In[58]:


fig, ax = plt.subplots(figsize = (6,4))
ax.set(yscale = 'log')
sns.countplot(x='home_ownership', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[59]:


fig, ax = plt.subplots(figsize = (12,8))
ax.set(xscale = 'log')
sns.countplot(y ='purpose', data=loan_data[loan_data.loan_status == 'Charged Off'])


# In[60]:


#creating bins for int_rate,open_acc,revol_util,total_acc
loan_data['int_rate_groups'] = pd.cut(loan_data['int_rate'], bins=5,precision =0,labels=['5%-9%','9%-13%','13%-17%','17%-21%','21%-24%'])
loan_data['open_acc_groups'] = pd.cut(loan_data['open_acc'],bins = 5,precision =0,labels=['2-10','10-19','19-27','27-36','36-44'])
loan_data['revol_util_groups'] = pd.cut(loan_data['revol_util'], bins=5,precision =0,labels=['0-20','20-40','40-60','60-80','80-100'])
loan_data['total_acc_groups'] = pd.cut(loan_data['total_acc'], bins=5,precision =0,labels=['2-20','20-37','37-55','55-74','74-90'])
loan_data['annual_inc_groups'] = pd.cut(loan_data['annual_inc'], bins=5,precision =0,labels =['3k-31k','31k-58k','58k-85k','85k-112k','112k-140k'])


# In[61]:


# Viewing new bins created
loan_data.head()


# In[62]:


fig, ax = plt.subplots(figsize = (15,10))
plt.subplot(221)
sns.countplot(x='int_rate_groups', data=loan_data[loan_data.loan_status == 'Charged Off'])
plt.xlabel('Interest Rate')
plt.subplot(222)
sns.countplot(x='emp_length', data=loan_data[loan_data.loan_status == 'Charged Off'])


# In[63]:


fig, ax = plt.subplots(figsize = (7,5))
ax.set_yscale('log')
sns.countplot(x='open_acc_groups', data=loan_data[loan_data.loan_status == 'Charged Off'])


# In[64]:


sns.countplot(x='revol_util_groups', data=loan_data[loan_data.loan_status == 'Charged Off'])


# In[65]:


fig, ax = plt.subplots(figsize = (8,6))
ax.set_yscale('log')
sns.countplot(x='total_acc_groups', data=loan_data[loan_data.loan_status == 'Charged Off'])


# In[66]:


fig, ax = plt.subplots(figsize = (10,6))
sns.countplot(x='annual_inc_groups', data=loan_data[loan_data.loan_status == 'Charged Off'])


# In[67]:


sns.countplot(x='verification_status', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[68]:


sns.countplot(y='term', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[69]:


fig,ax = plt.subplots(figsize = (10,8))
ax.set_yscale('log')
sns.countplot(x='inq_last_6mths', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[70]:


fig,ax = plt.subplots(figsize = (7,5))
ax.set_yscale('log')
sns.countplot(x='pub_rec', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[105]:


## Extracting month and year
df_month_year = loan_data['issue_d'].str.partition("-", True)     
loan_data['issue_month']=df_month_year[0]                       
loan_data['issue_year']='20' + df_month_year[2]


# In[106]:


loan_data.head()


# In[107]:


plt.figure(figsize=(15,15))
plt.subplot(221)
sns.countplot(x='issue_month', data=loan_data[loan_data['loan_status']=='Charged Off'])
plt.subplot(222)
sns.countplot(x='issue_year', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[ ]:





# In[74]:


fig,ax = plt.subplots(figsize = (12,5))
ax.set_yscale('log')
sns.countplot(x='funded_amnt_inv_group', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[75]:


fig,ax = plt.subplots(figsize = (15,6))
ax.set_yscale('log')
sns.countplot(x='loan_amnt_groups', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[76]:


sns.countplot(x='dti_groups', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[77]:


fig,ax = plt.subplots(figsize = (15,6))
ax.set_yscale('log')
sns.countplot(x='installment_groups', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[ ]:


# Summary of Key Factors Influencing Loan Default

# Financial Situation: Lower income, high debt-to-income ratio, and a history of missed payments are associated with increased default risk.
# Loan Characteristics: Loans with longer terms, higher interest rates, and larger amounts are more likely to result in default.
# Borrower Behavior: Renters, individuals using loans for debt consolidation, and those with multiple open credit accounts are at a higher risk of default.
# Credit History: A lack of credit inquiries and a clean public record, while seemingly positive, might paradoxically indicate a higher risk. This could be due to factors such as limited credit history or potential over-reliance on credit.
# Loan Grade: Loans categorized as 'B' with a sub-grade of 'B5' are associated with a higher default rate.

# These factors suggest that borrowers with financial strain, poor credit management habits, and a reliance on loans for immediate financial relief are more likely to encounter difficulties in repaying their debts.


# In[ ]:


# Annual income vs loan purpose


# In[78]:


plt.figure(figsize=(10,10))
sns.barplot(data =loan_data,x='loan_amnt', y='grade', hue ='loan_status',palette="pastel", order=['A','B','C','D','E','F','G'])
plt.show()


# In[79]:


plt.figure(figsize=(20,20))
plt.subplot(221)
sns.barplot(data =loan_data,y='loan_amnt', x='emp_length', hue ='loan_status',palette="pastel")
plt.subplot(222)
sns.barplot(data =loan_data,y='loan_amnt', x='verification_status', hue ='loan_status',palette="pastel")


# In[ ]:


grade vs interest rate


# In[80]:


plt.figure(figsize=(10,10))
sns.barplot(data =loan_data,x='int_rate', y='grade', hue ='loan_status',palette="pastel", order=['A','B','C','D','E','F','G'])
plt.show()


# In[81]:


# fig,ax = plt.subplots(figsize = (15,6))
plt.tight_layout()
sns.catplot(data =loan_data,y ='int_rate', x ='loan_amnt_groups', hue ='loan_status',palette="pastel",kind = 'box')


# In[ ]:


# Observation: Interest rate for charged off loans is higher than fully paid loans across all loan_amount groups.
# This could be a strong indicator of loan default.


# Findings
# Key Findings

# Income and Loan Purpose:
# - Home improvement loans for applicants earning 60-70k have a higher default rate.

# Income and Home Ownership:
# - Mortgage holders with income between 60-70k show a higher propensity to default.

# Income and Interest Rate:
# - Loans with interest rates in the 21-24% range for applicants earning 70-80k have a higher default risk.

# Loan Amount and Interest Rate:
# - Loans between 30k-35k with interest rates of 15-17.5% are associated with higher default rates.

# Loan Purpose and Amount:
# - Small business loans exceeding 14k have a higher default likelihood.

# Home Ownership and Loan Amount:
# - Mortgage holders with loans between 14-16k are more likely to default.

# Loan Grade and Amount:
# - Grade F loans between 15k-20k have a higher default risk.

# Employment Length and Loan Amount:
# - Individuals with 10 years of employment and loans between 12k-14k show a higher default tendency.

# Loan Verification and Amount:
# - Verified loans exceeding 16k have a higher default probability.

# Loan Grade and Interest Rate:
# - Grade G loans with interest rates above 20% are associated with higher default risk.