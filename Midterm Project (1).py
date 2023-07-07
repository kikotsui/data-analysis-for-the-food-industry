#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load data
data= pd.read_csv('Project.csv')

#check for missing values
print(data.isnull().sum())

#impute missing values with mean
data.fillna(data.mean(), inplace=True)

#Descriptive statistics
desc_stats= data.describe()
print(desc_stats)

# Plot histograms for each variable
data.hist(bins=10, figsize=(12,8))
plt.show()


# In[2]:


# Check the data types for the variables
print(data.dtypes)

# Convert date into Unix timestamp
data['DATE'] = pd.to_datetime(data['DATE']).astype(int)/10**9

print(data.dtypes)

# Calculate correlation matrix
corr_matrix = data[['Food Expenditures', 'DATE', 'GDP', 'S&P 500', 'Home Price Index', 
                    'Unemployed Rate', 'Real Personal Income', 'Retail Sales', 'CPI']].corr()

# Print correlation matrix
print(corr_matrix)


# In[3]:


# Calculate correlation coefficients with food_expenditure
corr_coeffs = data.corr()['Food Expenditures'].sort_values(ascending=False)
# Print correlation coefficients
print(corr_coeffs)


# In[15]:


# Define dependent and independent variables 
data = data.fillna(data.mean())
y= data['Food Expenditures']
X= data[['GDP', 'S&P 500', 'Home Price Index', 'Unemployed Rate', 'Real Personal Income', 'Retail Sales', 'CPI']]

# Add constant term to independet variables
X= sm.add_constant(X)

# Fit multiple linear regression model
model = sm.OLS(y, X).fit()

#print summary of regression result
print(model.summary())


# In[6]:


from sklearn.metrics import mean_squared_error
#Get the predicted values of the model
y_pred= model.predict(X)

# Plot the predicted values against the actual values
plt.scatter(y, y_pred)
plt.xlabel('Actual Food Expenditures')
plt.ylabel('Predicted Food Expenditures')
plt.title('OLS Regression Model')
plt.show()

mse = mean_squared_error(y, y_pred)

print("Mean Squared Error:", mse)


# In[7]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Split the data into X and y
X = data[['GDP', 'S&P 500', 'Home Price Index', 'Unemployed Rate', 'Real Personal Income', 'Retail Sales', 'CPI']]

# Calculate the VIF for each independent variable
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Print the VIF table
print(vif)


# In[8]:


from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# perform ridge regression with cross-validation to tune alpha parameter
alphas = np.logspace(-4, 4, 50)
ridge_cv = RidgeCV(alphas=alphas, cv=10)
ridge_cv.fit(X_train, y_train)

# evaluate model performance on test set
y_pred = ridge_cv.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
# print coefficients of independent variables
coef = ridge_cv.coef_
print("Coefficients:", coef)

# plot coefficients
import matplotlib.pyplot as plt
plt.plot(range(len(coef)), coef)
plt.xticks(range(len(coef)), X.columns, rotation=90)
plt.show()


# In[9]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# create a dataframe with your independent variables
X = data[['GDP', 'S&P 500', 'Home Price Index', 'Unemployed Rate', 'Real Personal Income', 'Retail Sales', 'CPI']]

# standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# perform PCA with 2 components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# create a new dataframe with the principal components
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# add the dependent variable 'food expenditures' to the new dataframe
principal_df['Food Expenditures'] = data['Food Expenditures']

# plot the principal components against the dependent variable
sns.scatterplot(data=principal_df, x='PC1', y='PC2', hue='Food Expenditures')


# In[10]:


#  X is data matrix
X = data[['GDP', 'S&P 500', 'Home Price Index', 'Unemployed Rate', 'Real Personal Income', 'Retail Sales', 'CPI']]

# Create a PCA object with n_components = number of features
pca = PCA(n_components=X.shape[1])

# Fit PCA on the data matrix X
pca.fit(X)

# Get the variance ratio of each principal component
var_ratio = pca.explained_variance_ratio_

# Print the variance ratio of each principal component
for i, ratio in enumerate(var_ratio):
    print(f"Variance ratio of PC{i + 1}: {ratio:.4f}")


# In[16]:


# create a new dataframe with the selected principal component and dependent variable
data = data.fillna(data.mean())
y= data['Food Expenditures']
X= data[['GDP','S&P 500','Unemployed Rate']]

# Add constant term to independet variables
X= sm.add_constant(X)

# Fit multiple linear regression model
model = sm.OLS(y, X).fit()

#print summary of regression result
print(model.summary())


#Get the predicted values of the model
y_pred= model.predict(X)

# Plot the predicted values against the actual values
plt.scatter(y, y_pred)
plt.xlabel('Actual Food Expenditures')
plt.ylabel('Predicted Food Expenditures')
plt.title('OLS Regression Model')
plt.show()

mse = mean_squared_error(y, y_pred)

print("Mean Squared Error:", mse)


# In[17]:


import statsmodels.api as sm

y= data['Food Expenditures']
X= data[['GDP', 'S&P 500', 'Home Price Index', 'Unemployed Rate', 'Real Personal Income', 'Retail Sales', 'CPI']]

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Get the p-values and t-values of the model
p_values = model.pvalues
t_values = model.tvalues

print("P values:", p_values)
print("t values:", t_values)


# In[ ]:




