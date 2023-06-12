#!/usr/bin/env python
# coding: utf-8

# In[73]:


from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor



import statsmodels.api as sm
 
from sklearn.model_selection import train_test_split


# In[9]:


bs = pd.DataFrame(boston.data)
bs


# In[5]:


bs


# In[10]:


bs.columns = boston.feature_names


# In[11]:


bs.columns


# In[12]:


bs.head()


# In[13]:


bs["price"] = boston["target"]


# In[14]:


bs


# In[16]:


x = bs.drop('price',axis = 1)


# In[20]:


bs.info()


# In[18]:


y = bs['price']


# In[19]:


y


# In[23]:


# split the independent and target variable into train subset and test subset
# set 'random_state' to generate the same dataset each time you run the code 
# 'test_size' returns the proportion of data to be included in the testing set
X_train_slr, X_test_slr, y_train_slr, y_test_slr = train_test_split(x,y, 
                                                    random_state=1, test_size = 0.3)

# check the dimensions of the train & test subset using 'shape'
# print dimension of train set
print('X_train_slr', X_train_slr.shape)
print('y_train_slr', y_train_slr.shape)

# print dimension of test set
print('X_test_slr', X_test_slr.shape)
print('y_test_slr', y_test_slr.shape)


# In[55]:


# convert the X_train to DataFrame
X_train = pd.DataFrame(X_train)

# insert a column of intercept to 'X_train_slr'
# create an array of ones with length equal to the total number of observations
c = np.ones(X_train.shape[0])

# A design matrix is a matrix of observations of independent variables whose first column is of ones
# If there are 'm' observations and 'n' variables, then the dimension of a design matrix is m * (n+1) 

# add the intercept 
# pass location index of intercept to the parameter 'loc'
# pass column name to the parameter 'column'
# pass the column value to the parameter 'value'
X_train.insert(loc = 0, column = 'intercept', value = c)

# display the first five rows of design matrix
X_train.head()


# In[25]:


# building model using stats model 
# building a model on the train dataset with an intercept
# fit() is used to fit the OLS model
SLR_model = sm.OLS(y_train_slr, X_train_slr).fit()

# print the summary output
print(SLR_model.summary())


# In[ ]:





# In[26]:


# predict the Total_Compensation using 'predict()' on training data
y_train_slr_pred = SLR_model.predict(X_train_slr)

# display the first five predicted values 
y_train_slr_pred.head()


# In[33]:


# calculate the SSR on train dataset
ssr = np.sum((y_train_slr_pred - y_train_slr.mean())**2)
ssr


# In[34]:


# calculate the SSE on train dataset
sse = np.sum((y_train_slr - y_train_slr_pred)**2)
sse


# In[35]:


# calculate the SST on train dataset
sst = np.sum((y_train_slr - y_train_slr.mean())**2)
sst


# In[36]:


# add the values of SSE and SSR
sse + ssr


# In[37]:


ssr/(sse+ssr)


# In[38]:


# calculate R-Squared on train dataset
# use 'rsquared' method from statsmodel
r_sq = SLR_model.rsquared

# print the R-squared value
r_sq


# In[39]:


# calculate R-Squared on train dataset using the formula
r_sq = ssr/sst

# print the R-squared value
r_sq


# In[40]:


# compute SEE using the below formula 
# see =  np.sqrt(sse/(len(train_data) - k))    

# for SLR take k = 2, as there are two coefficients (parameters) in the model
see = np.sqrt(sse/(len(X_train_slr) - 2))    
see


# In[72]:


# building model using sklearn

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

    # Make predictions
y_pred = regressor.predict(X_test)

    # Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)


   # r-square of model
regressor.score(X_test,y_test)


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define a list of k values to evaluate
k_values = [1, 3, 5, 7, 9]

# Perform KNN with different k values
for k in k_values:
    # Train the KNN regressor
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Calculate MAPE
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"MAPE for k={k}: {mape:.2f}")


# In[66]:


tree = DecisionTreeRegressor()

tree.fit(X_train,y_train)


# In[67]:


predtict_tree = tree.predict(X_test)


# In[69]:


mape_tree = mean_absolute_percentage_error(y_test,predtict_tree)

mape_tree


# In[71]:


tree.score(X_train,y_train)
tree.score(X_test,y_test)


# In[74]:


rand_forest = RandomForestRegressor()

rand_forest.fit(X_train,y_train)


# In[75]:


predtict_forest = tree.predict(X_test)


# In[77]:


mape_forest = mean_absolute_percentage_error(y_test,predtict_forest)

mape_forest


# In[80]:


#rand_forest.score(X_train,y_train)
rand_forest.score(X_test,y_test)


# In[ ]:




