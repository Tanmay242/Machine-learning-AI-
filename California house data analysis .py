#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
from sklearn.datasets import fetch_california_housing


import statsmodels.api as sm
 
from sklearn.model_selection import train_test_split


# In[14]:


df = fetch_california_housing(as_frame=True)
df


# In[ ]:





# In[37]:




# Split the data into features (X) and target variable (y)
X = df.data
y = df.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

    # Make predictions
y_pred = regressor.predict(X_test)

    # Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)


   # r-square of model
regressor.score(X_test,y_test)




# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


# In[ ]:





# In[40]:


from sklearn.tree import DecisionTreeRegressor


# In[41]:


tree_dec = DecisionTreeRegressor()


# In[42]:


tree_dec.fit(X_train,y_train)


# In[43]:


pred = tree_dec.predict(X_test)


# In[44]:


mape = mean_absolute_percentage_error(y_test, pred)
mape


# In[45]:


tree_dec.score(X_test,y_test)


# In[46]:


from sklearn.ensemble import RandomForestRegressor


# In[47]:


forest_model = RandomForestRegressor()
forest_model.fit(X_train,y_train)


# In[48]:


predF = forest_model.predict(X_test)


# In[49]:


#regressor.score(X_train,y_train)
forest_model.score(X_test,y_test)


# In[50]:


mapef = mean_absolute_percentage_error(y_test, pred)
mapef


# In[ ]:




