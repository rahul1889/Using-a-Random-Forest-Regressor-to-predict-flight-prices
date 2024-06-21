#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import sklearn as sk


# In[112]:


#load the dataset

df = pd.read_csv('/Users/rd/coding_stuff/Projects/Flights Price prediction/dataset/Clean_Dataset.csv')


# In[113]:


df


# In[15]:


df.airline.value_counts()


# In[16]:


df.source_city.value_counts()


# In[17]:


df.destination_city.value_counts()


# In[18]:


df.departure_time.value_counts()


# In[19]:


df.arrival_time.value_counts()


# In[20]:


df.stops.value_counts()


# In[22]:


df['class'].value_counts()


# In[23]:


df['duration'].min()


# In[26]:


df['duration'].max()


# In[27]:


df['duration'].median()


# In[29]:


## Preprocessing


# In[114]:


df = df.drop('Unnamed: 0', axis =1)
df = df.drop('flight', axis =1)

df['class'] = df['class'].apply(lambda x: 1 if x=='Business' else 0)


# In[115]:


df.stops = pd.factorize(df.stops)[0]


# In[116]:


df.stops


# In[117]:


df.head()


# In[118]:


df = df.join(pd.get_dummies(df.airline, dtype=int, prefix='airline')).drop('airline', axis=1)
df = df.join(pd.get_dummies(df.source_city,dtype=int, prefix='source')).drop('source_city', axis=1)
df = df.join(pd.get_dummies(df.destination_city,dtype=int, prefix='destination')).drop('destination_city', axis=1)
df = df.join(pd.get_dummies(df.arrival_time,dtype=int, prefix='arrival')).drop('arrival_time', axis=1)
df = df.join(pd.get_dummies(df.departure_time,dtype=int, prefix='departure')).drop('departure_time', axis=1)


# In[119]:


df


# In[120]:


# Training regression model


# In[123]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

x,y = df.drop('price', axis =1), df.price


# In[126]:


x_trian, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[128]:


reg = RandomForestRegressor(n_jobs= -1)

reg.fit(x_trian, y_train)


# In[131]:


#test the model
reg.score(x_test, y_test)


# In[142]:


#evaluate the model performance 
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = reg.predict(x_test)
print('R2', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))
print('MSE', mean_squared_error(y_test, y_pred))


# In[143]:


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Flight Price')
plt.ylabel('Predcited Flight price')
plt.title('Predicted vs Actual')


# In[144]:


df.price.describe()


# In[147]:


#find important features

importances = dict(zip(reg.feature_names_in_, reg.feature_importances_))

sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)


# In[148]:


sorted_importances


# In[156]:


#plot importance of features

plt.figure(figsize=(15,6))
plt.bar([x[0] for x in sorted_importances[:10]], [x[1] for x in sorted_importances[:10]])


# In[ ]:


#Hyper parameter tuning
#this will take forever to run as too many combinations, so lets just code and see what it does


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [2, 5, 10],
    'max_features': ['auto', 'sqrt']
}


grid_search = GridSearchCV(reg, param_grid, cv=5)
grid_search.fit(x_train, y_train)

best_params= grid_search.best_params_


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




