#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().system('pip install pandas')


# In[11]:


import pandas as pd


# In[12]:


get_ipython().system('pip install sklearn')


# In[13]:


from sklearn import linear_model


# In[18]:


import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[19]:


get_ipython().system('pip install matplotlib')


# In[20]:


import matplotlib.pyplot as plt


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


Emission= pd.read_csv("C:\\Users\\SHAIK.RAHEEM\\Documents\\Emission Data.csv")


# In[34]:


Emission


# In[42]:



x=Emission.drop(columns=['Co2 Emission'])
y=Emission['Co2 Emission']


# In[43]:


x


# In[44]:


y


# In[45]:


model=linear_model.LinearRegression()


# In[51]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model.fit(x_train,y_train)


# In[53]:


predictions=model.predict(x_test)


# In[54]:


score=r2_score(y_test,predictions)


# In[55]:


score


# In[61]:


print ('Coefficients: ', model.coef_)
print ('Intercept: ',model.intercept_)


# In[ ]:





# In[ ]:




