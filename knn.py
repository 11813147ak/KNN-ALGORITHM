#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[37]:


urlx='https://raw.githubusercontent.com/coding-blocks-archives/machine-learning-online-2018/master/Datasets/xdata.csv'
dfx1=pd.read_csv(urlx,error_bad_lines=False)


# In[38]:


dfx1.head()


# In[39]:


urly='https://raw.githubusercontent.com/coding-blocks-archives/machine-learning-online-2018/master/Datasets/ydata.csv'
dfy1=pd.read_csv(urly,error_bad_lines=False)


# In[40]:


dfy1.head()


# In[41]:


print(dfx1.shape,dfy1.shape)


# In[44]:


print(type(dfx1))


# In[56]:


dfx3=dfx1.values


# In[57]:


dfx3


# In[59]:


print(dfy1)


# In[60]:


print(type(dfy1))


# In[61]:


dfy2=dfy1.values


# In[62]:


dfy2


# In[105]:


plt.style.use('seaborn')
query_point=np.array([2,3])

plt.scatter(dfx3[:,0],dfx3[:,1],c='w')
plt.scatter(query_point[0],query_point[1],color='red')
plt.show()


# In[120]:


def dist(x1,x2):
    return np.sqrt(sum(x1-x2)**2)


def knn(dfx3,dfy2,querypoint,k=5):
    
    vals=[]
    m=dfx3.shape[0]
    
    for i in range(m):
        d=dist(querypoint,dfx3[i])
        vals.append((d,dfy2[i]))
        
    vals=sorted(vals)
    
    vals=vals[:k]
    
    vals=np.array(vals)
    
    print(vals)
    
    new_vals=np.unique(vals[:,1],return_counts=True)
    print(new_vals)
    index=new_vals[1].argmax()
    pred=new_vals[0][index]
    return pred
    
    


# In[125]:


knn(dfx3,dfy2,[1,2])


# In[ ]:





# In[100]:





# In[ ]:




