#!/usr/bin/env python
# coding: utf-8

# Description of the Data: https://developers.google.com/machine-learning/crash-course/california-housing-data-description

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


path = 'california_housing_train.csv'
df = pd.read_csv(path)
print(df.head())


# In[3]:


import matplotlib.pyplot as plt
bins = np.linspace(df['median_income'].min(),df['median_income'].max(),num=10)
plt.hist(df['median_income'], bins=bins)
plt.xticks(bins)
plt.title('Median Incomes of Housing Blocks in California in 1990')
plt.xlabel('Median Income (Tens Thousands of USD)')
plt.ylabel('Blocks')
plt.show()


# In[4]:


import seaborn as sns
sns.set_style('darkgrid')

sns.histplot(data=df, x='median_income', bins=10)
plt.xlabel('Median Income (Tens of Thousands of USD)')
plt.ylabel('Blocks')
plt.title('Median Incomes of Housing Blocks in California in 1990')
plt.show()


# In[5]:


import plotly.graph_objects as go

fig = go.Figure(data=go.Histogram(x=df['median_income'], nbinsx=10))   # nbinsx=10 to sort into 10 bins
fig.update_layout(
    title='Median Incomes of Housing Blocks in California in 1990',
    xaxis_title='Median Income (Tens of Thousands of USD)',
    yaxis_title='Blocks'
)


# In[6]:


import plotly.express as px

fig = px.histogram(df, x="median_income", nbins=16)
fig.show()


# In[ ]:




