#!/usr/bin/env python
# coding: utf-8

# # Project Title - change this
# 
# TODO - Write some introduction about your project here: describe the dataset, where you got it from, what you're trying to do with it, and which tools & techniques you're using. You can also mention about the course [Data Analysis with Python: Zero to Pandas](zerotopandas.com), and what you've learned from it.

# ### How to run the code
# 
# This is an executable [*Jupyter notebook*](https://jupyter.org) hosted on [Jovian.ml](https://www.jovian.ml), a platform for sharing data science projects. You can run and experiment with the code in a couple of ways: *using free online resources* (recommended) or *on your own computer*.
# 
# #### Option 1: Running using free online resources (1-click, recommended)
# 
# The easiest way to start executing this notebook is to click the "Run" button at the top of this page, and select "Run on Binder". This will run the notebook on [mybinder.org](https://mybinder.org), a free online service for running Jupyter notebooks. You can also select "Run on Colab" or "Run on Kaggle".
# 
# 
# #### Option 2: Running on your computer locally
# 
# 1. Install Conda by [following these instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Add Conda binaries to your system `PATH`, so you can use the `conda` command on your terminal.
# 
# 2. Create a Conda environment and install the required libraries by running these commands on the terminal:
# 
# ```
# conda create -n zerotopandas -y python=3.8 
# conda activate zerotopandas
# pip install jovian jupyter numpy pandas matplotlib seaborn opendatasets --upgrade
# ```
# 
# 3. Press the "Clone" button above to copy the command for downloading the notebook, and run it on the terminal. This will create a new directory and download the notebook. The command will look something like this:
# 
# ```
# jovian clone notebook-owner/notebook-id
# ```
# 
# 
# 
# 4. Enter the newly created directory using `cd directory-name` and start the Jupyter notebook.
# 
# ```
# jupyter notebook
# ```
# 
# You can now access Jupyter's web interface by clicking the link that shows up on the terminal or by visiting http://localhost:8888 on your browser. Click on the notebook file (it has a `.ipynb` extension) to open it.
# 

# ## Package install and Import
# 
# First we need to install and import compulsory modules.
# 
# 1.Jovian: to upload, save and share the content of my notebook.
# 
# 2.Pandas: pandas is python module to analysis of data and making dataframe.
# 
# 3.Numpy: numeric library, working with array
# 
# 4.Matplotlib:  Analyzing fun and interactive with the visualization library matplotlib
# 
# 5.seaborn: Adding more colours  into matplotlib visualization

# In[73]:


from distutils.sysconfig import get_python_inc
from sysconfig import get_python_version


get_python_version().system('pip install jovian opendatasets --upgrade --quiet')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_python_inc().run_line_magic('matplotlib', 'inline')
import matplotlib


# Let's read dataset using pd.read_csv file

# In[74]:


dh_df=pd.read_csv('output.csv')


# The dataset has been uploaded and ready to run

# Let us save and upload our work to Jovian before continuing.

# In[75]:


project_name = "DeseaseAndReason" 


# In[76]:





# In[77]:





# In[78]:


jovian.commit(project=project_name)


# # Loading the Dataset
# 
# We are ready to load and read the dataset, above we used command "read_csv" which is used for read .csv format files.
# Now, we load the dataset and take a look.

# In[79]:


dh_df


# As we can see the data of dataset rows number and columns names  

# ## Data Preparation and Cleaning
# 
# We already read and load the data.Now, we have to check for any null or wrong values present in dataset.
# 
# 

# In[80]:


dh_df.info()


# Numbers of Rows and Columns in dataset 

# In[81]:


print("There are {} of rows in dataset".format(dh_df.shape[0]))


# In[82]:


print("There are {} of columns in dataset".format(dh_df.shape[1]))


# In[83]:


dh_df.isna().any().any()


# isna().any().any() gives bool value 'True' that means there are some null(NaN) values.
# 

# In[84]:


dh_df.nunique(axis='columns')


# By specifying the column axis ( axis='columns' ), the nunique() method searches column-wise and returns the number of unique values for each row.

# # Handling Null Values 
# 
# We will check for null values in each columns of dataset

# In[85]:


dh_df.isnull().values.any() # return boolean value 'True' for NaN values


# In[86]:


dh_df.isnull().sum().sum() # return missing values in dataset


# In[87]:


sns.heatmap(dh_df.isnull(), cbar=False)
plt.title('Null Values Heatmap')
plt.show()


# In[88]:


dh_df.isnull().sum()


# Above in heatmap and table, we can see there are quite null values under "Code" column. There are 1860 null values.We will have to handle all null data points before we can dive into EDA and modeling

# In[89]:


dh_df.drop(['Code'],axis=1,inplace=True)


# As we have only one column which have null value so we drop that column.Now we check it again

# In[90]:


dh_df.isnull().any() #we can see the column is dropped and other column return FALSE boolean values
# and we have clean dataset 


# # Splitting the Dataset
# 
# In this section, we will split the dataset according to different disease or year or country and other aspects for deep dive and create new datasets

# In[92]:


dh_df.head(2) # we execute this statement by looking into columns better than scrolling 


# In[99]:


ind_df=dh_df[dh_df['Entity']=='India'].copy()


# New dataset based on india and will create few more on other countries 

# In[101]:


ind_df.head() # cheack ind_df data set and Entity column only show Country "India"


# In[102]:


ger_df=dh_df[dh_df['Entity']=='Germany'].copy()


# In[103]:


ger_df.head()


# In[104]:


jap_df=dh_df[dh_df['Entity']=='Japan'].copy()


# In[105]:


jap_df.head()


# Below dataset develop based on year

# In[109]:


dh_2019_df=dh_df[dh_df['Year']==2019].copy()   #this dataset based on year 2019 


# In[110]:


dh_2019_df.head()


# In[118]:


dh_2010_df=dh_df[dh_df['Year']==2010].copy() # this one is based on year 2010


# In[119]:


dh_2010_df


# In[136]:


ind_2019_df= dh_df[(dh_df.Entity=='India') & (dh_df.Year==2010)]  # this one on country india and year 2010


# In[137]:


ind_2019_df


# Call some new method of pandas on this dataset 

# In[149]:


dh_df.describe()


# The describe() method returns description of the data in the DataFrame. If the DataFrame contains numerical data, the description contains these information for each column: count - The number of not-empty values. mean - The average (mean) value. std - The standard deviation.

# In[152]:


dh_df.shape # Return 2 values first one is number of Rows and second values is Columns 


# In[153]:





# ## Exploratory Analysis and Visualization
# 
# **TODO** - write some explanation here.
# 
# 

# > Instructions (delete this cell)
# > 
# > - Compute the mean, sum, range and other interesting statistics for numeric columns
# > - Explore distributions of numeric columns using histograms etc.
# > - Explore relationship between columns using scatter plots, bar charts etc.
# > - Make a note of interesting insights from the exploratory analysis

# Let's begin by importing`matplotlib.pyplot` and `seaborn`.

# In[ ]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# Here we execute some operation like sum, mean and other interesting statistics

# Mean = the average value (the sum of all values divided by number of values)

# In[162]:


ind_df['Self-harm'].mean() #dataset based on india country on disease Self harm mean value 


# In[164]:


dh_df['Self-harm'].mean() #dataset based on complete country set on disease Self harm mean value 


# 
# 
# 
# 
# Mode = the value that appears most frequently.
# 

# In[166]:


dh_df['Self-harm'].mode()


# In[167]:


dh_df.corr()


# The corr() method calculates the relationship between each column in your data set.
# 
# Perfect Correlation:
# 
# We can see that "Duration" and "Duration" got the number 1.000000, which makes sense, each column always has a perfect relationship with itself.
# Good Correlation:
# 
# "Duration" and "Calories" got a 0.922721 correlation, which is a very good correlation, and we can predict that the longer you work out, the more calories you burn, and the other way around: if you burned a lot of calories, you probably had a long work out.
# Bad Correlation:
# 
# "Duration" and "Maxpulse" got a 0.009403 correlation, which is a very bad correlation, meaning that we can not predict the max pulse by just looking at the duration of the work out, and vice versa.

# # Data visualization
# 
# 
# In this section we execute operation which show result mostly in graphs and distributions of numeric columns using histograms
# 

# In[205]:


sns.set_context('poster', font_scale=0.5)

dh_df.hist(bins=25, grid=False, figsize=(25,18), color='#86bf91', zorder=2, rwidth=0.9)
plt.show()


# In[213]:


dh_df2=pd.read_csv('output.csv')


# Above, we can see graph divided on basis of difference diseases 

# In[214]:





# In[215]:


import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import plotly_express as px


# In[217]:




fig = px.choropleth(dh_df2,locations='Code',color='Cardiovascular diseases',scope='world',color_continuous_scale=px.colors.sequential.GnBu,
range_color=(10,50),title='Cardiovascular diseases',height=700,animation_frame ='Year')
fig.show()
sns.boxplot(dh_df2['Cardiovascular diseases'])


# **Above we introduce new libraries and functions and map the heart diesease chart according to year and color rate**

# **Below we used same functions but on Mental disorders**

# In[221]:



fig = px.choropleth(dh_df2,locations='Code',color='Mental disorders',scope='world',color_continuous_scale=px.colors.sequential.GnBu,
                    range_color=(10,50),title='Mental disorders',height=700,animation_frame ='Year')
fig.show()
sns.boxplot(dh_df['Mental disorders'])


# In[222]:




fig = px.choropleth(dh_df2,locations='Code',color='Neoplasms',scope='world',color_continuous_scale=px.colors.sequential.GnBu,
range_color=(10,32),title='Neoplasms',height=700,animation_frame ='Year')
fig.show()


# # Neonatal disorders' world map

# In[224]:


sns.boxplot(dh_df['Neonatal disorders'])


# In[227]:


dataset_1990_2019=dh_df2[(dh_df2['Year']==1990 )|( dh_df2['Year']==2019)]
dataset_1990_2019=dataset_1990_2019.dropna()
dataset_1990_2019=dataset_1990_2019[dataset_1990_2019['Entity'] != 'World']
dataset_1990_2019=dataset_1990_2019.drop('Entity',axis=1)


# In[228]:




from sklearn.cluster import KMeans
dataset_1990_2019_1=dataset_1990_2019.drop(['Code','Year'],axis=1)


# In[229]:


kmeans_model = KMeans(n_clusters=4, random_state=10).fit(dataset_1990_2019_1)


# In[230]:


labels = kmeans_model.labels_
dataset_1990_2019['labels']=labels


# In[231]:


sns.set_context('poster', font_scale=0.6)
fig = px.choropleth(dataset_1990_2019,locations='Code',color='labels',scope='world',color_continuous_scale=px.colors.sequential.GnBu,
                     range_color=(0,4),title='Clustering map',height=700,animation_frame ='Year')
fig.show()


# In[242]:


dataset_1990_2019.groupby('labels').mean().T


#  # Increase and decrease of clusters

# In[233]:



sns.set_context('poster', font_scale=0.6)
plt.figure(figsize=(22,12))
dataset_1990_2019.groupby(['labels','Year'])['labels'].count().plot.barh(color=['blue','red','blue','red','blue','red','blue','red'])


# 
# # Get the correlation
# 
# 
# 

# In[243]:



sns.set_context('poster', font_scale=0.5)
plt.figure(figsize=(22,12))
cor = dh_df2.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[236]:




#Correlation with output variable
cor_target = abs(cor["Neurological disorders"])
#Selecting highly correlated features
relevant_features = cor_target #[cor_target>0.5]
relevant_features


# In[237]:


relevant_features = cor_target [cor_target>0.5]
relevant_features


# # Most important Correlation with output variable

# In[238]:


data1=dataset_1990_2019[['Enteric infections','Respiratory infections and tuberculosis', 'Neonatal disorders',
                          'Neurological disorders','Sense organ diseases','Mental disorders','Musculoskeletal disorders']]


# In[239]:




plt.figure(figsize=(15,8))
sns.set_context('poster', font_scale=0.8)
sns.heatmap(data1.corr(),annot=True, cbar=False, cmap='Blues', fmt='.1f')


# # Scatter Plote¶
# 

# In[240]:




plt.figure(figsize=(20,15))
sns.set_context('poster', font_scale=1.0)
color_codes = {0:'red', 1:'blue', 2:'yellow',3:'black'}
colors = [color_codes[x] for x in labels]
pd.plotting.scatter_matrix(data1[data1.columns[0:]], figsize=(50,50), color=colors, alpha=0.8, diagonal='kde')
plt.show()









