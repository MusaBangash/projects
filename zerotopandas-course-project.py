#!/usr/bin/env python
# coding: utf-8

# # Analysis on Suicides Rates of different country,gender and age
# 
# Suicides is problem related to most mental health which effect person mentally and make the person hopeless.In the EDA explore the dataset and perform operation to understand that which country, age group, generation and gender commit this act.This dataset is get from kaggle platform.we are here explain, explore and visualize. I discover this program Zero to pandas very helping platform and also much other interesting programs.

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
# 4.Matplotlib: Analyzing fun and interactive with the visualization library matplotlib
# 
# 5.seaborn: Adding more colours into matplotlib visualization
# 

# In[1]:


get_ipython().system('pip install jovian opendatasets --upgrade --quiet')
import jovian
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib


# Let's read dataset using pd.read_csv file

# In[2]:


sr_df=pd.read_csv('master.csv')


# The dataset has been uploaded and ready to run
# 
# Let us save and upload our work to Jovian before continuing.
# 

# Let's begin by downloading the data, and listing the files within the dataset.

# The dataset has been downloaded and extracted.

# Let us save and upload our work to Jovian before continuing.

# In[3]:


project_name = "SuicideRates" 


# In[4]:


get_ipython().system('pip install jovian --upgrade -q')


# In[5]:


import jovian


# In[6]:


jovian.commit(project=project_name)


# ## Loading the Dataset

# We are ready to load and read the dataset, above we used command "read_csv" which is used for read .csv format files. Now, we load the dataset and take a look.

# In[7]:


sr_df


# As we can see the data of dataset rows number and columns names
# 

# In[8]:


sr_df.head(2)  


# with declared parameter it return number of rows from start

# In[9]:


sr_df.tail()


# without declaring parameter tail() or head() return automatic 5 rows 

# ## Data Preparation and Cleaning
# 
# **We already read and load the data.Now, we have to check for any null or wrong values present in dataset.**
# 

# In[10]:


sr_df.info()


# In[11]:


len(sr_df)  # number of rows 


# In[12]:


len(sr_df.columns) # number of columns 


# .info() print information about columns, contains number of rows,columns, columns names, data types, memory usage and
# other information.
# There are 27820 entries and 12 columns to work with EDA and also there are columns which have null values. 

# In[13]:


sr_df.shape # shape return number of rows first and number of columns second index 


# In[14]:


print("There are {} of rows in dataset".format(sr_df.shape[0]))


# In[15]:


print("There are {} of columns in dataset".format(sr_df.shape[1]))


# In[16]:


sr_df.columns


# It appears column names of dataset

# In[17]:


sr_df.describe()


# In[18]:


sr_df.iloc[:,1:5].describe()


# We can also print by specific column of our data set describe 

# In[ ]:





# describe() function is used to find some basic statistical information regarding a data frame in pandas. Above in dataset table it return count, minimum values, maximum and standard values over columns

# In[19]:


sr_df.nunique()


# The nunique() method returns the number of unique values for each column. 
# Number of unique values in columns like in sex there are 2 unique values because of gender MALE AND FEMALE

# ## Handling Null Values

# We can see that for each of the columns, there are a lot different unique values for some of them. 

# In[20]:


sr_df.isnull().values.any()


# isnull() return boolean values, True so there is some null values in dataset

# In[21]:


sr_df.isnull().sum()


# In[22]:


sr_df.isnull().sum().sum()


# we can check with multiple techniques there is one column HDI FOR YEAR HAVE 19456 null values 

# In[23]:


sns.heatmap(sr_df.isnull(), cbar=False)
plt.title('Null Values Heatmap')
plt.show()


# So in Heatmap also we can see visually that there are a lot of null values in HDI FOR YEAR

# In[24]:


sr_df_copy=sr_df.copy()


# There are a lot of missing values in one columns so we can do multiple operation
# 1. use of dropna() it create new dataframe but does not change the original by inplace=True the original will also change
# 2. drop the whole column
# 3. replace empty values
# 4. or value replace Using Mean, Median, or Mode

# In[25]:


new_sr_df=sr_df.dropna()


# In[26]:


new_sr_df.isnull().sum()


# As we can see there are not showing anymore null values 

# In[27]:


mean_sr_HDI=sr_df['HDI for year'].mean()
mode_sr_HDI=sr_df['HDI for year'].mode()[0]
median_sr_HDI=sr_df['HDI for year'].median()


# Now, after compute the mean, mode, median you can replace the null value of the specific column with any of these three but will will do just one in example 
# 
# 

# In[28]:


sr_df_copy["HDI for year"].fillna(mean_sr_HDI, inplace = True) 


# In[29]:


sr_df_copy


# So, HDI for year where there is null value it replace by mean value of column

# In[30]:


sr_df.sample(10) # this return random of rows declared in paramenter of method it will give you the basic idea about data 


# we can split the dataset according to our need of data and there is no limit to it 

# In[31]:


male_sr_df= sr_df[sr_df['sex']=='male'].copy()


# In[32]:


male_sr_df.head()


# In[33]:


female_sr_df=  sr_df[sr_df['sex']=='female'].copy()


# In[34]:


female_sr_df.head()


# In[35]:


Japan_sr_df=sr_df[sr_df['country']=='Japan'].copy()


# In[36]:


Japan_sr_df


# In[37]:


gdp_sr_df=sr_df[sr_df['gdp_per_capita ($)']>50000].copy()


# In[38]:


gdp_sr_df


# **As we can see there is no limit to you can split dataset according to your own requirement of data**

# In[39]:


import jovian


# In[40]:


jovian.commit()


# ## Exploratory Analysis and Visualization
# 
# Before we dive deep we will further do the exploratory analysis it is important to explore these variables to understand how we can represent our data more understandable.so we will convert or grouping operation on dataset
# 
# 

# > Instructions (delete this cell)
# > 
# > - Compute the mean, sum, range and other interesting statistics for numeric columns
# > - Explore distributions of numeric columns using histograms etc.
# > - Explore relationship between columns using scatter plots, bar charts etc.
# > - Make a note of interesting insights from the exploratory analysis

# Let's begin by importing`matplotlib.pyplot` and `seaborn`.

# In[41]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[ ]:





# In[42]:


sr_df.head()


# # Suicide Rates: Male and Female
# 
# comparison between Male and Female suicide rates in dataset to get which gender commit more in number 

# In[43]:


suicideGender1985=sr_df.groupby(['country','sex']).suicides_no.sum()


# In[44]:


suicideGender1985


# **Above we have the ratio between men and women suicide numbers of each country**
# **Now we will take the higest country with suicide rates**

# ## Countries with highest suicide numbers

# **we will draw a chart where we show the countries with highest suicides numbers**

# In[45]:


suicidesNo=[]
for country in sr_df.country.unique():
    suicidesNo.append(sum(sr_df[sr_df['country']==country].suicides_no))   


# In[46]:


suicidesNo=pd.DataFrame(suicidesNo,columns=['suicidesNo'])
country=pd.DataFrame(sr_df.country.unique(),columns=['country'])
data_suicide_countr=pd.concat([suicidesNo,country],axis=1)


# In[47]:


data_suicide_countr=data_suicide_countr.sort_values(['suicidesNo'],ascending=False)


# In[48]:


sns.barplot(y=data_suicide_countr.country[:15],x=data_suicide_countr.suicidesNo[:15])
plt.show()


# ## Age group between Male and Female with highest number of suicides 

# **In this section we can perform different type of operations First we will sort age groups with highest number of suicides and then perform on that division on gender base**

# In[49]:


group_df=sr_df.groupby(['age','sex'])['suicides_no'].sum().unstack()
group_df=group_df.reset_index().melt(id_vars='age')


# In[50]:


group_df_female=group_df.iloc[:6,:]
group_df_male=group_df.iloc[6:,:]


# In[51]:


group_df_female


# In[52]:


group_df_male


# In[53]:


female_=[175437,208823,506233,16997,430036,221984]
male_=[633105,915089,1945908,35267,1228407,431134]
plot_id = 0
for i,age in enumerate(['15-24 years','25-34 years','35-54 years','5-14 years','55-74 years','75+ years']):
    plot_id += 1
    plt.subplot(3,2,plot_id)
    plt.title(age)
    fig, ax = plt.gcf(), plt.gca()
    sns.barplot(x=['female','male'],y=[female_[i],male_[i]],color='blue')
    plt.tight_layout()
    fig.set_size_inches(10, 15)
plt.show()  


# **All data were analyzed. Graphical analysis was performed for all age rates for suicide rates.**
# 

# In[62]:


sr_df.head(2)


# # Generation Hue Gender Counter

# **First we execute generation counter and after than we will check on gender base each generation and find out the ratio**

# In[80]:


sns.countplot(sr_df.generation)
plt.title('Generation Counter')
plt.xticks(rotation=45)
plt.show()


# **We have 6 generation on dataset and we have count that how frequent these generation exists in dataset now we observe on gender base each generation**

# In[83]:


sns.countplot(sr_df.generation,hue=sr_df.sex)
plt.title('Generation hue Gender Counter')
plt.show()


# **Show the percentage of each generation on pie chart**

# In[114]:






f,ax=plt.subplots(1,2,figsize=(18,8))
sr_df['generation'].value_counts().plot.pie(explode=[0.1,0.1,0.1,0.1,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)



# In[117]:


sr_df.head(2)


# # Country with maximum and minimum suicides 
# 
# **We find out that which country have max suicides and min suicides the we polt that country on graph to check all related data**

# In[120]:


max(sr_df.suicides_no)


# In[121]:


min(sr_df.suicides_no)


# In[168]:


sr_df[sr_df.suicides_no==max(sr_df.suicides_no)]


# In[138]:


sr_df[sr_df['country']=='Russian Federation'].hist()
plt.tight_layout()
plt.show()


# The above graph show if you look at suicides and suicides/100 population this is now clear suicides ratio is decreasing and also if we check gdp per capita is also decease so it show us in Russian case that suicides are not directly link with economy. There should be other problems that cause people commit this act.
# 

# In[162]:


sr_df[sr_df.suicides_no==min(sr_df.suicides_no)]


# In[163]:


sr_df[sr_df['country']=='Albania'].hist()
plt.tight_layout()
plt.show()


# **Above we see here the same result that suicide rate down and also economy situation is not that good so in most cases suicide is not much related to economy.**

# The above graph show if you look at suicides and suicides/100 population this is now clear suicides ratio is decreasing and also if we check gdp per capita is also decease so it show us in Russian case that suicides are not directly link with economy. There should be other problems that cause people commit this act.

# Let us save and upload our work to Jovian before continuing

# In[164]:


import jovian


# In[165]:


jovian.commit()


# ## Asking and Answering Questions
# 
# **Here we can self ask interesting question,to further deep dive and understand the dataset**
# 

# In[166]:


sr_df.head()


# ## Q1: Find minimum and maximum year and then perform different operation to check the situation in minimum and maximum year present in dataset?

# In[196]:


min_year=min(sr_df.year)


# In[197]:


max_year=max(sr_df.year)


# In[198]:


print('Min Year :',min_year)


# In[200]:


print('Max Year :',max_year)


# In[207]:


sr_1985_df=sr_df[sr_df['year']==1985].copy()


# In[208]:


sr_1985_df


# In[ ]:


suicidesNo=[]
for country in sr_1985_df.country.unique():
    suicidesNo.append(sum(sr_1985_df[sr_1985_df['country']==country].suicides_no))   


# In[211]:


suicidesNo=pd.DataFrame(suicidesNo,columns=['suicidesNo'])
country=pd.DataFrame(sr_1985_df.country.unique(),columns=['country'])
data_suicide_countr=pd.concat([suicidesNo,country],axis=1)


# In[212]:


data_suicide_countr=data_suicide_countr.sort_values(['suicidesNo'],ascending=False)


# In[213]:


sns.barplot(y=data_suicide_countr.country[:15],x=data_suicide_countr.suicidesNo[:15])
plt.show()


# **Above the bar show that the minimum year in dataset these 7 countries had the most suicides rate**

# In[214]:


sr_2016_df=sr_df[sr_df['year']==2016].copy()


# In[215]:


sr_2016_df


# In[216]:


suicidesNo=[]
for country in sr_2016_df.country.unique():
    suicidesNo.append(sum(sr_2016_df[sr_2016_df['country']==country].suicides_no))   


# In[217]:


suicidesNo=pd.DataFrame(suicidesNo,columns=['suicidesNo'])
country=pd.DataFrame(sr_1985_df.country.unique(),columns=['country'])
data_suicide_countr=pd.concat([suicidesNo,country],axis=1)


# In[218]:


data_suicide_countr=data_suicide_countr.sort_values(['suicidesNo'],ascending=False)


# In[219]:


sns.barplot(y=data_suicide_countr.country[:15],x=data_suicide_countr.suicidesNo[:15])
plt.show()


# **In 2016 which is latest in dataset These country had the highest ratio**

# In[225]:


s1 = pd.merge(sr_1985_df, sr_2016_df, how='inner', on=['country'])


# In[226]:


s1


# **Above we just merged two dataframe on country**

# ## Q2: Plot GDP per capita and perform different operation on Gender base?

# In[228]:


sr_df['gdp_per_capita ($)'].unique()


# In[229]:


print("Max : ",max(sr_df['gdp_per_capita ($)'].unique()))
print('Min : ',min(sr_df['gdp_per_capita ($)'].unique()))


# In[232]:


sns.countplot(sr_df[sr_df['gdp_per_capita ($)']==251].sex)
plt.title("GdpPerCapitalMoney Gender (Male-Female)")
plt.show()


# In[233]:


sns.countplot(sr_df[sr_df['gdp_per_capita ($)']==126352].sex)
plt.title("GdpPerCapitalMoney Gender (Male-Female)")
plt.show()


# In[234]:


plt.figure(figsize=(10,5))
sns.countplot(sr_df.sex,hue=sr_df.age)
plt.title('Gender & Age')
plt.show()


# In[235]:


sr_df.groupby('age')['sex'].count()


# In[236]:


sns.barplot(x=sr_df.groupby('age')['sex'].count().index,y=sr_df.groupby('age')['sex'].count().values)
plt.xticks(rotation=90)
plt.show()


# ## Q3: Plot index popluation and perform difference operation?

# In[237]:


index_population=[]
for age in sr_df['age'].unique():
    index_population.append(sum(sr_df[sr_df['age']==age].population)/len(sr_df[sr_df['age']==age].population))
    
plt.bar(['15-24 years','35-54 years','75+ years','25-34 years','55-74 years','5-14 years'],index_population,align='center',alpha=0.5)
plt.xticks(rotation=90)
plt.show()


# In[238]:


index_population


# In[240]:


plt.figure(figsize=(10,5))
sns.set(style='whitegrid')
sns.boxplot(sr_df['population'])
plt.show()


# In[241]:


sns.set(style='whitegrid')
sns.boxplot(sr_df['gdp_per_capita ($)'])
plt.show()


# In[243]:


sns.set(style='whitegrid')
sns.boxplot(sr_df.year)
plt.show()


# In[ ]:





# In[ ]:





# ## Q4: Perform and plot using different data from dataset?

# In[245]:


sns.pairplot(sr_df,hue='generation')
plt.show()


# **This above operation give us details about every columns through the years according to generation**

# In[246]:


sns.pairplot(sr_df,hue='sex')
plt.show()


# **This is the same operation about this graph are according the gender base**

# ## Q5: Perform some operation using heatmap and plot?

# In[249]:


sns.distplot(sr_df[(sr_df['sex']=='female')].age.value_counts().values)
plt.show()


# In[252]:


sns.violinplot(x=sr_df['generation'],y=sr_df['population'])
plt.show()


# In[253]:


sns.heatmap(sr_df.corr(),cmap='YlGnBu',annot=True)
plt.show()


# 
# 
# # Draw the heatmap with the mask and correct aspect ratio

# In[254]:


sns.heatmap(sr_df.corr(), vmax=.3, center=1,
            square=True, linewidths=.5,annot=True)
plt.show()


# **We can perform thousand of these operation on dataset to more and more understanding the data.**

# In[256]:


cmap = sns.diverging_palette(220, 10, as_cmap=True)


# In[257]:


sns.heatmap(sr_df.corr(), cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


# In[258]:


sns.boxplot(x=sr_df['generation'],y=sr_df['population'])
plt.xticks(rotation=90)


# Let us save and upload our work to Jovian before continuing.

# In[259]:


import jovian


# In[260]:


jovian.commit()


# ## Inferences and Conclusion
# 
# 
# **It has been clear as we performed certain operation that suicides ratio is not directly related to economy.Over the time of period the suicide ratio is going slow and down. Currently Russia is the country with most suicides ratio.We find that early year available in data set US is the country with 6 other with high rate of suicide and the last available year 2016 only one country remain in top 10 all other countries improve and slow down the suicides. After perform various operation base on income, age we find out that gender also did not play big role in suicides but in certain age group male commit more suicides and in some group female commit more suicide but late in life people commit more suicide.**
# 
# 

# In[261]:


import jovian


# In[262]:


jovian.commit()


# ## References and Future Work
# 
# **Future Work**
# Practice more and more and start learning other Python module available on jovian and follow the road to data science.
# 
# **References**
# All helpful website to perform these operation
# 
# https://stackoverflow.com
# 
# https://pandas.pydata.org
# 
# https://www.w3schools.com
# 
# https://jovian.ai/learn/data-analysis-with-python-zero-to-pandas/lesson/lesson-5-data-visualization-with-matplotlib-and-seaborn
#  

# In[265]:


import jovian


# In[266]:


jovian.commit()


# In[ ]:




