#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter('ignore')


# In[2]:


df=pd.read_csv('Cinema_Statewise_UNCLEANED_Dataset_325_Rows.csv')
df


# In[3]:


#knowing no.of columns
df.columns


# In[4]:


#knowing number or rows and columns
df.shape


# In[5]:


#knowing info 
df.info()


# In[6]:


#knowing data types of each column 
df.dtypes


# In[7]:


#checking null values 
df.isnull().sum()


# In[8]:


#checking null values % column wise 
(df.isnull().sum()/len(df))*100


# In[9]:


df.describe()


# In[10]:


#checking unique 
df['City'].unique()


# In[11]:


#checking unique 
df['State'].unique()


# In[12]:


#checking unique 
df['Theatre_ID'].unique()


# In[13]:


df['Theatre_ID'].value_counts()


# In[14]:


df['Theatre_ID'].nunique()


# In[15]:


#checking unique 
df['Show_Time'].unique()


# In[16]:


#checking unique 
df['Seats_Available'].unique()


# In[17]:


#checking unique 
df['Seats_Occupied'].unique()


# In[18]:


#checking unique 
df['Weather'].unique()


# In[19]:


#checking unique 
df['Movie_Genre'].unique()


# In[20]:


#checking unique 
df['Day_Type'].unique()


# In[21]:


#checking unique 
df['Snack_Purchased'].unique()


# In[22]:


#checking unique 
df['Snack_Spend_INR'].unique()


# In[23]:


#checking unique 
df['Customer_Type'].unique()


# In[24]:


#Data cleaning 


# In[25]:


#STEP 1 ====> Fixing data tupes
df.dtypes


# In[26]:


#changing object data type to =====> categorical
data_types=['Theatre_ID','Show_Time','Movie_Genre','Movie_Type','Snack_Purchased','Customer_Type','Weather','Day_Type']
df[data_types]=df[data_types].astype('category')
#changing data datatype(object) to =====> Datetime
df['Date']=pd.to_datetime(df['Date'],errors='coerce')


# In[27]:


df.dtypes


# In[28]:


#STEP 2 ====> REMOVING DUPLICATES


# In[29]:


#checking duplicates
df.duplicated().sum()


# In[30]:


#removing duplicates 
df.drop_duplicates(inplace=True)


# In[31]:


#checking wheather duplicates removed or not
df.duplicated().sum()


# In[32]:


##STEP 3 ====> FINDING INVALID VALUES AND FILLING THOSE VALUES WITH NAN


# In[33]:


#Finding invalid values
df.select_dtypes(include='number')\
.where(df.select_dtypes(include='number')<=0)\
.stack()
#invalid values ===> [0,-50,-100,-3]


# In[34]:


#Filling invalid values with NaN
invalid_num=df.select_dtypes(include='number').columns
df[invalid_num]=df[invalid_num].where(df.select_dtypes(include='number')>0,np.nan)


# In[35]:


#checking wheather invalid values filled with NaN or Not 
df.select_dtypes(include='number')\
.where(df.select_dtypes(include='number')<=0)\
.stack() 
# values filled with NaN


# In[36]:


#checking in dataset 
df


# In[37]:


df['Snack_Spend_INR'].unique() #======> erliar invalid value -100 is there but now replaced with nan


# In[38]:


#STEP 4 ====> FINDING SKEW AND OUTLIER 


# In[39]:


# Finding skew
df.select_dtypes(include='number').skew()
# skew = 0 ====> symmetric
# skew > 0.5 ====> High 
# skew < 0.5 =====> Low


# In[40]:


# Checking outliers
df.select_dtypes(include='number').boxplot(figsize=(12,5))


# In[41]:


#STEP 5 ====> FILLING NULL VALUES (NaN) 


# In[42]:


# I am filling null values with mean because data does't contain any outlier and also skew is low

# Filling numerical columns NaN Values

num_cols=df.select_dtypes(include='number').columns
df[num_cols]=df[num_cols].fillna(df[num_cols].mean())  #=====> Filling with mean

# Filling categorical and object columns NaN Values

cat_cols=df.select_dtypes(include=['object','category']).columns
df[cat_cols]=df[cat_cols].fillna(df[cat_cols].mode().iloc[0]) #======> Filling with mode


# In[43]:


#checking null values

df.isnull().sum()


# In[44]:


df


# In[45]:


# creating column 
df['Total_Seats']=df['Seats_Available']+df['Seats_Occupied']
df


# In[46]:


# Creating group column 
df['Ticket_Price_Range']=pd.cut(df['Ticket_Price_INR'],bins=[0, 150, 220,df['Ticket_Price_INR'].max()],labels=['Low (0-150)','Medium (150-220)','High (220+)'])
df


# In[47]:


#creating quarter column
df['Quarter'] = df['Date'].dt.to_period('Q')
df


# In[48]:


#creating 'total revenue '  column
df['Total_Revenue']=df['Seats_Occupied']*df['Ticket_Price_INR']
df


# In[49]:


# creating columnn for occupancy rate 
df['Occupancy_Rate']=df['Seats_Occupied']/df['Total_Seats']
df


# In[50]:


df


# In[51]:


# Revenue per Seat
df['Revenue_per_Seat'] = df['Total_Revenue'] / df['Total_Seats']
df


# In[52]:


#occupacy level
df['Occupancy_Level']=pd.cut(df['Occupancy_Rate'],bins=[0,0.4,0.7,1],labels=['Low (Occupancy)','Medium (Occupancy)','High (Occupancy)'])
df


# In[53]:


df.drop('City',axis=1,inplace=True)
df


# In[54]:


df


# In[55]:


#State wise revenue
df.groupby('State')['Total_Revenue'].sum().plot(kind='barh')
plt.title('State wise revenue ')
plt.show()


# In[56]:


# State wise revenue per seat
rp=df.groupby('State')['Revenue_per_Seat'].mean()
rp.plot(kind='bar')
plt.title('State wise revenue per seat ')
plt.show()


# In[57]:


# Volume vs Pricing Trade-off Across States =====> Certain states earn more overall because they sell more seats, 
#while others earn more from each seat even with lower total revenue.
#conclusion ===> Total revenue differences across states are driven by two distinct factors: audience volume and pricing efficiency, 
#meaning states should not be evaluated on revenue alone.


# In[58]:


#Occupancy rate by ticket price range
oc=df.groupby('Ticket_Price_Range')['Occupancy_Rate'].mean()
oc.plot(kind='line',marker='o')
plt.title('Occupancy rate by ticket price range')
plt.show()


# In[59]:


#========> The analysis shows that medium ticket prices maximize occupancy, while higher prices reduce demand, 
#indicating a clear pricing sweet spot and the risk of overpricing.


# In[60]:


#revenue by ticket price range
price_revenue_per_seat=df.groupby('Ticket_Price_Range')['Total_Revenue'].mean()
price_revenue_per_seat.plot(kind='line',marker='o',color='green')
plt.title('Revenue by ticket price range')
plt.show()


# In[61]:


#Lower ticket prices attract more viewers and increase occupancy, 
#but revenue remains lower because each seat is sold at a cheaper price, 
#while higher-priced tickets generate more revenue despite fewer viewers.


# In[62]:


#occupancy level and queue waiting time 
que=df.groupby('Occupancy_Level')['Queue_Time_Minutes'].mean()
que.plot(kind='bar')
plt.title('occupancy level and queue waiting time ')
plt.show()


# In[63]:


#=====> Queue problems are not driven by crowd size, but by operational readiness.


# In[64]:


# show time occupancy rate 
sh=df.groupby('Show_Time')['Occupancy_Rate'].mean()
sh.plot(kind='bar')
plt.title('show time occupancy rate ')
plt.show()


# In[65]:


shq=df.groupby('Show_Time')['Queue_Time_Minutes'].mean()
shq.plot(kind='line',marker='o',color='red')
plt.title('show time Queue waiting time ')
plt.show()


# In[66]:


# ====> Peak show times are under-staffed or poorly managed, causing long queues even before occupancy becomes high.


# In[67]:


pivot = df.pivot_table(
    values='Occupancy_Rate',
    index='Show_Time',
    columns='Day_Type',
    aggfunc='mean'
)

sns.heatmap(pivot, annot=True, cmap='YlOrRd')
plt.title('Day type , showtime and occupancy rate')
plt.show()


# In[68]:


# ======> Weakend evening having more occupancy rate


# In[69]:


pivot = df.pivot_table(
    values='Occupancy_Rate',
    index='Show_Time',
    columns='Weather',
    aggfunc='mean'
)

sns.heatmap(pivot, annot=True, cmap='YlOrRd')
plt.title('Wheather , showtime and occupancy rate')
plt.show()


# In[70]:


# Even though in rainy season evening show occupancy rate is high ... in ovarall we can conclude that evening show is our major show time 


# In[71]:


# genre occupancy rate 
gen=df.groupby('Movie_Genre')['Occupancy_Rate'].sum()
e=[0.1,0,0,0,0]
gen.plot(kind='pie',autopct="%0.f%%",explode=e,shadow=True)
plt.title('Genre occupancy rate')
plt.show()


# In[72]:


#========> The analysis shows that medium ticket prices maximize occupancy, while higher prices reduce demand, 
#indicating a clear pricing sweet spot and the risk of overpricing.


# In[73]:


pivot = df.pivot_table(
    values='Occupancy_Rate',
    index='Movie_Genre',
    columns='State',
    aggfunc='mean'
)

sns.heatmap(pivot, annot=True)
plt.title('Movie genre occupancy rate by state wise')
plt.show()


# In[74]:


#======> Movie genre performance is highly region specific  a genre that works well in one state performs poorly in another.


# In[76]:


# Average of revenue per seat by movie type 
revenue=df.groupby('Movie_Type')['Revenue_per_Seat'].mean()
revenue.plot(kind='bar')
plt.title(' Average of revenue per seat by movie type ')
plt.show()


# In[ ]:


# =====> For blockbuster shows we are getting high revenue per seat 


# In[77]:


# Average of snack spend by show time 
snack = df.groupby('Show_Time')['Snack_Spend_INR'].mean()
snack.plot(kind='barh')
plt.title(' Average of snack spend by show time ')
plt.show()


# In[ ]:


#===> Night , morning and evening having more snack spend


# In[82]:


# Average of total revenue by month 
revmonth=df.groupby('Quarter')['Total_Revenue'].mean()
revmonth.plot(kind='bar')
plt.show()


# In[ ]:


#=====> Q2 generated high revenue

