#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[ ]:





# In[20]:


cs=pd.read_csv("customer_shopping_data (1).csv")


# In[21]:


cs.head()


# In[22]:


plt.figure(figsize=(10,5))
for gender in cs['gender'].unique():
    subset = cs[cs['gender'] == gender]

    plt.hist(subset['category'], alpha=1, label=gender, bins=len(subset['category'].unique()))
    plt.legend()
    plt.grid()
    plt.xlabel('Category')
    plt.ylabel('')


# In[23]:


#this will represent the most of the product that sold to the repested gender
#the in all the case female is having high amount of the product purchase
#it will also help to repesent product sell


# In[24]:


total_rev=cs.groupby('shopping_mall')['price'].sum().round()


# In[25]:


#total revenue by shopping mall , in which mall we can open the shop
plt.figure(figsize=(10,6))
total_rev.plot(kind='bar', color='red')
plt.xlabel('shopping mall')
plt.ylabel('Total revenue in lakhs')
plt.title(" Total revenue according to the shopping malls")


# In[26]:


# the above graph show the highest amount revenue of shopping mall gather till the date


# In[27]:


cs.head()


# In[28]:


# the 


# In[29]:


payments=cs.groupby('payment_method')['invoice_no'].count().round()


# In[30]:


payments.plot(kind='pie' ,autopct='%1.0f%%', explode=(0.1,0,0))
plt.legend(payments)





# In[31]:


# The above pie chart show the percentage of payments method of all shopping mall 


# In[32]:


plt.figure(figsize=(15,5))
sns.boxplot(x='shopping_mall',y='age', data=cs)
 


# In[ ]:





# In[33]:


plt.figure(figsize=(10,5))
cs.groupby('category')['invoice_no'].count().nlargest(10).plot(kind="bar" , color ='red')
plt.xlabel("Category")
plt.ylabel("Count of invoice")
plt.title(' Amount of product  sold')


# In[35]:


cs.columns


# In[36]:





# In[61]:


df1=pd.DataFrame(cs.shopping_mall.value_counts())
df1


# In[63]:


df1 = pd.DataFrame(cs.shopping_mall.value_counts())
df1.columns = ['count_of_cust']


# In[64]:


df1


# In[65]:


plt.pie(df1.count_of_cust, labels=df1.index);


# In[55]:


df1.index


# In[56]:


df1.count


# In[72]:


cs.head()


# In[73]:


import pandas as pd


# In[76]:


cs=pd.read_csv("customer_shopping_data (1).csv")


# In[77]:


cs.head()


# In[85]:


ab = pd.DataFrame(cs.category.value_counts())
ab.columns=['count_of_cate']
plt.pie(ab.count_of_cate,labels=ab.index, explode=(.1,0,0,0,0,0,0,0), autopct='%1.1f%%');
plt.title("count of each category")


# In[93]:


plt.bar(df1.index, df1.count_of_cust,)
plt.xticks(rotation=90);


# In[97]:


ab = pd.DataFrame(cs.category.value_counts())
ab.columns = ['count_of_cust']


# In[101]:


plt.bar(ab.index, ab.count_of_cust,color= ['red','black','darkred','navy','blue','violet','lime','purple'])
plt.xticks(rotation=90);


# In[107]:


sns.barplot(x=ab.index ,y=ab.count_of_cust)
plt.xticks(rotation=90);


# In[109]:


cs.head()


# In[122]:


sns.countplot(x='shopping_mall',data=cs,order=cs['shopping_mall'].value_counts().index)#always give the double cot to the column in x axis
plt.xticks(rotation=90);
#order=cs['shopping_mall'].value_counts().index=======used to sort the grap by the ase order
#this grap represnt the count of the vistors in shopping mall
#the kanyon adn mall of istanbull have the huge of vistor according to the data 


# In[127]:


sns.countplot(x='category',data=cs,order=cs['category'].value_counts().index)
#sns.countplot(cs.category) it can also be aplicable same as above one
plt.xticks(rotation=90);
#the grap represnt the count


# In[128]:


sns.countplot(x='gender',data=cs,order=cs['gender'].value_counts().index)


# In[129]:


lc=pd.read_csv("LungCapData.csv")


# In[130]:


lc.head()


# In[131]:


sns.boxplot(x='Smoke',y='LungCap' ,data=lc, hue='Gender')


# In[134]:


sns.countplot(x='shopping_mall',data=cs,order=cs['shopping_mall'].value_counts().index, hue='gender')
plt.xticks(rotation=90);


# In[141]:


plt.figure(figsize=(15,8))
sns.countplot(x='shopping_mall',data=cs,order=cs['shopping_mall'].value_counts().index, hue='category')
plt.xticks(rotation=90);
#this graphs represent the which mall have the largest amount of category sale


# In[143]:


cs.groupby('category').price.sum().round()


# In[155]:


#df3=pd.DataFrame(cs.groupby('category').price.sum().round().nlargest())
df3=pd.DataFrame(cs.groupby('category').price.sum().round())

df3


# In[151]:


plt.pie(df3.price, labels=df3.index); #it consider all 8 cate


# In[154]:


plt.pie(df3.price, labels=df3.index);# its only consider only 5


# In[157]:


plt.pie(df3.price, labels=df3.index ,autopct='%1.01f%%');


# In[158]:


df3=pd.DataFrame(cs.groupby('category').price.sum().round())
df3=df3.sort_values('price', ascending=False)
df3


# In[161]:


plt.pie(df3.price[0:4], labels=df3.index[0:4] ,autopct='%1.01f%%'); #we can also used to find out the top 4 or more


# In[164]:


df3=pd.DataFrame(cs.groupby('category').price.sum().round().nsmallest(4))
plt.pie(df3.price, labels=df3.index ,autopct='%1.01f%%');


# In[172]:


df3=pd.DataFrame(cs.groupby('category').price.sum().round().nlargest(5))
df3.columns=['price']
df3


# In[173]:


sns.barplot(x=df3.price, y=df3.index)


# In[175]:


df3=pd.DataFrame(cs.groupby('category').price.sum().round())
df3['price_in_million']=df3.price/1000000
sns.barplot(x=df3.price_in_million, y=df3.index)


# In[177]:


ist = cs[cs.shopping_mall == 'Mall of Istanbul']


# In[178]:


sns.boxplot(x = 'category' , y = 'price' , data = ist)
plt.xticks(rotation = 75)
plt.title("price range for each category")


# In[179]:


sns.displot(cs.age)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


tr=pd.read_csv("train.csv")

tr.head()
# In[4]:


tr.head()


# In[5]:


sns.displot(tr.talk_time , color='green' )


# In[6]:


sns.lineplot(x ='price_range', y ='battery_power', data=tr,marker='o') 
plt.title('clock_speed vs price range',size=6)
plt.grid()
#the price range shows the higher battery back
#price of mobile represent the battery backup


# In[1]:


import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns


# In[45]:


ad=pd.read_csv('Adidas US Sales Datasets.csv')


# In[46]:


ad.info()


# In[48]:


ad.Total Sales.value_counts()


# In[17]:


plt.pie( Units Sold, labels=Sales Method ,autopct='%1.0f%%')


# In[14]:


import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
tr=pd.read_csv("train.csv")
df1=pd.DataFrame(tr.groupby('price_range').ram.mean().round())
df1.index=['low','med','high','veryhigh']
plt.pie(df1.ram, labels=df1.index ,data=tr)


# In[18]:


sns.barplot(df1.index, df1.ram )


# In[19]:


import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
pp=pd.read_csv("Property_Price_Train.csv")


# In[26]:


aa=pd.DataFrame(pp.isnull().sum()[pp.isnull().sum()>0])
aa.columns=['nullscount']
aa=aa.sort_values('nullscount', ascending=False)


# In[27]:


aa


# In[28]:


plt.figure(figsize=(10,5))
sns.barplot(x=aa.index,y=aa.nullscount)
plt.xticks(rotation=90)


# In[29]:


ab=pd.DataFrame(pp.isnull().sum()*100/pp.shape[0])
ab.columns=['nullscount']
##ab.nullscount=ab.nullscount*100/
ab=aa.sort_values('nullscount', ascending=False)


# In[30]:


plt.figure(figsize=(10,5))
sns.barplot(x=aa.index,y=aa.nullscount)
plt.xticks(rotation=90)


# In[34]:


mm=['jan','feb','mar','apr','may','june','july','july','aug','sep']
profit=[500,700,-100,-150,300,450,550,400,430,460]


# In[43]:


col_list=[]
for i in profit:
    if i<0:
        col_list.append('red')
    else:
        col_list.append('green')
#by this we can divide the value and give the color


# In[44]:


plt.bar(mm , profit , color=col_list)
plt.xlabel('months')
plt.ylabel('profit in cr')
plt.xticks(rotation=75)
plt.axhline(y=0, color="red")#it show the axis line


# In[183]:


cs.head()


# In[191]:


payment_method_counts = cs['payment_method'].value_counts()


plt.figure(figsize=(6,6))
plt.pie(payment_method_counts, labels=payment_method_counts.index, autopct='%1.1f%%', startangle=140)


plt.title('Distribution of category of items')


plt.show()


# In[190]:


cs.groupby('gender').price.sum().round()


# In[197]:


aa=pd.DataFrame(cs.groupby('gender').price.sum().round()/1000000) #in millions


# In[198]:


aa


# In[201]:


plt.pie(aa.price, labels=aa.index, autopct='%1.1f%%', startangle=140);


# In[203]:


cs.head()


# In[214]:


#cheak the purchasing power of age group less than 30 compare to 40 to 50
ab=cs.groupby('age').price.sum().round()
ab=pd.DataFrame(ab[ab.index<=30])
#bc=pd.DataFrame(ab[ab.index>=40 and ab.index<=50])

ab


# In[2]:


import pandas as pd


# In[3]:


import plotly 


# In[4]:


import cufflinks as cf


# In[5]:


from plotly import __version__
print(__version__)


# In[6]:


import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()


# In[7]:


df=pd.read_csv("LungCapData.csv")


# In[8]:


df.head()


# In[9]:


df.iplot(kind='bar',x='Age', y= 'LungCap',mode="markers")


# In[10]:


df.iplot(kind='line',x='Age', y= 'LungCap',mode="markers")


# In[11]:


df.iplot(kind='box')


# In[12]:


cr=pd.read_csv('CreditRisk.csv')


# In[13]:


cr.head()


# In[14]:


cr.iplot(kind='box' , x='ApplicantIncome')


# In[15]:


df.iplot(kind="hist")


# In[16]:


df.LungCap.iplot(kind="hist")


# In[17]:


count_gpd=pd.DataFrame({'GDP':(19.4 ,11.8, 4.8, 3.4, 2.5, 2.4),
                       "countries": ('us', 'china', 'japna', 'germ', 'uk' , 'india')})


# In[18]:


count_gpd
import  matplotlib.pyplot as plt
import seaborn as sns


# In[19]:


count_gpd


# In[20]:


import plotly.express
plotly.express.pie(count_gpd , names="countries", values='GDP')
#never put colun  at last


# In[21]:


plotly.express.bar(count_gpd , x="countries", y='GDP')


# In[22]:


cs=pd.read_csv("customer_shopping_data (1).csv")
cs.head(1)


# In[23]:


cs['salesvalue']=cs.price*cs.quantity


# In[24]:


import plotly.express as px


# In[25]:


fig = px.sunburst(cs, path=['shopping_mall', 'category'], values='salesvalue', width=1000, height=1000)
fig.show()


# In[26]:


fig = px.treemap(cs, path=['shopping_mall', 'category'], values='salesvalue', width=1000, height=1000)
fig.show()


# In[27]:


prod_sales=pd.DataFrame()
prod_sales['Num_of_prod']=[6,14,21,18,25]
prod_sales['Sales_in_million']=[10,16.7,28,20,34]


# In[28]:


prod_sales


# In[29]:


plt.scatter(prod_sales['Num_of_prod'],prod_sales['Sales_in_million'],
            s=prod_sales['Sales_in_million']*30, color='Red', alpha=.5)


# In[30]:


#this graph is also know as the bubble graph
#here the alpha is used to give the brightness to the dot in the graph


# In[31]:


#who work on dataset and u created a many plots
#example mobile price data set, but only want to show only few graph
#all on one place
cs.head()


# In[ ]:





# In[32]:


df1=pd.DataFrame(cs.gender.value_counts())
df1.columns=['gender_count']
df2=pd.DataFrame(cs.shopping_mall.value_counts())
df2.columns=['shopping_mall_count']
df3=pd.DataFrame(cs.category.value_counts())
df3.columns=['category_count']
df4=pd.DataFrame(cs.payment_method.value_counts())
df4.columns=['payment_method']


# In[33]:


fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(10,8),constrained_layout=True)
ax[0,0].bar(df1.index,df1.gender_count,color='green')
ax[0,1].bar(df2.index,df2.shopping_mall_count)
ax[1,0].bar(df3.index,df3.category_count , color='red')
ax[1,1].bar(df4.index,df4.payment_method , color='lime')
ax[0,0].tick_params(axis='x',rotation=90)
ax[0,1].tick_params(axis='x',rotation=90)
ax[1,0].tick_params(axis='x',rotation=90)
ax[1,1].tick_params(axis='x',rotation=90)


# In[34]:


tr=pd.read_csv('train.csv')
tr.head()


# df5 = pd.DataFrame(tr.dual_sim.value_counts())
# df5.columns = ['count_of_dualsim']

# In[35]:


df5 = pd.DataFrame(tr.dual_sim.value_counts())
df5.columns = ['count_of_dualsim']
df6 = pd.DataFrame(tr.four_g.value_counts())
df6.columns = ['count_of_4G']
df7 = pd.DataFrame(tr.three_g.value_counts())
df7.columns = ['count_of_3G']
df8 = pd.DataFrame(tr.touch_screen.value_counts())
df8.columns = ['count_of_touch']
df9 = pd.DataFrame(tr.wifi.value_counts())
df9.columns = ['count_of_wifi']
df10 = pd.DataFrame(tr.blue.value_counts())
df10.columns = ['count_of_blue']


# In[36]:


fig, ax = plt.subplots(nrows = 3 , ncols = 2 , figsize = (5,8) , constrained_layout = True)
ax[0,0].bar(df5.index , df5.count_of_dualsim , color = 'darkmagenta')
ax[0,1].bar(df6.index , df6.count_of_4G , color = 'red')
ax[1,0].bar(df7.index , df7.count_of_3G , color = 'lightblue')
ax[1,1].bar(df8.index , df8.count_of_touch , color = 'lime')
ax[2,0].bar(df9.index , df9.count_of_wifi , color = 'black')
ax[2,1].bar(df10.index , df10.count_of_blue , color = 'skyblue')

ax[0,1].tick_params(axis = 'x' , rotation = 75)
ax[1,0].tick_params(axis = 'x' , rotation = 75)
ax[1,1].tick_params(axis = 'x' , rotation = 75)
ax[2,0].tick_params(axis = 'x' , rotation = 75)
ax[2,1].tick_params(axis = 'x' , rotation = 75)

ax[0,0].title.set_text("Sales based on Dual Sim")
ax[0,1].title.set_text("Sales based on 4G")
ax[1,0].title.set_text("Sales based on 3G")
ax[1,1].title.set_text("Sales based on Touch Screen")
ax[2,0].title.set_text("Sales based on Wifi")
ax[2,1].title.set_text("Sales based on Blue");


# In[ ]:




