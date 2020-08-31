#!/usr/bin/env python
# coding: utf-8

# ![title](Header__0008_2.png "Header")

# In[ ]:


# Installing missing packages.... Run it for the very first time only

# !pip install feather
# !pip install kmodes
# !pip install openpyxl


# In[1]:


# Importing the libraries
import os
import pandas as pd
from pandas import DataFrame
import numpy as np
import warnings
import openpyxl
import feather
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
matplotlib.pyplot.boxplot
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


import scipy.stats as stats
import math


# In[ ]:


# Setting working directory
wd_path = r'D:\ABI Office Work\POC Insights\Korea SOT Analysis\EDA'
os.chdir(wd_path)


# In[2]:


# Setting working directory
os.getcwd()
pd.set_option('display.max_columns',50)
path = 'D:/ABI Office Work/POC Insights/Korea WS Analysis'

pd.set_option('display.max_columns',500)

# Reading Data
Data1 = pd.read_csv(path+'/Data/sellin_201901_03.csv', sep=';' ,header=None)
Data2 = pd.read_csv(path+'/Data/sellin_201904_06.csv', sep=';' ,header=None)
Data3 = pd.read_csv(path+'/Data/sellin_2019_0708.csv', sep=';' ,header=None)
Data4 = pd.read_csv(path+'/Data/sellin_2019_06.csv', sep=';' ,header=None)
Data5 = pd.read_csv(path+'/Data/sellin_201801_03.csv', sep=';' ,header=None)
Data6 = pd.read_csv(path+'/Data/sellin_201804_06.csv', sep=';' ,header=None)
Data7 = pd.read_csv(path+'/Data/sellin_201807_09.csv', sep=';' ,header=None)
Data8 = pd.read_csv(path+'/Data/sellin_201810_12.csv', sep=';' ,header=None)

#Data18 = pd.DataFrame(columns=colname)

colname = ['BAR_ID','BIZNUMBER','CUSTOMERNAME','LIQUORTYPE','MLIQUORTYPE',
           'MPRODUCT','MPURCHASER','PRODUCTNAME','REVENUE','YEARMONTH',
           'BIZNUMBERX', 'PRODUCT','VOLUME_1L', 'package','SKU_ID',
           'SKU_COUNT','OUTLETCODE']

df_list = [Data1, Data2, Data3, Data4, Data5, Data6, Data7, Data8]

#Data = pd.DataFrame()
for df in df_list:
    df.columns = colname

Data2 = Data2[(Data2['YEARMONTH'] < 201906)]

# Merging all datasets
Data = pd.concat([Data1,Data2,Data3,Data4,Data5,Data6,Data7,Data8])

# Removing -ive sales
Data = Data[(Data['VOLUME_1L'] >0)]
Data['REVENUE'] = np.where(Data['REVENUE']<0 ,0,Data['REVENUE'])

######************************************************************************
## ***************************************************************************

# Getting Maker name : MPURCHASER  data
MPurchase = pd.read_csv(path+'/Data/MPURCHASER_CODE.csv')
Data['MPURCHASER'].astype
Data['MPURCHASER'].fillna(0,inplace=True)
Data['MPURCHASER'].astype(int)
# Merging Purchaser data
Data = Data.merge(MPurchase,right_on ='MPURCHASER_CDE',
                          left_on='MPURCHASER', how='left')
# Removing reduntant columns
Data.drop('MPURCHASER_CDE', axis = 1, inplace=True)

newData = Data

newData['LIQUORTYPE'].fillna(0,inplace=True)
newData['LIQUORTYPE']=np.where((newData['LIQUORTYPE']== '#REF!'),
       100000,newData['LIQUORTYPE'])
newData['LIQUORTYPE'] = pd.to_numeric(newData['LIQUORTYPE'], 
       downcast='integer') 
newData['LIQUORTYPE'].unique()

newData['MLIQUORTYPE'].fillna(0,inplace=True)
newData['MLIQUORTYPE'] = pd.to_numeric(newData['MLIQUORTYPE'], 
       downcast='integer') 
newData['MLIQUORTYPE'].unique()

others = [1,2,3,4,6,13]
soju = [7,8]
spirit = [9,10,11,12]
mliqtp = [100,400,200,300]

newData['category_name'] = np.where(
        (newData['LIQUORTYPE'] >=1) & (newData['LIQUORTYPE'] <=13),
        np.where(newData['LIQUORTYPE'].isin(others),'Others',
                 np.where(newData['LIQUORTYPE']==5,'Beer',
                          np.where(newData['LIQUORTYPE']==6,'Wine',
                                   np.where(newData['LIQUORTYPE'].isin(soju),'Soju',
                                            np.where(newData['LIQUORTYPE'].isin(spirit),'Spirit','NAB'))))),
           np.where(newData['MLIQUORTYPE'].isin(mliqtp),
                     np.where(newData['MLIQUORTYPE']==400,'Others',
                              np.where(newData['MLIQUORTYPE'] == 100, 'Beer',
                                       np.where(newData['MLIQUORTYPE']==200,'Soju',
                                                np.where(newData['MLIQUORTYPE']==300,'Spirit','NAB')))),'NAB'))
   
newData['is_comp'] = np.where(newData['Maker_NM']=='OBC',0,1)

newData['is_beer'] = np.where(newData['category_name'] == 'Beer', 0 ,1 )

# ******* New definition of SKU  ****** 

newData['MPRODUCT'] = newData['MPRODUCT'].fillna(0)
false_mproduct = [0, 'nan', ' ',]
newData['MPRODUCT'] = np.where(newData['MPRODUCT'].isin(false_mproduct), newData['PRODUCTNAME'], newData['MPRODUCT'] )
newData['Maker_NM'] = newData['Maker_NM'].fillna("No manufacturer available", inplace = True)


newData.drop(['SKU_ID','SKU_COUNT'],axis=1, inplace=True)
# newData['MPRODUCT'] = Data['MPRODUCT'].astype(int)
newData['SKU'] = newData['MPRODUCT'].map(str) + "_" + newData['package']


########################### LINE BELOW FOR PURPOSE 
# Counting SKU by brand

sell_in_data['SKU_ID'] = sell_in_data['MPRODUCT'].astype(str).str.cat(sell_in_data['package'], sep ='_')
data_sku = sell_in_data.groupby(['BAR_ID', 'YEARMONTH','MPRODUCT'])['SKU_ID'].nunique().reset_index()
colnames_3 = ['BAR_ID','YEARMONTH','MPRODUCT','BRANDSKU']
data_sku.columns = colnames_3

sell_in_data = pd.merge(sell_in_data, data_sku, on = ['BAR_ID','YEARMONTH','MPRODUCT'])

########### Calculating SKU at POC level


data_sku_all = sell_in_data.groupby(['BAR_ID', 'YEARMONTH'])['SKU_ID'].nunique().reset_index()
colnames_4 = ['BAR_ID','YEARMONTH','Total_SKU_COUNT']
data_sku_all.columns = colnames_4

sell_in_data = pd.merge(sell_in_data, data_sku_all, on = ['BAR_ID','YEARMONTH'])

sell_in_data['RelDistribution'] = sell_in_data['BRANDSKU'] / sell_in_data['Total_SKU_COUNT']


# In[3]:


data_2019_july_aug=pd.read_csv('C:/Users/40102156/Korea_SOT_Analysis/Sell_in_data/sellin_july_august_2019/SellInData_julyAug2019/SellInData_julyAug2019.csv',sep=",")


# In[4]:


#taking only data till june from this dataset
data_2019=data_2019[data_2019['YEARMONTH']<=201906]


# In[5]:


#concating the data for all the years
base_data=pd.concat([data_2018,data_2019,data_2019_july_aug])
del data_2018
del data_2019
del data_2019_july_aug
import gc
gc.collect()


# In[6]:


#taking all the channels except off trade
base_data1=base_data[base_data['OUTLETCODE']!='K4']
base_data1.head()


# In[7]:


#reading product name mapping for all the beers
product_mapping= pd.read_csv('C:/Users/40102156/Korea_SOT_Analysis/MPRODUCT_CODE.csv')
product_mapping.head()


# In[8]:


#adding the brand name to the base data
base_data2=base_data1.merge(product_mapping[['MPRODUCT','cass_flag','focus_brands','terra_flag']],on='MPRODUCT',how='left')
base_data2['cass_flag'].fillna(0,inplace= True)
base_data2['focus_brands'].fillna(0,inplace= True)
base_data2['terra_flag'].fillna(0,inplace= True)
base_data2.head()


# In[9]:


#adding quarter column based on year month
base_data2['quarter']=np.where((base_data2.YEARMONTH>=201801) & (base_data2.YEARMONTH<=201803) ,201801,0)
base_data2['quarter']=np.where((base_data2.YEARMONTH>=201804) & (base_data2.YEARMONTH<=201806) ,201802,base_data2.quarter)
base_data2['quarter']=np.where((base_data2.YEARMONTH>=201807) & (base_data2.YEARMONTH<=201809) ,201803,base_data2.quarter)
base_data2['quarter']=np.where((base_data2.YEARMONTH>=201810) & (base_data2.YEARMONTH<=201812) ,201804,base_data2.quarter)
base_data2['quarter']=np.where((base_data2.YEARMONTH>=201901) & (base_data2.YEARMONTH<=201903) ,201901,base_data2.quarter)
base_data2['quarter']=np.where((base_data2.YEARMONTH>=201904) & (base_data2.YEARMONTH<=201906) ,201902,base_data2.quarter)
base_data2['quarter']=np.where((base_data2.YEARMONTH>=201907) & (base_data2.YEARMONTH<=201909) ,201903,base_data2.quarter)


#     ##################### DEEPESH ADDITION ###################################

# In[2]:


base_data2 = pd.read_csv(r'D:\ABI Office Work\POC Insights\Korea WS Analysis\Data\base_data2.csv')
base_data2.head()


# In[3]:


#Finding the ABI SOT for each PoC (across POC type)
grouping=base_data2[(base_data2['quarter']==201902) & (base_data2['category_name']=='Beer') & (base_data2['VOLUME_1L']>0)].groupby(['OUTLETCODE','BAR-ID','Maker_NM'])['VOLUME_1L'].sum().reset_index(name='bar_maker_beer_vol')
grouping_rollup=grouping.groupby(['OUTLETCODE','BAR-ID'])['bar_maker_beer_vol'].sum().reset_index(name='bar_beer_vol')
grouping1=grouping.merge(grouping_rollup,on=['OUTLETCODE','BAR-ID'],how='left')
grouping1['ABI_SoT']=grouping1['bar_maker_beer_vol']/grouping1['bar_beer_vol']
grouping1.head()


# In[4]:


#Finding the Beer SOT for each PoC (across POC type)
beer_SOT=base_data2[(base_data2['quarter']==201902) & (base_data2['VOLUME_1L']>0)].groupby(['OUTLETCODE','BAR-ID','category_name'])['VOLUME_1L'].sum().reset_index(name='bar_catg_vol')
beer_SOT_rollup=beer_SOT.groupby(['OUTLETCODE','BAR-ID'])['bar_catg_vol'].sum().reset_index(name='bar_vol')
beer_SOT1=beer_SOT.merge(beer_SOT_rollup,on=['OUTLETCODE','BAR-ID'],how='left')
beer_SOT1['Beer_SoT']=beer_SOT1['bar_catg_vol']/beer_SOT1['bar_vol']
beer_SOT1.head()


# In[5]:


#Combining the beer and ABI SOT
bar_universe=base_data2[base_data2['quarter']==201902]
bar_universe=bar_universe[['OUTLETCODE','BAR-ID']].drop_duplicates()
#removing POCs which have multiple outlet code
cnt_outlets=bar_universe.groupby('BAR-ID').size().reset_index(name='cnt')
print (cnt_outlets.groupby('cnt').size())
inc_list=cnt_outlets[cnt_outlets['cnt']==1]
bar_universe=bar_universe.merge(inc_list[['BAR-ID']],on='BAR-ID',how='inner')
ABI_SOT=grouping1[grouping1['Maker_NM']=='OBC']
beer_SOT2=beer_SOT1[beer_SOT1['category_name']=='Beer']
bar_universe1=bar_universe.merge(ABI_SOT,on=['OUTLETCODE','BAR-ID'],how='left')
bar_universe2=bar_universe1.merge(beer_SOT2,on=['OUTLETCODE','BAR-ID'],how='left')
bar_universe2['ABI_SoT'].fillna(0,inplace=True)
bar_universe2['Beer_SoT'].fillna(0,inplace=True)
bar_universe2.head()


# In[6]:


bar_universe2 = bar_universe2[bar_universe2.bar_maker_beer_vol >=0 ]
bar_universe2 = bar_universe2[bar_universe2.bar_beer_vol >=0 ]

bar_universe2.loc[bar_universe2['BAR-ID']=='211001-00000001258']


# In[7]:


# Deleting files to release memory
import gc
del bar_universe
del beer_SOT
del beer_SOT_rollup
del beer_SOT1
# del base_data2
gc.collect 
bar_universe = pd.DataFrame()
beer_SOT = pd.DataFrame()
beet_SOT_rollup = pd.DataFrame()
beer_SOT1 = pd.DataFrame()


# ##  What does K1,K2,K3 distribution signify? What is the distribution of volume across these restaurants? 

# In[8]:


### Finding distrubution of K1,K2,K3,K4 restaurants
K_distribution = pd.DataFrame(bar_universe2.OUTLETCODE.value_counts())
K_distribution.columns = ['Count']
K_distribution['Total_Count'] =  bar_universe2.OUTLETCODE.value_counts().sum()
K_distribution['Distribution perc'] = K_distribution['Count'] / K_distribution['Total_Count']

K_distribution['OUTLETCODE']= K_distribution.index

group2_data = bar_universe2[['OUTLETCODE','bar_maker_beer_vol','bar_beer_vol']].groupby(['OUTLETCODE'],
                            as_index=False).agg({'bar_maker_beer_vol':['sum'],
                                                 'bar_beer_vol': ['sum']})
group2_data.columns = ['OUTLETCODE','bar_maker_beer_vol','bar_beer_vol']
group2_data['Total maker Vol'] = group2_data.bar_maker_beer_vol.sum()
group2_data['Total beer Vol'] = group2_data.bar_beer_vol.sum()

group2_data['Total maker Vol perc'] = group2_data['bar_maker_beer_vol'] / group2_data['Total maker Vol']
group2_data['Total beer Vol perc'] = group2_data['bar_beer_vol'] / group2_data['Total beer Vol']

K_distribution = K_distribution.merge(group2_data, on = 'OUTLETCODE')
K_distribution = K_distribution.iloc[:,[3,0,1,2,4,5,8,9]]
K_distribution['MS'] = K_distribution['bar_maker_beer_vol']/ K_distribution['bar_beer_vol']
K_distribution.head()


# ## Univariate Plot: Checking outlier, distribution of variables

# In[29]:


# x = bar_universe2['bar_maker_beer_vol']
# fig, axs = plt.subplots(nrows=1, ncols=3)
def plot_histogram (dataset, colindex):
    K1_data=dataset[dataset['OUTLETCODE']=='K1']
    print('Analysis for',K1_data.columns[colindex])
    print('\n',pd.cut(K1_data.iloc[:,colindex],5).value_counts().sort_index())
    plt.title("Histogram with 'auto' bins")
    plt.hist(K1_data.iloc[:,colindex], bins=5)
    plt.show()
    plt.title("Line plot with unique values")
    print('Line plot of',K1_data.columns[colindex])
    plt.plot(K1_data.iloc[:,colindex].unique())
    threshold = K1_data.iloc[:,colindex].unique().mean()
    plt.axhline(y=threshold,linewidth=1, color='red')
    plt.show()
    df = pd.DataFrame(K1_data.iloc[:,colindex].unique())
    df.boxplot()

plot_histogram(bar_universe2,7)


# # OUTLIER TREATMENT

# In[10]:


# x = bar_universe2['bar_maker_beer_vol']
# dynamic filters --- should come automatically

K3_data=bar_universe2[bar_universe2['bar_maker_beer_vol'] < 1000] # 1000 dynamic 
print(pd.cut(K3_data['bar_maker_beer_vol'],4).value_counts().sort_index())
plt.hist(K3_data['bar_maker_beer_vol'], bins=4)


# In[11]:


K3_data.columns


# ### Multicollinearity Check
# #### Looking for multicollinearity among variables

# In[11]:


import seaborn as sns
df = K3_data.iloc[:,[0,3,4,5,7,8,9]]
sns.pairplot(df, hue="OUTLETCODE")


# In[24]:


# sns.heatmap(df.iloc[:,1:6], annot = True)
df = K3_data.iloc[:,[0,3,4,5,7,8,9]]


# ### Finding range to define High Beer SOT and High ABI SOT

# In[28]:


# df.describe(include = ['ABI_SoT'])
df2 = df.iloc[:,[3,6]]
def_percentiles = [.25,.33,.5,.75,.9,1]
df2.describe (percentiles = def_percentiles)


# In[139]:


import warnings
warnings.filterwarnings("ignore")
#Getting the High, medium, low buckets for ABI and Beer SoT
K1_data=bar_universe2[bar_universe2['OUTLETCODE']=='K1']
print (K1_data.shape)

ABI_sot_33=K1_data['ABI_SoT'].quantile(0.33)
print ('33rd percentile=',ABI_sot_33)
ABI_sot_50=K1_data['ABI_SoT'].quantile(0.50)
print ('50th percentile=',ABI_sot_50)
# K1_data['ABI_SoT_grp']=np.where(K1_data['ABI_SoT']<ABI_sot_33,'Low','NA')
# K1_data['ABI_SoT_grp']=np.where((K1_data['ABI_SoT']>=ABI_sot_33) & (K1_data['ABI_SoT']<ABI_sot_66) ,'Med',K1_data.ABI_SoT_grp)
K1_data['ABI_SoT_grp']=np.where(K1_data['ABI_SoT']<ABI_sot_50 ,'High','Low')
K1_data.head()


# In[140]:


dist1=K1_data.groupby('ABI_SoT_grp').size().reset_index(name="cnt_POCs")
print ("Distribution of POCs basis ABI SoT =",'\n',dist1)


# In[186]:


import warnings
warnings.filterwarnings("ignore")
#Beer SoT

beer_sot_33=K1_data['Beer_SoT'].quantile(0.33)
print ('33rd percentile=',beer_sot_33)
beer_sot_66=K1_data['Beer_SoT'].quantile(0.66)
print ('66th percentile=',beer_sot_66)
K1_data['beer_SoT_grp']=np.where(K1_data['Beer_SoT']<=beer_sot_33,'Low','NA')
K1_data['beer_SoT_grp']=np.where((K1_data['Beer_SoT']>beer_sot_33) & (K1_data['Beer_SoT']<=beer_sot_66) ,'Med',K1_data.beer_SoT_grp)
K1_data['beer_SoT_grp']=np.where(K1_data['Beer_SoT']>beer_sot_66 ,'High',K1_data.beer_SoT_grp)
K1_data.head()


# In[187]:


dist=K1_data.groupby('beer_SoT_grp').size().reset_index(name="cnt_POCs")
print ("Distribution of POCs basis Beer SoT =",'\n',dist)

print('\n')
dist3=K1_data.groupby(['beer_SoT_grp','ABI_SoT_grp']).size().reset_index(name="cnt_POCs")
dist3=dist3.pivot_table(index=['beer_SoT_grp'], columns='ABI_SoT_grp', values='cnt_POCs').fillna(0).reset_index()
print ("Distribution of POCs basis Beer SoT and ABI SoT =",'\n',dist3)


# In[188]:


#Getting the IVs for all the PoCs

#Pricing of terra
PTR_terra=base_data2[(base_data2['terra_flag']==1) & (base_data2['quarter']==201902) ].groupby('BAR-ID')['VOLUME_1L','REVENUE'].sum().reset_index().rename(columns={'VOLUME_1L':'terra_vol','REVENUE':'terra_revenue'
              })
PTR_terra['terra_PTR']=PTR_terra['terra_revenue']/PTR_terra['terra_vol']

PTR_cass=base_data2[(base_data2['cass_flag']==1) & (base_data2['quarter']==201902) ].groupby('BAR-ID')['VOLUME_1L','REVENUE'].sum().reset_index().rename(columns={'VOLUME_1L':'cass_vol','REVENUE':'cass_revenue'
              })
PTR_cass['cass_PTR']=PTR_cass['cass_revenue']/PTR_cass['cass_vol']

bar_universe3=bar_universe2.merge(PTR_terra[['BAR-ID','terra_PTR']],on='BAR-ID',how='left')
bar_universe3=bar_universe3.merge(PTR_cass[['BAR-ID','cass_PTR']],on='BAR-ID',how='left')
bar_universe3['cass_terra_PTR_index']=bar_universe3['cass_PTR']/bar_universe3['terra_PTR']

bar_universe3=bar_universe3.replace([np.inf, -np.inf], np.nan)
bar_universe3.fillna(0,inplace=True)


# In[189]:


#RD of terra and cass
#Adding the SKU cnt(terra vs overall) in the bar
SKU_def=base_data2[(base_data2['category_name']=='Beer')]
SKU_def=SKU_def[['BAR-ID','quarter','YEARMONTH','MPRODUCT','package']].drop_duplicates()
SKU_beer=SKU_def[(SKU_def['quarter']==201902)].groupby(['BAR-ID','YEARMONTH']).size().reset_index(name='SKU_all_beer')
SKU_beer.head()
# SKU_beer=SKU_beer.groupby(['BAR-ID','YEARMONTH'])['SKU_all_beer'].mean().reset_index()
# SKU_beer=SKU_beer.pivot_table(index=['BAR-ID'], columns='quarter', values='SKU_all_beer').fillna(0).reset_index()
# SKU_beer.columns=['BAR-ID','SKU_beer_02','SKU_beer_03']

# bar_universe4=bar_universe3.merge(SKU_beer[['BAR-ID','SKU_all_beer']],on=['BAR-ID'],how='left')
# bar_universe4.head()


# In[190]:


#Getting the SKU cnt for terra and RD of terra and cass
#Terra 
SKU_terra=base_data2[base_data2['terra_flag']==1]
SKU_terra=SKU_terra[['BAR-ID','quarter','YEARMONTH','MPRODUCT','package']].drop_duplicates()
SKU_terra=SKU_terra[(SKU_terra['quarter']==201902)].groupby(['BAR-ID','YEARMONTH']).size().reset_index(name='SKU_terra')
SKU_terra.head()

#Cass
SKU_cass=base_data2[base_data2['cass_flag']==1]
SKU_cass=SKU_cass[['BAR-ID','quarter','YEARMONTH','MPRODUCT','package']].drop_duplicates()
SKU_cass=SKU_cass[(SKU_cass['quarter']==201902)].groupby(['BAR-ID','YEARMONTH']).size().reset_index(name='SKU_cass')


SKU_beer1=SKU_beer.merge(SKU_terra,on=['BAR-ID','YEARMONTH'],how='left')
SKU_beer2=SKU_beer1.merge(SKU_cass,on=['BAR-ID','YEARMONTH'],how='left')
SKU_beer2.fillna(0,inplace=True)
SKU_beer2['RD_cass']=SKU_beer2['SKU_cass']/SKU_beer2['SKU_all_beer']
SKU_beer2['RD_terra']=SKU_beer2['SKU_terra']/SKU_beer2['SKU_all_beer']
SKU_beer3=SKU_beer2.groupby('BAR-ID')['SKU_all_beer','SKU_terra','SKU_cass','RD_cass','RD_terra'].mean().reset_index()
print (SKU_beer3.head())

bar_universe5=bar_universe3.merge(SKU_beer3,on='BAR-ID',how='left')

# bar_universe5=bar_universe5.replace([np.inf, -np.inf], np.nan)
# bar_universe5.fillna(0,inplace=True)

bar_universe5.head()


# In[198]:


#adding the credit card data
credit_card= pd.read_csv('C:/Users/40102156/Korea_SOT_Analysis/Credit_Card/CreditCard2019_jan_to_aug/CreditCard2019_jan_to_aug.csv')

#excluding the duplicates at poc level
cc_include=credit_card[credit_card['QUARTER']==2].groupby('poc_id').size().reset_index(name='cnt')
cc_include=cc_include[cc_include['cnt']==1]
credit_card_Q2=credit_card[credit_card['QUARTER']==2].merge(cc_include[['poc_id']],on='poc_id',how='inner')
#k[k['cnt']==2] #211005-00000001097,211005-00000001402,211005-00000008493
#credit_card.loc[credit_card['poc_id']=='211005-00000001402']
print (credit_card_Q2.shape)
credit_card.columns


# In[199]:


bar_universe6=bar_universe5.merge(credit_card_Q2[['poc_id', 'YYYYMM', 'PMS_POC_ID', 'YYYY', 'QUARTER', 'DAY_07_YN',
       'DAY_30_YN', 'DAY_90_YN', 'RANK_1_AMT_3M', 'RANK_2_AMT_3M',
       'RANK_3_AMT_3M', 'RANK_4_AMT_3M', 'RANK_5_AMT_3M', 'IAD_RT_AMT_3M',
       'RANK_1_AMT_1M', 'RANK_2_AMT_1M', 'RANK_3_AMT_1M', 'RANK_4_AMT_1M',
       'RANK_5_AMT_1M', 'IAD_RT_AMT_1M', 'RANK_1_CNT_3M', 'RANK_2_CNT_3M',
       'RANK_3_CNT_3M', 'RANK_4_CNT_3M', 'RANK_5_CNT_3M', 'IAD_RT_CNT_3M',
       'RANK_1_CNT_1M', 'RANK_2_CNT_1M', 'RANK_3_CNT_1M', 'RANK_4_CNT_1M',
       'RANK_5_CNT_1M', 'IAD_RT_CNT_1M', 'AVG_AMT_3M', 'AVG_AMT_1M', 'WK_RT',
       'WE_RT', 'TIME_0510_RT', 'TIME_1114_RT', 'TIME_1517_RT', 'TIME_1819_RT',
       'TIME_2021_RT', 'TIME_2224_RT', 'TIME_0104_RT', 'PERS_RT', 'CORP_RT',
       'F20_RT', 'F30_RT', 'F40_RT', 'F50_RT', 'F60_RT', 'M20_RT', 'M30_RT',
       'M40_RT', 'M50_RT', 'M60_RT', 'REVISIT_RT', 'WRK_DT', 'CD_GRADE',
       'TP_GRADE_LEVEL', 'CNT_GRADE_STD', 'AMT_ESTIMATE_SALES', 'TP_VOLUME',
       'VOLUME_BEER_OB', 'VOLUME_BEER_OTHERS', 'VOLUME_SOJU', 'VOLUME_OTHERS',
       'SOT_BEER_OB', 'SOT_BEER_OTHERS', 'SOT_SOJU', 'SOT_OTHERS',
       'VOLUME_TOTAL', 'SOT_BEER', 'SOT_OB']],left_on='BAR-ID',right_on='poc_id',how='inner')
#gender
bar_universe6['male_ratio']=bar_universe6['M20_RT']+bar_universe6['M30_RT']+bar_universe6['M40_RT']+bar_universe6['M50_RT']+bar_universe6['M60_RT']
bar_universe6['female_ratio']=bar_universe6['F20_RT']+bar_universe6['F30_RT']+bar_universe6['F40_RT']+bar_universe6['F50_RT']+bar_universe6['F60_RT']
#age
bar_universe6['age_20_to_40']=bar_universe6['F20_RT']+bar_universe6['F30_RT']+bar_universe6['F40_RT']+bar_universe6['M20_RT']+bar_universe6['M30_RT']+bar_universe6['M40_RT']
bar_universe6['age_50_to_60']=bar_universe6['F50_RT']+bar_universe6['F60_RT']+bar_universe6['M50_RT']+bar_universe6['M60_RT']

bar_universe6=bar_universe6.replace([np.inf, -np.inf], np.nan)
bar_universe6.fillna(0,inplace=True)

bar_universe6.head()


# In[200]:


#Creating variables for matching test and control pairs

# #New vs Old POC (wrt to Beer)
bar_association_beer=base_data2[['BAR-ID','YEARMONTH','category_name']].drop_duplicates()
bar_association_beer=bar_association_beer[bar_association_beer['category_name']=='Beer']
bar_association_beer=bar_association_beer.sort_values(['BAR-ID','YEARMONTH'])
bar_association_beer['months_of_business_beer'] = bar_association_beer.groupby(by=['BAR-ID'])['YEARMONTH'].transform(lambda x: x.rank())
bar_beer1=bar_association_beer[bar_association_beer['YEARMONTH']<=201906].groupby('BAR-ID')['YEARMONTH'].max().reset_index()
bar_association_beer1=bar_association_beer.merge(bar_beer1,on=['BAR-ID','YEARMONTH'],how='inner')
print (bar_association_beer1.head(),'\n')

#adding the above to the base data
bar_universe7=bar_universe6.merge(bar_association_beer1[['BAR-ID','months_of_business_beer']],on=['BAR-ID'],how='left')
bar_universe7.head()


# In[210]:


#adding region as categorical variable
#adding regions as one hot encoding
poc_region_map=base_data2[['region(EN)','BAR-ID']].drop_duplicates()
bar_universe7['Soju_other_catg_vol']=bar_universe7['bar_vol']-bar_universe7['bar_beer_vol']
bar_universe8=bar_universe7.merge(poc_region_map,on=['BAR-ID'],how='left')

poc_channel1= pd.read_csv('C:/Users/40102156/Korea_SOT_Analysis/POC_BRN_Type.csv')
poc_channel2= pd.read_csv('C:/Users/40102156/Korea_SOT_Analysis/TBD_POC_201901_03.csv',sep=';')
poc_channel2=poc_channel2.merge(poc_channel1[['pms_poc_id','POC_Channel']],on='pms_poc_id',how='inner')
bar_universe8=bar_universe8.merge(poc_channel2[['BAR-ID','POC_Channel']],on=['BAR-ID'],how='left')
bar_universe8['POC_Channel'].fillna('Others',inplace=True)

# one_hot = pd.get_dummies(poc_region_map['region(EN)'])
# poc_region_map = poc_region_map.drop('region(EN)',axis = 1)
# poc_region_map = poc_region_map.join(one_hot)
# bar_universe8=bar_universe8.merge(poc_region_map,on=['BAR-ID'],how='left')

#Adding the Volume of Soju (All alcoholic beverages - Beer Vol)


bar_universe8.head()


# poc_region_map.head()


# In[211]:


bar_universe8.columns


# In[234]:


#Getting High Beer SoT and Low ABI SoT
K1_data2=bar_universe8.merge(K1_data[['BAR-ID','ABI_SoT_grp','beer_SoT_grp']],on='BAR-ID',how='inner')

K1_high_low=K1_data2[(K1_data2['beer_SoT_grp']=='High') & (K1_data2['ABI_SoT_grp']=='Low')]
K1_high_low['TC_flag']=0
print('Size of Control POCs - High-Low',K1_high_low.shape)

K1_high_high=K1_data2[(K1_data2['beer_SoT_grp']=='High') & (K1_data2['ABI_SoT_grp']=='High')]
K1_high_high['TC_flag']=1
print('Size of Test POCs - High-High',K1_high_high.shape)

K1_hl_to_hh=pd.concat([K1_high_high,K1_high_low])


K1_hl_to_hh=K1_hl_to_hh.replace([np.inf, -np.inf], np.nan)
K1_hl_to_hh.fillna(0,inplace=True)

K1_hl_to_hh.head()


# In[87]:


# importing all packages
import sys
import time
get_ipython().system('pip install kmodes')
# !pip install kprototypes
from   kmodes import kmodes
from   kmodes import kprototypes
import os
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from IPython.display import display


# In[235]:


#Drop unnecesary columns

K1_hl_to_hh1=K1_hl_to_hh.drop(columns = ['OUTLETCODE','BAR-ID','Maker_NM','bar_maker_beer_vol','bar_beer_vol','ABI_SoT','category_name',
                                    'Beer_SoT','cass_PTR','poc_id','YYYYMM','PMS_POC_ID','YYYY','QUARTER',
                        'DAY_07_YN','DAY_30_YN','DAY_90_YN','RANK_1_AMT_3M','RANK_2_AMT_3M','RANK_3_AMT_3M',
                        'RANK_4_AMT_3M','RANK_5_AMT_3M','IAD_RT_AMT_3M','RANK_1_AMT_1M','RANK_2_AMT_1M',
                        'RANK_3_AMT_1M','RANK_4_AMT_1M','RANK_5_AMT_1M','IAD_RT_AMT_1M','RANK_1_CNT_3M',
                        'RANK_2_CNT_3M','RANK_3_CNT_3M','RANK_4_CNT_3M','RANK_5_CNT_3M','IAD_RT_CNT_3M',
                        'RANK_1_CNT_1M','RANK_2_CNT_1M','RANK_3_CNT_1M','RANK_4_CNT_1M','RANK_5_CNT_1M',
                        'IAD_RT_CNT_1M','F20_RT','F30_RT','F40_RT','F50_RT','F60_RT','M20_RT','M30_RT',
                        'M40_RT','M50_RT','M60_RT','TIME_0510_RT','TIME_1114_RT','TIME_1517_RT','TIME_1819_RT',
                        'TIME_2021_RT','TIME_2224_RT','TIME_0104_RT','WRK_DT','CD_GRADE','TP_GRADE_LEVEL',
                        'CNT_GRADE_STD','TP_VOLUME','VOLUME_BEER_OB','VOLUME_BEER_OTHERS',
                        'VOLUME_OTHERS','SOT_BEER_OB','SOT_BEER_OTHERS','SOT_SOJU','SOT_OTHERS','VOLUME_TOTAL',
                        'SOT_BEER','SOT_OB','male_ratio','female_ratio','age_20_to_40','age_50_to_60','VOLUME_SOJU',
                        'bar_catg_vol','bar_vol'])


K1_hl_to_hh1.columns.to_list()


# In[236]:


## Update this setting before running the model

DEBUG = 2                                  # set to 1 to debug, 2 for more
verbose = 0                               # kmodes debugging
# beer_sot = ['High']                       # Running Analysis for High-High and High-Low analysis

#These are the "categorical" fields in CSV

categorical_field_names = [17,18]
numerical_field_names = list(range(17))
# numerical_field_names = list(range(1,20))

init = 'Huang'                    # init can be 'Cao', 'Huang' or 'random'
n_clusters = 8                   # how many clusters (hyper parameter)
max_iter = 100                    # default 100


# In[237]:


### strip whitespace (should get this done at export time)

K1_hl_to_hh1.rename(columns=lambda x: x.strip(), inplace = True)

### Drop NA and NaN values
K1_hl_to_hh1 =K1_hl_to_hh1.replace([np.inf, -np.inf], np.nan)
K1_hl_to_hh1.fillna(0,inplace=True)

# ### Ensure things are dtype="category" (cast)

# for c in categorical_field_names:
#     K1_hl_to_hh[c] = K1_hl_to_hh[c].astype('category')


# In[238]:


# QCing if only numerica fileds getting selected
K1_hl_to_hh1.iloc[:,numerical_field_names].head(2)


# In[239]:


# QCing if only categorical fileds getting selected
K1_hl_to_hh1.iloc[:,categorical_field_names].head(2)


# In[242]:


## Scalling the data. We have two methods written down.

# df = scaler.fit_transform(df.iloc[:,0:19])   #Keep this line commented if using below method
K1_hl_to_hh_vars=K1_hl_to_hh1.copy()
K1_hl_to_hh_vars.iloc[:,numerical_field_names] =K1_hl_to_hh_vars.iloc[:,numerical_field_names].apply(lambda x: (x - x.mean()) / np.std(x))
K1_hl_to_hh_vars=K1_hl_to_hh_vars.iloc[:,:19]
K1_hl_to_hh_vars.head()


# In[243]:


### Get the model
data_cats_matrix =K1_hl_to_hh_vars.as_matrix()
kproto = kprototypes.KPrototypes(n_clusters=n_clusters,init=init,verbose=verbose)
clusters = kproto.fit_predict(data_cats_matrix,categorical=categorical_field_names)

# Adding the predicted clusters to the main dataset
# K1_hl_to_hh['cluster_id'] = clusters
# print (K1_hl_to_hh.head())

# # Checking the clusters created
# datasetdf = pd.DataFrame(df['cluster_id'].value_counts())
# print (datasetdf)

# print("\n")
# print('Cluster cost = ',kproto.cost_)

# # df.to_csv('ClusteringData_V2.csv', index= False, encoding = 'utf_8_sig')

# print("\n")


# In[244]:


#Getting the distribution of POCs across clusters
K1_hl_to_hh=K1_hl_to_hh.reset_index()
cluster_map1=pd.DataFrame(clusters,columns=['cluster'])
print (cluster_map1.groupby('cluster').size())
K1_hl_to_hh_cluster=K1_hl_to_hh.join(cluster_map1)
K1_hl_to_hh_cluster.head()


# In[250]:


cluster_dist=K1_hl_to_hh_cluster.groupby('cluster').size().reset_index(name='cnt_of_POCs')
sns.barplot(x=cluster_dist['cluster'], y=cluster_dist['cnt_of_POCs'])


# In[251]:


print (K1_hl_to_hh_cluster.shape)
print (K1_hl_to_hh_cluster.groupby('TC_flag').size())
print (K1_hl_to_hh_cluster.groupby('cluster').size())
K1_hl_to_hh_cluster.groupby(['cluster','TC_flag']).size().reset_index()


# In[252]:


# #silhouette width 
# from sklearn.metrics import silhouette_score
# silhouette_score(K1_hl_to_hh_features,cluster_map, metric='euclidean')


# In[253]:


# ******************************************** Cluster Profiling ****************************************************


# In[256]:


#Comparing cluster 2 and 4
K1_hl_to_hh_cluster2_4=K1_hl_to_hh_cluster[(K1_hl_to_hh_cluster['cluster']==2) | (K1_hl_to_hh_cluster['cluster']==4)]
print ('shape of cluster=',K1_hl_to_hh_cluster2_4.shape)
K1_hl_to_hh_cluster2_4.groupby('cluster').size()


# In[265]:


K1_hl_to_hh_cluster2_4.head()


# In[268]:


# Getting the variables for comparison and flagging 1 & 0
K1_hl_to_hh_cluster2_4_vars= K1_hl_to_hh_cluster2_4[['terra_PTR','cass_terra_PTR_index', 'SKU_all_beer',
 'SKU_terra', 'SKU_cass','RD_cass','RD_terra','AVG_AMT_3M','AVG_AMT_1M','WK_RT','WE_RT','PERS_RT','CORP_RT','REVISIT_RT',
'AMT_ESTIMATE_SALES','months_of_business_beer', 'Soju_other_catg_vol','region(EN)','POC_Channel','cluster']]
K1_hl_to_hh_cluster2_4_vars['cluster_flag']=np.where(K1_hl_to_hh_cluster2_4_vars.cluster==4,1,0)
K1_hl_to_hh_cluster2_4_vars.head()


# In[286]:


#Running a classification model to find what differentiates the 2 clusters
#Xgboost 
import xgboost as xgb

#splitting the dataset
X, y = K1_hl_to_hh_cluster2_4_vars.iloc[:,0:17],K1_hl_to_hh_cluster2_4_vars.iloc[:,20]

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


cluster_model = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.5, learning_rate = 0.03,n_jobs=-1,
                max_depth =6, alpha =10, n_estimators = 200)


cluster_model.fit(X_train, y_train)


# In[287]:


#Finding the factors differentiating the 2 clusters
imp=cluster_model.get_booster().get_score(importance_type= 'gain')

var_imp=pd.DataFrame(list(imp.items()), columns=['Feature', 'Gain'])
var_imp.sort_values('Gain',ascending=False)


# In[297]:


#Finding the parameters and its impact on cluster 4  and 2 
print ('Weekend Footfall' ,'\n',K1_hl_to_hh_cluster2_4_vars.groupby('cluster')['WE_RT'].mean())

print ('\n')

print ('Weekday Footfall' ,'\n',K1_hl_to_hh_cluster2_4_vars.groupby('cluster')['WK_RT'].mean())


# In[305]:


# ************************************************* Recommendation Block ***************************************************


# In[335]:


#Model for High Low to High High for cluster 4
K1_hl_to_hh_cluster4=K1_hl_to_hh_cluster[K1_hl_to_hh_cluster['cluster']==4]
print ('shape of cluster 4 is :',K1_hl_to_hh_cluster4.shape)
print ('\n')
print ('distribution of HL and HH in cluster 4 :','\n',K1_hl_to_hh_cluster4.groupby('TC_flag').size().reset_index(name='cnt_of_POCs'))


# In[355]:


#Taking the variables for recommendation:
K1_hl_to_hh_cluster4_1=K1_hl_to_hh_cluster4[['TC_flag',
        'cass_PTR','SKU_cass', 'RD_cass', 'male_ratio', 'female_ratio', 'age_20_to_40','age_50_to_60',
        'F20_RT', 'F30_RT', 'F40_RT','F50_RT', 'F60_RT', 'M20_RT', 'M30_RT','M40_RT', 'M50_RT', 'M60_RT',
        'TIME_0510_RT','TIME_1114_RT','TIME_1517_RT','TIME_1819_RT','TIME_2021_RT','TIME_2224_RT','TIME_0104_RT','BAR-ID']]


K1_hl_to_hh_cluster4_1=K1_hl_to_hh_cluster4_1.replace([np.inf, -np.inf], np.nan)
K1_hl_to_hh_cluster4_1.fillna(0,inplace=True)


# In[360]:


K1_hl_to_hh_cluster4_1.shape


# In[361]:


#Xgboost 
import xgboost as xgb

#splitting the dataset
X, y = K1_hl_to_hh_cluster4_1.iloc[:,1:25],K1_hl_to_hh_cluster4_1.iloc[:,0]

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


poc_model = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.5, learning_rate = 0.03,n_jobs=-1,
                max_depth =6, alpha =10, n_estimators = 200)


poc_model.fit(X_train, y_train)


# In[362]:


#Evaluating training performance
X_train_pred=poc_model.predict(X_train)
print(confusion_matrix(y_train,X_train_pred))

print(classification_report(y_train,X_train_pred))


# In[363]:


#Evaluating the performance on test data
X_test_pred=poc_model.predict(X_test)
print(confusion_matrix(y_test,X_test_pred))

print(classification_report(y_test,X_test_pred))


# In[383]:


# imp=poc_model.get_booster().get_score(importance_type= 'gain')

# var_imp=pd.DataFrame(list(imp.items()), columns=['Feature', 'Gain'])
# var_imp.sort_values('Gain',ascending=False)


# In[373]:


from xgboost import plot_importance
from matplotlib import pyplot as plt

# plot feature importance


plot_importance(poc_model) 
plt.rcParams["figure.figsize"] = (5,15)
plt.show()


# In[351]:


K1_hl_to_hh_cluster4_1.groupby('TC_flag')['TIME_1819_RT'].mean()


# In[348]:


# ************************************** N=1 Recommendation ********************************


# In[380]:


K1_hl_to_hh_cluster4_recom=K1_hl_to_hh_cluster4_1[K1_hl_to_hh_cluster4_1['TC_flag']==0]
K1_hl_to_hh_cluster4_recom=K1_hl_to_hh_cluster4_recom[['BAR-ID','RD_cass','cass_PTR','TIME_1819_RT','TIME_2021_RT','age_20_to_40']]

#Indexing the HL with HH metrics
K1_hl_to_hh_cluster4_HH=K1_hl_to_hh_cluster4_1[K1_hl_to_hh_cluster4_1['TC_flag']==1]
#Getting the average
RD_cass_HH=K1_hl_to_hh_cluster4_HH['RD_cass'].mean()
cass_PTR_HH=K1_hl_to_hh_cluster4_HH['cass_PTR'].mean()
TIME_1819_RT_HH=K1_hl_to_hh_cluster4_HH['TIME_1819_RT'].mean()
TIME_2021_RT_HH=K1_hl_to_hh_cluster4_HH['TIME_2021_RT'].mean()
age_20_to_40_HH=K1_hl_to_hh_cluster4_HH['age_20_to_40'].mean()

K1_hl_to_hh_cluster4_recom['RD_cass_index']=K1_hl_to_hh_cluster4_recom['RD_cass']/RD_cass_HH
K1_hl_to_hh_cluster4_recom['cass_PTR_index']=K1_hl_to_hh_cluster4_recom['cass_PTR']/cass_PTR_HH
K1_hl_to_hh_cluster4_recom['TIME_1819_RT_index']=K1_hl_to_hh_cluster4_recom['TIME_1819_RT']/TIME_1819_RT_HH
K1_hl_to_hh_cluster4_recom['TIME_2021_RT_index']=K1_hl_to_hh_cluster4_recom['TIME_2021_RT']/TIME_2021_RT_HH
K1_hl_to_hh_cluster4_recom['age_20_to_40_index']=K1_hl_to_hh_cluster4_recom['age_20_to_40']/age_20_to_40_HH

K1_hl_to_hh_cluster4_recom.head()


# In[ ]:





# In[ ]:


#***************************** End of Recommendations ***************************************************************


# In[182]:


imp=poc_model.get_booster().get_score(importance_type= 'gain')

var_imp=pd.DataFrame(list(imp.items()), columns=['Feature', 'Gain'])
var_imp.sort_values('Gain',ascending=False)
# names = list(D.keys())
# values = list(D.values())
# plt.xlabel('features', fontsize=10)
# #tick_label does the some work as plt.xticks()
# plt.bar(range(0,48),values,tick_label=names)

# plt.rcParams["figure.figsize"] = (40,3)
# plt.show()


# In[203]:


cluster0_data_for_model.groupby('TC_flag')['RD_cass'].mean()


# In[ ]:





# In[30]:


#trying out KNN classifier
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors=20)
# classifier.fit(X_scaled, y)


# In[ ]:


# y_pred = classifier.predict_proba(X_scaled)
# y_pred


# In[32]:


K1_hl_to_hm_scaled= scaler.fit_transform(K1_hl_to_hm[['months_of_business_beer',
       'RANK_5_AMT_3M', 'RANK_5_CNT_1M', 'WK_RT', 'IAD_RT_AMT_3M',
      'IAD_RT_CNT_1M', 'WE_RT', 'TIME_1114_RT', 'TIME_2021_RT',
      'TIME_2224_RT', 'PERS_RT', 'CORP_RT', 'F20_RT', 'F30_RT', 'F40_RT',
      'F50_RT', 'F60_RT', 'M20_RT', 'M30_RT', 'AVG_AMT_1M', 'AVG_AMT_3M',
      'M40_RT', 'M50_RT', 'M60_RT', 'REVISIT_RT']])

K1_hl_to_hm_scaled=pd.DataFrame(K1_hl_to_hm_scaled,columns=['months_of_business_beer',
       'RANK_5_AMT_3M', 'RANK_5_CNT_1M', 'WK_RT', 'IAD_RT_AMT_3M',
      'IAD_RT_CNT_1M', 'WE_RT', 'TIME_1114_RT', 'TIME_2021_RT',
      'TIME_2224_RT', 'PERS_RT', 'CORP_RT', 'F20_RT', 'F30_RT', 'F40_RT',
      'F50_RT', 'F60_RT', 'M20_RT', 'M30_RT', 'AVG_AMT_1M', 'AVG_AMT_3M',
      'M40_RT', 'M50_RT', 'M60_RT', 'REVISIT_RT'])
K1_hl_to_hm_scaled=pd.concat([K1_hl_to_hm_scaled,TC['TC_flag']],axis=1)
d1=K1_hl_to_hm_scaled[K1_hl_to_hm_scaled['TC_flag']==1]
d0=K1_hl_to_hm_scaled[K1_hl_to_hm_scaled['TC_flag']==0]
# import scipy

# dist= scipy.spatial.distance.cdist(d0.iloc[:,0:24], d1.iloc[:,0:24], metric='mahalanobis')
# #euclidean
# dist1=pd.DataFrame(dist)
# dist1.head()


# In[33]:


# #Control POCs
# ctrl_pocs=K1_hl_to_hm[K1_hl_to_hm['TC_flag']==0]
# ctrl_pocs=ctrl_pocs[['BAR-ID']].drop_duplicates().reset_index()
# #test POCs
# test_pocs=K1_hl_to_hm[K1_hl_to_hm['TC_flag']==1]
# test_pocs=test_pocs[['BAR-ID']].drop_duplicates().reset_index()
# test_pocs_id=test_pocs['BAR-ID'].values
# print (test_pocs_id)
# print ('control poc shape =',ctrl_pocs.shape)
# dist1.columns=test_pocs_id
# dist2=pd.concat([dist1,ctrl_pocs['BAR-ID']],axis=1)
# dist2.head()
# #dist1.min(axis=1, skipna=None, level=None, numeric_only=None)


# In[34]:


# dist3=dist2.melt(id_vars ='BAR-ID')
# nearest_poc=dist3.sort_values(['BAR-ID','value'])
# nearest_poc1=dist3.merge(nearest_poc,left_on=['BAR-ID','value'],right_on=['BAR-ID','min_value'],how='inner')
# nearest_poc['rank_of_test_bar'] = nearest_poc.groupby(by=['BAR-ID'])['value'].transform(lambda x: x.rank())


# In[35]:


# nearest_poc=dist3.sort_values(['BAR-ID','value'])


# In[36]:


# dist3['rank_of_test_bar'] = dist3.groupby(by=['BAR-ID'])['value'].transform(lambda x: x.rank())


# In[37]:


# #How has been the mapping been done
# l=nearest_poc_match.groupby('variable').size().reset_index(name='cnt')
# l.groupby('cnt').size().reset_index(name='cnt_test_pocs')


# In[38]:


#Weird case of rank =1.5 (need to come back here)
# dist3.loc[(dist3['BAR-ID']=='301033-00000005615') & (dist3['rank_of_test_bar']==1.5)] 


# In[68]:


# Finding out how well they have been matched
# nearest_poc_match.columns=['BAR-ID_ctrl','BAR-ID_matched','distance','rank_of_test_bar']
# K1_hl_to_hm.columns
# nearest_poc_comp=nearest_poc_match.merge(K1_hl_to_hm[['BAR-ID','months_of_business_beer','RANK_5_AMT_3M','TIME_2021_RT','TIME_2224_RT','F20_RT','TIME_2224_RT','F20_RT','F30_RT']],
#                                          left_on='BAR-ID_ctrl',right_on='BAR-ID',how='left')

# nearest_poc_comp.columns=['BAR-ID_ctrl', 'BAR-ID_matched', 'distance', 'rank_of_test_bar','BAR-ID_ctrl', 'months_of_business_beer_ctrl', 'RANK_5_AMT_3M_ctrl', 'TIME_2021_RT_ctrl',
#        'TIME_2224_RT_ctrl', 'F20_RT_ctrl', 'TIME_2224_RT_ctrl', 'F20_RT_ctrl', 'F30_RT_ctrl']

# nearest_poc_comp1=nearest_poc_comp.merge(K1_hl_to_hm[['BAR-ID','months_of_business_beer','RANK_5_AMT_3M','TIME_2021_RT','TIME_2224_RT','F20_RT','TIME_2224_RT','F20_RT','F30_RT']],
#                                          left_on='BAR-ID_matched',right_on='BAR-ID',how='left')
# nearest_poc_comp1.head()


# In[381]:


K1_hl_to_hh_cluster4_recom.to_csv('K1_hl_to_hh_cluster4_recom.csv',index=None)

