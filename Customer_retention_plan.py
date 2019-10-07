#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import warnings
warnings.filterwarnings(action="ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# In[2]:


data = pd.read_csv("D:/new_data/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[3]:


len(data)


# * 一共7043条数据

# In[4]:


len(data.columns)


# * 一共21个特征

# In[5]:


data.columns


# 含义：
# * customerID:客户ID
# * gender：性别
# * SeniorCitizen：老年人（1是，0不是）
# * Partner：合作伙伴（Yes、No）
# * Dependents：家属
# * tenure：任期
# * PhoneService：电话服务
# * MultipleLines：多线？
# * InternetService：互联网服务
# * OnlineSecurity：在线安全
# * OnlineBackup：在线备份
# * DeviceProtection：设备保护
# * TechSupport：技术支持
# * StreamingTV：流媒体电视
# * StreamingMovies：流媒体电影
# * Contract：合约方式
# * PaperlessBilling：无纸账单
# * PaymentMethod：付款方式
# * MonthlyCharges：每月费用
# * TotalCharges：总费用
# * Churn：用户流失

# 上面是每个特征的含义，方便理解

# In[7]:


data.isnull().sum()


# * 数据中不存在缺失的情况

# 从数据的获取途径可以基本判断数据中是不存在异常情况的

# 下面要对数据做一些基本的处理

# In[8]:


#转换数据类型
data['TotalCharges'] = data["TotalCharges"].replace(" ",0)
data['TotalCharges'] = data['TotalCharges'].astype(float)


# 有些特征中的取值存在一定的重复，需要进行统一的处理

# In[9]:


cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']

for col in cols:
    data[col] = data[col].replace({"No internet service":'No'})


# 下面对特征值进行适当的转换

# In[10]:


data['SeniorCitizen'] = data['SeniorCitizen'].replace({1:'Yes',0:'No'})


# In[11]:


#转换tenure
def tenure(data):
    if data <= 12:
        return 't_0_12'
    elif (data > 12) & (data <= 24):
        return 't_12_24'
    elif (data > 24) & (data <= 48):
        return 't_24_48'
    elif (data > 48) & (data <= 60):
        return 't_48_60'
    else:
        return 't_60'


# In[12]:


data["tenure_group"] = data['tenure'].apply(tenure)


# In[13]:


f,ax = plt.subplots(1,1,figsize=(14,6))
colors = ["#3791D7","#D72626"]
labels = ["No Churn","Churn"]

data["Churn"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax,shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax.set_title('流失客户与非流失客户的分布情况', fontsize=14)


# * 结果显示有27%的客户已经流失了，73%的客户还在使用我们的服务

# 下面观察不同特征维度与是否流失之间的关系

# 性别

# In[14]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["#3791D7","#D72626"]
labels = ["Female","Male"]

data[data.Churn == 'Yes']["gender"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中性别的分布情况', fontsize=14)

data[data.Churn == 'No']["gender"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中性别的分布情况', fontsize=14)


# * 从上面的结果可以看出，在不同性别的数据中，流失与非流失的比例是基本一致的，所以性别对于是否流失是没有影响的

# 观察是否是老年人与客户流失之间的关系

# In[15]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["#3791D7","#D72626"]
labels = ["No","Yes"]

data[data.Churn == 'Yes']["SeniorCitizen"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中是否是老年人的分布情况', fontsize=14)

data[data.Churn == 'No']["SeniorCitizen"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中是否是老年人的分布情况', fontsize=14)


# * 在这个特征中可以看出一点有价值的信息，在老年人这类群体中，流失的比例要更高，占了25%；而非老年人中的流失比例只有13%
# * 结论：老年人群体的流失要严重的多，需要针对老年人进行保留计划的制定

# 观察是否是合伙人与流失之间的关系

# In[16]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["#3791D7","#D72626"]
labels = ["No","Yes"]

data[data.Churn == 'Yes']["Partner"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中合伙人的分布情况', fontsize=14)

data[data.Churn == 'No']["Partner"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中合伙人的分布情况', fontsize=14)


# * 可以发现，有合伙人的流失比例要少，是36%；而没有合伙人中的流失比例为47%
# * 结论：如果没有合伙人，那么流失的情况要更严重

# 观察家属与是否流失之间的关系

# In[17]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["#3791D7","#D72626"]
labels = ["No","Yes"]

data[data.Churn == 'Yes']["Dependents"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中是否有家属的分布情况', fontsize=14)

data[data.Churn == 'No']["Dependents"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中是否有家属的分布情况', fontsize=14)


# * 家属客户群的流失比例为17%，而非家属客户群中的流失比例为34%
# * 结论：如果是非家属，则客户流失的要更严重

# 观察电话服务与是否流失之间的关系

# In[67]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["#3791D7","#D72626"]
labels = ["No","Yes"]

data[data.Churn == 'Yes']["PhoneService"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中电话服务使用情况的分布', fontsize=14)

data[data.Churn == 'No']["PhoneService"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中电话服务使用情况的分布', fontsize=14)


# * 差异不明显，所以这个特征对是否流失并没有影响

# In[68]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["r","g",'y']
labels = ["Yes","No","No Phone Service"]

data[data.Churn == 'Yes']["MultipleLines"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中MultipleLines的分布情况', fontsize=14)

data[data.Churn == 'No']["MultipleLines"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中MultipleLines的分布情况', fontsize=14)


# * 差异不明显，所以这个特征对是否流失是没有影响的

# 观察使用不同因特网服务的客户群中流失情况

# In[69]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["r","g",'y']
labels = ["Fiber optic","DSL","No"]

data[data.Churn == 'Yes']["InternetService"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中使用不同服务的分布情况', fontsize=14)

data[data.Churn == 'No']["InternetService"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中使用不同服务的分布情况', fontsize=14)


# * 在流失的客户群中，使用Fiber optic的比例非常高，占了69%；而未流失的客户群中，三者的比例是比较均匀的
# * 结论：从流失数据中可以看出使用Fiber optic服务后会更容易流失，所以需要针对这种类型的服务进行改进

# 下面都是为客户提供的一些附加服务，或者是售后服务

# 下面观察在线安全对是否流失的影响

# In[70]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["#3791D7","#D72626"]
labels = ["No","Yes"]

data[data.Churn == 'Yes']["OnlineSecurity"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中使用在线安全情况的分布', fontsize=14)

data[data.Churn == 'No']["OnlineSecurity"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中使用在线安全情况的分布', fontsize=14)


# * 在流失客户群中在线安全使用率比较低，只占了16%；而非流失客户群众的占比是33%
# * 结论：拥有在线安全服务的客户更不容易流失

# 下面观察在线备份功能对客户流失的影响

# In[71]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["#3791D7","#D72626"]
labels = ["No","Yes"]

data[data.Churn == 'Yes']["OnlineBackup"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中使用在线备份情况的分布', fontsize=14)

data[data.Churn == 'No']["OnlineBackup"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中使用在线备份情况的分布', fontsize=14)


# * 结论：拥有在线备份服务的客户更不容易流失

# 观察设备保护对于客户流失的影响

# In[72]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["#3791D7","#D72626"]
labels = ["No","Yes"]

data[data.Churn == 'Yes']["DeviceProtection"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中使用设备保护情况的分布', fontsize=14)

data[data.Churn == 'No']["DeviceProtection"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中使用设备保护情况的分布', fontsize=14)


# * 结论：拥有设备保护服务的客户更不容易流失

# 下面观察技术支持对客户流失的影响

# In[73]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["#3791D7","#D72626"]
labels = ["No","Yes"]

data[data.Churn == 'Yes']["TechSupport"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中技术支持情况的分布', fontsize=14)

data[data.Churn == 'No']["TechSupport"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中技术支持情况的分布', fontsize=14)


# * 在流失的客户群中使用技术支持的人很少，只占了17%，而非流失的客户群众占了34%
# * 结论：拥有技术支持服务的客户更不容易流失

# 观察流媒体电视对客户流失的影响

# In[74]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["#3791D7","#D72626"]
labels = ["No","Yes"]

data[data.Churn == 'Yes']["StreamingTV"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中使用流媒体电视情况的分布', fontsize=14)

data[data.Churn == 'No']["StreamingTV"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中使用流媒体电视情况的分布', fontsize=14)


# * 差异并不明显，说明是否使用流媒体电视对于客户的流失没有多大的影响

# 下面观察流媒体电影对客户流失的影响

# In[75]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["#3791D7","#D72626"]
labels = ["No","Yes"]

data[data.Churn == 'Yes']["StreamingMovies"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中使用流媒体电影情况的分布', fontsize=14)

data[data.Churn == 'No']["StreamingMovies"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中使用流媒体电影情况的分布', fontsize=14)


# 差异不明显，说明是否使用流媒体电影对于客户的流失没有多大的影响

# 下面观察合约方式对于客户流失的影响

# In[76]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["r","g","b"]
labels = ["Month-to-month","One year","Two year"]

data[data.Churn == 'Yes']["Contract"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中不同合约方式的情况分布', fontsize=14)

data[data.Churn == 'No']["Contract"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中不同合约方式的情况分布', fontsize=14)


# * 流失客户群众多数都在用按月的方式，占了89%；而非流失的客户群众，One year和Two year的方式要更多
# * 结论：短期合约的客户流失较高

# 下面观察无纸账单对客户流失的影响

# In[30]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["#3791D7","#D72626"]
labels = ["Yes","No"]

data[data.Churn == 'Yes']["PaperlessBilling"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中使用无纸账单的情况分布', fontsize=14)

data[data.Churn == 'No']["PaperlessBilling"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中使用无纸账单的情况分布', fontsize=14)


# * 在流失的客户群体中使用无纸账单的占比为75%，而非流失客户群中的占比未54%
# * 结论：对于这样服务方式还需要进行改进

# 下面观察支付方式对客户流失的影响

# In[32]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["r","g","b","y"]
labels = ["Electronic check","Mailed check","Bank transfer","Credit card"]

data[data.Churn == 'Yes']["PaymentMethod"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中不同的付款方式的分布情况', fontsize=14)

data[data.Churn == 'No']["PaymentMethod"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中不同的付款方式的分布情况', fontsize=14)


# * 在流失客户群中使用电子发票的方式的人比较多，占了57%，而其他的就偏少；而在非流失的客户群众，各种方式的分布情况是比较均匀的
# * 结论：受到了电子发票这种方式的影响

# 下面观察使用期限的分组对客户流失的影响

# In[34]:


f,ax = plt.subplots(1,2,figsize=(14,6))
colors = ["r","g","b","y","black"]
labels = ["t_0_12","t_24_48","t_12_24","t_48_60","t_60"]

data[data.Churn == 'Yes']["tenure_group"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[0],shadow=True,
                                      colors=colors,labels=labels, fontsize=12, startangle=70)
ax[0].set_title('流失客户中不同分组的分布情况', fontsize=14)

data[data.Churn == 'No']["tenure_group"].value_counts().plot.pie(autopct='%1.2f%%',ax=ax[1],shadow=True,
                                      colors=colors,labels=labels,fontsize=12, startangle=70)
ax[1].set_title('非流失客户中不同分组的分布情况', fontsize=14)


# * 在流失客户群中，使用期限很少的占比很高，占了55%，这也符合实际规律，因为流失客户的使用时间一般会比较短；而非流失的客户群众，各类别的客户的分布是比较均匀的
# * 短期合约的客户流失较高

# 下面观察tenure与客户流失之间的关系

# * 在流失中的客户群体中，tenure的值都是很小的，说明使用期限很短

# 下面观察tenure、MonthlyChargeds和TotalCharges对客户是否流失的影响

# In[78]:


#折线图
def kdeplot(feature):
    plt.figure(figsize=(9,4))
    plt.title("KDE for {}".format(feature))
    ax0 = sns.kdeplot(data[data['Churn'] == 'No'][feature].dropna(), color= 'navy', label= 'Churn: No')
    ax1 = sns.kdeplot(data[data['Churn'] == 'Yes'][feature].dropna(), color= 'orange', label= 'Churn: Yes')

kdeplot('tenure')
kdeplot('MonthlyCharges')
kdeplot('TotalCharges')


# * 新近的客户更容易流失
# * 月费较高的人也更容易流失

# 下面为了构建模型，需要先将数据进行一些适当的转换

# In[89]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# In[90]:


Id_col = ['customerID']
target_col = ["Churn"]

#类别特征
cat_cols   = data.nunique()[data.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]

#数值特征
num_cols   = [x for x in data.columns if x not in cat_cols + target_col + Id_col]

#二分值特征
bin_cols   = data.nunique()[data.nunique() == 2].keys().tolist()

#多列值
multi_cols = [i for i in cat_cols if i not in bin_cols]


# In[91]:


#进行二值处理
le = LabelEncoder()
for col in bin_cols:
    data[col] = le.fit_transform(data[col])
    
#独热编码    
data = pd.get_dummies(data = data,columns = multi_cols)


# In[92]:


#对数值进行标准化处理
#对数值进行缩放
#减少特征之间量纲的影响
#加快算法的运行效果
ss = StandardScaler()
ss_data = ss.fit_transform(data[num_cols])
ss_data = pd.DataFrame(ss_data,columns=num_cols)

#重组数据
copy_data = data.copy()
data = data.drop(columns = num_cols,axis = 1)
data = data.merge(ss_data,left_index=True,right_index=True,how = "left")


# In[93]:


#输出特征之间的相关性
#介于-1到1之间，-1是完全负相关，1是完全正相关
data.corr()


# In[104]:


from sklearn.ensemble import RandomForestClassifier

params = {'random_state': 0, 'n_jobs': 4, 'n_estimators': 5000, 'max_depth': 8}

X, y = data.drop(['Churn','customerID'],axis=1),data['Churn']
# Fit RandomForest Classifier
clf = RandomForestClassifier(**params)
clf = clf.fit(X,y)
# Plot features importances
imp = pd.Series(data=clf.feature_importances_,index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,12))
plt.title("Feature importance")
ax = sns.barplot(y=imp.index,x=imp.values,palette="Blues_d",orient='h')


# * 上面是通过随机森林算法计算出了所有特征在分类问题上的评分，而排位越靠前的特征对结果的影响力就越大
# * 在后面构建预测模型的时候，上面的整个分析过程是有帮助的

# 这份数据中的流失和非流失的数据比例是不一致的，所以这是一个不平衡数据，对于不平衡数据是需要进行处理的，否则构建的预测模型会受到影响，导致预测不准确

# In[105]:


data['Churn'].value_counts()


# In[235]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[213]:


from imblearn.over_sampling import SMOTE
from sklearn import metrics

cols = [col for col in data.columns if col not in Id_col + target_col]

smote_X = data[cols]
smote_y = data[target_col]

#分割数据
smote_train_X,smote_test_X,smote_train_y,smote_test_y = train_test_split(smote_X,smote_y,test_size=0.2,random_state=31)

s = SMOTE(random_state=0)
s_X,s_y = s.fit_sample(smote_train_X,smote_train_y)
s_X = pd.DataFrame(data=s_X,columns=cols)
s_y = pd.DataFrame(data=s_y,columns=target_col)


# In[214]:


#逻辑回归
lr = LogisticRegression()
lr.fit(s_X,s_y)

p_test = lr.predict(smote_test_X)
#预测准确率
print(metrics.accuracy_score(smote_test_y, p_test))


# In[223]:


#随机森林
rfc = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
rfc.fit(s_X,s_y)
prediction_test = rfc.predict(smote_test_X)
#预测准确率
print (metrics.accuracy_score(smote_test_y, prediction_test))


# In[222]:


#XGBOOST模型
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(s_X,s_y)
preds = model.predict(smote_test_X)
#预测准确率
metrics.accuracy_score(smote_test_y,preds)


# In[228]:


importances = model.feature_importances_
weights = pd.Series(importances,
                 index=smote_X.columns.values)


# In[233]:


weights.sort_values()[-10:].plot(kind = 'barh')


# In[234]:


weights.sort_values()[:10].plot(kind = 'barh')


# 总结：

# * 老年人群体的流失要严重的多，需要针对老年人进行保留计划的制定
# * 如果没有合伙人，那么流失的情况要更严重
# * 如果没有家属，则客户流失的要更严重
# * 从流失数据中可以看出使用Fiber optic服务后会更容易流失，所以需要针对这种类型的服务进行改进
# * 保证提供给客户的一些附加服务，或者是售后服务（安全服务、备份服务、设备保护服务和技术支持服务）的质量可以降低客户流失
# * 短期合约的客户流失较高
# * 对于无纸服务的方式还需要进行改进
# * 受到了电子发票这种方式的影响导致流失
# * 新近的客户更容易流失
# * 月费较高的人更容易流失
