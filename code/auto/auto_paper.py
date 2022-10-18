#!/usr/bin/env python
# coding: utf-8

# # Paper_Classifier_3

# ### Label이 기타(2)인 데이터의 Label을 0으로 바꿔서 학습시킨 모델

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# 인공신경망
from sklearn.neural_network import MLPClassifier
# 의사결정나무
from sklearn.tree import DecisionTreeClassifier 
# 랜덤포레스트, 그래디언트 부스팅
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier,HistGradientBoostingClassifier
from xgboost import XGBClassifier
# 그리드 서치
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,recall_score,precision_score,classification_report
import time
import datetime
import sys

# # 데이터 불러오기

# In[2]:
args = sys.argv

# 정지전 sp분까지 label을 1로; inter의 크기로 merge; 정지후 재가동 시간 이후 rm분까지 제거.
sp, inter, rm = args[1],args[2]args[3]

# filepath 입력
filepath = "../../data/정제데이터/reg_data/sp_{}__inter_{}__rm_{}.csv".format(args[1],args[2]args[3])
# filepath = "/home/piai/test/Big_Data_분석_기초/제지공장 프로젝트/data/정제데이터/reg_data/sp_5__inter_30__rm_30.csv"
# 레이블 칼럼명 입력
label_col= "result"


# In[3]:


df = pd.read_csv(filepath,index_col="Unnamed: 0")

# In[5]:


col_num = "003 010 019 036 039 051 055 061 081 082 083 091 092 093 094 095 096 097 098 117 118 121 124 125 126 173 176 196".split(' ')


# In[6]:


col_num = ["TAG_{}".format(num) if len(col_num[0])==3 else num for num in col_num ]


# In[7]:


df_drop = df.drop([*col_num],axis=1)


# In[8]:


df_drop.loc[(df_drop[label_col]==2),label_col]=0
df_droped=df_drop
# df_drop.loc[(df_drop[label_col]==0)]
# df_droped[label_col].value_counts()


# In[9]:


df_drop.fillna(df_drop.mean(),inplace=True)


# In[10]:


df_x = df_droped.drop(label_col,axis=1,inplace=False)
df_y = df_droped[label_col]


# In[11]:


df_train_x,df_test_x, df_train_y, df_test_y = train_test_split(df_x,df_y,test_size=0.3,random_state=1234)


# raw_data

# In[14]:


df_raw = pd.read_csv("../../data/원본데이터/pivot_df_y.csv",index_col="Unnamed: 0")


# In[15]:


df_raw.index=pd.to_datetime(df_raw.index)


# In[16]:


df_raw.fillna(df_raw.mean(),inplace=True)


# In[17]:


df_raw_drop = df_raw.drop([*col_num],axis=1)


# In[18]:


df_raw_x = df_raw_drop.drop(label_col,axis=1,inplace=False)
df_raw_y = df_raw_drop[label_col]


# In[19]:


minutes = datetime.timedelta(minutes=30)


# In[20]:


df_info = pd.read_csv("../data/원본데이터/03_중지리스트.csv")

# start_date, end_date를 datetime 데이터형으로 변경
df_info["start_date"],df_info["end_date"]= list(map(pd.to_datetime,[df_info["start_date"],df_info["end_date"]]))
# df_info를 start_date 기준으로 오름차순.
df_info.sort_values("start_date",inplace=True)


# # 모델링

# 각 모델에서 필요한 파라미터

# In[21]:


para_leaf=[n_leaf * 1 for n_leaf in range(1,21)]
para_split=[n_split * 2 for n_split in range(1,21)]
para_depth=[depth for depth in range(2,11)]
para_n_tree=[n_tree*1 for n_tree in range(1,11)]
para_lr=[n_lr *0.1 for n_lr in range(1,11)]


# ## ANN

# In[22]:


start = time.time()
rand_ann = MLPClassifier(random_state=1234, hidden_layer_sizes=(20, 20),activation="relu")
rand_ann.fit(df_train_x,df_train_y)


# In[23]:


print("ANN 소요시간 :", time.time() - start)


# In[24]:
ann_result = rand_ann

# ## DT

# In[25]:


start = time.time()
estimator = DecisionTreeClassifier(random_state=1234)
param_rand={
            "max_depth":para_depth,
            "min_samples_split":para_split,
            "min_samples_leaf":para_leaf,
#             "n_estimators":para_n_tree,
#             "learning_rate":para_lr
           }
rand_dt = RandomizedSearchCV(estimator, param_rand, n_iter = 5, cv = 3, scoring="f1_micro", n_jobs=-1,verbose=0)
rand_dt.fit(df_train_x,df_train_y)


# In[26]:


print("DT 소요시간 :", time.time() - start)


# In[27]:


best_dt = rand_dt.best_estimator_

# In[28]:


dt_result = best_dt.fit(df_train_x,df_train_y)


# ## RF

# In[29]:


start = time.time()
estimator = RandomForestClassifier(random_state=1234)
param_rand={
            "max_depth":para_depth,
            "min_samples_split":para_split,
            "min_samples_leaf":para_leaf,
            "n_estimators":para_n_tree,
#             "learning_rate":para_lr
           }
rand_rf = RandomizedSearchCV(estimator, param_rand, n_iter = 5, cv = 3, scoring="f1_micro", n_jobs=-1,verbose=0)
rand_rf.fit(df_train_x,df_train_y)


# In[30]:


print("RF 소요시간 :", time.time() - start)


# In[31]:


best_rf = rand_rf.best_estimator_
best_rf


# In[32]:


rf_result = best_rf.fit(df_train_x,df_train_y)


# ## GB

# In[33]:


start = time.time()
estimator = GradientBoostingClassifier(random_state=1234)
param_rand={
            "max_depth":para_depth,
            "min_samples_split":para_split,
            "min_samples_leaf":para_leaf,
            "n_estimators":para_n_tree,
            "learning_rate":para_lr
           }
rand_gb = RandomizedSearchCV(estimator, param_rand, n_iter = 5, cv = 3, scoring="f1_micro", n_jobs=-1,verbose=0)
rand_gb.fit(df_train_x,df_train_y)


# In[34]:


print("GB 소요시간 :", time.time() - start)


# In[35]:


best_gb = rand_gb.best_estimator_
best_gb


# In[36]:


gb_result = best_gb.fit(df_train_x,df_train_y)
print("Score on training set  {:.3f}".format(best_gb.score(df_train_x, df_train_y)))
print("Score on testing set  {:.3f}".format(best_gb.score(df_test_x, df_test_y)))


# ## HGB

# In[37]:


start = time.time()
estimator = HistGradientBoostingClassifier(random_state=1234)
param_rand={
            "max_depth":para_depth,
#             "min_samples_split":para_split,
            "min_samples_leaf":para_leaf,
#             "n_estimators":para_n_tree,
            "learning_rate":para_lr
           }
rand_hgb = RandomizedSearchCV(estimator, param_rand, n_iter = 5, cv = 3, scoring="recall_micro", n_jobs=-1,verbose=0)
rand_hgb.fit(df_train_x,df_train_y)


# In[38]:


print("HGB 소요시간 :", time.time() - start)


# In[39]:


best_hgb = rand_hgb.best_estimator_
best_hgb


# In[40]:


hgb_result = best_hgb.fit(df_train_x,df_train_y)



# ## XGB

# In[41]:


start = time.time()


# In[42]:


estimator = XGBClassifier(random_state=1234,use_missing=False)
param_rand={
            "learning_rate":para_lr,    
            "max_depth":para_depth,
            "n_estimators":para_n_tree
           }
rand_xgb = RandomizedSearchCV(estimator, param_rand, n_iter = 5, cv = 3, scoring="f1_micro", n_jobs=-1,verbose=0)
rand_xgb.fit(df_train_x,df_train_y)


# In[43]:


print("XGB 소요시간 :", time.time() - start)


# In[44]:


best_xgb = rand_xgb.best_estimator_
best_xgb


# In[45]:


xgb_result = best_xgb.fit(df_train_x,df_train_y)



# # 변수 중요도

# In[106]:


# 변수중요도 상위 몇 개?
top_importance = 20


# In[107]:


###### v_feature_name = df_train_x.columns
v_feature_name = df_train_x.columns
df_importance = pd.DataFrame()
df_importance["Feature"] = v_feature_name
# df_importance["ANN_Importance"] = rand_ann.feature_importances_
df_importance["DT_Importance"] = best_dt.feature_importances_
df_importance["RF_Importance"] = best_rf.feature_importances_
df_importance["GB_Importance"] = best_gb.feature_importances_
df_importance["XGB_Importance"] = best_xgb.feature_importances_
df_importance.sort_values("XGB_Importance", ascending=True, inplace=True)
# coordinates_ann = [i for i in range(len(df_importance))]
coordinates_dt = [i for i in range(len(df_importance))]
coordinates_rf = [i+0.2 for i in range(len(df_importance))]
coordinates_gb = [i+0.4 for i in range(len(df_importance))]
coordinates_xgb = [i+0.6 for i in range(len(df_importance))]
plt.figure(figsize=(15,7))
plt.bar(x=coordinates_dt[-top_importance:], height=df_importance['DT_Importance'][-top_importance:],width=0.2,label="DT")
plt.bar(x=coordinates_rf[-top_importance:], height=df_importance['RF_Importance'][-top_importance:],width=0.2,label="RF")
plt.bar(x=coordinates_gb[-top_importance:], height=df_importance['GB_Importance'][-top_importance:],width=0.2,label="GB")
plt.bar(x=coordinates_xgb[-top_importance:], height=df_importance['XGB_Importance'][-top_importance:],width=0.2,label="XGB")
plt.legend(fontsize=15)
plt.title("모델별 변수 중요도 - ALL Data",fontsize=20)
plt.xticks(coordinates_rf[-top_importance:],df_importance['Feature'][-top_importance:],rotation=90,fontsize=15)
plt.ylabel("변수 중요도",fontsize=15)
plt.xlabel("변수",fontsize=15)
plt.savefig("./feature_importance_{}_{}_{}.png".format(sp,inter,rm)
plt.show()

ann_y_pred = ann_result.predict(df_test_x)
dt_y_pred = dt_result.predict(df_test_x)
rf_y_pred = rf_result.predict(df_test_x)
gb_y_pred = gb_result.predict(df_test_x)
hgb_y_pred = hgb_result.predict(df_test_x)
xgb_y_pred = xgb_result.predict(df_test_x)



# ## 각 모델별 성능 비교

# In[78]:


models=['ANN','DT','RF','GB','HGB','XGB']
precision,accuracy,recall,f1=[],[],[],[]


# In[79]:


precision.append(precision_score(df_test_y,ann_y_pred))

accuracy.append(accuracy_score(df_test_y,ann_y_pred))

recall.append(recall_score(df_test_y,ann_y_pred))

f1.append(f1_score(df_test_y,ann_y_pred))


# In[80]:


precision.append(precision_score(df_test_y,dt_y_pred))

accuracy.append(accuracy_score(df_test_y,dt_y_pred))

recall.append(recall_score(df_test_y,dt_y_pred))

f1.append(f1_score(df_test_y,dt_y_pred))


# In[81]:


precision.append(precision_score(df_test_y,rf_y_pred))

accuracy.append(accuracy_score(df_test_y,rf_y_pred))

recall.append(recall_score(df_test_y,rf_y_pred))

f1.append(f1_score(df_test_y,rf_y_pred))


# In[82]:


precision.append(precision_score(df_test_y,gb_y_pred))

accuracy.append(accuracy_score(df_test_y,gb_y_pred))

recall.append(recall_score(df_test_y,gb_y_pred))

f1.append(f1_score(df_test_y,gb_y_pred))


# In[83]:


precision.append(precision_score(df_test_y,hgb_y_pred))

accuracy.append(accuracy_score(df_test_y,hgb_y_pred))

recall.append(recall_score(df_test_y,hgb_y_pred))

f1.append(f1_score(df_test_y,hgb_y_pred))


# In[84]:


precision.append(precision_score(df_test_y,xgb_y_pred))

accuracy.append(accuracy_score(df_test_y,xgb_y_pred))

recall.append(recall_score(df_test_y,xgb_y_pred))

f1.append(f1_score(df_test_y,xgb_y_pred))


# In[105]:


fig,ax=plt.subplots(2,2,figsize=(13,10))
colors=['r','g','b','c','m']
fig.suptitle("각 모델별 성능 비교",fontsize=40)

ax[0,0].bar(models,accuracy,color=colors)
ax[0,0].set_title("Accuracy",fontsize=25)
ax[0,0].set_ylim(bottom=min(accuracy)-0.05,top=max(accuracy)+0.05)
ax[0,0].tick_params(axis='x', labelsize=20)
ax[0,0].tick_params(axis='y', labelsize=20)

ax[0,1].bar(models,precision,color=colors)
ax[0,1].set_title("precision",fontsize=25)
ax[0,1].set_ylim(bottom=min(precision)-0.05,top=max(precision)+0.05)
ax[0,1].tick_params(axis='x', labelsize=20)
ax[0,1].tick_params(axis='y', labelsize=20)

ax[1,0].bar(models,recall,color=colors)
ax[1,0].set_title("Recall",fontsize=25)
ax[1,0].set_ylim(bottom=min(recall)-0.05,top=max(recall)+0.05)
ax[1,0].tick_params(axis='x', labelsize=20)
ax[1,0].tick_params(axis='y', labelsize=20)

ax[1,1].bar(models,f1,color=colors)
ax[1,1].set_title("F1",fontsize=25)
ax[1,1].set_ylim(bottom=min(f1)-0.05,top=max(f1)+0.05)
ax[1,1].tick_params(axis='x', labelsize=20)
ax[1,1].tick_params(axis='y', labelsize=20)

plt.savefig("./모델_성능_{}_{}_{}.png".format(sp,inter,rm)


# In[103]:


print(*["{}:\n{}\n{}\n".format(i,classification_report(df_test_y,globals()["{}_y_pred".format(i.lower())]),"="*len("              precision    recall  f1-score   support")) for i in models])


            
# 모델 저장

path = "./trained_model/sp_{}_inter_{}_rm_{}".format(sp,inter,rm)
cmd = "mkdir {}".format(path)
os.system(cmd)

# ann
trained_file = "{}/{}_{}_{}_{}.pickle".format(path,"ann",sp,inter,rm)
with open(trained_file,'wb') as fw:
    pickle.dump(rand_ann, fw)

# rf
trained_file = "{}/{}_{}_{}_{}.pickle".format(path,"rf",sp,inter,rm)
with open(trained_file,'wb') as fw:
    pickle.dump(rand_rf, fw)

# gb
trained_file = "{}/{}_{}_{}_{}.pickle".format(path,"gb",sp,inter,rm)
with open(trained_file,'wb') as fw:
    pickle.dump(rand_gb, fw)

# hgb
trained_file = "{}/{}_{}_{}_{}.pickle".format(path,"hgb",sp,inter,rm)
with open(trained_file,'wb') as fw:
    pickle.dump(rand_hgb, fw)

# xgb
trained_file = "{}/{}_{}_{}_{}.pickle".format(path,"xgb",sp,inter,rm)
with open(trained_file,'wb') as fw:
    pickle.dump(rand_xgb, fw)
# ### plotting

# In[86]:


# min1 = datetime.timedelta(minutes = 600)
# min2 = datetime.timedelta(minutes = 1440)
# start_temp = False
# for start,end in zip(df_info[(df_info["cause"]=="불량중지")]["start_date"],df_info[(df_info["cause"]=="불량중지")]["end_date"]):
#     if start_temp:
#         if ((start_temp+min2)>start):
#             continue
#         n_diff = 2
        
#         xgb_y_before_30 = xgb_result.predict_proba(df_raw_x.loc[((df_raw_x.index>=start-min1)&(df_raw_x.index<=start))])
#         x_ind = df_raw_x.loc[((df_raw_x.index>=start-min1)&(df_raw_x.index<=start))].index
#         y_ind = xgb_y_before_30[:,1]
#         revised_y = (y_ind - np.roll(y_ind, n_diff))[n_diff:]
#         revised_x = np.roll(np.array(x_ind), n_diff)[n_diff:]
#         plt.plot(revised_x,revised_y) # 차분
# #         plt.plot(x_ind,y_ind)
#         plt.xticks(rotation=45)
#         plt.ylim(0,1)
#         plt.show()
#     start_temp =end


# In[87]:


# np.roll(np.array([1,2,3,4,5]), 1)[1:]


# In[88]:


# df_raw_predict = xgb_result.predict(df_raw_x)
# df_raw_proba = xgb_result.predict_proba(df_raw_x)
# print(float(df_raw_predict.sum())/len(df_raw_predict))
# plt.plot(df_raw_x.index,df_raw_proba[:,1])

