import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from sklearn.linear_model import SGDClassifier, LogisticRegression



dfoff = pd.read_csv('ccf_offline_stage1_train.csv')
dftest = pd.read_csv('ccf_offline_stage1_test_revised.csv')
dfon = pd.read_csv('ccf_online_stage1_train.csv')
print('data read end.')

def getDiscountType(row):
    if pd.isnull(row):
        return np.nan
    elif ':' in row:
        return 1
    else:
        return 0

def convertRate(row):
    # 将折扣转化为打折比率
    if pd.isnull(row):
        return 1.0
    elif ':' in str(row):
        rows = row.split(':')
        return 1.0 - float(rows[1])/float(rows[0])
    else:
        return float(row)

def getDiscountMan(row):
	# 获取优惠券面额
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0

def getDiscountJian(row):
	# 获取优惠券减免额度
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0
print("tool is ok.")

def processData(df):
    # 通过Discount_rate得到discount的一系列特征
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    print(df['discount_rate'].unique())
    # 对distance进行转换
    df['distance'] = df['Distance'].fillna(-1).astype(int)
    return df

dfoff = processData(dfoff)
dftest = processData(dftest)

date_received = dfoff['Date_received'].unique()
date_received = sorted(date_received[pd.notnull(date_received)])

date_buy = dfoff['Date'].unique()
date_buy = sorted(date_buy[pd.notnull(date_buy)])

couponbydate = dfoff[dfoff['Date_received'].notnull()][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
couponbydate.columns = ['Date_received', 'count']

buybydate = dfoff[(dfoff['Date'].notnull()) & (dfoff['Date_received'].notnull())][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
buybydate.columns = ['Date_received', 'count']

print('discount initial end')

def getWeekday(row):
	if row == 'nan':
		return np.nan
	else:
		return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1

dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)

# weekday_type: 周六和周日为1，其他为0
dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x: 1 if x in [6,7] else 0)
dftest['weekday_type'] = dftest['weekday'].apply(lambda x: 1 if x in [6,7] else 0)

# 对weekday进行one-hot编码（独热编码）
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
tmpdf = pd.get_dummies(dfoff['weekday'].replace('nan', np.nan))
tmpdf.columns = weekdaycols
dfoff[weekdaycols] = tmpdf

tmpdf = pd.get_dummies(dftest['weekday'].replace('nan', np.nan))
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf

def label(row):
	if pd.isnull(row['Date_received']):
		return -1
	if pd.notnull(row['Date']):
		td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
		if td <= pd.Timedelta(15, 'D'):
			return 1
	return 0

dfoff['label'] = dfoff.apply(label, axis=1)

print('label end')

# data split
print('----------data split--------')
df = dfoff[dfoff['label'] != -1].copy()
train = df[(df['Date_received'] < 20160516)].copy()
valid = df[(df['Date_received'] >= 20160516) & (df['Date_received'] <= 20160615)].copy()
print('data split end')

original_features = ['discount_rate', 'discount_type', 'discount_man', 'discount_jian', 'distance', 'weekday', 'weekday_type'] + weekdaycols

print('---------train---------')
model = SGDClassifier(
	loss = 'log',
	penalty = 'elasticnet',
	fit_intercept = True,
	max_iter = 100,
	shuffle = True,
	alpha = 0.01,
	l1_ratio = 0.01,
	n_jobs = 1,
	class_weight = None
)

model.fit(train[original_features], train['label'])

# 预测以及结果评价
print(model.score(valid[original_features], valid['label']))

print('----------save model-----------')
# 把生成的算法模型保存下来
with open('1_model.pkl', 'wb') as f:
	pickle.dump(model, f)
with open('1_model.pkl', 'rb') as f:
	model = pickle.load(f)

# 对dftest进行预测并保存预测结果
y_test_pred = model.predict_proba(dftest[original_features])
dftest1 = dftest[['User_id', 'Coupon_id', 'Date_received']].copy()
dftest1['label'] = y_test_pred[:,1]
dftest1.to_csv('submit1.csv', index=False, header=False)
