# 泰坦尼克号乘客生存预测
## 1. 简介
通过本次的数据集分析乘客中获救的乘客有哪些特点，判断出什么样的人更容易获救。同时，要利用机器学习来预测在这场灾难中哪些人最终获救。
## 2. 数据样本
本项目数据集分为两份：`titanic_train.csv`和`titanic_test.csv`

`titanic_train.csv`:训练集，包含891条数据

`titanic_test.csv`:测试集，包含418条数据

### 数据特征

字段|字段说明
:--|:--
PassengerId|乘客编号
Survived|存活情况（存活：1，死亡：0）
Pclass|客舱等级
Name|乘客姓名
Sex|性别
Age|年龄
SibSp|同乘的兄弟姐妹/配偶书
Parch|同乘的父母/小孩数
Ticket|船票编号
Fare|船票价格
Cabin|客舱号
Embarked|登船港口

`PassengerId`是数据唯一序号，`Survived`是存活情况，为预测标记特征。其余10个为原始数据特征。

## 3.数据处理
首先读取`titanic_train.csv`和`titanic_test.csv`文件。

```python
#导入数据分析库 pandas
import pandas as pd
#导入科学计算库 numpy
import numpy as np
from pandas import Series, DataFrame

data_train = pd.read_csv('titanic_train.csv')
data_test = pd.read_csv('titanic_test.csv')
```

```python
data_train.head(10)
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/titanic/figure/data_trian.head(10).png?raw=true)

这是训练集中前10条数据，典型的`dataframe`风格，其中`Survived`是否存活（存活：1，死亡：0）是标签列，其余列为特征列。

```python
data_test.head(10)
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/titanic/figure/data_test.head(10).png?raw=true)

这是测试集中前10条数据，共11列，需要根据测试集中的特征数据，以及通过训练集得到的算法模型来预测`Survived`标签列的值，即乘客是否存活。

同时可以利用`data_trian.info()`函数来获取更多关于数据的信息。

```python
data_train.info()
```

![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/titanic/figure/data_train.info().png?raw=true)

从以上数据我们可以看出，特征`Age`、`Cabin`以及`Embarked`数据有缺失。其中`Age`有714条数据，缺失117条数据；`Cabin`有204条数据，缺失687条数据；`Embarked`有889条数据，缺失2条数据，其余列都是891条数据。

通过`data_train.describe()`我们可以得到诸如平均值、中位数、最大最小值等信息。

```python
data_train.describe()
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/titanic/figure/data_train.describe().png?raw=true)

从`Survived`的`mean`字段中我们可以看出大概有38%的人获救。乘客平均年龄29岁。

## 4.特征提取
采用最快速的方法预测一个结果。
### 数据空值处理
1.	客舱号`Cabin`存在大量空值，如果直接对空值进行填空，带来的误差会比较大，所以舍弃`Cabin`字段。
2. 年龄列对于一个乘客是否能够存活是一个重要的判断标准，采用`Age`中位数对空值进行填充。
3. `PassengerId`是一个连续的序列，对于是否能够存活没有任何影响，故舍弃。

```python
#Age列中的缺失值用Age中位数进行填充
data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())
data_train.describe()
```

![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/titanic/figure/data_train.describe()-Age.png?raw=true)

## 5.线性回归算法

```python
#线性回归
from sklearn.linear_model import LinearRegression

#训练集交叉验证，得到平均值
from sklearn.model_selection import KFold

#选取简单的可用输入特征
predictors = ['Pclass','Age','SibSp','Parch','Fare']

#初始化线性回归算法
alg = LinearRegression()
#样本平均分为互斥的3份，3折交叉验证
kf = KFold(n_splits=3, shuffle=False, random_state=1)

predictions = []

for train,test in kf.split(data_train):
	train_predictors = (data_train[predictors].iloc[train,:])
	#得到训练集标记train_target
	train_target = data_train['Survived'].iloc[train]
	#使用train_predictors和train_targe训练算法模型
	alg.fit(train_predictors, train_target)
	test_predictions = alg.predict(data_train[predictors].iloc[test,:])
	#每个验证集通过验证，得到一个array。
	predictions.append(test_predictions)
```

```python
import numpy as np

#将得到的三个array链接在一起
predictions = np.concatenate(predictions, axis=0)

#得到算法模型预测结果
predictions[predictions>.5] = 1
predictions[predictions<=.5] = 0
accuracy = sum(predictions == data_train['Survived']) / len(predictions)
print('准确率为：', accuracy)
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/titanic/figure/linearregression_accuracy.png?raw=true)

## 6.逻辑回归算法
```python
from sklearn import model_selection
#逻辑回归
from sklearn.linear_model import LogisticRegression

#初始化逻辑回归算法
LogRegAlg = LogisticRegression(random_state=1)
re = LogRegAlg.fit(data_train[predictors], data_train['Survived'])

#使用sklearn库里的交叉验证函数获取预测准确率分数
scores = model_selection.cross_val_score(LogRegAlg, data_train[predictors], data_train['Survived'], cv=3)

#使用交叉验证分数的平均值作为最终的准确率
print('准确率为：', scores.mean())
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/titanic/figure/logacuracy.png?raw=true)

### 增加特征 Sex 和 Embarked 列，查看对预测的影响
对性别 Sex 列和登船港口 Embarked 列进行字符处理。

```python
#Sex性别列处理：male用0，female用1
data_train.loc[data_train['Sex'] == 'male', 'Sex'] = 0
data_train.loc[data_train['Sex'] ==  'female', 'Sex'] = 1
```

```python
#缺失值用最多的S进行填充
data_train['Embarked'] = data_train['Embarked'].fillna('S')
#地点用0,1,2
data_train.loc[data_train['Embarked'] == 'S', 'Embarked'] = 0
data_train.loc[data_train['Embarked'] == 'C', 'Embarked'] = 1
data_train.loc[data_train['Embarked'] == 'Q', 'Embarked'] = 2
```

增加2个特征Sex和Embarked，继续使用逻辑回归算法进行预测。

```python
predictors = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

LogRegAlg = LogisticRegression(random_state=1)
#为每一个fold计算准确率。
re = LogRegAlg.fit(data_train[predictors], data_train['Survived'])
scores = model_selection.cross_val_score(LogRegAlg, data_train[predictors], data_train['Survived'], cv=3)
#使用scores的均值作为最终的准确率。
print('准确率为：',scores.mean())
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/titanic/figure/Sexacuracy.png?raw=true)

通过增加2个特征，模型的准确率提高到78.78%，提高了8.86%，说明好的特征有利于提升模型的预测能力。

```python
data_test.describe()
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/titanic/figure/data_test.describe()-1.png?raw=true)

```python
data_test.head(10)
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/titanic/figure/data_test.head(10).png?raw=true)

```python
#Age列中的缺失值用Age均值进行填充
data_test['Age'] = data_test['Age'].fillna(data_test['Age'].median())
#Fare列中的缺失值用Fare最大值进行填充
data_test['Fare'] = data_test['Fare'].fillna(data_test['Fare'].max())

#Sex性别列处理：male用0，female用1
data_test.loc[data_test['Sex'] == 'male', 'Sex'] = 0
data_test.loc[data_test['Sex'] == 'female', 'Sex'] = 1

#缺失值用最多的S进行填充
data_test.loc[data_test['Embarked'] == 'S', 'Embarked'] = 0
data_test.loc[data_test['Embarked'] == 'C', 'Embarked'] = 1
data_test.loc[data_test['Embarked'] == 'Q', 'Embarked'] = 2

test_features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
#构造测试集的Survived列
data_test['Survived'] = -1

test_predictors = data_test[test_features]
data_test['Survived'] = LogRegAlg.predict(test_predictors)
```

```python
data_test.head()
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/titanic/figure/data_test.head().png?raw=true)

## 7.使用随机森林算法
```python
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

predictors = ['Pclass','Sex,'Age','SibSp','Parch','Fare','Embarked']

#10可决策树，停止条件：样本个数为2，叶子节点数为1
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

kf = model_selection.KFold(n_splits=3, shuffle=Fasle, random_state=1)

scores = model_selection.cross_val_score(alg, data_train[predictors], data_train['Survived'], cv=kf)

print(scores)
print(scores.mean())
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/titanic/figure/10rf_scores.png?raw=true)

增加决策树个数到30，交叉验证方法采用10折交叉验证。

```python
#30棵决策树，停止条件：样本个数为2，叶子节点个数为1。
alg = RandomForestClassifier(random_state=1, n_estimators=30, min_samples_split=2, min_samples_leaf=1)

kf = model_selection.KFold(n_splits=10, shuffle=False, random_state=1)

scores = model_selection(alg, data_test[predictors], data_test['Survived'], cv=kf)

print(scores)
print(scores.mean())
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/titanic/figure/30rf_scores.png?raw=true)
