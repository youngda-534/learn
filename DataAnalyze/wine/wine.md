# 对红酒数据集的分析
## 简介
红酒通用数据集中包含1599个样本，11个红酒的理化性质，以及红酒的品质（0到10，0最差，10最好）。本实验主要是为了熟悉常见python包和数据可视化。主要内容分为：单变量、双变量和多变量分析。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#颜色
color = sns.color_palette()
#数据输出精度
pd.set_option('precision', 3)
```
读取数据

```python
df = pd.read_csv('winequality-red.csv', sep=';')
df.head(5)
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/df.head(5).png?raw=true)

```python
df.info()
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/df.info.png?raw=true)

## 单变量分析
```python
#简单的数据统计
df.describe()
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/df.describe.png?raw=true)

```python
#设置图的风格
plt.style.use('ggplot')
```

```python
colnm = df.columns.tolist()
fig = plt.figure(figsize=(10,6))

for i in range(12):
	plt.subplot(2,6,i+1)
	sns.boxplot(df[colnm[i], orient="v", width=0.5, color=color[0])
	plt.ylabel(colnm[i], fontsize=12)
plt.tight_layout()
```

![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/Unicariate%20Boxplots.png?raw=true)

```python
colnm = df.columns.tolist()
plt.figure(figsize=(10,8))

for i in range(12):
	plt.subplot(4,3,i+1)
	df[colnm[i]].hist(bins=100, color=color[0])
	plt.xlabel(colnm[i], fontsize=12)
	plt.ylabel('Frequency')
plt.tight_layout()
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/Unicariate%20Histograms.png?raw=true)

### 品质
这个数据集的目的使研究红酒品质和理化性质之间的关系。品质的评价范围是0-10，这个数据集中范围是3到8，大部分集中在5或6。
### 酸度相关的特征
本数据集有7个酸度相关的特征：fixed acidity, volatile acidity, citric acid, free sulfur dioxide, total sulfur dioxide, sulphates, pH。前6个特征都与红酒的pH有关。pH是在对数的尺度，下面对前6个特征取对数然后作histogram。另外，pH值主要是与fixed acidity有关，fixed acidity比volatile acidity和citric acid高1到2个数量级，比free sulfur dioxide, total sulfur dioxide, sulphates高3个数量级。一个新特征total acid来自于前三个特征的和。
	
```python
acidityFeat = ['fixed acidity', 'volatile acidity', 'crtric acid', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates']

plt.figure(figsize = (10, 4))

for i in range(6):
	ax = plt.subplot(2,3,i+1)
	v = np.log10(np.clip(df[acidityFeat[i]].values, a_min = 0.001, a_max = None))
	# np.clip()用于剪辑数组，给定间隔，间隔外的值将被剪切到间隔边缘。
	plt.hist(v, bins=50, color = color[0])
	plt.xlabel('log(' + acidityFeat[i] + ')', fontsize = 12)
	plt.ylabel('Frequency')
```

![Acidity Features in log10 Scale](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/Acidity%20Features%20in%20log10%20Scale.png?raw=true)

```python
plt.figure(figsize=(6,3))

bins = 10**(np.linspace(-2, 2))
#np.linspace()在指定的间隔内返回均匀间隔的数字。num默认为50。
plt.hist(df['fixed acidity'], bins = bins, edgecolor = 'k', label = 'Fixed Acidity')
plt.hist(df['volatile acidity'], bins = bins, edgecolor = 'k', label = 'Volatile Acidity')
plt.hist(df['citric acid'], bins = bins, edgecolor = 'k', alpha = 0.8, label = 'Citric Acid')
plt.xscale('log')
#对x轴采用对数刻度
plt.xlabel('Acid Concentration (g/dm^3)')
plt.ylabel('Frequency')
plt.title('Histogram of Acid Concentration')
plt.legend()
#plt.legend()显示图例位置，默认为upper left。
plt.tight_layout()
```
![Histogram of Acid Concentration](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/Histogram%20of%20Acid%20Concentration.png?raw=true)

```python
# 总酸度
df['total acid'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
```
```python
plt.figure(figsize = (8,3))

plt.subplot(121)
plt.hist(df['total acid'], bins=50, color=color[0])
plt.xlabel('total acid')
plt.ylabel('Frequency')
plt.subplot(122)
plt.hist(np.log(df['total acid']), bins = 50, color = color[0])
plt.xlabel('log(total acid)')
plt.ylabel('Frequency')
plt.tight_layout()
```
![Total Acid Histogram](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/Total%20Acid%20Histogram.png?raw=true)

### 甜度
Residual sugar与酒的甜度相关，通常用来区别各种红酒， 干红（<= 4 g/L），半干（4-12 g/L），半甜（12-45 g/L）。这个数据集中，主要为干红，没有甜葡萄酒。

```python
# Residual sugar
df['sweetness'] = pd.cut(df['residual sugar'], bins = [0, 4, 12, 45], labels=["dry", "medium dry", "semi-sweet"])
#pd.cut()切分数据集，将剩余糖数值位于0-4之间的标记为'dry'，4-12之间的标记为'medium dry'，12-45之间的标记为'semi-sweet'。
```
```python
plt.figure(figsize = (5,3))
df['sweetness'].value_counts().plot(kind = 'bar', color = color[0])
plt.xticks(rotation=0)
plt.xlabel('sweetness', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.tight_layout()
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/Sweetness.png?raw=true)

## 双变量分析
### 红酒品质和理化特征的关系
由下文两图可以看出：

* 品质好的就有更高的柠檬酸，硫酸盐和酒精度数。硫酸盐（硫酸钙）的加入通常是调整酒的酸度的。其中酒精度数和品质的相关性最高。
* 品质好的酒有较低的挥发性酸类，密度和pH。
* 残留糖分、氯离子、二氧化硫对酒的品质影响不大。

```python
sns.set_style('ticks')
sns.set_context("notebook", font_scale=1.1)

colnm = df.columns.tolist()[:11] + ['total acid']
plt.figure(figsize = (10,8))

for i in range(12):
	plt.subplot(4,3,i+1)
	sns.boxplot(x = 'quality', y = colnm[i], data = df, color = color[1], width = 0.6)
	plt.ylabel(colnm[i], fontsize=12)
plt.tight_layout()
#自动调整子图参数，使之填充整个图像区域。
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/Physicochemical%20Properties%20and%20Wine%20Quality%20by%20Boxplot.png?raw=true)

```python
sns.set_style("dark")

plt.figure(figsize = (10,8))
colnm = df.columns.tolist()[:11] + ['total acid', 'quality']
mcorr = df[colnm].corr()
# 生成dataframe特征之间的相关系数。
mask = np.zeors_like(mcorr, dtype=np.bool)
# 返回与给定数组具有相同形状和类型的零数组。
mask[np.triu_indices_from(mask)] = True
# np.triu_indices_from()返回矩阵的上三角矩阵。
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# 生成一个调色板，并返回一个colormap对象。
# colormap:数值到颜色空间的映射。
g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f') 
# 生成一个热力图，mask为True的单元格直接被屏蔽，annot为True则将数值写在单元格上，fmt设置注释时要使用的字符串格式代码。
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/Pairwise%20Correlation%20Plot.png?raw=true)

### 密度和酒精浓度
密度和酒精浓度是相关的，物理上，二者不是线性关系。密度还与酒中其他物质的含量有关，但是关系很小。

```python
# style
sns.set_style('ticks')
sns.set_context("notebook", font_scale=1.4)

# plot figure
plt.figure(figsize = (6,4))
sns.regplot(x='density', y='alcohol', data=df, scatter_kws={'s':10}, color=color[1])
plt.xlim(0.989, 1.005)
plt.ylim(7, 16)
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/Density%20vs%20Alcohol.png?raw=true)

### 酸性物质含量和pH
pH和非挥发性酸性物质有-0.683的相关性。因为非挥发性酸性物质的含量远远高于其他酸性物质，总酸性物质（total acid）这个特征并没有太多意义。

```python
acidity_related = ['fixed acidity', 'volatile acidity', 'total sulfur dioxide', 'sulphates', 'total acid']

plt.figure(figsize = (10,6))

for i in range(5):
	plt.subplot(2,3,i+1)
	sns.replot(x='pH', y=acidity_related[i], data=df, scatter_kws={'s':10}, color = color[1])
plt.tight_layout()
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/pH%20vs%20acid.png?raw=true)

## 多变量分析
与品质相关性最高的三个特征是酒精浓度、挥发性酸度和柠檬酸。下图中显示了酒精浓度、挥发性酸和品质的关系。
### 酒精浓度、挥发性酸和品质
对于好久（7、8）以及差酒（3、4），关系很明显。但是对于中等酒（5、6），酒精浓度的挥发性酸度有很大程度的交叉。

```python
plt.style.use('ggplot')

sns.lmplot(x = 'alcohol', y = 'volatile acidity', hue = 'quality', data = df, fit_reg = False, scatter_kws={'s':10}, size=5)
# 若fit_reg为True将估计并绘制x与y变量相关的回归模型。
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/Scatter%20Plots%20of%20Alcohol,%20Volatile%20Acid%20and%20Quality.png?raw=true)

```python
sns.lmplot(x='alcohol', y='volatile acidity', col='quality', hue='quality', data=df, fit_reg=False, size=3, aspect=0.9, col_wrap=3, scatter_kws={'s':20})
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/Scatter%20Plots%20of%20Alcohol,%20Volatile%20Acid%20and%20Quality%202.png?raw=true)

### pH、非挥发性酸和柠檬酸
pH和非挥发性酸以及柠檬酸有相关性。整体趋势也很合理，即浓度越高、pH越低。

```python
#style
sns.set_style('ticks')
sns.set_context("notebook", font_scale=1.4)

plt.figure(figsize=(6,5))
cm = plt.cm.get_cmap('RdBu')
sc = plt.scatter(df['fixed acidity'], df['citric acid'], c=df['pH'], vmin=2.5, vmax=4, s=15, cmap=cm)
bar = plt.colorbar(sc)
bar.set_label('pH', rotation = 0)
plt.xlabel('fixed acidity')
plt.ylabel('citric acid')
plt.xlim(4,18)
plt.ylim(0,1)
```
![](https://github.com/youngda-534/machine-learning/blob/master/DataAnalyze/wine/figure/pH%20with%20Fixed%20Acidity%20and%20Citric%20Acid.png?raw=true)

## 总结
整体而言，红酒的品质主要与酒精浓度、挥发性酸和柠檬酸有关。