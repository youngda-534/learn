import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


color = sns.color_palette()
pd.set_option('precision', 3)


df = pd.read_csv('winequality-red.csv', sep=';')

plt.style.use('ggplot')

acidityFeat = ['fixed acidity', 'volatile acidity', 'citric acid',
               'free sulfur dioxide', 'total sulfur dioxide', 'sulphates']
			   
df['total acid'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']

'''
plt.figure(figsize = (8,3))

plt.subplot(121)
plt.hist(df['total acid'], bins = 50, color = color[0])
plt.xlabel('total acid')
plt.ylabel('Frequency')

plt.subplot(122)
plt.hist(np.log(df['total acid']), bins=50, color=color[0])
plt.xlabel('log(total acid)')
plt.ylabel('Frequency')

plt.tight_layout()

plt.show()
'''

'''
df['sweetness'] = pd.cut(df['residual sugar'], bins=[0,4,12,45],labels=["dry", "medium dry", "semi_sweet"]) #将序列离散化，‘residual sugar’值在0~4的归为‘dry’，4~12的归为‘medium dry’，12~45的归为‘semi_sweet’。

plt.figure(figsize=(5,3))
df['sweetness'].value_counts().plot(kind='bar',color=color[0])
plt.xticks(rotation=0)
plt.xlabel('sweetness', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()
'''

sns.set_style('ticks')
sns.set_context("notebook", font_scale=1.4)

plt.figure(figsize=(6,5))
cm = plt.cm.get_cmap('RdBu')
sc = plt.scatter(df['fixed acidity'], df['citric acid'], c=df['pH'], vmin=2.6, vmax=4, s=15, cmap=cm)
bar = plt.colorbar(sc)
bar.set_label('pH', rotation=0)
plt.xlabel('fixed acidity')
plt.ylabel('citric acid')
plt.xlim(4,18)
plt.ylim(0,1)
plt.show()
