## Matplotlib
### 1. Matplotlib.pyplot简单介绍
* Matplotlib是一些命令行风格函数的集合，使得matplotlib以类似于MATLAB的方式工作。每个pyplot函数对一幅图片做出一些改动：比如创建新图片，在图片上创建新的作图区域，在一个作图区域内画直线，给图片添加标签等。根据不同的风格，Matplotlib图又可以分为很多类型，分别为：条形图、直方图、散点图、面积图、饼图。同样也可以将多个图合并为一张。
* Basemap：一个地图绘图工具包，包含各种地图投影，海岸线和政治边界。
* Cartopy：一个映射库，具有面向对象的地图投影定义，以及任意点、线、多边形和图像转换功能。
* Excel tool：Matplotlib提供与Microsoft Excel交换数据的实用程序。
* Mplot3d：用于三维图。
* Natgrid：natgrid库的接口，用于对间隔数据进行不规则网格化。
![Matplotlib创建的部分图](http://aliyuntianchipublic.cn-hangzhou.oss-pub.aliyun-inc.com/public/files/image/null/1533539350136_lJDOK3SCeq.jpg)

### 2. Matplotlib生成的一些简单图形
```	python
#导入Matplotlib库
import matplotlib.pyplot as plt
#画布上画图
plt.plot([1,2,3],[4,5,3])
plt.show()
```
![](/Users/liyongda/desktop/code/matplotlib/plot.png)

* 使用三行代码，就可以创建出一个简单的图形。同时，我们还可以为其添加标题、标签，以使得图片更易读取。

```python
import matplotlib.pyplot as plt
x = [3,4,5]
y = [2,16,9]
plt.plot(x,y)
#图片的标题
plt.title('Image')
#坐标轴Y轴
plt.ylabel('Y axis')
#坐标轴X轴
plt.xlabel('X axis')
plt.show()
```
![](/Users/liyongda/desktop/code/matplotlib/plot2.png)

* 也可以使用一些特定的方法来改变指定线条的粗细和颜色，增加网格，添加样式等。此时需要从matplotlib库中导入样式包，然后使用样式函数。

```python
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
x = [5,8,10]
y = [2,10,4]
x2 = [6,8,12]
y2 = [1,8,5]
plt.plot(x,y,'g',label='line one', linewidth=5)
plt.plot(x2,y2,'r',label='line two', linewidth=5)
plt.title('Style')
plt.ylabel('Y axis')
plt.xlabel('X axis')
#设置图例位置
plt.legend()
plt.grid(True, color='k')
plt.show()
```
![](/Users/liyongda/desktop/code/matplotlib/style.png)

### 3. 条形图
* 条形图主要用来比较不同类别之间的数据，也可以观察某个变量在一段时间内的变化。可以水平或者垂直表示，条形越长，则表示的价值越大。
	
```python
import matplotlib.pyplot as plt
plt.bar([0.25,1.25,2.25,3.25,4.25],[50,40,70,80,20],label="CocaCola", color='b',width=.5)
plt.bar([.75,1.75,2.75,3.75,4.75],[80,20,20,50,60],label="Pepsi",color='r',width=.5)
plt.legend()
plt.xlabel('Days')
plt.ylabel('Sales')
plt.title('Information')
plt.show()
```
![](/Users/liyongda/desktop/code/matplotlib/bar.png)

### 4. 直方图
* 直方图用于显示分布，而条形图用于比较不同的实体。直方图对于阵列或者长列表有很好的效果。

```python
import matplotlib.pyplot as plt
population_age = [22,55,62,45,21,22,34,42,42,4,2,102,95,85,55,110,120,70,65,55,111,115,80,75,65,54,44,43,42,48]
bins = [0,10,20,30,40,50,60,70,80,90,100]
plt.hist(population_age, bins, histtype='bar', color='b', rwidth=0.8)
plt.xlabel('age groups')
plt.ylabel('Number of people')
plt.title('Histogram')
plt.show()
```
![](/Users/liyongda/desktop/code/matplotlib/histogram.png)

### 5. 散点图
* 散点图经常被用来比较变量。数据显示为点的集合。每个点具有一个变量的值，该变量确定水平轴上的位置，另一个变量的值确定垂直轴上的位置。

```python
import matplotlib.pyplot as plt
x = [1,1.5,2,2.5,3,3.5,3.6]
y = [7.5,8,8.5,9,9.5,10,10.5]
x1 = [8,8.5,9,9.5,10,10.5,11]
y1 = [3,3.5,3.7,4,4.5,5,5.2]
plt.scatter(x,y,label='high income low saving', color='r')
plt.scatter(x1,y1,label='low income high savings',color='b')
plt.xlabel('saving*1000')
plt.ylabel('income*1000')
plt.title('Scatter Plot')
plt.legend()
plt.show()
```
![](/Users/liyongda/desktop/code/matplotlib/point2.png)

### 6. 面积图
* 面积图又被称为堆栈图。可用于跟踪构成一个整体类别的两个或多个相关组随时间的变化情况。例如，我们每天24个小时，分别花费在睡觉、吃饭、工作和玩耍上面的时间和比重。

```python
import matplotlib.pyplot as plt
days = [1,2,3,4,5]
sleeping = [7,8,6,11,7]
eating = [2,3,4,3,2]
working = [7,8,7,2,2]
playing = [8,5,7,8,13]
plt.plot([],[],color='m',label='Sleeping',linewidth=5)
plt.plot([],[],color='c','label='Eating',linewidth=5)
plt.plot([],[],color='r',label='Working',linewidth=5)
plt.plot([],[],color='k',label='Playing',linewidth=5) 
plt.stackplot(days, sleeping, eating, working, playing, colors=['m','c','r','k'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Stack Plot')
plt.legend()
plt.show()
```
![](/Users/liyongda/desktop/code/matplotlib/stack.png)

* 面积图或堆栈图用于显示不同属性随时间变化的趋势。

### 7. 饼图
* 饼图指圆饼图，被分解成段。用于显示百分比或比例数据，其中每个饼图片代表一个类别。

```python
import matplotlib.pyplot as plt
days = [1,2,3,4,5]
sleeping = [7,8,6,11,7]
eating = [2,3,4,3,2]
working = [7,8,7,2,2]
playing = [8,5,7,8,13]
slices = [7,2,2,13]
activities = ['sleeping','eating','working','playing']
cols = ['c','m','r','b']
plt.pie(slices, labels=activities, colors=cols, startangle=90, shadow=True, explode=(0,0.1,0,0), autopct='%1.1f%%')
plt.title('Pie Plot')
plt.show()
```
![](/Users/liyongda/desktop/code/matplotlib/pie.png)

* 上图中将圆圈划分为4个扇区或切片，分别代表相应的类别（吃饭、睡觉、玩耍和工作）以及它们所占的百分比。饼图切片的计算会自动完成。

### 8. 多图合并
* 下面将讨论如何将多种类型的图进行合并，同时处理和展示。

```python
import numpy as np
import matplotlib.pyplot as plt
def f(t):
	return np.exp(-t) * np.cos(2*np.pi*t)
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
plt.plot(t1, f(t1), 'bo', t2, f(t2))
plt.subplot(222)
plt.plot(t2, np.cos(2*np.pi*t2))
plt.show()
```
![](/Users/liyongda/desktop/code/matplotlib/multi.png)



	
	